"""FastAPI application serving the offline chat UI and its API.

Routes are intentionally few: the page, one status endpoint, and one streamed
chat endpoint. Anything that mutates the knowledge base stays out of this
process — the agent only reads.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent_service.chat import ChatTurn, run_turn
from agent_service.config import AgentConfig
from agent_service.llm import probe

STATIC_DIR = Path(__file__).resolve().parent / "static"

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


_rag_status_cache: Dict[tuple, Dict[str, Any]] = {}
_rag_status_lock = asyncio.Lock()


def _rag_status_key(config: AgentConfig) -> tuple:
    """Identity of the retrieval deployment a probe result describes.

    Keyed rather than a single global: two `create_app` instances in one process
    (tests, or a supervisor hosting more than one configuration) must not answer
    for each other's knowledge base.
    """
    return (
        config.rag_enabled,
        config.rag_knowledge_base,
        os.getenv("RAG_KB_ROOT") or os.getenv("KB_ROOT_PATH") or "",
        os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
        os.getenv("RAG_SERVICE_URL", ""),
    )


async def _rag_status(config: AgentConfig) -> Dict[str, Any]:
    """Report whether retrieval is usable, without failing the status page.

    Reporting has to match what a query will actually do:

    - Inspect the knowledge base the agent will *use*, not the retrieval
      module's default. Otherwise the row count of one KB is displayed under the
      name of another.
    - Treat an index that reports itself unavailable as a failure. `status()`
      returns the failure as a value (`dense_path="unavailable"`), so a naive
      reader turns "no index at all" into a green, exit-0 readiness signal.
    - Ask the embedding provider too. A dead provider leaves every query on the
      lexical fallback, which is a different (degraded) state from ready.
    - When `RAG_SERVICE_URL` is configured the index deliberately lives in
      another process: ask that service instead of opening a local index the
      operator moved away on purpose.

    A successful result is memoized per deployment (the probe opens the
    converted index, and the UI polls status on every page load); failures are
    re-checked on every call, since the index may simply not be built yet.
    """
    if not config.rag_enabled:
        return {"enabled": False}

    key = _rag_status_key(config)
    cached = _rag_status_cache.get(key)
    if cached is not None:
        return cached

    async with _rag_status_lock:
        cached = _rag_status_cache.get(key)
        if cached is not None:
            return cached
        try:
            if os.getenv("RAG_SERVICE_URL"):
                result = await asyncio.to_thread(_probe_remote_service, config)
            else:
                result = await asyncio.to_thread(_probe_local_index, config)
        except Exception as exc:  # noqa: BLE001 - status must degrade, not raise
            return {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}
        if result.get("error"):
            # Failure-shaped result: reported every call, never memoized.
            return result
        _rag_status_cache[key] = result
        return result


def _probe_local_index(config: AgentConfig) -> Dict[str, Any]:
    from rag_service import RagConfig
    from rag_service.backends.faiss import FaissBackend, describe_status
    from rag_service.errors import RagIndexNotReadyError

    rag_config = RagConfig.from_environment()
    knowledge_base = config.rag_knowledge_base or rag_config.default_knowledge_base
    # Check the gate before inspecting: `status()` reports failures as a value
    # that keeps only the exception class name, which would surface an
    # allow-list rejection as a breadcrumb about build_cosine.
    rag_config.require_allowed(knowledge_base)
    backend = FaissBackend(rag_config)
    try:
        status = backend.status(knowledge_base)
    finally:
        backend.close()
    if status.get("error") or status.get("dense_path") == "unavailable":
        raise RagIndexNotReadyError(
            f"knowledge base '{knowledge_base}' index is not readable "
            f"({status.get('error', 'unavailable')}); run "
            "`python -m rag_service.build_cosine`"
        )
    return {
        "enabled": True,
        "mode": "in-process",
        "knowledge_base": knowledge_base,
        "status": describe_status(status),
        # A dead embedding provider degrades every query to lexical-only, which
        # is not "ready". Report it here rather than only in warning text on the
        # first query.
        "embedding": _embedding_status(rag_config),
    }


def _embedding_status(rag_config: Any) -> str:
    """Cheap readiness of the embedding provider, for the status page.

    Deliberately NOT a real embedding call: the MCP server can afford one at
    startup, but `/api/health` runs on every page load and a cold embed costs
    seconds (measured ~24 s for the first check). Listing the provider's models
    answers the question that matters -- reachable, and does it have the model
    the index was built with -- in tens of milliseconds.
    """
    import httpx

    base = rag_config.ollama_base_url.rstrip("/")
    model = rag_config.embedding_model or "bge-m3"
    try:
        response = httpx.get(
            f"{base}/api/tags",
            timeout=min(float(rag_config.embedding_timeout), 5.0),
        )
    except Exception as exc:  # noqa: BLE001 - status must degrade, not raise
        return f"unavailable ({type(exc).__name__})"
    if response.status_code >= 400:
        return f"unavailable (HTTP {response.status_code})"
    try:
        names = {item.get("name", "") for item in response.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return "unknown (unexpected response)"
    # Ollama reports "bge-m3:latest" for a model pulled as "bge-m3".
    if any(name == model or name.split(":")[0] == model.split(":")[0] for name in names):
        return f"ready ({model})"
    return f"unavailable ({model} not in the provider's model list)"


def _probe_remote_service(config: AgentConfig) -> Dict[str, Any]:
    import httpx

    base = os.environ["RAG_SERVICE_URL"].rstrip("/")
    headers = {}
    token = os.getenv("RAG_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = httpx.get(f"{base}/v1/rag/health", headers=headers, timeout=10.0)
    if response.status_code == 401:
        return {
            "enabled": True,
            "mode": "http",
            "url": base,
            "error": "HTTP 401：RAG_API_TOKEN 被拒绝",
        }
    if response.status_code >= 400:
        return {
            "enabled": True,
            "mode": "http",
            "url": base,
            "error": f"HTTP {response.status_code}",
        }
    payload = response.json()
    capabilities = payload.get("capabilities") or {}
    return {
        "enabled": True,
        "mode": "http",
        "url": base,
        # The remote service does not know which KB this agent asks for, and it
        # is the agent's choice that a query will use.
        "knowledge_base": config.rag_knowledge_base
        or os.getenv("RAG_DEFAULT_KNOWLEDGE_BASE", ""),
        "status": describe_remote_status(capabilities),
    }


def describe_remote_status(capabilities: Dict[str, Any]) -> str:
    if not capabilities:
        return "remote service reachable (no capability detail)"
    if capabilities.get("error"):
        return f"remote service degraded: {capabilities['error']}"
    rows = capabilities.get("rows")
    path = capabilities.get("dense_path", "unknown")
    suffix = f" rows={rows}" if rows is not None else ""
    return f"remote dense_path={path}{suffix}"


def create_app(config: AgentConfig) -> FastAPI:
    app = FastAPI(title="Local Knowledge Base Agent", version="0.1.0")

    def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if config.api_token is None:
            return
        if authorization != f"Bearer {config.api_token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/config")
    async def public_config() -> Dict[str, Any]:
        payload = config.as_public_dict()
        payload["auth_required"] = config.api_token is not None
        return payload

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        checks = await asyncio.gather(
            *(probe(provider) for provider in config.providers), return_exceptions=True
        )
        providers = []
        for provider, outcome in zip(config.providers, checks):
            if isinstance(outcome, BaseException):
                ok, detail = False, f"{type(outcome).__name__}: {outcome}"
            else:
                ok, detail = outcome
            providers.append(
                {
                    "id": provider.id,
                    "label": provider.label,
                    "offline": provider.offline,
                    "ok": ok,
                    "detail": detail,
                    "models": list(provider.models),
                }
            )
        return {
            "status": "ok",
            "providers": providers,
            "rag": await _rag_status(config),
        }

    @app.post("/api/chat", dependencies=[Depends(authorize)])
    async def chat(turn: ChatTurn, request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            try:
                async for event in run_turn(config, turn):
                    if await request.is_disconnected():
                        return
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - the stream must always terminate cleanly
                yield _sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
