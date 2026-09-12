"""FastAPI application serving the offline chat UI and its API.

Routes are intentionally few: the page, one status endpoint, and one streamed
chat endpoint. Anything that mutates the knowledge base stays out of this
process — the agent only reads.
"""
from __future__ import annotations

import asyncio
import json
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


async def _rag_status(config: AgentConfig) -> Dict[str, Any]:
    """Report whether the local index is usable, without failing the status page."""
    if not config.rag_enabled:
        return {"enabled": False}

    def _inspect() -> Dict[str, Any]:
        from rag_service import RagConfig
        from rag_service.backends.faiss import FaissBackend, describe_status

        rag_config = RagConfig.from_environment()
        backend = FaissBackend(rag_config)
        return {
            "enabled": True,
            "knowledge_base": config.rag_knowledge_base or rag_config.default_knowledge_base,
            "status": describe_status(backend.status()),
        }

    try:
        return await asyncio.to_thread(_inspect)
    except Exception as exc:  # noqa: BLE001 - status must degrade, not raise
        return {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}


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
