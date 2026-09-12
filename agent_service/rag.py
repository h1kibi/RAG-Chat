"""RAG capability bridge used by the agent and by external callers.

Set ``RAG_SERVICE_URL`` to keep FAISS artifacts out of the agent process. If
that variable is absent, the bridge opens the standalone backend in-process,
which requires ``RAG_KB_ROOT`` (or ``KB_ROOT_PATH``) to point at the
``knowledge_base`` directory.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Mapping

_service_instance = None
_service_lock = threading.Lock()


def _service():
    """Return the single in-process service for the active runtime configuration."""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            from rag_service import RagConfig, RagService
            from rag_service.backends.faiss import FaissBackend

            config = RagConfig.from_environment()
            _service_instance = RagService(config, FaissBackend(config))
        return _service_instance


def _http_timeout() -> float:
    """Read ``RAG_HTTP_TIMEOUT``, naming the variable when it does not parse.

    A bare ``float()`` reported `could not convert string to float: '30s'`,
    and the agent's degradation path relayed that verbatim as
    "检索不可用，已降级为纯对话" -- so a typo in this variable looked like a
    broken retrieval service rather than a configuration mistake. Every other
    numeric knob in the repository names itself; this one is read here rather
    than in a config module, which is how it was missed.
    """
    raw = os.getenv("RAG_HTTP_TIMEOUT", "120").strip()
    if not raw:
        return 120.0
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"RAG_HTTP_TIMEOUT must be a number, got {raw!r}") from exc


def _http_search(**kwargs: Any) -> dict:
    from rag_service.http_client import RagHttpClient

    client = RagHttpClient(
        os.environ["RAG_SERVICE_URL"],
        api_token=os.getenv("RAG_API_TOKEN") or None,
        timeout=_http_timeout(),
    )
    return client.search_dict(**kwargs)


def search_knowledge_base(
    query: str,
    knowledge_base: str = "",
    top_k: int = 5,
    limit: int | None = None,
    score_threshold: float | None = None,
    filters: Mapping[str, Any] | None = None,
    merge_neighbors: bool | None = None,
    lexical_weight: float | None = None,
    strip_images: bool | None = None,
    snippet_chars: int | None = None,
    cursor: str | None = None,
) -> dict:
    filter_values = dict(filters or {})
    overrides: dict[str, Any] = {}
    if limit is not None:
        overrides["limit"] = limit
    if score_threshold is not None:
        overrides["score_threshold"] = score_threshold
    if merge_neighbors is not None:
        overrides["merge_neighbors"] = merge_neighbors
    if lexical_weight is not None:
        overrides["lexical_weight"] = lexical_weight
    if strip_images is not None:
        overrides["strip_images"] = strip_images
    if snippet_chars is not None:
        overrides["snippet_chars"] = snippet_chars
    if cursor:
        overrides["cursor"] = cursor
    if os.getenv("RAG_SERVICE_URL"):
        kwargs = {
            "query": query,
            "knowledge_base": knowledge_base,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        if filter_values:
            kwargs["filters"] = filter_values
        kwargs.update(overrides)
        return _http_search(**kwargs)

    from rag_service import RetrievalRequest

    service = _service()
    return service.search(
        RetrievalRequest(
            query=query,
            knowledge_base=knowledge_base or service.config.default_knowledge_base,
            top_k=top_k,
            filters=filter_values,
            **overrides,
        )
    ).model_dump()
