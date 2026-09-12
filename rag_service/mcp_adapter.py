from __future__ import annotations

from typing import Any

from rag_service.models import RetrievalRequest
from rag_service.service import RagService


def register_mcp_tool(server: Any, service: RagService, *, name: str = "search_knowledge_base") -> None:
    """Register the shared read-only service in an existing MCP server.

    The recommended standalone CTF entry point is ``rag_service.mcp_server``
    with the ``ctf_rag`` tool. This helper is only for applications that already
    own an MCP server and want an in-process generic retrieval tool.
    """
    @server.tool(name=name)
    async def search_knowledge_base(
        query: str,
        knowledge_base: str = "cybersec",
        top_k: int = 5,
        limit: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        response = service.search(
            RetrievalRequest(
                query=query,
                knowledge_base=knowledge_base,
                top_k=top_k,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters or {},
                cursor=cursor,
            )
        )
        return response.model_dump()
