"""Register the shared retrieval service in an existing MCP server.

NOTE: this module deliberately does **not** use ``from __future__ import
annotations``. The MCP SDK builds a tool from a function by walking
``inspect.signature(fn)`` and calling ``issubclass(param.annotation, Context)``
on each parameter that has no generic origin. With postponed evaluation every
annotation is a *string*, so that call raises
``TypeError: issubclass() arg 1 must be a class`` and registration fails before
the server ever starts. Don't add the future import back.

``rag_service.mcp_server`` (the standalone ``ctf_rag`` entry point) is unaffected
for the same reason: it never postponed its annotations.
"""
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
        from pydantic import ValidationError

        from rag_service.models import describe_validation_error

        try:
            request = RetrievalRequest(
                query=query,
                knowledge_base=knowledge_base,
                top_k=top_k,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters or {},
                cursor=cursor,
            )
        except ValidationError as exc:
            # Same boundary rule as ctf_rag, the OpenAI adapter, the LangChain
            # tool and the HTTP handler: an MCP caller gets the field and the
            # reason, not pydantic's dump with an errors.pydantic.dev URL.
            raise ValueError(
                f"invalid search_knowledge_base arguments: {describe_validation_error(exc)}"
            ) from None
        return service.search(request).model_dump()
