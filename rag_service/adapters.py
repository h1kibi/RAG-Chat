from __future__ import annotations

from typing import Any, Dict, Optional

from rag_service.models import RetrievalRequest
from rag_service.service import RagService


def create_langchain_tool(service: RagService, *, name: str = "search_knowledge_base", description: Optional[str] = None):
    """Create a LangChain-compatible tool without coupling the service to an LLM."""
    from langchain_core.tools import StructuredTool

    def search_knowledge_base(
        query: str = "",
        knowledge_base: str = "",
        top_k: int = 5,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        merge_neighbors: Optional[bool] = None,
        lexical_weight: Optional[float] = None,
        strip_images: Optional[bool] = None,
        snippet_chars: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        request = RetrievalRequest(
            query=query,
            knowledge_base=knowledge_base or service.config.default_knowledge_base,
            top_k=top_k,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters or {},
            merge_neighbors=merge_neighbors,
            lexical_weight=lexical_weight,
            strip_images=strip_images,
            snippet_chars=snippet_chars,
            cursor=cursor,
        )
        return service.search(request).as_tool_text()

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name=name,
        description=description
        or "Search the approved read-only knowledge base and return cited evidence.",
    )


def create_openai_tool_schema(*, name: str = "search_knowledge_base", description: Optional[str] = None) -> Dict[str, Any]:
    """Return a provider-neutral OpenAI function/tool declaration."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description
            or "Search the approved read-only knowledge base and return cited evidence.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 0, "maxLength": 8000},
                    "knowledge_base": {"type": "string", "default": "cybersec"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "browse listing page size; defaults to top_k and is ignored for queries",
                    },
                    "score_threshold": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.45},
                    "merge_neighbors": {"type": "boolean", "description": "default true; false returns single chunks"},
                    "lexical_weight": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.35},
                    "strip_images": {"type": "boolean", "description": "default true"},
                    "snippet_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50000,
                        "default": 800,
                        "description": "0 disables query-centered cropping; hard RAG_MAX_CONTENT_CHARS cap remains",
                    },
                    "cursor": {
                        "type": "string",
                        "default": "",
                        "description": "browse listing pagination token from the previous next_cursor",
                    },
                    "filters": {
                        "type": "object",
                        "additionalProperties": False,
                        "description": (
                            "category = first path segment of source, e.g. 14_ctf_wp, 15_butian, "
                            "09_hacktricks, 13_xianzhi; source_prefix = path prefix; "
                            "source = exact relative path; chunk_id = kb:row of one chunk to "
                            "verify a citation; year = year anywhere in the path "
                            "(incl. CVE numbers); exclude_source_prefix = drop sources under "
                            "this prefix"
                        ),
                        "properties": {
                            "category": {"type": "string", "minLength": 1},
                            "source_prefix": {"type": "string", "minLength": 1},
                            "source": {"type": "string", "minLength": 1},
                            "chunk_id": {"type": "string", "minLength": 1},
                            "year": {"type": "integer", "minimum": 1900, "maximum": 2100},
                            "exclude_source_prefix": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
                "required": [],
            },
        },
    }


def dispatch_openai_tool_call(service: RagService, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute validated arguments from an OpenAI-compatible tool call."""
    from pydantic import ValidationError

    from rag_service.models import describe_validation_error

    try:
        request = RetrievalRequest.model_validate(arguments)
    except ValidationError as exc:
        # An LLM caller should be told what to fix, not handed pydantic's dump
        # (which echoes any oversized value back into the model's context).
        raise ValueError(f"invalid arguments: {describe_validation_error(exc)}") from None
    response = service.search(request)
    return response.model_dump()
