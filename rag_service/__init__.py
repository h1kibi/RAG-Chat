"""Small, transport-neutral public API for the standalone RAG service."""
from __future__ import annotations

from .config import RagConfig
from .http_client import RagHttpClient
from .models import RetrievalRequest, RetrievalResponse, SearchResult
from .service import RagService

__all__ = [
    "RagConfig",
    "RagHttpClient",
    "RagService",
    "RetrievalRequest",
    "RetrievalResponse",
    "SearchResult",
]
