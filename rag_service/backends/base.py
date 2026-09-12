from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag_service.models import RetrievalRequest, SearchResult


class RetrievalBackend(ABC):
    """Minimal backend interface used by the standalone service."""

    name = "backend"

    @abstractmethod
    def search(self, request: RetrievalRequest) -> List[SearchResult]:
        raise NotImplementedError
