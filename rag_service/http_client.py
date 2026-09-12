from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from .models import RetrievalRequest, RetrievalResponse


class RagHttpClient:
    """Small synchronous HTTP client for agents running in another process."""

    def __init__(self, base_url: str, *, api_token: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout

    def search(self, request: RetrievalRequest) -> RetrievalResponse:
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        response = httpx.post(
            f"{self.base_url}/v1/rag/search",
            json=request.model_dump(),
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return RetrievalResponse.model_validate(response.json())

    def search_dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.search(RetrievalRequest.model_validate(kwargs)).model_dump()
