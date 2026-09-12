from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, List

import httpx

from .errors import RagEmbeddingError


@dataclass(frozen=True)
class OllamaEmbeddingClient:
    """Minimal Ollama client used by the standalone RAG process.

    This deliberately uses Ollama's native ``/api/embed`` endpoint instead of
    importing the application's model factory. The standalone service can
    therefore run without the Agent or Chatchat packages being initialized.

    Transient provider failures (HTTP 5xx, connection errors) are retried
    with bounded backoff before falling back to provider-safe text variants.
    """

    base_url: str = "http://127.0.0.1:11434"
    model: str = "bge-m3"
    timeout: float = 120.0
    max_retries: int = 2
    retry_delay: float = 1.0
    keep_alive: str = "30m"
    _client: Any = field(default=None, compare=False, repr=False, init=False)

    def embed_query(self, text: str) -> List[float]:
        candidates = [text]
        safe_text = text.replace(".c_str()", " c_str ")
        if safe_text != text:
            candidates.append(safe_text)
        ascii_text = " ".join(text.encode("ascii", errors="ignore").decode("ascii").split())
        if ascii_text and ascii_text not in candidates:
            candidates.append(ascii_text)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                return self._request(candidate)
            except Exception as exc:  # try the bounded provider-safe candidates
                last_error = exc
        raise RagEmbeddingError(
            f"embedding provider unreachable at {self.base_url} for model "
            f"'{self.model}': {type(last_error).__name__}. Embeddings are required for "
            "query mode only — empty-query browse and document listings still work. "
            "Check that Ollama is running and that the model is pulled "
            f"(`ollama pull {self.model}`)."
        ) from last_error

    def _request(self, text: str) -> List[float]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(text)
            except (httpx.HTTPError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status >= 500
                if isinstance(exc, ValueError):
                    retryable = False
                if attempt >= self.max_retries or not retryable:
                    raise
                last_error = exc
                time.sleep(self.retry_delay * (2**attempt))
        raise last_error  # pragma: no cover - defensive

    def _request_once(self, text: str) -> List[float]:
        # httpx bounds connect/read/write/pool, whereas requests' timeout leaves
        # a stalled upload unbounded: if the provider restarts mid-request the
        # call can hang forever and the caller never sees an error to retry.
        import httpx

        response = self._http_client().post(
            f"{self.base_url.rstrip('/')}/api/embed",
            json={"model": self.model, "input": [text], "keep_alive": self.keep_alive},
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise ValueError("Ollama returned an invalid embedding response")
        vector = embeddings[0]
        if not vector or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("Ollama returned an empty or non-finite embedding")
        return [float(value) for value in vector]

    def _http_client(self):
        if self._client is None:
            import httpx

            object.__setattr__(
                self,
                "_client",
                httpx.Client(
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=float(self.timeout),
                        write=float(self.timeout),
                        pool=10.0,
                    ),
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                ),
            )
        return self._client
    def close(self) -> None:
        """Close the pooled HTTP connection used by this embedder."""
        client = self._client
        if client is not None:
            client.close()
            object.__setattr__(self, "_client", None)
