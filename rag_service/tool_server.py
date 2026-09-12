from __future__ import annotations

import os
import sys

from rag_service.backends.faiss import FaissBackend, describe_status
from rag_service.config import RagConfig
from rag_service.http_api import create_app
from rag_service.service import RagService


def embedding_preflight(config) -> str:
    """Probe the embedding provider once and describe the outcome.

    Cold-starting an MCP server with a dead provider used to surface only on the
    first query, which reads as "the corpus has nothing". Reporting it at startup
    (and in health) makes the state visible before any retrieval happens.
    Browse never needs embeddings, so a failure here is a warning, not fatal.
    """
    from rag_service.embeddings import OllamaEmbeddingClient

    client = OllamaEmbeddingClient(
        base_url=config.ollama_base_url,
        model=config.embedding_model or "bge-m3",
        timeout=min(float(config.embedding_timeout), 15.0),
        max_retries=0,
    )
    try:
        client.embed_query("preflight")
    except Exception as exc:
        return f"embedding=unavailable ({type(exc).__name__}); query mode will degrade to lexical-only, browse unaffected"
    finally:
        client.close()
    return f"embedding=ready ({config.embedding_model})"


def build_app():
    """Construct the HTTP application after runtime environment is available."""
    config = RagConfig.from_environment()
    backend = FaissBackend(config)
    service = RagService(config, backend)
    app = create_app(service, api_token=os.getenv("RAG_API_TOKEN") or None)
    print(describe_status(backend.status()), file=sys.stderr)
    print(embedding_preflight(config), file=sys.stderr)
    return app


def __getattr__(name: str):
    # Uvicorn's ``module:app`` loader still works, while importing this module
    # alone no longer requires RAG_KB_ROOT to be configured.
    if name == "app":
        return build_app()
    raise AttributeError(name)
