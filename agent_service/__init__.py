"""Agent-side integration facade.

This package owns two responsibilities:

- ``chat`` / ``llm`` / ``server``: the offline-first web chat agent that talks
  to a local Ollama model (or a cloud model when a key is supplied), optionally
  grounding answers in the local knowledge base.
- ``rag``: the single bridge through which anything else consumes the
  standalone ``rag_service``, so the retrieval backend can be swapped
  (in-process, HTTP, or remote) without touching callers.
"""
from __future__ import annotations

from agent_service.chat import ChatMessage, ChatTurn, run_turn
from agent_service.config import AgentConfig, LlmProvider
from agent_service.errors import (
    AgentError,
    MissingCredentialError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from agent_service.rag import search_knowledge_base
from agent_service.server import create_app

__all__ = [
    "AgentConfig",
    "AgentError",
    "ChatMessage",
    "ChatTurn",
    "LlmProvider",
    "MissingCredentialError",
    "ProviderResponseError",
    "ProviderUnavailableError",
    "create_app",
    "run_turn",
    "search_knowledge_base",
]
