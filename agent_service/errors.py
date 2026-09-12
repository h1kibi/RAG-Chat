"""Error types shared by the agent runtime.

Every failure carries an actionable sentence: the web UI shows these verbatim, so
"连接被拒绝" alone would leave an offline operator guessing which command fixes it.
"""
from __future__ import annotations


class AgentError(RuntimeError):
    """Base class for agent-runtime failures."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict:
        payload = {"error": self.message, "error_type": type(self).__name__}
        if self.hint:
            payload["hint"] = self.hint
        return payload


class ProviderUnavailableError(AgentError):
    """The chat endpoint could not be reached at all."""


class ProviderResponseError(AgentError):
    """The endpoint answered, but not with a usable chat completion."""


class MissingCredentialError(AgentError):
    """A provider that needs an API key was selected without one."""
