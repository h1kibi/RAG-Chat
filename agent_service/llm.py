"""OpenAI-compatible chat client used by the agent server.

Only the ``/chat/completions`` and ``/models`` shapes are needed, which every
target speaks: Ollama's OpenAI shim, Zhipu's ``bigmodel`` endpoint, and any
other OpenAI-compatible gateway. Keeping one code path means the offline and
online modes differ only in which ``LlmProvider`` is selected.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Iterable, Mapping, Sequence

import httpx

from agent_service.config import LlmProvider
from agent_service.errors import (
    MissingCredentialError,
    ProviderResponseError,
    ProviderUnavailableError,
)

Message = Mapping[str, str]

_OFFLINE_HINT = (
    "确认本地 Ollama 已启动（ollama serve），并已拉取模型（ollama pull {model}）；"
    "再用 AGENT_OLLAMA_BASE_URL 指向正确地址。"
)
_CLOUD_HINT = (
    "云端模式需要联网与有效的 API Key。设置 AGENT_CLOUD_API_KEY（或 AGENT_CLOUD_API_KEY_ENV "
    "指向的环境变量），或把 AGENT_DEFAULT_PROVIDER 改回 ollama 以离线运行。"
)


def _headers(provider: LlmProvider) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    return headers


def _body_snippet(response: httpx.Response, limit: int = 400) -> str:
    text = response.text.strip()
    if len(text) > limit:
        text = f"{text[:limit]}…"
    return text or "<empty body>"


def _raise_for_status(provider: LlmProvider, response: httpx.Response, model: str) -> None:
    if response.status_code < 400:
        return
    detail = _body_snippet(response)
    if response.status_code in (401, 403):
        raise ProviderResponseError(
            f"{provider.label} 拒绝认证（HTTP {response.status_code}）：{detail}",
            hint=_CLOUD_HINT if not provider.offline else "检查本地端点是否需要 API Key。",
        )
    if response.status_code == 404:
        raise ProviderResponseError(
            f"{provider.label} 没有找到接口或模型（HTTP 404）：{detail}",
            hint=f"确认模型名 {model!r} 已在该端点注册（ollama list）。",
        )
    raise ProviderResponseError(
        f"{provider.label} 返回 HTTP {response.status_code}：{detail}",
        hint="端点可达但拒绝了请求，先确认模型名与请求参数。",
    )


def _wrap_transport_error(provider: LlmProvider, model: str, exc: Exception) -> ProviderUnavailableError:
    hint = _OFFLINE_HINT.format(model=model) if provider.offline else _CLOUD_HINT
    return ProviderUnavailableError(
        f"无法连接 {provider.label}（{provider.base_url}）：{type(exc).__name__}: {exc}",
        hint=hint,
    )


def _extract_delta(payload: dict) -> str:
    """Pull the incremental text out of one streaming chunk."""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    text = delta.get("content")
    if text:
        return text
    # Some gateways omit `delta` on the final chunk and use `message`.
    message = choice.get("message") or {}
    return message.get("content") or ""


def _iter_sse_lines(raw: Iterable[str]) -> Iterable[str]:
    for line in raw:
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[len("data:") :].strip()


async def stream_chat(
    provider: LlmProvider,
    model: str,
    messages: Sequence[Message],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> AsyncIterator[str]:
    """Yield answer text chunks for one chat request.

    Errors are raised before any chunk is yielded when possible; a mid-stream
    failure propagates as ``ProviderResponseError`` so the caller can keep the
    text already shown rather than replacing it with a blank answer.
    """
    if not provider.offline and not provider.api_key:
        raise MissingCredentialError(
            f"{provider.label} 需要 API Key，但未配置。",
            hint=_CLOUD_HINT,
        )

    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", provider.chat_url, json=payload, headers=_headers(provider)
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _raise_for_status(provider, response, model)

                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    # A gateway ignored `stream`; return the whole answer at once.
                    await response.aread()
                    yield _read_non_streaming(provider, response, model)
                    return

                async for line in response.aiter_lines():
                    for data in _iter_sse_lines([line]):
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            raise ProviderResponseError(
                                f"{provider.label} 返回了无法解析的流式数据：{data[:200]}",
                                hint="该端点可能不是 OpenAI 兼容接口。",
                            ) from None
                        text = _extract_delta(chunk)
                        if text:
                            yield text
    except httpx.HTTPError as exc:
        raise _wrap_transport_error(provider, model, exc) from exc


def _read_non_streaming(provider: LlmProvider, response: httpx.Response, model: str) -> str:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderResponseError(
            f"{provider.label} 返回了非 JSON 响应：{_body_snippet(response)}",
            hint="该端点可能不是 OpenAI 兼容接口。",
        ) from exc
    choices = payload.get("choices") or []
    if not choices:
        raise ProviderResponseError(
            f"{provider.label} 的响应没有 choices 字段：{_body_snippet(response)}",
            hint="确认模型名正确，且端点支持 /chat/completions。",
        )
    return ((choices[0] or {}).get("message") or {}).get("content") or ""


async def probe(provider: LlmProvider, *, timeout: float = 5.0) -> tuple[bool, str]:
    """Reachability check for one provider, used by ``/api/health``.

    Returns an ``(ok, detail)`` pair instead of raising so a status page can
    report every provider at once, including the ones that are down.

    ``GET /models`` is the cheap check, but not every OpenAI-compatible gateway
    implements it. Reporting such an endpoint as down would be wrong -- chat
    works -- so a 404/405 falls back to one minimal completion, which is
    definitive.
    """
    if not provider.offline and not provider.api_key:
        return False, "未配置 API Key"
    url = f"{provider.base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=_headers(provider))
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code in (401, 403):
        return False, f"HTTP {response.status_code}（API Key 被拒绝）"
    if response.status_code < 400:
        return True, "ok"
    if response.status_code in (404, 405):
        return await _probe_via_chat(provider, timeout=timeout)
    return False, f"HTTP {response.status_code}"


async def _probe_via_chat(provider: LlmProvider, *, timeout: float) -> tuple[bool, str]:
    """Confirm the endpoint can actually answer a one-token completion.

    Used when ``/models`` is unavailable. Costs a negligible request and only
    runs for providers whose model listing is unsupported.
    """
    payload = {
        "model": provider.models[0] if provider.models else "unknown",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                provider.chat_url, json=payload, headers=_headers(provider)
            )
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code in (401, 403):
        return False, f"HTTP {response.status_code}（API Key 被拒绝）"
    if response.status_code < 400:
        return True, "ok（端点不提供 /models，已用最小请求确认）"
    if response.status_code == 404:
        return False, (
            f"HTTP 404（{provider.chat_url} 不存在：检查 base_url 是否缺少路径段，"
            f"如 /v1 或 /v4）"
        )
    return False, f"HTTP {response.status_code}（最小请求被拒绝）"
