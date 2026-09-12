"""Provider probing and streaming: the two places a gateway's quirks show up.

`GET /models` is the cheap readiness check, but not every OpenAI-compatible
gateway implements it, and reporting such an endpoint as down would be wrong
when chat works. The streaming path likewise has to tolerate a gateway that
ignores `stream`, and must not silently truncate an answer it cannot parse.
"""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from agent_service.config import LlmProvider
from agent_service.errors import ProviderResponseError
from agent_service.chat import ChatTurn, run_turn
from agent_service.config import AgentConfig
from agent_service.llm import probe, stream_chat


def _provider(offline=False, api_key="k", models=("m",)):
    return LlmProvider(
        id="cloud" if not offline else "ollama",
        label="Test",
        base_url="http://127.0.0.1:1/v1",
        api_key=api_key,
        models=models,
        offline=offline,
    )


class _Handler(httpx.MockTransport):
    """Serve canned responses so no network is touched."""

    def __init__(self, routes):
        def handler(request: httpx.Request) -> httpx.Response:
            key = (request.method, request.url.path)
            response = routes.get(key)
            if response is None:
                return httpx.Response(404, json={"error": f"no route for {key}"})
            if callable(response):
                return response(request)
            return response

        super().__init__(handler)


def _patched_client(transport):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    return patch("httpx.AsyncClient", side_effect=factory)


class ProbeTests(unittest.TestCase):
    def test_models_endpoint_answers_ok(self):
        routes = {("GET", "/v1/models"): httpx.Response(200, json={"data": []})}
        with _patched_client(_Handler(routes)):
            ok, detail = asyncio.run(probe(_provider()))
        self.assertTrue(ok)
        self.assertEqual(detail, "ok")

    def test_a_gateway_without_models_is_probed_by_a_real_completion(self):
        # Chat works; /models does not exist. Reporting this as down would be a
        # false alarm in a status page an operator is meant to trust.
        routes = {
            ("GET", "/v1/models"): httpx.Response(404, json={"error": "not found"}),
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            ),
        }
        with _patched_client(_Handler(routes)):
            ok, detail = asyncio.run(probe(_provider()))
        self.assertTrue(ok, detail)
        self.assertIn("最小请求", detail)

    def test_a_gateway_rejecting_the_completion_stays_down(self):
        routes = {
            ("GET", "/v1/models"): httpx.Response(405, json={"error": "nope"}),
            ("POST", "/v1/chat/completions"): httpx.Response(500, json={"error": "boom"}),
        }
        with _patched_client(_Handler(routes)):
            ok, _ = asyncio.run(probe(_provider()))
        self.assertFalse(ok)

    def test_a_rejected_key_is_reported_as_such(self):
        routes = {("GET", "/v1/models"): httpx.Response(401, json={"error": "bad key"})}
        with _patched_client(_Handler(routes)):
            ok, detail = asyncio.run(probe(_provider()))
        self.assertFalse(ok)
        self.assertIn("API Key", detail)

    def test_a_cloud_provider_without_a_key_is_not_probed(self):
        ok, detail = asyncio.run(probe(_provider(offline=False, api_key="")))
        self.assertFalse(ok)
        self.assertIn("API Key", detail)


def _sse(*chunks: str) -> httpx.Response:
    body = "".join(f"data: {chunk}\n\n" for chunk in chunks)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def _collect(provider, routes):
    texts = []
    with _patched_client(_Handler(routes)):
        async for chunk in stream_chat(
            provider, "m", [{"role": "user", "content": "hi"}],
            temperature=0.2, max_tokens=10, timeout=5,
        ):
            texts.append(chunk)
    return texts


class StreamTests(unittest.TestCase):
    def test_text_deltas_are_yielded_and_done_terminates(self):
        chunk = json.dumps({"choices": [{"delta": {"content": "你好"}}]})
        routes = {("POST", "/v1/chat/completions"): _sse(chunk, chunk, "[DONE]")}
        self.assertEqual(asyncio.run(_collect(_provider(), routes)), ["你好", "你好"])

    def test_a_gateway_that_ignores_stream_still_returns_the_whole_answer(self):
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200,
                json={"choices": [{"message": {"content": "non-streamed answer"}}]},
                headers={"content-type": "application/json"},
            )
        }
        self.assertEqual(
            asyncio.run(_collect(_provider(), routes)), ["non-streamed answer"]
        )

    def test_an_error_status_raises_before_any_text_is_yielded(self):
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                401, json={"error": "bad key"}, headers={"content-type": "application/json"}
            )
        }
        with self.assertRaises(ProviderResponseError) as ctx:
            asyncio.run(_collect(_provider(), routes))
        self.assertIn("API Key", str(ctx.exception.hint))

    def test_an_unparseable_frame_is_reported_rather_than_skipped(self):
        # Silently dropping it would return a truncated answer that looks whole.
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200,
                text="data: {not json}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        }
        with self.assertRaises(ProviderResponseError):
            asyncio.run(_collect(_provider(), routes))

    def test_a_cloud_provider_without_a_key_fails_before_connecting(self):
        provider = _provider(offline=False, api_key="")
        with self.assertRaises(Exception) as ctx:
            asyncio.run(_collect(provider, {}))
        self.assertIn("API Key", str(getattr(ctx.exception, "hint", "")))

    def test_keepalive_and_comment_lines_are_ignored(self):
        body = ": keepalive\n\ndata: " + json.dumps(
            {"choices": [{"delta": {"content": "x"}}]}
        ) + "\n\ndata: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        self.assertEqual(asyncio.run(_collect(_provider(), routes)), ["x"])


class FinishReasonTests(unittest.TestCase):
    """A provider that stops at `max_tokens` must be distinguishable.

    Reported from real use: a DeepSeek reply was cut at 1024 tokens mid-list and
    displayed as if complete, which reads like a stalled stream.
    """

    def _collect_with_state(self, routes, offline=True):
        state = {}

        async def run():
            provider = _provider(offline=offline)
            texts = []
            with _patched_client(_Handler(routes)):
                async for chunk in stream_chat(
                    provider, "m", [{"role": "user", "content": "hi"}],
                    temperature=0.2, max_tokens=10, timeout=5, state=state,
                ):
                    texts.append(chunk)
            return texts

        return asyncio.run(run()), state

    def test_a_length_finish_reason_is_reported(self):
        body = "".join(
            f"data: {json.dumps(chunk)}\n\n"
            for chunk in (
                {"choices": [{"delta": {"content": "半句"}}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
            )
        ) + "data: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        texts, state = self._collect_with_state(routes)
        self.assertEqual(texts, ["半句"])
        self.assertEqual(state["finish_reason"], "length")

    def test_a_normal_stop_is_reported_as_stop(self):
        body = "".join(
            f"data: {json.dumps(chunk)}\n\n"
            for chunk in (
                {"choices": [{"delta": {"content": "完整"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        ) + "data: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        _, state = self._collect_with_state(routes)
        self.assertEqual(state["finish_reason"], "stop")

    def test_a_non_streaming_reply_reports_its_finish_reason_too(self):
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "truncated"}, "finish_reason": "length"}
                    ]
                },
                headers={"content-type": "application/json"},
            )
        }
        texts, state = self._collect_with_state(routes)
        self.assertEqual(texts, ["truncated"])
        self.assertEqual(state["finish_reason"], "length")


class AnnotationIntegrityTests(unittest.TestCase):
    def test_annotations_are_evaluatable(self):
        # `from __future__ import annotations` turns every annotation into a
        # string, so a name that was never imported still compiles. That hid a
        # real defect in mcp_adapter (the SDK introspects annotations) and had
        # left `Optional`/`Dict`/`Any` unimported here. Resolving them catches
        # the whole class.
        import inspect
        import typing

        from agent_service import llm

        for name, value in vars(llm).items():
            if not (inspect.isclass(value) or callable(value)):
                continue
            module = getattr(value, "__module__", "")
            if not module or not module.startswith("agent_service"):
                continue
            try:
                typing.get_type_hints(value)
            except NameError as exc:
                self.fail(f"{name} has an unresolvable annotation: {exc}")


class TruncationNoticeTests(unittest.TestCase):
    """A truncated answer must be labelled, not passed off as complete."""

    def _turn(self):
        return {"messages": [{"role": "user", "content": "hi"}], "use_rag": False}

    def test_hitting_max_tokens_produces_a_warning_event(self):
        import asyncio

        chunk = json.dumps({"choices": [{"delta": {"content": "half"}}]})
        done = json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]})
        body = f"data: {chunk}\n\ndata: {done}\n\ndata: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        events = []

        async def run():
            with _patched_client(_Handler(routes)):
                async for event in run_turn(_config(), ChatTurn(**self._turn())):
                    events.append(event)

        asyncio.run(run())
        warnings = [e["message"] for e in events if e["type"] == "warning"]
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("max_tokens", warnings[0])
        self.assertIn("截断", warnings[0])
        self.assertIn("AGENT_MAX_TOKENS", warnings[0])
        # The partial text is still delivered, and the stream still terminates.
        self.assertEqual([e["text"] for e in events if e["type"] == "delta"], ["half"])
        self.assertEqual(events[-1]["type"], "done")

    def test_a_complete_answer_carries_no_truncation_warning(self):
        import asyncio

        chunk = json.dumps({"choices": [{"delta": {"content": "done"}}]})
        stop = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        body = f"data: {chunk}\n\ndata: {stop}\n\ndata: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        events = []

        async def run():
            with _patched_client(_Handler(routes)):
                async for event in run_turn(_config(), ChatTurn(**self._turn())):
                    events.append(event)

        asyncio.run(run())
        self.assertEqual([e for e in events if e["type"] == "warning"], [])

    def test_a_reasoning_model_that_never_reached_the_answer_says_so(self):
        # deepseek-flash streams hidden `reasoning_content` on the same token
        # budget; with a small max_tokens the visible reply is empty and the user
        # otherwise just sees nothing.
        import asyncio

        think = json.dumps({"choices": [{"delta": {"reasoning_content": "嗯" * 50}}]})
        cut = json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]})
        body = f"data: {think}\n\ndata: {cut}\n\ndata: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        events = []

        async def run():
            with _patched_client(_Handler(routes)):
                async for event in run_turn(_config(), ChatTurn(**self._turn())):
                    events.append(event)

        asyncio.run(run())
        warnings = [e["message"] for e in events if e["type"] == "warning"]
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("推理模型", warnings[0])
        self.assertIn("思考", warnings[0])
        self.assertEqual([e for e in events if e["type"] == "delta"], [])
        self.assertEqual(events[-1]["type"], "done")

    def test_hidden_reasoning_never_leaks_into_the_answer(self):
        # It is not the answer and it is not cited evidence.
        import asyncio

        think = json.dumps({"choices": [{"delta": {"reasoning_content": "SECRET-THOUGHT"}}]})
        chunk = json.dumps({"choices": [{"delta": {"content": "visible"}}]})
        stop = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        body = f"data: {think}\n\ndata: {chunk}\n\ndata: {stop}\n\ndata: [DONE]\n\n"
        routes = {
            ("POST", "/v1/chat/completions"): httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        }
        events = []

        async def run():
            with _patched_client(_Handler(routes)):
                async for event in run_turn(_config(), ChatTurn(**self._turn())):
                    events.append(event)

        asyncio.run(run())
        text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
        self.assertEqual(text, "visible")
        self.assertNotIn("SECRET-THOUGHT", str(events))


if __name__ == "__main__":
    unittest.main()

def _config() -> AgentConfig:
    """A minimal agent config for the turn-level tests (no retrieval)."""
    return AgentConfig(
        providers=(_provider(),),
        default_provider="cloud",
        default_model="m",
        rag_enabled=False,
    )
