"""Agent HTTP surface: status reporting, auth, and the streamed chat endpoint."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent_service.config import AgentConfig, LlmProvider
from agent_service.server import create_app


def _config(**overrides):
    values = {
        "providers": (
            LlmProvider(
                id="ollama",
                label="Ollama",
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama",
                models=("qwen2.5:7b",),
                offline=True,
            ),
        ),
        "rag_enabled": False,
    }
    values.update(overrides)
    return AgentConfig(**values)


def _events(response):
    """Decode an SSE body into the list of event objects it carried."""
    return [
        json.loads(line[len("data:") :].strip())
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


def _fake_stream(*chunks):
    async def _stream(*_args, **_kwargs):
        for chunk in chunks:
            yield chunk

    return _stream


class StaticPageTests(unittest.TestCase):
    def test_page_and_config_are_served(self):
        with TestClient(create_app(_config())) as client:
            page = client.get("/")
            config = client.get("/api/config").json()

        self.assertEqual(page.status_code, 200)
        self.assertIn("本地资料库智能问答", page.text)
        self.assertEqual(config["default_provider"], "ollama")
        self.assertFalse(config["auth_required"])

    def test_config_never_leaks_provider_credentials(self):
        config = _config(
            providers=(
                LlmProvider(
                    id="cloud",
                    label="Cloud",
                    base_url="https://example.invalid/v1",
                    api_key="super-secret-key",
                    models=("glm-5.3-flash",),
                    offline=False,
                ),
            ),
            default_provider="cloud",
            default_model="glm-5.3-flash",
        )
        with TestClient(create_app(config)) as client:
            body = client.get("/api/config").text

        self.assertNotIn("super-secret-key", body)


class HealthTests(unittest.TestCase):
    def test_health_reports_each_provider_and_the_index_state(self):
        config = _config(rag_enabled=True)
        with patch("agent_service.server.probe", new=AsyncMock(return_value=(True, "ok"))), patch(
            "agent_service.server._rag_status",
            new=AsyncMock(return_value={"enabled": True, "knowledge_base": "cybersec", "status": "sq8"}),
        ):
            with TestClient(create_app(config)) as client:
                health = client.get("/api/health").json()

        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["providers"][0]["ok"])
        self.assertEqual(health["rag"]["knowledge_base"], "cybersec")

    def test_a_dead_provider_is_reported_without_failing_the_page(self):
        config = _config()
        with patch(
            "agent_service.server.probe",
            new=AsyncMock(return_value=(False, "ConnectError: refused")),
        ), patch(
            "agent_service.server._rag_status",
            new=AsyncMock(return_value={"enabled": True, "error": "RagIndexNotReadyError: stale"}),
        ):
            with TestClient(create_app(config)) as client:
                response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        health = response.json()
        self.assertFalse(health["providers"][0]["ok"])
        self.assertIn("stale", health["rag"]["error"])


class ChatStreamTests(unittest.TestCase):
    def _turn(self, text="你好"):
        return {"messages": [{"role": "user", "content": text}], "use_rag": False}

    def test_streaming_answer_is_delivered_as_delta_events(self):
        config = _config()
        with patch("agent_service.chat.stream_chat", new=_fake_stream("你", "好")):
            with TestClient(create_app(config)) as client:
                events = _events(client.post("/api/chat", json=self._turn()))

        self.assertEqual(events[0]["type"], "start")
        self.assertEqual([e["text"] for e in events if e["type"] == "delta"], ["你", "好"])
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["model"], "qwen2.5:7b")

    def test_provider_failure_arrives_as_a_terminal_error_event(self):
        from agent_service.errors import ProviderUnavailableError

        async def _failing(*_args, **_kwargs):
            raise ProviderUnavailableError("无法连接 Ollama", hint="先启动 ollama serve")
            yield  # pragma: no cover - makes this an async generator

        config = _config()
        with patch("agent_service.chat.stream_chat", new=_failing):
            with TestClient(create_app(config)) as client:
                events = _events(client.post("/api/chat", json=self._turn()))

        error = events[-1]
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error_type"], "ProviderUnavailableError")
        self.assertIn("无法连接 Ollama", error["error"])
        self.assertIn("ollama serve", error["hint"])

    def test_retrieval_failure_degrades_to_plain_chat(self):
        def _broken(**_kwargs):
            raise ConnectionError("index offline")

        config = _config(rag_enabled=True)
        with patch("agent_service.rag.search_knowledge_base", new=_broken), patch(
            "agent_service.chat.stream_chat", new=_fake_stream("answer")
        ):
            with TestClient(create_app(config)) as client:
                events = _events(
                    client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
                )

        warning = next(event for event in events if event["type"] == "warning")
        self.assertIn("检索不可用", warning["message"])
        self.assertIn("answer", [event.get("text") for event in events])

    def test_retrieval_events_expose_citable_sources(self):
        def _fake_search(**_kwargs):
            return {
                "knowledge_base": "cybersec",
                "total": 1,
                "results": [
                    {
                        "source": "14_ctf_wp/by-year/2014/booty.md",
                        "chunk_id": "cybersec:12",
                        "score": 0.8,
                        "content": "evidence",
                    }
                ],
            }

        config = _config(rag_enabled=True)
        with patch("agent_service.rag.search_knowledge_base", new=_fake_search), patch(
            "agent_service.chat.stream_chat", new=_fake_stream("ok")
        ):
            with TestClient(create_app(config)) as client:
                events = _events(
                    client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
                )

        retrieval = next(event for event in events if event["type"] == "retrieval")
        self.assertEqual(retrieval["results"][0]["chunk_id"], "cybersec:12")
        self.assertEqual(retrieval["results"][0]["source"], "14_ctf_wp/by-year/2014/booty.md")

    def test_a_turn_not_ending_on_the_user_is_rejected(self):
        config = _config()
        with TestClient(create_app(config)) as client:
            response = client.post(
                "/api/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                },
            )
        self.assertEqual(response.status_code, 422)


class AuthTests(unittest.TestCase):
    def test_chat_requires_the_configured_bearer_token(self):
        config = _config(api_token="local-token")
        with patch("agent_service.chat.stream_chat", new=_fake_stream("ok")):
            with TestClient(create_app(config)) as client:
                denied = client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "hi"}], "use_rag": False},
                )
                allowed = client.post(
                    "/api/chat",
                    json={"messages": [{"role": "user", "content": "hi"}], "use_rag": False},
                    headers={"Authorization": "Bearer local-token"},
                )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_status_and_page_stay_open_so_an_operator_can_diagnose(self):
        config = _config(api_token="local-token")
        with patch("agent_service.server.probe", new=AsyncMock(return_value=(True, "ok"))), patch(
            "agent_service.server._rag_status", new=AsyncMock(return_value={"enabled": False})
        ):
            with TestClient(create_app(config)) as client:
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertTrue(client.get("/api/config").json()["auth_required"])


if __name__ == "__main__":
    unittest.main()
