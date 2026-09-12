
"""Provider selection and environment parsing for the web chat agent."""
import os
import unittest
from unittest.mock import patch

from agent_service.config import AgentConfig


def _env(**values):
    """Patch the environment, clearing the keys this config reads."""
    watched = {
        "AGENT_HOST", "AGENT_PORT", "AGENT_API_TOKEN", "AGENT_DEFAULT_PROVIDER",
        "AGENT_DEFAULT_MODEL", "AGENT_OLLAMA_MODELS", "AGENT_OLLAMA_BASE_URL",
        "AGENT_CLOUD_BASE_URL", "AGENT_CLOUD_MODELS", "AGENT_CLOUD_LABEL",
        "AGENT_CLOUD_API_KEY", "AGENT_CLOUD_API_KEY_ENV", "AGENT_RAG_ENABLED",
        "AGENT_RAG_KNOWLEDGE_BASE", "AGENT_RAG_TOP_K", "AGENT_RAG_SCORE_THRESHOLD",
        "AGENT_MAX_TOKENS", "AGENT_HISTORY_LEN", "ZAI_API_KEY",
    }
    cleaned = {key: value for key, value in os.environ.items() if key not in watched}
    cleaned.update({key: str(value) for key, value in values.items()})
    return patch.dict(os.environ, cleaned, clear=True)
class AgentCliDiagnosticsTests(unittest.TestCase):
    def test_invalid_environment_returns_field_error_without_traceback(self):
        import contextlib
        import io

        from agent_service import __main__ as cli

        stderr = io.StringIO()
        with _env(AGENT_PORT="not-a-port"), contextlib.redirect_stderr(stderr):
            code = cli.main(["--print-config"])

        self.assertEqual(code, 2)
        self.assertIn("AGENT_PORT", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())



class OfflineDefaultsTests(unittest.TestCase):
    def test_defaults_need_no_network_and_no_api_key(self):
        with _env():
            config = AgentConfig.from_environment()

        self.assertEqual([provider.id for provider in config.providers], ["ollama"])
        self.assertTrue(config.providers[0].offline)
        self.assertEqual(config.default_model, "qwen2.5:7b")
        self.assertEqual(config.host, "127.0.0.1")
        self.assertIsNone(config.api_token)

    def test_ollama_base_url_gains_the_openai_path(self):
        with _env(AGENT_OLLAMA_BASE_URL="http://192.168.1.9:11434/"):
            config = AgentConfig.from_environment()

        self.assertEqual(config.providers[0].base_url, "http://192.168.1.9:11434/v1")
        self.assertEqual(
            config.providers[0].chat_url, "http://192.168.1.9:11434/v1/chat/completions"
        )


class CloudProviderTests(unittest.TestCase):
    def test_cloud_provider_is_absent_without_endpoint_configuration(self):
        with _env(AGENT_CLOUD_API_KEY="secret"):
            config = AgentConfig.from_environment()

        self.assertEqual([provider.id for provider in config.providers], ["ollama"])

    def test_a_half_configured_cloud_provider_fails_loudly(self):
        # Dropping it silently would leave the operator with no cloud option and
        # no explanation of why.
        with _env(AGENT_CLOUD_BASE_URL="https://example.invalid/v1"):
            with self.assertRaisesRegex(ValueError, "AGENT_CLOUD_MODELS"):
                AgentConfig.from_environment()
        with _env(AGENT_CLOUD_MODELS="glm-5.3-flash"):
            with self.assertRaisesRegex(ValueError, "AGENT_CLOUD_BASE_URL"):
                AgentConfig.from_environment()

    def test_cloud_provider_reads_the_key_from_an_indirect_variable(self):
        with _env(
            AGENT_CLOUD_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
            AGENT_CLOUD_MODELS="glm-5.3-flash",
            AGENT_CLOUD_API_KEY_ENV="ZAI_API_KEY",
            ZAI_API_KEY="key-from-env",
        ):
            config = AgentConfig.from_environment()

        cloud = config.provider("cloud")
        self.assertEqual(cloud.api_key, "key-from-env")
        self.assertEqual(cloud.chat_url, "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.assertEqual(config.default_provider, "ollama")

    def test_public_config_never_exposes_the_api_key(self):
        with _env(
            AGENT_CLOUD_BASE_URL="https://example.invalid/v1",
            AGENT_CLOUD_MODELS="some-model",
            AGENT_CLOUD_API_KEY="top-secret",
        ):
            payload = AgentConfig.from_environment().as_public_dict()

        self.assertNotIn("top-secret", str(payload))
        cloud = next(item for item in payload["providers"] if item["id"] == "cloud")
        local = next(item for item in payload["providers"] if item["id"] == "ollama")
        self.assertTrue(cloud["requires_key"])
        self.assertFalse(cloud["offline"])
        self.assertFalse(local["requires_key"])
        self.assertTrue(local["offline"])

    def test_requiring_cloud_as_default_without_cloud_configured_fails_loudly(self):
        with _env(AGENT_DEFAULT_PROVIDER="cloud"):
            with self.assertRaisesRegex(ValueError, "AGENT_DEFAULT_PROVIDER=cloud"):
                AgentConfig.from_environment()


class ModelResolutionTests(unittest.TestCase):
    def test_requested_model_overrides_the_default(self):
        with _env(AGENT_OLLAMA_MODELS="qwen2.5:7b,llama3.1:8b"):
            config = AgentConfig.from_environment()

        provider, model = config.resolve("", "llama3.1:8b")
        self.assertEqual(provider.id, "ollama")
        self.assertEqual(model, "llama3.1:8b")

    def test_empty_selection_falls_back_to_the_configured_default(self):
        config = AgentConfig(
            providers=(_ollama_stub(),), default_provider="ollama", default_model="qwen2.5:7b"
        )
        provider, model = config.resolve("", "")
        self.assertEqual((provider.id, model), ("ollama", "qwen2.5:7b"))

    def test_unknown_provider_is_rejected_rather_than_silently_ignored(self):
        config = AgentConfig(providers=(_ollama_stub(),))
        with self.assertRaisesRegex(ValueError, "unknown provider"):
            config.resolve("nonexistent", "")


class ValidationTests(unittest.TestCase):
    def test_config_without_providers_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one LLM provider"):
            AgentConfig(providers=())

    def test_default_provider_must_be_configured(self):
        with self.assertRaisesRegex(ValueError, "not configured"):
            AgentConfig(providers=(_ollama_stub(),), default_provider="cloud")

    def test_health_port_range_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "port"):
            AgentConfig(providers=(_ollama_stub(),), port=70_000)

    def test_rag_flag_can_be_switched_off_for_a_pure_chat_deployment(self):
        with _env(AGENT_RAG_ENABLED="0"):
            config = AgentConfig.from_environment()
        self.assertFalse(config.rag_enabled)

    def test_a_blank_flag_means_unset_and_keeps_the_default(self):
        # `AGENT_RAG_ENABLED=` in a shell profile must not silently disable
        # retrieval; a blank value is "not configured", not "false".
        with _env(AGENT_RAG_ENABLED=""):
            config = AgentConfig.from_environment()
        self.assertTrue(config.rag_enabled)

    def test_unrecognised_boolean_names_the_variable(self):
        with _env(AGENT_RAG_ENABLED="maybe"):
            with self.assertRaisesRegex(ValueError, "AGENT_RAG_ENABLED"):
                AgentConfig.from_environment()

    def test_unparseable_numbers_name_the_variable(self):
        with _env(AGENT_PORT="not-a-port"):
            with self.assertRaisesRegex(ValueError, "AGENT_PORT"):
                AgentConfig.from_environment()

    def test_blank_numeric_values_fall_back_to_defaults(self):
        with _env(AGENT_PORT="", AGENT_TEMPERATURE=""):
            config = AgentConfig.from_environment()
        self.assertEqual(config.port, 8801)
        self.assertEqual(config.temperature, 0.2)


def _ollama_stub():
    from agent_service.config import LlmProvider

    return LlmProvider(
        id="ollama",
        label="Ollama",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        models=("qwen2.5:7b",),
        offline=True,
    )


if __name__ == "__main__":
    unittest.main()
