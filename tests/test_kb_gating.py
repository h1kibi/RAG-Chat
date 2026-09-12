"""Cross-module configuration: the agent's KB choice vs retrieval's allow-list.

The agent selects a knowledge base with `AGENT_RAG_KNOWLEDGE_BASE`; the
retrieval service gates access with `RAG_ALLOWED_KNOWLEDGE_BASES`. The two are
independent, so a mismatch is a normal operator mistake -- and the error has to
say which one to change, because "knowledge base is not allowed: cybersec" does
not.
"""
import tempfile
import unittest
from pathlib import Path

from rag_service import RagConfig, RagService, RetrievalRequest
from rag_service.backends.faiss import FaissBackend


class DisallowedKnowledgeBaseTests(unittest.TestCase):
    def _config(self, root: Path, allowed) -> RagConfig:
        return RagConfig(
            knowledge_base_root=root,
            allowed_knowledge_bases=frozenset(allowed),
            embedding_model="bge-m3",
        )

    def test_the_error_names_the_allow_list_variable_and_the_permitted_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(Path(temp_dir), {"samples"})
            backend = FaissBackend(config)
            try:
                with self.assertRaises(ValueError) as ctx:
                    RagService(config, backend).search(
                        RetrievalRequest(query="x", knowledge_base="cybersec")
                    )
            finally:
                backend.close()

            message = str(ctx.exception)
            self.assertIn("cybersec", message)
            self.assertIn("RAG_ALLOWED_KNOWLEDGE_BASES", message)
            self.assertIn("samples", message)

    def test_a_malformed_name_is_explained_as_a_naming_problem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # No allow-list, so gating is pure name validation.
            config = self._config(Path(temp_dir), set())
            backend = FaissBackend(config)
            try:
                with self.assertRaises(ValueError) as ctx:
                    RagService(config, backend).search(
                        RetrievalRequest(query="x", knowledge_base="../etc")
                    )
            finally:
                backend.close()

            message = str(ctx.exception)
            self.assertIn("usable knowledge base name", message)
            self.assertNotIn("allow-list", message)

    def test_an_allow_listed_name_passes_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(Path(temp_dir), {"cybersec"})
            # The index does not exist here, so this fails later and differently
            # -- the point is that the gate itself does not reject the name.
            self.assertTrue(config.is_allowed_knowledge_base("cybersec"))
            self.assertFalse(config.is_allowed_knowledge_base("samples"))


if __name__ == "__main__":
    unittest.main()
