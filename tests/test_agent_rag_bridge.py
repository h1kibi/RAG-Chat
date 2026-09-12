import os
import unittest
from unittest.mock import MagicMock, patch

from agent_service.rag import search_knowledge_base
from rag_service.models import RetrievalRequest, RetrievalResponse, SearchResult


class AgentRagBridgeTests(unittest.TestCase):
    def test_in_process_search_invokes_rag_service(self):
        fake_response = RetrievalResponse(
            query="test query",
            knowledge_base="cybersec",
            results=[
                SearchResult(
                    content="evidence",
                    score=0.9,
                    source="01_web/example.md",
                    chunk_id="cybersec:0",
                )
            ],
            total=1,
            backend="faiss",
            embedding_model="bge-m3",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_SERVICE_URL", None)
            with patch("agent_service.rag._service") as mock_service_accessor:
                mock_service = MagicMock()
                mock_service.config.default_knowledge_base = "cybersec"
                mock_service.search.return_value = fake_response
                mock_service_accessor.return_value = mock_service

                result = search_knowledge_base(
                    query="test query",
                    knowledge_base="",
                    top_k=3,
                    score_threshold=0.2,
                )

                self.assertEqual(result["total"], 1)
                self.assertEqual(result["backend"], "faiss")
                self.assertEqual(result["results"][0]["chunk_id"], "cybersec:0")
                mock_service.search.assert_called_once()
                sent_request = mock_service.search.call_args[0][0]
                self.assertEqual(sent_request.knowledge_base, "cybersec")
                self.assertEqual(sent_request.top_k, 3)

    def test_http_search_dispatches_when_service_url_configured(self):
        with patch.dict(os.environ, {"RAG_SERVICE_URL": "http://127.0.0.1:8791", "RAG_API_TOKEN": "token-xyz"}):
            with patch("rag_service.http_client.RagHttpClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.search_dict.return_value = {
                    "total": 1,
                    "backend": "faiss",
                    "results": [{"chunk_id": "cybersec:42"}],
                }
                mock_client_cls.return_value = mock_client

                result = search_knowledge_base(
                    query="remote search",
                    knowledge_base="cybersec",
                    top_k=4,
                    score_threshold=0.1,
                    filters={"category": "14_ctf_wp", "year": 2014},
                )

                self.assertEqual(result["total"], 1)
                self.assertEqual(result["results"][0]["chunk_id"], "cybersec:42")
                mock_client_cls.assert_called_once_with(
                    "http://127.0.0.1:8791",
                    api_token="token-xyz",
                    timeout=120.0,
                )
                mock_client.search_dict.assert_called_once_with(
                    query="remote search",
                    knowledge_base="cybersec",
                    top_k=4,
                    score_threshold=0.1,
                    filters={"category": "14_ctf_wp", "year": 2014},
                )

    def test_a_malformed_http_timeout_names_the_variable(self):
        # The agent relays this message verbatim as its degradation warning, so
        # a bare float() error read as "retrieval is broken".
        with patch.dict(
            os.environ,
            {"RAG_SERVICE_URL": "http://127.0.0.1:8791", "RAG_HTTP_TIMEOUT": "30s"},
        ):
            with self.assertRaisesRegex(ValueError, "RAG_HTTP_TIMEOUT") as ctx:
                search_knowledge_base(query="x", knowledge_base="cybersec")

        self.assertIn("'30s'", str(ctx.exception))

    def test_a_blank_http_timeout_falls_back_to_the_default(self):
        with patch.dict(
            os.environ,
            {"RAG_SERVICE_URL": "http://127.0.0.1:8791", "RAG_HTTP_TIMEOUT": ""},
        ):
            with patch("rag_service.http_client.RagHttpClient") as mock_client_cls:
                mock_client_cls.return_value.search_dict.return_value = {"total": 0}
                search_knowledge_base(query="x", knowledge_base="cybersec")

        self.assertEqual(mock_client_cls.call_args.kwargs["timeout"], 120.0)


if __name__ == "__main__":
    unittest.main()
