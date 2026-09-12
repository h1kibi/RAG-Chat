"""One-shot CLI: argument translation and observable output.

The CLI exists so an agent without an MCP transport can still query the
read-only index; it must therefore build exactly the same request contract and
print the same cited evidence block as the MCP tool.
"""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from rag_service import __main__ as cli
from rag_service.models import SearchResult


class _FakeBackend:
    name = "fake"

    def __init__(self, config=None):
        self.request = None
        self.closed = False

    def search(self, request):
        self.request = request
        return [
            SearchResult(
                content="evidence body",
                score=0.71,
                source="15_butian/example.md",
                chunk_id="cybersec:7",
            )
        ]

    def close(self):
        self.closed = True


class CliParsingTests(unittest.TestCase):
    def test_blank_query_browse_request_carries_filters_and_limit(self):
        args = cli._build_parser().parse_args(
            ["search", "", "--filters", "category=15_butian", "--limit", "20"]
        )
        request = cli.build_request(args)

        self.assertEqual(request.query, "")
        self.assertEqual(request.filters, {"category": "15_butian"})
        self.assertEqual(request.limit, 20)

    def test_year_filter_is_coerced_to_integer(self):
        args = cli._build_parser().parse_args(
            ["search", "x", "--filters", "year=2021", "--filters", "source_prefix=14_ctf_wp/"]
        )
        request = cli.build_request(args)

        self.assertEqual(request.filters["year"], 2021)
        self.assertEqual(request.filters["source_prefix"], "14_ctf_wp/")

    def test_exclude_source_prefix_accepts_comma_separated_list(self):
        args = cli._build_parser().parse_args(
            ["search", "x", "--filters", "exclude_source_prefix=12_security_learning,13_xianzhi"]
        )
        request = cli.build_request(args)

        self.assertEqual(
            request.filters["exclude_source_prefix"],
            ["12_security_learning", "13_xianzhi"],
        )

    def test_cursor_is_normalized_to_none_when_absent(self):
        args = cli._build_parser().parse_args(["search", ""])
        self.assertIsNone(cli.build_request(args).cursor)

    def test_unsupported_filter_is_rejected(self):
        with self.assertRaises(SystemExit):
            cli._build_parser().parse_args(["search", "x", "--filters", "unknown=value"])

    def test_malformed_filter_is_rejected(self):
        with self.assertRaises(SystemExit):
            cli._build_parser().parse_args(["search", "x", "--filters", "category"])

    def test_missing_subcommand_defaults_to_serve(self):
        args = cli._build_parser().parse_args(["--host", "0.0.0.0", "--port", "9000"])

        self.assertIsNone(args.command)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)


class CliPortTests(unittest.TestCase):
    def test_invalid_rag_port_environment_is_a_usage_error(self):
        captured = io.StringIO()
        with patch.dict(os.environ, {"RAG_PORT": "not-a-port"}, clear=False):
            with contextlib.redirect_stderr(captured):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["search", "x"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("RAG_PORT", captured.getvalue())
        self.assertNotIn("Traceback", captured.getvalue())


class CliSearchTests(unittest.TestCase):
    def test_search_prints_cited_evidence_and_closes_the_backend(self):
        backend = _FakeBackend()
        captured = io.StringIO()

        with patch.dict(
            os.environ,
            {"RAG_KB_ROOT": "C:/RAG-Agent-Data/data/knowledge_base"},
            clear=False,
        ), patch(
            "rag_service.backends.faiss.FaissBackend",
            lambda config: backend,
        ):
            with contextlib.redirect_stdout(captured):
                exit_code = cli.main(["search", "JNDI log4shell", "--top-k", "3"])

        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("source=15_butian/example.md", output)
        self.assertIn("chunk_id=cybersec:7", output)
        self.assertIn("evidence body", output)
        self.assertEqual(backend.request.top_k, 3)
        self.assertTrue(backend.closed)

    def test_json_flag_emits_the_raw_response(self):
        backend = _FakeBackend()
        captured = io.StringIO()

        with patch.dict(
            os.environ,
            {"RAG_KB_ROOT": "C:/RAG-Agent-Data/data/knowledge_base"},
            clear=False,
        ), patch(
            "rag_service.backends.faiss.FaissBackend",
            lambda config: backend,
        ):
            with contextlib.redirect_stdout(captured):
                cli.main(["search", "x", "--json"])

        payload = captured.getvalue()
        self.assertIn('"chunk_id": "cybersec:7"', payload)
        self.assertIn('"score": 0.71', payload)


if __name__ == "__main__":
    unittest.main()
