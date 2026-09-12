"""The CLI's error contract, and the sq8 fallback's diagnostic honesty.

Two failure-reporting gaps, both of which send the operator to the wrong fix:

- `python -m rag_service search` used to leak a ten-line traceback with exit 1
  when `RAG_KB_ROOT` was unset -- the most common first-run mistake -- instead
  of the field-level usage error every other entry point produces.
- A numpy scan was always explained as "the sq8 artifact is missing", even when
  the artifact was present and simply could not be read. The prescribed remedy
  (build_cosine) reports the artifacts as current and changes nothing.
"""
import contextlib
import io
import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

import rag_service.backends.faiss as backend_module
from rag_service import __main__ as cli
from rag_service.backends.faiss import FaissBackend, describe_status
from rag_service.build_cosine import build_cosine_files
from rag_service.config import RagConfig


def _build_store(root: Path) -> Path:
    index_path = root / "cybersec" / "vector_store" / "bge-m3"
    index_path.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(4)
    index.add(np.eye(4, 4, dtype="float32"))
    faiss.write_index(index, str(index_path / "index.faiss"))
    store = {
        f"cybersec:{row}": Document(
            page_content="检索用的正文内容，足够长以通过低信息过滤。" * 3,
            metadata={"source": f"01_web/doc-{row}.md", "id": f"cybersec:{row}"},
        )
        for row in range(4)
    }
    with (index_path / "index.pkl").open("wb") as handle:
        pickle.dump(
            (InMemoryDocstore(store), {row: f"cybersec:{row}" for row in range(4)}),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    connection = sqlite3.connect(root / "info.db")
    try:
        connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
        connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
        connection.commit()
    finally:
        connection.close()
    build_cosine_files(root, "cybersec", "bge-m3")
    return index_path


def _config(root: Path) -> RagConfig:
    return RagConfig(
        knowledge_base_root=root,
        allowed_knowledge_bases=frozenset({"cybersec"}),
        embedding_model="bge-m3",
    )


class CliErrorContractTests(unittest.TestCase):
    """Usage problems must read as usage problems, with the documented code."""

    def _run(self, argv, env):
        import os
        from unittest.mock import patch

        stderr = io.StringIO()
        with patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stderr(stderr):
                try:
                    code = cli.main(argv)
                except SystemExit as exc:  # pragma: no cover - argparse path
                    code = int(exc.code or 0)
        return code, stderr.getvalue()

    def test_missing_knowledge_base_root_is_a_usage_error_not_a_traceback(self):
        code, stderr = self._run(["search", "glibc tcache", "--top-k", "3"], {})
        self.assertEqual(code, 2, stderr)
        self.assertIn("RAG_KB_ROOT", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_an_unparsable_variable_names_the_variable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            code, stderr = self._run(
                ["search", "x"],
                {
                    "RAG_KB_ROOT": str(Path(temp_dir)),
                    "RAG_DEFAULT_TOP_K": "not-a-number",
                },
            )
        self.assertEqual(code, 2, stderr)
        self.assertIn("RAG_DEFAULT_TOP_K", stderr)

    def test_out_of_range_arguments_stay_field_level(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            code, stderr = self._run(
                ["search", "x", "--top-k", "0"], {"RAG_KB_ROOT": str(Path(temp_dir))}
            )
        self.assertEqual(code, 2, stderr)
        self.assertIn("top_k", stderr)


class Sq8FallbackDiagnosticTests(unittest.TestCase):
    """Explain *why* the numpy scan is in use, because the fixes differ."""

    def _status(self, root: Path) -> str:
        backend_module._LOGGED_PATHS.clear()
        backend = FaissBackend(_config(root))
        try:
            return describe_status(backend.status("cybersec"))
        finally:
            backend.close()

    def test_a_healthy_store_reports_the_fast_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_store(root)
            self.assertIn("dense_path=sq8", self._status(root))

    def test_a_healthy_store_does_not_claim_the_artifact_is_unreadable(self):
        # `sq8_state` answers "why did we fall back?", so it is only meaningful
        # when a fallback happened. Published unconditionally it made
        # /v1/rag/health report dense_path=sq8 and sq8_state=unreadable at the
        # same time, which is a false alarm for anything reading the JSON.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_store(root)
            backend_module._LOGGED_PATHS.clear()
            backend = FaissBackend(_config(root))
            try:
                status = backend.status("cybersec")
            finally:
                backend.close()

        self.assertEqual(status["dense_path"], "sq8")
        self.assertEqual(status["sq8_state"], "loaded")

    def test_an_absent_artifact_points_at_build_cosine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = _build_store(root)
            (index_path / "vectors.cos.sq8").unlink()
            status = self._status(root)
        self.assertIn("dense_path=numpy", status)
        self.assertIn("missing", status)
        self.assertNotIn("--force", status)

    def test_an_unreadable_artifact_is_not_reported_as_missing(self):
        # A present-but-unreadable artifact (corrupt, or too large to map) is the
        # memory-shortage case; build_cosine would report it as current and fix
        # nothing, so the message must say so.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = _build_store(root)
            artifact = index_path / "vectors.cos.sq8"
            artifact.write_bytes(b"\x00" * artifact.stat().st_size)
            status = self._status(root)
        self.assertIn("dense_path=numpy", status)
        self.assertIn("exists but could not be loaded", status)
        self.assertIn("--force", status)


if __name__ == "__main__":
    unittest.main()
