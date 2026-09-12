"""Operational visibility and degraded-mode behaviour.

Two silent failures motivated these:

1. Running under an interpreter without faiss silently used a numpy
   dequantization scan — measured 2.7-3.7 s per query against 0.15 s through
   faiss. A tuning run under the wrong python looked like a slow service.
2. A dead embedding provider failed the whole query, even though the corpus and
   the identifier postings were perfectly readable. Callers read the failure as
   "the corpus has nothing".
"""
import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from rag_service.backends.faiss import (
    FaissBackend,
    describe_dense_path,
    faiss_available,
)
from rag_service.build_cosine import build_cosine_files
from rag_service.config import RagConfig
from rag_service.errors import RagEmbeddingError
from rag_service.models import RetrievalRequest
from rag_service.service import RagService

DIMENSION = 4


class CapabilityReportingTests(unittest.TestCase):
    def _index(self, root: Path) -> Path:
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(DIMENSION)
        vectors = np.zeros((2, DIMENSION), dtype="float32")
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))
        store = {
            "cybersec:0": Document(
                page_content="第一篇文章正文内容，包含足够的字符以通过低信息过滤。" * 4,
                metadata={"source": "13_xianzhi/a.md", "id": "cybersec:0"},
            ),
            "cybersec:1": Document(
                page_content="第二篇文章正文内容，包含 CVE-2021-3490 标识符。" * 4,
                metadata={"source": "13_xianzhi/b.md", "id": "cybersec:1"},
            ),
        }
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(store), {0: "cybersec:0", 1: "cybersec:1"}), handle)
        connection = sqlite3.connect(root / "info.db")
        try:
            connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
            connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
            connection.commit()
        finally:
            connection.close()
        return index_path

    def _config(self, root: Path, **overrides) -> RagConfig:
        values = {
            "knowledge_base_root": root,
            "allowed_knowledge_bases": frozenset({"cybersec"}),
            "embedding_model": "bge-m3",
            "lexical_weight": 0.35,
        }
        values.update(overrides)
        return RagConfig(**values)

    def test_status_reports_the_scan_path_actually_in_use(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            backend = FaissBackend(self._config(root))
            try:
                status = backend.status()
            finally:
                backend.close()
            self.assertEqual(status["dense_path"], "sq8")
            self.assertEqual(status["rows"], 2)
            self.assertTrue(status["postings"])
            self.assertEqual(status["faiss_available"], faiss_available())

    def test_describe_status_renders_a_status_dict(self):
        # Regression: build_app passed a status dict where a store was
        # expected, raising KeyError('vectors') at startup.
        from rag_service.backends.faiss import describe_status

        text = describe_status(
            {"dense_path": "sq8", "rows": 1016721, "postings": True, "faiss_available": True}
        )
        self.assertIn("dense_path=sq8", text)
        self.assertIn("rows=1016721", text)

        missing = describe_status(
            {"dense_path": "numpy", "rows": 10, "postings": False, "faiss_available": True}
        )
        self.assertIn("postings=missing", missing)

        unknown = describe_status({})
        self.assertIn("dense_path=unknown", unknown)

    def test_a_present_but_unusable_postings_sidecar_is_not_called_missing(self):
        # The sidecar fingerprints the document file by size+mtime, so copying a
        # knowledge base always invalidates it. Reporting that as "missing"
        # sends the operator looking for a file that is right there.
        from rag_service.backends.faiss import _postings_state, describe_status

        with tempfile.TemporaryDirectory() as temp_dir:
            docs = Path(temp_dir) / "docs.cos.jsonl"
            docs.write_text("{}\n", encoding="utf-8")
            self.assertEqual(_postings_state(docs), "missing")

            (docs.parent / "docs.cos.postings").write_bytes(b"")
            (docs.parent / "docs.cos.postings.idx.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_postings_state(docs), "unusable")

        rendered = describe_status(
            {
                "dense_path": "sq8",
                "rows": 10,
                "postings": False,
                "postings_state": "unusable",
                "faiss_available": True,
            }
        )
        self.assertIn("postings=unusable", rendered)
        self.assertNotIn("postings=missing", rendered)

    def test_an_empty_postings_index_is_not_rejected_for_a_stale_fingerprint(self):
        # It holds no offsets, so there is nothing to go stale: it means the
        # corpus has no identifier tokens. Copying a KB rewrites mtime, so the
        # fingerprint always mismatches afterwards -- reporting the fixture as
        # broken ("postings=unusable") when it is merely empty.
        from rag_service.backends.faiss import _open_postings

        with tempfile.TemporaryDirectory() as temp_dir:
            docs = Path(temp_dir) / "docs.cos.jsonl"
            docs.write_text('{"text":"a"}\n', encoding="utf-8")
            (docs.parent / "docs.cos.postings").write_bytes(b"")
            (docs.parent / "docs.cos.postings.idx.json").write_text(
                json.dumps(
                    {
                        "format": "postings-v1",
                        "tokens": 0,
                        "postings": 0,
                        "max_df": 500,
                        "marks": [],
                        # deliberately not the current file state
                        "source": {"size": 1, "mtime_ns": 1},
                    }
                ),
                encoding="utf-8",
            )
            loaded = _open_postings(docs)

        self.assertIsNotNone(loaded, "an empty index must load, not report a defect")
        self.assertEqual(loaded["marks"], [])

    def test_a_populated_postings_index_still_rejects_a_stale_fingerprint(self):
        # The converse: real offsets are only safe while they match the file.
        from rag_service.backends.faiss import _open_postings

        with tempfile.TemporaryDirectory() as temp_dir:
            docs = Path(temp_dir) / "docs.cos.jsonl"
            docs.write_text('{"text":"a"}\n', encoding="utf-8")
            (docs.parent / "docs.cos.postings").write_bytes(b"cve-2021\t0\n")
            (docs.parent / "docs.cos.postings.idx.json").write_text(
                json.dumps(
                    {
                        "format": "postings-v1",
                        "tokens": 1,
                        "postings": 1,
                        "max_df": 500,
                        "marks": [["cve-2021", 0]],
                        "source": {"size": 1, "mtime_ns": 1},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(_open_postings(docs))

    def test_status_reports_numpy_when_sq8_is_missing(self):
        # The degradation must be visible rather than inferred from latency.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            (index_path / "vectors.cos.sq8").unlink()
            backend = FaissBackend(self._config(root))
            try:
                status = backend.status()
            finally:
                backend.close()
            self.assertEqual(status["dense_path"], "numpy")

    def test_describe_dense_path_reports_once_then_stays_silent(self):
        # A server states its capability at startup, not on every query.
        import rag_service.backends.faiss as backend_module

        backend_module._LOGGED_PATHS.clear()
        numpy_store = {"sq8": None, "vectors": np.zeros((2, DIMENSION), dtype="float32")}
        first = describe_dense_path(numpy_store)
        self.assertIn("dense_path=numpy", first)
        self.assertIn("rows=2", first)
        self.assertIsNone(describe_dense_path(numpy_store))

        sq8_store = {"sq8": object(), "vectors": np.zeros((2, DIMENSION), dtype="float32")}
        self.assertIn("dense_path=sq8", describe_dense_path(sq8_store))
        self.assertIsNone(describe_dense_path(sq8_store))


class LexicalFallbackTests(unittest.TestCase):
    """A dead provider must not turn into "the corpus has nothing"."""

    def _setup(self, root: Path) -> FaissBackend:
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(DIMENSION)
        vectors = np.zeros((2, DIMENSION), dtype="float32")
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))
        store = {
            "cybersec:0": Document(
                page_content="无关文章，讨论完全不同的主题内容以便区分。" * 4,
                metadata={"source": "13_xianzhi/other.md", "id": "cybersec:0"},
            ),
            "cybersec:1": Document(
                page_content="## eBPF漏洞CVE-2021-3490分析与利用\n\n" + "漏洞源于边界追踪缺陷。" * 4,
                metadata={"source": "13_xianzhi/10613-util.md", "id": "cybersec:1"},
            ),
        }
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(store), {0: "cybersec:0", 1: "cybersec:1"}), handle)
        connection = sqlite3.connect(root / "info.db")
        try:
            connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
            connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
            connection.commit()
        finally:
            connection.close()
        build_cosine_files(root, "cybersec", "bge-m3")
        return FaissBackend(
            RagConfig(
                knowledge_base_root=root,
                allowed_knowledge_bases=frozenset({"cybersec"}),
                embedding_model="bge-m3",
            )
        )

    def test_identifier_is_still_returned_when_embeddings_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._setup(root)
            try:
                with patch.object(backend, "_query_embedding", side_effect=RagEmbeddingError("down")):
                    results = backend.search(
                        RetrievalRequest(query="CVE-2021-3490", knowledge_base="cybersec", top_k=3)
                    )
            finally:
                backend.close()

            self.assertTrue(results)
            self.assertIn("CVE-2021-3490", results[0].content or "")
            self.assertEqual(results[0].metadata["degraded"], "lexical-only")

    def test_fallback_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._setup(root)
            backend.config = RagConfig(
                knowledge_base_root=root,
                allowed_knowledge_bases=frozenset({"cybersec"}),
                embedding_model="bge-m3",
                lexical_fallback=False,
            )
            try:
                with patch.object(
                    backend, "_query_embedding", side_effect=RagEmbeddingError("down")
                ):
                    with self.assertRaises(RagEmbeddingError):
                        backend.search(
                            RetrievalRequest(
                                query="CVE-2021-3490", knowledge_base="cybersec", top_k=3
                            )
                        )
            finally:
                backend.close()

    def test_service_marks_the_response_degraded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._setup(root)
            service = RagService(backend.config, backend)
            try:
                with patch.object(
                    backend, "_query_embedding", side_effect=RagEmbeddingError("down")
                ):
                    response = service.search(
                        RetrievalRequest(query="CVE-2021-3490", knowledge_base="cybersec", top_k=3)
                    )
            finally:
                backend.close()

            self.assertEqual(response.degraded, "lexical-only")
            self.assertTrue(any("lexical-only" in w for w in response.warnings))
            text = response.as_tool_text()
            # The score column no longer means cosine, so the caller must be told
            # before reading the numbers.
            self.assertIn("DEGRADED", text)
            self.assertIn("NOT a cosine similarity", text)


    def test_empty_fallback_response_still_reports_degraded_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._setup(root)
            service = RagService(backend.config, backend)
            try:
                with patch.object(
                    backend, "_query_embedding", side_effect=RagEmbeddingError("down")
                ):
                    response = service.search(
                        RetrievalRequest(
                            query="zzzz_no_indexed_anchor_991827",
                            knowledge_base="cybersec",
                            top_k=3,
                        )
                    )
            finally:
                backend.close()

            self.assertTrue(response.no_match)
            self.assertEqual(response.degraded, "lexical-only")
            self.assertTrue(any("lexical-only" in warning for warning in response.warnings))
            self.assertIn("DEGRADED", response.as_tool_text())


if __name__ == "__main__":
    unittest.main()
