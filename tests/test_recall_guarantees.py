"""Recall guarantees: exact identifiers must not depend on dense rank.

The reported failure: a document stating ``CVE-2021-3490`` in its body had a
dense rank of 628 in a 1,016,721-row index, one slot outside the 600-row rerank
pool, and its *filename* carried a different number so the path signal missed it
too. The query returned ``no_match`` — a false negative the caller reads as "the
corpus does not cover this".

These tests pin both halves: an identifier whose document lies outside the dense
pool, and the filter semantics that a typo used to bypass silently.
"""
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

from rag_service.backends.faiss import FaissBackend, _lookup_postings, _open_postings
from rag_service.build_cosine import build_cosine_files
from rag_service.config import RagConfig
from rag_service.models import RetrievalRequest

DIMENSION = 8


def _write_metadata(root: Path, kb_name: str, embed_model: str) -> None:
    connection = sqlite3.connect(root / "info.db")
    try:
        connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
        connection.execute(
            "INSERT INTO knowledge_base VALUES (?, ?)", (kb_name, embed_model)
        )
        connection.commit()
    finally:
        connection.close()


class RecallGuaranteeTests(unittest.TestCase):
    """The identifier document must be found although dense ranks it last."""

    NOISE_ROWS = 30

    def _build_index(self, root: Path) -> Path:
        """One body-only identifier document buried under high-scoring noise."""
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(DIMENSION)
        vectors = np.zeros((self.NOISE_ROWS + 1, DIMENSION), dtype="float32")
        # noise: all identical and ranked above the target, but at the observed
        # top score (0.70) rather than 1.0, so the lexical term can overturn it.
        noise = np.asarray([0.70, 0.0, 0.7141], dtype="float32")
        noise /= np.linalg.norm(noise)
        vectors[: self.NOISE_ROWS, :3] = noise
        # target: genuinely relevant but ranked below the noise, mirroring the
        # reported case (dense 0.5063 vs a 0.70 top score). Low enough to fall
        # outside a small pool, high enough that lexical evidence lifts it to
        # the top once it is in the candidate set.
        target = np.asarray([0.55, 0.8352], dtype="float32")
        target /= np.linalg.norm(target)
        vectors[self.NOISE_ROWS, :2] = target
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))

        store = {}
        mapping = {}
        for row in range(self.NOISE_ROWS):
            doc_id = f"cybersec:{row}"
            store[doc_id] = Document(
                page_content=f"泛化的漏洞利用笔记，与查询无关。编号 {row}。",
                metadata={"source": f"13_xianzhi/noise-{row}.md", "id": doc_id},
            )
            mapping[row] = doc_id
        target_id = f"cybersec:{self.NOISE_ROWS}"
        # The filename deliberately carries a *different* number, which is the
        # detail that made the path-recall guarantee useless in the report.
        store[target_id] = Document(
            page_content=(
                "# eBPF Verifier 分析\n\n"
                "## eBPF漏洞CVE-2021-3490分析与利用\n\n"
                "该漏洞源于 verifier 对 32 位运算的边界追踪缺陷，可导致越界读写。\n"
            ),
            metadata={"source": "13_xianzhi/10613-CVE-2021-3493分析与利用.md", "id": target_id},
        )
        mapping[self.NOISE_ROWS] = target_id
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(store), mapping), handle, protocol=pickle.HIGHEST_PROTOCOL)

        _write_metadata(root, "cybersec", "bge-m3")
        return index_path

    def _config(self, root: Path) -> RagConfig:
        return RagConfig(
            knowledge_base_root=root,
            allowed_knowledge_bases=frozenset({"cybersec"}),
            embedding_model="bge-m3",
            lexical_weight=0.35,
            # Pool of 5 guarantees the target (dense rank 31) is outside it.
            candidate_pool=5,
            path_recall_limit=0,
        )

    def _query_vector(self) -> np.ndarray:
        vector = np.zeros(DIMENSION, dtype="float32")
        vector[0] = 1.0
        return vector

    def test_identifier_is_recalled_from_outside_the_dense_pool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            self.assertTrue((index_path / "docs.cos.postings").is_file())

            config = self._config(root)
            backend = FaissBackend(config)
            try:
                with patch.object(backend, "_embedder") as mock_embedder:
                    mock_embedder.return_value.embed_query.return_value = self._query_vector()
                    results = backend.search(
                        RetrievalRequest(
                            query="CVE-2021-3490",
                            knowledge_base="cybersec",
                            top_k=3,
                            score_threshold=0.0,
                        )
                    )
            finally:
                backend.close()

            sources = [result.source for result in results]
            self.assertIn("13_xianzhi/10613-CVE-2021-3493分析与利用.md", sources)

    def test_without_postings_the_target_stays_out_of_reach(self):
        # Control: the guarantee is what finds it, not the dense ranking. Delete
        # the postings index and the same query misses, proving the test would
        # fail if the lookup regressed.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            (index_path / "docs.cos.postings").unlink()
            (index_path / "docs.cos.postings.idx.json").unlink()

            config = self._config(root)
            backend = FaissBackend(config)
            try:
                with patch.object(backend, "_embedder") as mock_embedder:
                    mock_embedder.return_value.embed_query.return_value = self._query_vector()
                    results = backend.search(
                        RetrievalRequest(
                            query="CVE-2021-3490",
                            knowledge_base="cybersec",
                            top_k=3,
                            score_threshold=0.0,
                        )
                    )
            finally:
                backend.close()

            self.assertNotIn(
                "13_xianzhi/10613-CVE-2021-3493分析与利用.md",
                [result.source for result in results],
            )

    def test_postings_lookup_returns_the_indexed_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")

            docs_path = index_path / "docs.cos.jsonl"
            postings = _open_postings(docs_path)
            self.assertIsNotNone(postings)
            rows = _lookup_postings(postings, "cve-2021-3490", 10)
            self.assertEqual(rows, [self.NOISE_ROWS])

    def test_missing_token_yields_no_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            postings = _open_postings(index_path / "docs.cos.jsonl")
            self.assertEqual(_lookup_postings(postings, "cve-9999-99999", 10), [])

    def test_frequent_tokens_are_not_indexed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            postings = _open_postings(index_path / "docs.cos.jsonl")
            # "编号" appears in every noise chunk; digits alone must not be
            # indexed either, or a query for "2021" would pull the whole corpus.
            self.assertEqual(_lookup_postings(postings, "1121", 10), [])

    def test_sq8_index_is_built_and_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            self.assertTrue((index_path / "vectors.cos.sq8").is_file())

            config = self._config(root)
            backend = FaissBackend(config)
            try:
                store = backend._load_store("cybersec", "bge-m3")
                self.assertIsNotNone(store.get("sq8"))
                self.assertEqual(store["sq8"].ntotal, self.NOISE_ROWS + 1)
            finally:
                backend.close()


class ChunkResolutionTests(unittest.TestCase):
    """A citation must be verifiable: source + chunk_id is the quoting pair."""

    def _backend(self, root: Path) -> FaissBackend:
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
                page_content="第一块内容，讨论完全不同的话题以免混淆。" * 3,
                metadata={"source": "13_xianzhi/a.md", "id": "cybersec:0"},
            ),
            "cybersec:1": Document(
                page_content="## 目标小节\n\n这里是被引用的那一块正文。" * 3,
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
        build_cosine_files(root, "cybersec", "bge-m3")
        return FaissBackend(
            RagConfig(
                knowledge_base_root=root,
                allowed_knowledge_bases=frozenset({"cybersec"}),
                embedding_model="bge-m3",
            )
        )

    def test_chunk_id_returns_exactly_the_cited_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = self._backend(Path(temp_dir))
            try:
                results = backend.search(
                    RetrievalRequest(
                        query="", knowledge_base="cybersec", filters={"chunk_id": "cybersec:1"}
                    )
                )
            finally:
                backend.close()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].source, "13_xianzhi/b.md")
            self.assertEqual(results[0].chunk_id, "cybersec:1")
            self.assertIn("被引用的那一块正文", results[0].content or "")
            self.assertTrue(results[0].metadata["chunk"])

    def test_malformed_chunk_id_is_rejected_with_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = self._backend(Path(temp_dir))
            try:
                with self.assertRaisesRegex(ValueError, "knowledge_base"):
                    backend.search(
                        RetrievalRequest(
                            query="", knowledge_base="cybersec", filters={"chunk_id": "nonsense"}
                        )
                    )
            finally:
                backend.close()

    def test_out_of_range_chunk_id_returns_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = self._backend(Path(temp_dir))
            try:
                results = backend.search(
                    RetrievalRequest(
                        query="",
                        knowledge_base="cybersec",
                        filters={"chunk_id": "cybersec:99999999"},
                    )
                )
            finally:
                backend.close()
            self.assertEqual(results, [])

    def test_chunk_id_from_another_knowledge_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = self._backend(Path(temp_dir))
            try:
                with self.assertRaisesRegex(ValueError, "must match the request"):
                    backend.search(
                        RetrievalRequest(
                            query="",
                            knowledge_base="cybersec",
                            filters={"chunk_id": "samples:1"},
                        )
                    )
            finally:
                backend.close()

    def test_chunk_id_filter_is_accepted_by_the_request_model(self):
        request = RetrievalRequest(query="", filters={"chunk_id": "cybersec:1"})
        self.assertEqual(request.filters["chunk_id"], "cybersec:1")

    def test_chunk_id_is_exposed_by_every_entry_surface(self):
        # Regression: chunk_id was added to the model and backend but not to the
        # MCP tool signature, so the documented verify-a-citation workflow was
        # unreachable from the agent's only entry point.
        import inspect

        from rag_service.mcp_server import ctf_rag

        self.assertIn("chunk_id", inspect.signature(ctf_rag).parameters)
        self.assertIn("chunk_id", inspect.getdoc(ctf_rag))

        from rag_service.__main__ import _FILTER_KEYS

        self.assertIn("chunk_id", _FILTER_KEYS)

        from rag_service.adapters import create_openai_tool_schema

        properties = create_openai_tool_schema()["function"]["parameters"]["properties"]
        self.assertIn("chunk_id", properties["filters"]["properties"])

    def test_unknown_filter_key_still_rejected(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RetrievalRequest(query="", filters={"chunk": "1"})


class FilterSemanticsTests(unittest.TestCase):
    """Prefix filters are literal, so a partial segment cannot silently no-op."""

    def test_exclude_matches_partial_segment(self):
        from rag_service.backends.faiss import _matches_filters

        source = "13_xianzhi/17572-某文章.md"
        self.assertFalse(_matches_filters(source, {"exclude_source_prefix": "13_xianzhi/17"}))
        self.assertFalse(_matches_filters(source, {"exclude_source_prefix": ["13_xianzhi/175"]}))
        self.assertFalse(_matches_filters(source, {"exclude_source_prefix": "13_xianzhi"}))

    def test_unrelated_exclude_keeps_the_document(self):
        from rag_service.backends.faiss import _matches_filters

        source = "13_xianzhi/17572-某文章.md"
        self.assertTrue(_matches_filters(source, {"exclude_source_prefix": "12_security_learning"}))

    def test_category_stays_segment_scoped(self):
        # category is documented as the first path segment, so a sibling
        # directory sharing the prefix must not match.
        from rag_service.backends.faiss import _matches_filters

        self.assertFalse(_matches_filters("13_xianzhi_notes/x.md", {"category": "13_xianzhi"}))
        self.assertTrue(_matches_filters("13_xianzhi/x.md", {"category": "13_xianzhi"}))

    def test_source_prefix_is_literal(self):
        from rag_service.backends.faiss import _matches_filters

        self.assertTrue(
            _matches_filters("14_ctf_wp/by-year/2014/x.md", {"source_prefix": "14_ctf_wp/by-year/2014/"})
        )
        self.assertTrue(
            _matches_filters("14_ctf_wp/by-year/2014/x.md", {"source_prefix": "14_ctf_wp/by-year/201"})
        )
        self.assertFalse(
            _matches_filters("14_ctf_wp/by-year/2015/x.md", {"source_prefix": "14_ctf_wp/by-year/2014"})
        )


if __name__ == "__main__":
    unittest.main()
