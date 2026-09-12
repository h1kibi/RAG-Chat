"""Agent-facing boundary: argument validation, browsing without a query,
score-component visibility, and the concurrency of the embedding path.

Each test pins a defect reported from real MCP usage.
"""
import concurrent.futures as futures
import inspect
import pickle
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from rag_service.backends.faiss import FaissBackend
from rag_service.build_cosine import build_cosine_files
from rag_service.config import RagConfig
from rag_service.errors import RagEmbeddingError
from rag_service.mcp_server import _check_arguments, ctf_rag
from rag_service.models import (
    RetrievalRequest,
    RetrievalResponse,
    SearchResult,
    _format_signals,
    describe_validation_error,
)
from rag_service.service import RagService

DIMENSION = 4


def _config(**overrides) -> RagConfig:
    values = {
        "knowledge_base_root": Path("."),
        "allowed_knowledge_bases": frozenset({"cybersec"}),
        "embedding_model": "bge-m3",
    }
    values.update(overrides)
    return RagConfig(**values)


class ToolSignatureTests(unittest.TestCase):
    def test_query_is_optional_so_browse_modes_need_no_argument(self):
        # The docstring advertises four browse modes that "omit query"; a
        # required parameter made every one of them fail schema validation.
        parameter = inspect.signature(ctf_rag).parameters["query"]
        self.assertEqual(parameter.default, "")
        self.assertIn("query", inspect.getdoc(ctf_rag))
        self.assertIn("Browse modes (omit query", inspect.getdoc(ctf_rag))

    def test_chunk_id_filter_is_documented(self):
        self.assertIn("chunk_id", inspect.getdoc(ctf_rag))

    def test_mcp_loads_the_same_environment_policy_as_http(self):
        import os
        import rag_service.mcp_server as mcp_server

        with patch.dict(
            os.environ,
            {
                "RAG_KB_ROOT": ".",
                "RAG_ALLOWED_KNOWLEDGE_BASES": "cybersec",
                "RAG_LEXICAL_WEIGHT": "0.17",
                "RAG_DEFAULT_SCORE_THRESHOLD": "0.51",
                "RAG_LEXICAL_FALLBACK": "0",
                "RAG_SNIPPET_CHARS": "123",
            },
            clear=False,
        ):
            config, service = mcp_server._service()
        try:
            self.assertEqual(config.lexical_weight, 0.17)
            self.assertEqual(config.default_score_threshold, 0.51)
            self.assertFalse(config.lexical_fallback)
            self.assertEqual(config.snippet_chars, 123)
        finally:
            service.backend.close()


class ArgumentValidationTests(unittest.TestCase):
    """Failures must be field-level and must not echo oversized input."""

    def _check(self, **overrides):
        kwargs = {
            "query": "x",
            "top_k": 5,
            "limit": None,
            "lexical_weight": None,
            "score_threshold": None,
            "snippet_chars": None,
        }
        kwargs.update(overrides)
        return _check_arguments(_config(max_top_k=50), **kwargs)

    def test_valid_arguments_pass(self):
        self._check()
        self._check(query="", limit=10, lexical_weight=0.35, score_threshold=0.45, snippet_chars=0)

    def test_top_k_out_of_range_names_the_field_and_range(self):
        with self.assertRaisesRegex(ValueError, r"top_k must be an integer in 1\.\.50"):
            self._check(top_k=0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            self._check(top_k=9999)

    def test_lexical_weight_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "lexical_weight must be a number in 0..1"):
            self._check(lexical_weight=5)

    def test_score_threshold_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "score_threshold"):
            self._check(score_threshold=3)

    def test_snippet_chars_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "snippet_chars"):
            self._check(snippet_chars=-1)

    def test_limit_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            self._check(limit=0)

    def test_oversized_query_is_reported_without_being_echoed(self):
        long_query = "A" * 9000
        with self.assertRaises(ValueError) as caught:
            self._check(query=long_query)
        message = str(caught.exception)
        self.assertIn("9000 characters", message)
        self.assertNotIn("AAAA", message)
        self.assertNotIn("pydantic", message)

    def test_non_string_query_is_rejected_without_type_error(self):
        with self.assertRaisesRegex(ValueError, "query must be a string"):
            self._check(query=123)

    def test_all_problems_are_reported_together(self):
        with self.assertRaises(ValueError) as caught:
            self._check(top_k=0, lexical_weight=9)
        message = str(caught.exception)
        self.assertIn("top_k", message)
        self.assertIn("lexical_weight", message)

class ScoreComponentTests(unittest.TestCase):
    def test_components_are_rendered_so_the_split_is_visible(self):
        # When the service warns "top score is low", the caller needs to know
        # whether that score is semantic or a literal word match.
        rendered = _format_signals({"dense_score": 0.722, "lexical_score": 0.236})
        self.assertIn("dense=0.722", rendered)
        self.assertIn("lex=0.236", rendered)

    def test_advisory_flags_still_render_alongside_components(self):
        rendered = _format_signals(
            {
                "dense_score": 0.5,
                "lexical_score": 0.5,
                "truncated": True,
                "merged_chunks": 3,
                "chunk_id_range": "cybersec:5-cybersec:7",
                "has_screenshots": 2,
                "degraded": "lexical-only",
            }
        )
        for token in (
            "dense=0.500",
            "lex=0.500",
            "truncated",
            "merged=3",
            "range=cybersec:5-cybersec:7",
            "shots=2",
            "degraded=",
        ):
            self.assertIn(token, rendered)

    def test_absent_components_add_nothing(self):
        self.assertEqual(_format_signals({}), "")
        self.assertEqual(_format_signals(None), "")

    def test_response_header_carries_components(self):
        response = RetrievalResponse(
            query="q",
            knowledge_base="cybersec",
            total=1,
            backend="faiss",
            results=[
                SearchResult(
                    content="body",
                    score=0.6,
                    source="a/b.md",
                    chunk_id="cybersec:1",
                    metadata={"dense_score": 0.7, "lexical_score": 0.3},
                )
            ],
        )
        header = next(
            line for line in response.as_tool_text().splitlines() if line.startswith("[1]")
        )
        self.assertIn("dense=0.700", header)
        self.assertIn("lex=0.300", header)


class LexicalWeightCouplingTests(unittest.TestCase):
    """Raising lexical_weight also raises the gate; that must be stated."""

    def _warning(self, weight):
        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        return service._lexical_weight_warning(_request(weight))

    def test_raised_weight_warns_about_the_effective_dense_floor(self):
        warning = self._warning(0.6)
        self.assertIsNotNone(warning)
        self.assertIn("also raises the score_threshold gate", warning)
        # 0.45 / (1 - 0.6) = 1.125: impossible for any cosine to reach.
        self.assertIn("1.125", warning)

    def test_weight_at_the_default_is_silent(self):
        self.assertIsNone(self._warning(0.35))

    def test_weight_below_the_default_is_silent(self):
        # Lowering the weight only loosens the gate; it cannot empty results.
        self.assertIsNone(self._warning(0.2))

    def test_omitted_weight_is_silent(self):
        self.assertIsNone(self._warning(None))

    def test_weight_of_one_reports_that_nothing_can_pass(self):
        warning = self._warning(1.0)
        self.assertIn("no score can pass at all", warning)


def _request(weight):
    return RetrievalRequest(
        query="x", knowledge_base="cybersec", top_k=1, lexical_weight=weight
    )


class _CountingEmbedder:
    def __init__(self, vector=None, error=None, delay=0.0):
        self.calls = 0
        self._vector = vector if vector is not None else [1.0, 0.0]
        self._error = error
        self._delay = delay

    def embed_query(self, text):
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._vector


class EmbeddingPathTests(unittest.TestCase):
    def test_identical_concurrent_queries_embed_once(self):
        # A subagent fan-out asking the same question must not pay N provider
        # calls; this is the single-flight guarantee.
        backend = FaissBackend(_config())
        counting = _CountingEmbedder(delay=0.2)
        backend._embedder = lambda model: counting

        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            vectors = list(
                pool.map(lambda _: backend._query_embedding("bge-m3", "same query"), range(6))
            )

        self.assertEqual(counting.calls, 1)
        self.assertTrue(all(vector == [1.0, 0.0] for vector in vectors))

    def test_distinct_queries_each_embed(self):
        backend = FaissBackend(_config())
        counting = _CountingEmbedder()
        backend._embedder = lambda model: counting

        backend._query_embedding("bge-m3", "first")
        backend._query_embedding("bge-m3", "second")

        self.assertEqual(counting.calls, 2)

    def test_failure_is_memoised_within_the_ttl(self):
        # A dead provider otherwise costs a full connect/retry cycle per call
        # (measured 9-12 s in degraded mode).
        backend = FaissBackend(_config(embedding_failure_ttl=30.0))
        counting = _CountingEmbedder(error=RagEmbeddingError("down"))
        backend._embedder = lambda model: counting

        for _ in range(3):
            with self.assertRaises(RagEmbeddingError):
                backend._query_embedding("bge-m3", "same query")

        self.assertEqual(counting.calls, 1)

    def test_failure_is_retried_after_the_ttl(self):
        backend = FaissBackend(_config(embedding_failure_ttl=0.0))
        counting = _CountingEmbedder(error=RagEmbeddingError("down"))
        backend._embedder = lambda model: counting

        for _ in range(2):
            with self.assertRaises(RagEmbeddingError):
                backend._query_embedding("bge-m3", "same query")

        self.assertEqual(counting.calls, 2)

    def test_success_after_failure_is_cached(self):
        backend = FaissBackend(_config(embedding_failure_ttl=0.0))
        counting = _CountingEmbedder()
        backend._embedder = lambda model: counting

        backend._query_embedding("bge-m3", "q")
        backend._query_embedding("bge-m3", "q")

        self.assertEqual(counting.calls, 1)

    def test_concurrent_failures_share_one_attempt(self):
        backend = FaissBackend(_config(embedding_failure_ttl=30.0))
        counting = _CountingEmbedder(error=RagEmbeddingError("down"), delay=0.2)
        backend._embedder = lambda model: counting

        def attempt(_):
            try:
                backend._query_embedding("bge-m3", "same query")
            except RagEmbeddingError:
                return "failed"
            return "ok"

        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(attempt, range(6)))

        self.assertEqual(counting.calls, 1)
        self.assertEqual(set(outcomes), {"failed"})


class DegenerateCombinationTests(unittest.TestCase):
    """Combinations that cannot mean anything must fail, not silently no-op."""

    def test_chunk_id_with_a_query_is_rejected(self):
        # Silently ignoring chunk_id returned unrelated ranked results while the
        # caller believed it had fetched one exact chunk.
        from rag_service.backends.faiss import FaissBackend

        backend = FaissBackend(_config())
        with self.assertRaisesRegex(ValueError, "cannot be combined with a query"):
            backend.search(
                RetrievalRequest(
                    query="rsync",
                    knowledge_base="cybersec",
                    top_k=2,
                    filters={"chunk_id": "cybersec:1"},
                )
            )

    def test_chunk_id_alongside_other_filters_is_rejected(self):
        from rag_service.backends.faiss import FaissBackend

        backend = FaissBackend(_config())
        with self.assertRaisesRegex(ValueError, "ignores other filters"):
            backend.search(
                RetrievalRequest(
                    query="",
                    knowledge_base="cybersec",
                    filters={"chunk_id": "cybersec:1", "source": "a/b.md"},
                )
            )


class FailureReplayTests(unittest.TestCase):
    def test_each_caller_gets_its_own_exception_object(self):
        # Re-raising one stored instance shares mutable traceback state across
        # callers; a fan-out would see each other's frames in the traceback.
        backend = FaissBackend(_config(embedding_failure_ttl=30.0))
        counting = _CountingEmbedder(error=RagEmbeddingError("down"), delay=0.2)
        backend._embedder = lambda model: counting

        held = []

        def attempt(_):
            try:
                backend._query_embedding("bge-m3", "shared")
            except RagEmbeddingError as exc:
                held.append(exc)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(counting.calls, 1)
        self.assertEqual(len(held), 5)
        self.assertEqual(len({id(item) for item in held}), 5)
        self.assertEqual(len({id(item.__traceback__) for item in held}), 5)
        self.assertTrue(all(str(item) == "down" for item in held))

    def test_interrupt_is_propagated_and_not_memoised(self):
        # Caching a BaseException replayed a phantom KeyboardInterrupt on later
        # queries without even contacting the provider.
        backend = FaissBackend(_config(embedding_failure_ttl=30.0))
        counting = _CountingEmbedder(error=KeyboardInterrupt("ctrl-c"))
        backend._embedder = lambda model: counting

        for attempt_no in (1, 2):
            with self.assertRaises(KeyboardInterrupt):
                backend._query_embedding("bge-m3", "same")

        self.assertEqual(counting.calls, 2)
        self.assertEqual(len(backend._embed_failures), 0)

    def test_degraded_metadata_renders_a_lexical_component(self):
        # The rendering contract: with no dense term present, the line still
        # shows the lexical value that produced the score.
        rendered = _format_signals({"lexical_score": 0.45, "degraded": "lexical-only"})
        self.assertIn("lex=0.450", rendered)
        self.assertIn("degraded=lexical-only", rendered)
        self.assertNotIn("dense=", rendered)


class ScorerConsistencyTests(unittest.TestCase):
    """The subset path must use the same quantized scorer as the full scan.

    The full scan uses faiss SQ8 while subsets used the independent per-row int8
    file, so the same document scored differently depending on whether a filter
    was present — enough to flip a near-tie in the reported glibc query.
    """

    def _index(self, root: Path) -> None:
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)
        rows = 6
        index = faiss.IndexFlatIP(DIMENSION)
        rng = np.random.default_rng(11)
        vectors = rng.normal(size=(rows, DIMENSION)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))
        store = {}
        mapping = {}
        for row in range(rows):
            doc_id = f"cybersec:{row}"
            store[doc_id] = Document(
                page_content=f"第 {row} 篇文档正文，包含足够的字符以通过低信息过滤。" * 6,
                metadata={"source": f"13_xianzhi/doc-{row}.md", "id": doc_id},
            )
            mapping[row] = doc_id
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(store), mapping), handle)
        connection = sqlite3.connect(root / "info.db")
        try:
            connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
            connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
            connection.commit()
        finally:
            connection.close()
        build_cosine_files(root, "cybersec", "bge-m3")

    def test_subset_and_full_scan_scores_are_identical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index(root)
            backend = FaissBackend(
                RagConfig(
                    knowledge_base_root=root,
                    allowed_knowledge_bases=frozenset({"cybersec"}),
                    embedding_model="bge-m3",
                    # equal pools so only the scorer can differ
                    candidate_pool=6,
                    filtered_candidate_limit=6,
                )
            )
            try:
                store = backend._load_store("cybersec", "bge-m3")
                self.assertIsNotNone(store.get("sq8"))
                query = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="float32")
                rows = np.arange(6, dtype=np.int64)

                from rag_service.backends.faiss import _score_rows

                subset = _score_rows(store, query, rows)
                full = store["sq8"].search(
                    np.ascontiguousarray(query.reshape(1, -1)), 6
                )
                by_row = {int(row): float(score) for row, score in zip(full[1][0], full[0][0])}
                for row, score in zip(rows, subset):
                    self.assertAlmostEqual(
                        float(score), by_row[int(row)], places=5,
                        msg=f"row {row}: subset={score} full={by_row[int(row)]}",
                    )
            finally:
                backend.close()

    def test_filtered_and_unfiltered_agree_when_pools_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index(root)
            backend = FaissBackend(
                RagConfig(
                    knowledge_base_root=root,
                    allowed_knowledge_bases=frozenset({"cybersec"}),
                    embedding_model="bge-m3",
                    candidate_pool=6,
                    filtered_candidate_limit=6,
                )
            )
            try:
                with patch.object(backend, "_embedder") as mock_embedder:
                    mock_embedder.return_value.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]
                    plain = backend.search(
                        RetrievalRequest(
                            query="文档", knowledge_base="cybersec", top_k=3,
                            score_threshold=0.0,
                        )
                    )
                    # a filter that excludes nothing must not change ranking
                    filtered = backend.search(
                        RetrievalRequest(
                            query="文档", knowledge_base="cybersec", top_k=3,
                            score_threshold=0.0,
                            filters={"exclude_source_prefix": "zzz_no_such_prefix"},
                        )
                    )
            finally:
                backend.close()

            self.assertEqual([r.chunk_id for r in plain], [r.chunk_id for r in filtered])
            for left, right in zip(plain, filtered):
                self.assertAlmostEqual(left.score, right.score, places=6)

    def test_subset_scoring_falls_back_without_faiss(self):
        # With no sq8 index the int8 numpy path must still answer, so the
        # degraded environment keeps working.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index(root)
            backend = FaissBackend(
                RagConfig(
                    knowledge_base_root=root,
                    allowed_knowledge_bases=frozenset({"cybersec"}),
                    embedding_model="bge-m3",
                )
            )
            try:
                store = backend._load_store("cybersec", "bge-m3")
                store.pop("sq8")
                query = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="float32")
                from rag_service.backends.faiss import _score_rows

                scores = _score_rows(store, query, np.arange(6, dtype=np.int64))
                self.assertEqual(scores.shape, (6,))
                self.assertTrue(np.isfinite(scores).all())
            finally:
                backend.close()


class YearFilterScopingTests(unittest.TestCase):
    """`year` is a source-path filter and must be scoped like the others.

    It was excluded from the row narrowing, so ``year=2014`` searched only the
    global candidate pool (600 rows) while 1452 rows actually matched, and
    returned 3 hits where 5 were reachable.
    """

    def _index(self, root: Path) -> None:
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)
        docs = [
            ("# 2014 exploit analysis\n\n" + "漏洞位于边界检查缺失处。" * 12, "14_ctf_wp/by-year/2014/a.md"),
            ("# 2014 second\n\n" + "利用方式为堆布局操控。" * 12, "13_xianzhi/CVE-2014-3153.md"),
            ("# 2021 unrelated\n\n" + "与该年份无关的内容。" * 12, "13_xianzhi/2021-topic.md"),
        ]
        index = faiss.IndexFlatIP(DIMENSION)
        vectors = np.zeros((len(docs), DIMENSION), dtype="float32")
        vectors[:, 0] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))
        store = {}
        mapping = {}
        for row, (text, source) in enumerate(docs):
            doc_id = f"cybersec:{row}"
            store[doc_id] = Document(page_content=text, metadata={"source": source, "id": doc_id})
            mapping[row] = doc_id
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(store), mapping), handle)
        connection = sqlite3.connect(root / "info.db")
        try:
            connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
            connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
            connection.commit()
        finally:
            connection.close()
        build_cosine_files(root, "cybersec", "bge-m3")

    def _search(self, backend, **overrides):
        """Search with a deterministic 4-dim query vector (the fixture width)."""
        kwargs = {
            "query": "漏洞 利用",
            "knowledge_base": "cybersec",
            "top_k": 10,
            "score_threshold": 0.0,
        }
        kwargs.update(overrides)
        with patch.object(backend, "_embedder") as mock_embedder:
            mock_embedder.return_value.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]
            return backend.search(RetrievalRequest(**kwargs))

    def test_year_is_treated_as_a_path_filter(self):
        from rag_service.backends.faiss import _has_path_filters

        self.assertTrue(_has_path_filters({"year": 2014}))

    def test_year_filter_returns_every_matching_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index(root)
            backend = FaissBackend(
                RagConfig(
                    knowledge_base_root=root,
                    allowed_knowledge_bases=frozenset({"cybersec"}),
                    embedding_model="bge-m3",
                )
            )
            try:
                results = self._search(backend, filters={"year": 2014})
            finally:
                backend.close()

            sources = {result.source for result in results}
            self.assertEqual(
                sources,
                {"14_ctf_wp/by-year/2014/a.md", "13_xianzhi/CVE-2014-3153.md"},
            )

    def test_year_filter_excludes_other_years(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index(root)
            backend = FaissBackend(
                RagConfig(
                    knowledge_base_root=root,
                    allowed_knowledge_bases=frozenset({"cybersec"}),
                    embedding_model="bge-m3",
                )
            )
            try:
                results = self._search(backend, filters={"year": 2021})
            finally:
                backend.close()
            self.assertEqual([r.source for r in results], ["13_xianzhi/2021-topic.md"])


class _FakeHandle:
    def __init__(self, log):
        self._log = log

    def close(self):
        self._log.append("closed")


class _FakeMemmap:
    """Mirrors numpy's memmap surface: a ``_mmap`` attribute with close()."""

    def __init__(self, log):
        self._mmap = _FakeHandle(log)


class StoreLifecycleTests(unittest.TestCase):
    """Rebuilds must not accumulate resident index generations."""

    def _backend(self, root: Path, limit: int) -> FaissBackend:
        YearFilterScopingTests()._index(root)
        return FaissBackend(
            RagConfig(
                knowledge_base_root=root,
                allowed_knowledge_bases=frozenset({"cybersec"}),
                embedding_model="bge-m3",
                store_cache_limit=limit,
            )
        )

    def test_store_cache_is_bounded_and_releases_evicted_generations(self):
        # Each generation holds an open memmap over the full vector matrix, so
        # an unbounded cache leaks GBs per rebuild into a long-running service.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._backend(root, limit=2)
            closes = []
            generations = {"n": 0}

            def fake_open(*_args, **_kwargs):
                generations["n"] += 1
                return {
                    "vectors": _FakeMemmap(closes),
                    "quantized": None,
                    "sq8": None,
                    "documents": {},
                }

            def fake_manifest(*_args, **_kwargs):
                # Distinct fingerprint per generation => distinct cache key,
                # exactly as a real rebuild produces.
                return {
                    "source": {"index.faiss": {"size": 1, "mtime_ns": generations["n"]}},
                    "artifacts": {"vectors_bytes": 1},
                }

            try:
                with patch(
                    "rag_service.backends.faiss._read_current_manifest",
                    side_effect=fake_manifest,
                ), patch.object(
                    FaissBackend, "_open_store", staticmethod(fake_open)
                ):
                    for _ in range(6):
                        backend._load_store("cybersec", "bge-m3")
                self.assertEqual(len(backend._stores), 2)
                # 6 opened, 2 retained, 4 must have been released
                self.assertEqual(len(closes), 4)
            finally:
                backend.close()

    def test_close_releases_every_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._backend(root, limit=3)
            backend._load_store("cybersec", "bge-m3")
            self.assertEqual(len(backend._stores), 1)
            backend.close()
            self.assertEqual(len(backend._stores), 0)

    def test_closed_store_drops_native_handles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = self._backend(root, limit=3)
            backend._load_store("cybersec", "bge-m3")
            backend.close()
            # _release_store must null the handles, not merely forget the dict
            self.assertEqual(backend._stores, {})

    def test_cache_put_survives_concurrent_eviction(self):
        # Module caches are shared process-wide; an unguarded eviction loop can
        # raise StopIteration when another thread empties the dict.
        import rag_service.backends.faiss as backend_module

        backend_module._RANGES_CACHE.clear()
        errors = []

        def writer(thread_id):
            try:
                for index in range(50):
                    backend_module._cache_put(
                        backend_module._RANGES_CACHE,
                        (f"t{thread_id}-{index}", index, index, index),
                        {"a": (0, 1)},
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        size = len(backend_module._RANGES_CACHE)
        backend_module._RANGES_CACHE.clear()
        self.assertEqual(errors, [])
        self.assertLessEqual(size, backend_module._CACHE_ENTRIES)


class ModuleCacheBoundTests(unittest.TestCase):
    """Fingerprint-keyed caches must not retain every generation forever.

    A long-lived service that survives rebuilds used to accumulate one entry per
    index generation: each source-range map holds ~20k entries and nothing
    evicted it, not even ``close()``.
    """

    def test_ranges_cache_evicts_oldest_beyond_the_bound(self):
        import rag_service.backends.faiss as backend_module

        backend_module._RANGES_CACHE.clear()
        try:
            for generation in range(backend_module._CACHE_ENTRIES + 5):
                backend_module._cache_put(
                    backend_module._RANGES_CACHE,
                    (f"docs-{generation}", generation, generation, generation),
                    {"a.md": (0, 1)},
                )
            self.assertEqual(
                len(backend_module._RANGES_CACHE), backend_module._CACHE_ENTRIES
            )
            self.assertNotIn(
                ("docs-0", 0, 0, 0), backend_module._RANGES_CACHE
            )
            newest = backend_module._CACHE_ENTRIES + 4
            self.assertIn(
                (f"docs-{newest}", newest, newest, newest), backend_module._RANGES_CACHE
            )
        finally:
            backend_module._RANGES_CACHE.clear()

    def test_postings_cache_evicts_oldest_beyond_the_bound(self):
        import rag_service.backends.faiss as backend_module

        backend_module._POSTINGS_CACHE.clear()
        try:
            for generation in range(backend_module._CACHE_ENTRIES + 3):
                backend_module._cache_put(
                    backend_module._POSTINGS_CACHE, (f"p-{generation}", 1, 2), {}
                )
            self.assertEqual(
                len(backend_module._POSTINGS_CACHE), backend_module._CACHE_ENTRIES
            )
        finally:
            backend_module._POSTINGS_CACHE.clear()


class DegradedPathParityTests(unittest.TestCase):
    """The lexical fallback must apply the same content guards as ranking.

    The postings index reaches low-information rows: version-like tokens
    ("1.13") occur inside the large coordinate/binary dumps, so a degraded
    lookup without the guards handed back raw debug dumps as evidence.
    """

    def _index_with_dump(self, root: Path) -> None:
        """One junk row (numeric dump) plus a two-chunk real document."""
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)
        rows = 3
        index = faiss.IndexFlatIP(DIMENSION)
        vectors = np.zeros((rows, DIMENSION), dtype="float32")
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        vectors[2, 1] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))

        dump = "{60.91, 22106301.01}, {0.00, 2.41}, {2.84, 5.35}, {1.13, 9.99}\n" * 4
        store = {
            "cybersec:0": Document(
                page_content=dump,
                metadata={"source": "14_ctf_wp/debug/params.txt", "id": "cybersec:0"},
            ),
            "cybersec:1": Document(
                page_content=(
                    "## Form Maker 1.13.3 SQL injection analysis\n\n"
                    "The plugin passes the parameter into a query without binding, so 1.13 "
                    "is exploitable.\n"
                )
                * 2,
                metadata={"source": "13_xianzhi/form-maker-1.13.md", "id": "cybersec:1"},
            ),
            "cybersec:2": Document(
                page_content=(
                    "## Form Maker 1.13.3 remediation\n\n"
                    "Bind the parameter and upgrade past 1.13.3 to close the issue.\n"
                )
                * 2,
                metadata={"source": "13_xianzhi/form-maker-1.13.md", "id": "cybersec:2"},
            ),
        }
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump(
                (InMemoryDocstore(store), {0: "cybersec:0", 1: "cybersec:1", 2: "cybersec:2"}),
                handle,
            )
        connection = sqlite3.connect(root / "info.db")
        try:
            connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
            connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
            connection.commit()
        finally:
            connection.close()
        build_cosine_files(root, "cybersec", "bge-m3")

    def _config(self, root: Path, **overrides) -> RagConfig:
        values = {
            "knowledge_base_root": root,
            "allowed_knowledge_bases": frozenset({"cybersec"}),
            "embedding_model": "bge-m3",
        }
        values.update(overrides)
        return RagConfig(**values)

    def test_degraded_path_drops_low_information_rows(self):
        from rag_service.errors import RagEmbeddingError as EmbeddingError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index_with_dump(root)
            backend = FaissBackend(self._config(root))
            try:
                with patch.object(
                    backend, "_query_embedding", side_effect=EmbeddingError("down")
                ):
                    results = backend.search(
                        RetrievalRequest(query="1.13", knowledge_base="cybersec", top_k=5)
                    )
            finally:
                backend.close()

            bodies = [result.content or "" for result in results]
            self.assertTrue(bodies, "degraded path returned nothing")
            self.assertTrue(
                all("22106301.01" not in body for body in bodies),
                f"degraded path returned the numeric dump: {bodies}",
            )

    def test_degraded_path_merges_neighbours_like_the_main_path(self):
        from rag_service.errors import RagEmbeddingError as EmbeddingError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._index_with_dump(root)
            backend = FaissBackend(self._config(root))
            try:
                with patch.object(
                    backend, "_query_embedding", side_effect=EmbeddingError("down")
                ):
                    results = backend.search(
                        RetrievalRequest(query="1.13", knowledge_base="cybersec", top_k=2)
                    )
            finally:
                backend.close()

            self.assertTrue(results)
            self.assertEqual(results[0].metadata.get("degraded"), "lexical-only")
            # merge must actually run, not be silently skipped
            self.assertIn("merged_chunks", results[0].metadata)


class BrowseCombinationTests(unittest.TestCase):
    def test_cursor_with_chunk_id_is_rejected(self):
        # One chunk is not a listing; accepting a pagination token silently
        # implied it had been honoured.
        from rag_service.backends.faiss import FaissBackend

        backend = FaissBackend(_config())
        with self.assertRaisesRegex(ValueError, "cannot be paginated"):
            backend.search(
                RetrievalRequest(
                    query="",
                    knowledge_base="cybersec",
                    cursor="15_butian/x.md",
                    filters={"chunk_id": "cybersec:1"},
                )
            )


class InProcessAdapterTests(unittest.TestCase):
    """`register_mcp_tool` is a documented entry point (MCP.md §12).

    It had no coverage, which let two defects live: registration raised before
    the host server ever started, and validation leaked pydantic's dump.
    """

    @staticmethod
    def _empty_response() -> RetrievalResponse:
        return RetrievalResponse(
            query="q",
            knowledge_base="cybersec",
            results=[],
            total=0,
            backend="faiss",
        )

    def _call(self, name="rag_lookup", **arguments):
        """Register on a REAL FastMCP server and invoke through it.

        Returns ``(structured_payload, listed_tools)``. ``call_tool`` yields a
        ``(content_blocks, payload)`` pair; the payload is what an MCP client
        receives as structured content.
        """
        import asyncio

        from mcp.server.fastmcp import FastMCP

        from rag_service.mcp_adapter import register_mcp_tool

        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        server = FastMCP("host-app")
        try:
            register_mcp_tool(server, service, name=name)
            listed = asyncio.run(server.list_tools())
            _, payload = asyncio.run(server.call_tool(name, arguments))
            return payload, listed
        finally:
            backend.close()

    def test_registration_succeeds_on_a_real_mcp_server(self):
        # Regression: postponed annotations made every parameter a string, and
        # the SDK's `issubclass(param.annotation, Context)` then raised
        # TypeError during registration -- the server never came up.
        with patch.object(FaissBackend, "search", return_value=[]), patch.object(
            RagService, "search", return_value=self._empty_response()
        ):
            _, listed = self._call(query="q")
        self.assertEqual([tool.name for tool in listed], ["rag_lookup"])
        properties = listed[0].inputSchema.get("properties", {})
        self.assertIn("query", properties)
        self.assertIn("filters", properties)

    def test_a_valid_call_returns_the_service_payload(self):
        result = SearchResult(
            content="evidence", score=0.9, source="a/b.md", chunk_id="cybersec:1"
        )
        response = RetrievalResponse(
            query="q",
            knowledge_base="cybersec",
            results=[result],
            total=1,
            backend="faiss",
        )
        with patch.object(RagService, "search", return_value=response):
            payload, _ = self._call(query="q")
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["results"][0]["chunk_id"], "cybersec:1")

    def test_invalid_arguments_report_a_fixable_message(self):
        with self.assertRaises(Exception) as ctx:
            self._call(query="x", top_k=0)
        message = str(ctx.exception)
        self.assertIn("top_k", message)
        self.assertNotIn("errors.pydantic.dev", message)


class ValidationSurfaceTests(unittest.TestCase):
    """Every entry point must fail with a usable message, not pydantic's dump."""

    def test_formatter_is_field_level_and_drops_echoed_input(self):
        from pydantic import ValidationError

        try:
            RetrievalRequest(query="A" * 9000)
        except ValidationError as exc:
            message = describe_validation_error(exc)
        self.assertIn("query", message)
        self.assertIn("8000", message)
        self.assertNotIn("AAAA", message)
        self.assertNotIn("pydantic", message.lower())

    def test_openai_dispatch_reports_a_fixable_message(self):
        from rag_service.adapters import dispatch_openai_tool_call

        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        with self.assertRaisesRegex(ValueError, "top_k"):
            dispatch_openai_tool_call(service, {"query": "x", "top_k": 0})

    def test_langchain_tool_reports_a_fixable_message(self):
        # The last entry point to be translated: it used to hand back pydantic's
        # dump, naming an errors.pydantic.dev URL instead of the field.
        from rag_service.adapters import create_langchain_tool

        backend = FaissBackend(_config())
        tool = create_langchain_tool(RagService(backend.config, backend))
        with self.assertRaisesRegex(ValueError, "top_k") as ctx:
            tool.invoke({"query": "x", "top_k": 0})
        self.assertNotIn("pydantic.dev", str(ctx.exception))

    def test_openai_dispatch_still_validates_successfully(self):
        from unittest.mock import patch

        from rag_service.adapters import dispatch_openai_tool_call

        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        with patch.object(backend, "search", return_value=[]):
            payload = dispatch_openai_tool_call(service, {"query": "x"})
        self.assertEqual(payload["total"], 0)

    def test_http_validation_error_does_not_echo_the_payload(self):
        from fastapi.testclient import TestClient

        from rag_service.http_api import create_app

        backend = FaissBackend(_config())
        app = create_app(RagService(backend.config, backend))
        client = TestClient(app)

        response = client.post("/v1/rag/search", json={"query": "A" * 9000})
        self.assertEqual(response.status_code, 422)
        body = response.text
        self.assertNotIn("A" * 100, body)
        self.assertNotIn("errors.pydantic.dev", body)
        self.assertIn("8000", response.json()["detail"])

    def test_http_still_returns_400_for_service_level_errors(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from rag_service.http_api import create_app

        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        app = create_app(service)
        client = TestClient(app)

        with patch.object(service, "search", side_effect=ValueError("bad filters")):
            response = client.post("/v1/rag/search", json={"query": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad filters")


if __name__ == "__main__":
    unittest.main()
