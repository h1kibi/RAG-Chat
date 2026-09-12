"""Backend operational state: the degraded marker, and sidecar freshness.

Two silent-wrongness hazards are pinned here.

*The degraded marker is per call, not per thread.* It used to persist on the
worker thread for the process lifetime, so a single query served during a
transient provider outage permanently labelled every later successful query
"DEGRADED ... scores are not cosine similarities" -- telling callers to distrust
correct dense scores.

*Browse must enforce the same sidecar freshness query mode does.* Browse read
`vectors.cos.json` for its row count but never compared the recorded source
fingerprint, so an index rebuilt without re-running `build_cosine` left every
browse path serving documents from the previous index while queries correctly
refused -- stale evidence in the mode used to verify citations.
"""
import pickle
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from rag_service import RagConfig, RagService
from rag_service.backends.faiss import FaissBackend
from rag_service.build_cosine import build_cosine_files
from rag_service.errors import RagEmbeddingError, RagIndexNotReadyError
from rag_service.models import RetrievalRequest

DIMENSION = 4
SOURCES = ["01_web/a.md", "01_web/b.md", "01_web/c.md"]


def _build_index(root: Path, rows: int = 3) -> Path:
    """A genuine FAISS index + docstore + info.db, as the upstream tool writes it."""
    index_path = root / "cybersec" / "vector_store" / "bge-m3"
    index_path.mkdir(parents=True, exist_ok=True)
    content = root / "cybersec" / "content" / "01_web"
    content.mkdir(parents=True, exist_ok=True)
    for name in ("a.md", "b.md", "c.md"):
        (content / name).write_text("seed", encoding="utf-8")

    index = faiss.IndexFlatIP(DIMENSION)
    index.add(np.eye(rows, DIMENSION, dtype="float32"))
    faiss.write_index(index, str(index_path / "index.faiss"))

    store = {}
    mapping = {}
    for row in range(rows):
        doc_id = f"cybersec:{row}"
        store[doc_id] = Document(
            page_content=f"第 {row} 篇文章正文，用于检索与浏览。",
            metadata={"source": SOURCES[row], "id": doc_id},
        )
        mapping[row] = doc_id
    with (index_path / "index.pkl").open("wb") as handle:
        pickle.dump((InMemoryDocstore(store), mapping), handle, protocol=pickle.HIGHEST_PROTOCOL)

    connection = sqlite3.connect(root / "info.db")
    try:
        connection.execute("CREATE TABLE knowledge_base (kb_name TEXT, embed_model TEXT)")
        connection.execute("INSERT INTO knowledge_base VALUES ('cybersec','bge-m3')")
        connection.commit()
    finally:
        connection.close()
    return index_path


def _config(root: Path) -> RagConfig:
    return RagConfig(
        knowledge_base_root=root,
        allowed_knowledge_bases=frozenset({"cybersec"}),
        embedding_model="bge-m3",
    )


class DegradedMarkerTests(unittest.TestCase):
    """A recovered provider must stop being reported as degraded."""

    def _query_vector(self) -> np.ndarray:
        vector = np.zeros(DIMENSION, dtype="float32")
        vector[0] = 1.0
        return vector

    def test_a_transient_outage_does_not_taint_later_successful_queries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            config = _config(root)
            backend = FaissBackend(config)
            service = RagService(config, backend)

            calls = {"count": 0}

            def flaky(model: str, query: str):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RagEmbeddingError("simulated provider outage")
                return self._query_vector()

            request = RetrievalRequest(query="任意文件读取", knowledge_base="cybersec", top_k=2)
            try:
                with patch.object(backend, "_query_embedding", side_effect=flaky):
                    during_outage = service.search(request)
                    after_recovery = service.search(request)
            finally:
                backend.close()

            self.assertEqual(during_outage.degraded, "lexical-only")
            self.assertIsNone(
                after_recovery.degraded,
                "a query served on the dense path must not inherit the previous call's state",
            )
            self.assertNotIn("DEGRADED", after_recovery.as_tool_text())
            # The dense path genuinely ran, so the un-degraded label is correct.
            self.assertTrue(after_recovery.results)
            self.assertIn("dense_score", after_recovery.results[0].metadata)

    def test_browse_does_not_inherit_a_previous_query_degradation(self):
        # Browse never embeds, so it must never report a degraded marker either.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            config = _config(root)
            backend = FaissBackend(config)
            service = RagService(config, backend)

            def always_fail(model: str, query: str):
                raise RagEmbeddingError("simulated provider outage")

            try:
                with patch.object(backend, "_query_embedding", side_effect=always_fail):
                    service.search(
                        RetrievalRequest(query="任意文件读取", knowledge_base="cybersec", top_k=2)
                    )
                listed = service.search(
                    RetrievalRequest(query="", knowledge_base="cybersec", limit=2)
                )
            finally:
                backend.close()

            self.assertIsNone(listed.degraded)


class SidecarFreshnessTests(unittest.TestCase):
    """Every read path must refuse a sidecar whose source index has changed."""

    def _attempt(self, service: RagService, **kwargs) -> str:
        try:
            service.search(RetrievalRequest(**kwargs))
        except RagIndexNotReadyError:
            return "refused"
        return "returned"

    def test_all_read_paths_serve_a_fresh_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            config = _config(root)
            service = RagService(config, FaissBackend(config))
            try:
                self.assertEqual(self._attempt(service, query=""), "returned")
                self.assertEqual(
                    self._attempt(service, query="", filters={"category": "01_web"}), "returned"
                )
                self.assertEqual(
                    self._attempt(service, query="", filters={"source": "01_web/a.md"}), "returned"
                )
                self.assertEqual(
                    self._attempt(service, query="", filters={"chunk_id": "cybersec:0"}), "returned"
                )
            finally:
                service.backend.close()

    def test_every_read_path_refuses_after_a_rebuild_without_reconversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")

            # The upstream index is rebuilt (different row count, new mtime)
            # without re-running build_cosine, so the sidecar is now stale.
            rebuilt = faiss.IndexFlatIP(DIMENSION)
            rebuilt.add(np.eye(2, DIMENSION, dtype="float32"))
            faiss.write_index(rebuilt, str(index_path / "index.faiss"))

            config = _config(root)
            service = RagService(config, FaissBackend(config))
            try:
                self.assertEqual(
                    self._attempt(service, query="", filters={"category": "01_web"}),
                    "refused",
                    "source listing served documents from the previous index",
                )
                self.assertEqual(
                    self._attempt(service, query="", filters={"source": "01_web/a.md"}),
                    "refused",
                    "document page served documents from the previous index",
                )
                self.assertEqual(
                    self._attempt(service, query=""),
                    "refused",
                    "corpus index listing served documents from the previous index",
                )
                self.assertEqual(
                    self._attempt(service, query="", filters={"chunk_id": "cybersec:0"}),
                    "refused",
                )
            finally:
                service.backend.close()


class StoreLifetimeTests(unittest.TestCase):
    """A release must never unmap a store a search is still reading.

    The store is shared across threads (asyncio.to_thread, FastAPI's threadpool,
    the MCP server). `_release_store` closes the underlying memmap, which unmaps
    the file even while another thread is scanning it -- an access violation, not
    a catchable exception. Read paths therefore take a lease, and release is
    deferred while any lease is held.
    """

    def _backend(self, root: Path) -> FaissBackend:
        return FaissBackend(_config(root))

    def test_release_is_deferred_while_a_reader_holds_the_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            backend = self._backend(root)
            try:
                with backend._reader_lease():
                    store = backend._load_store("cybersec", "bge-m3")
                    expected = store["vectors"].shape[0]
                    backend.close()
                    # Still mapped: the reader can finish its scan.
                    self.assertIsNotNone(
                        store["vectors"],
                        "close() unmapped a store a reader is still scanning",
                    )
                    self.assertEqual(store["vectors"].shape[0], expected)
                    self.assertEqual(int(store["vectors"][0][0]), 1)
                del store
            finally:
                backend.close()

    def test_an_idle_release_still_unmaps(self):
        # The guard must not defeat the documented behaviour: with no reader
        # active, close() releases the mapping (so the index can be rebuilt).
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            backend = self._backend(root)
            store = backend._load_store("cybersec", "bge-m3")
            backend.close()
            self.assertIsNone(store["vectors"])
            del store

    def test_concurrent_searches_survive_a_mid_flight_close(self):
        # The reported failure mode: a release landing while other threads are
        # scanning raised AttributeError out of search().
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_index(root)
            build_cosine_files(root, "cybersec", "bge-m3")
            service = RagService(_config(root), self._backend(root))

            request = RetrievalRequest(
                query="", knowledge_base="cybersec", filters={"category": "01_web"}
            )
            failures: list[str] = []
            barrier = threading.Barrier(5)

            def worker():
                try:
                    barrier.wait(timeout=10)
                    for _ in range(8):
                        results = service.search(request)
                        if not results:
                            failures.append("empty result")
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{type(exc).__name__}: {exc}")

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads:
                thread.start()
            try:
                barrier.wait(timeout=10)
                # Release repeatedly while the readers are mid-scan.
                for _ in range(4):
                    service.backend.close()
            finally:
                for thread in threads:
                    thread.join(timeout=30)

            self.assertEqual(failures, [], "a concurrent release broke an in-flight search")
            service.backend.close()
            del service



if __name__ == "__main__":
    unittest.main()