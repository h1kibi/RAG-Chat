"""The bundled demo index must stay loadable and browsable.

`examples/demo-kb` is a shipped artifact, so it is a contract: if the sidecar
format, the manifest handshake, or the accessor layout changes, a fresh clone
stops being usable and nothing else in the suite would notice -- the other tests
build their own stores in temp directories.

These assertions deliberately avoid the embedding provider (browse mode and
`status()` need no vectors for a query), so the suite keeps running offline.
"""
import json
import unittest
from pathlib import Path

from rag_service.backends.faiss import FaissBackend, describe_status
from rag_service.config import RagConfig
from rag_service.models import RetrievalRequest
from rag_service.service import RagService

DEMO_ROOT = Path(__file__).resolve().parent.parent / "examples" / "demo-kb"
INDEX_DIR = DEMO_ROOT / "cybersec" / "vector_store" / "bge-m3"

REQUIRED_FILES = (
    "vectors.cos.f32",
    "vectors.cos.sq8",
    "vectors.cos.int8",
    "vectors.cos.scales.f32",
    "vectors.cos.json",
    "docs.cos.jsonl",
    "docs.cos.offsets.u64",
    "docs.cos.ranges.json",
)

SOURCE_FILES = ("index.faiss", "index.pkl")
"""Deliberately not shipped: the manifest records a source fingerprint of
``size+mtime_ns``, and Git cannot preserve mtime, so a committed set that still
referenced a source index could never validate after a clone. The set is
published with ``build_cosine --prebuilt`` instead."""


def _config() -> RagConfig:
    return RagConfig(
        knowledge_base_root=DEMO_ROOT,
        allowed_knowledge_bases=frozenset({"cybersec"}),
        embedding_model="bge-m3",
    )


class DemoIndexTests(unittest.TestCase):
    def test_the_demo_index_is_committed_complete(self):
        # A partially committed fixture fails at runtime with a confusing
        # "stale or unreadable" message, so catch it here instead.
        self.assertTrue(DEMO_ROOT.is_dir(), f"missing demo root: {DEMO_ROOT}")
        for name in REQUIRED_FILES:
            path = INDEX_DIR / name
            self.assertTrue(path.is_file(), f"missing demo artifact: {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty demo artifact: {path}")
        self.assertTrue((DEMO_ROOT / "info.db").is_file())

    def test_the_demo_index_ships_without_its_source_index(self):
        # Shipping the source pair would make the fixture fail on every fresh
        # clone: the manifest's mtime fingerprint cannot survive a checkout.
        for name in SOURCE_FILES:
            self.assertFalse(
                (INDEX_DIR / name).exists(),
                f"{name} must not be committed; regenerate with "
                "scripts/build-demo-index.py (it publishes --prebuilt)",
            )
        manifest = json.loads((INDEX_DIR / "vectors.cos.json").read_text(encoding="utf-8"))
        self.assertIsNone(
            manifest.get("source"),
            "a committed artifact set must declare itself prebuilt",
        )

    def test_the_demo_index_is_not_reported_as_stale(self):
        # `status()` runs the same manifest fingerprint handshake a query does;
        # a mismatched pair surfaces as dense_path="unavailable".
        backend = FaissBackend(_config())
        try:
            status = backend.status("cybersec")
        finally:
            backend.close()

        self.assertNotIn("error", status, describe_status(status))
        self.assertEqual(status["dense_path"], "sq8", describe_status(status))
        self.assertGreater(status["rows"], 0)

    def test_the_demo_index_can_be_browsed_without_an_embedding_provider(self):
        # Browse needs no embedding, so this covers the shipped artifact on a
        # machine with no Ollama running.
        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        try:
            corpus = service.search(RetrievalRequest(query="", knowledge_base="cybersec"))
            listing = service.search(
                RetrievalRequest(
                    query="", filters={"category": "05_pentest_method"}, knowledge_base="cybersec"
                )
            )
        finally:
            backend.close()

        # The empty-query corpus index enumerates the template categories.
        self.assertTrue(corpus.results, "demo corpus index returned nothing")
        self.assertEqual(
            corpus.results[0].metadata.get("categories"),
            [
                "00_foundations",
                "01_web_security",
                "02_network_security",
                "03_linux_windows",
                "04_vulnerability",
                "05_pentest_method",
                "06_ctf",
                "07_defense",
            ],
        )
        # A category listing carries the document paths in its content.
        self.assertTrue(listing.results, "demo category listing returned nothing")
        self.assertIn(
            "- 05_pentest_method/authorized-pentest-workflow.md",
            listing.results[0].content or "",
        )

    def test_every_template_document_is_indexed(self):
        # The fixture is derived from knowledge-base/cybersec/content; if a
        # template document is added without regenerating the index, the demo
        # silently under-represents the template.
        content_root = DEMO_ROOT.parent.parent / "knowledge-base" / "cybersec" / "content"
        expected = sorted(
            path.relative_to(content_root).as_posix() for path in content_root.rglob("*.md")
        )

        backend = FaissBackend(_config())
        service = RagService(backend.config, backend)
        try:
            corpus = service.search(RetrievalRequest(query="", knowledge_base="cybersec"))
            categories = corpus.results[0].metadata.get("categories") or []
            indexed = []
            for category in categories:
                page = service.search(
                    RetrievalRequest(
                        query="",
                        filters={"category": category},
                        knowledge_base="cybersec",
                        limit=50,
                    )
                )
                for line in (page.results[0].content or "").splitlines():
                    if line.startswith("- "):
                        indexed.append(line[2:].strip())
        finally:
            backend.close()

        self.assertEqual(sorted(indexed), expected)


if __name__ == "__main__":
    unittest.main()
