"""Standalone cosine sidecar must never be adopted without a matching manifest.

A newly committed FAISS index paired with leftover ``vectors.cos.*`` would
return vectors that do not correspond to any indexed document. The builder may
only skip conversion when a manifest proves the artifacts were derived from the
current ``index.faiss``/``index.pkl`` fingerprints.
"""
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss

from rag_service.build_cosine import build_cosine_files, quantize_int8

_U64 = struct.Struct("<Q")


class BuildCosineTests(unittest.TestCase):
    def _make_index_pair(self, root: Path) -> Path:
        index_path = root / "cybersec" / "vector_store" / "bge-m3"
        index_path.mkdir(parents=True, exist_ok=True)
        (index_path / "index.faiss").write_bytes(b"placeholder" * 8)
        (index_path / "index.pkl").write_bytes(b"placeholder" * 8)
        return index_path

    def _write_stale_artifacts(self, index_path: Path) -> None:
        """Internally consistent artifacts left over from an earlier index."""
        docs = [
            {"text": "old chunk 0", "metadata": {"source": "01_a/old.md"}},
            {"text": "old chunk 1", "metadata": {"source": "01_a/old.md"}},
        ]
        payloads = [
            json.dumps(item, ensure_ascii=False).encode("utf-8") + b"\n"
            for item in docs
        ]
        (index_path / "docs.cos.jsonl").write_bytes(b"".join(payloads))
        offsets = []
        cursor = 0
        for payload in payloads:
            offsets.append(cursor)
            cursor += len(payload)
        (index_path / "docs.cos.offsets.u64").write_bytes(
            b"".join(_U64.pack(offset) for offset in offsets)
        )
        (index_path / "vectors.cos.f32").write_bytes(b"\x00" * (2 * 4 * 2))
        (index_path / "vectors.cos.json").write_text(
            json.dumps(
                {
                    "format": "cosine-v2",
                    "source": {
                        "index.faiss": {"size": 1, "mtime_ns": 1},
                        "index.pkl": {"size": 1, "mtime_ns": 1},
                    },
                    "rows": 2,
                    "dimension": 4,
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )

    def test_stale_artifacts_are_reconverted_not_adopted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._make_index_pair(root)
            self._write_stale_artifacts(index_path)

            with patch.object(
                faiss,
                "read_index",
                return_value=SimpleNamespace(ntotal=0, d=0),
            ) as read_index:
                with self.assertRaises(ValueError):
                    build_cosine_files(root, "cybersec", "bge-m3")

            read_index.assert_called_once()
            manifest = json.loads(
                (index_path / "vectors.cos.json").read_text(encoding="utf-8")
            )
            # the stale manifest must not have been rewritten as current
            self.assertEqual(manifest["source"]["index.faiss"]["size"], 1)

    def _write_real_index(self, root: Path, rows: list[tuple[str, str]]) -> Path:
        """A genuine FAISS index + docstore so conversion actually runs."""
        import pickle

        import numpy as np
        from langchain.docstore.document import Document
        from langchain_community.docstore.in_memory import InMemoryDocstore

        index_path = self._make_index_pair(root)
        vectors = np.zeros((len(rows), 2), dtype="float32")
        for row in range(len(rows)):
            vectors[row] = np.asarray([1.0, float(row) / 10.0], dtype="float32")
        index = faiss.IndexFlatL2(2)
        index.add(vectors)
        faiss.write_index(index, str(index_path / "index.faiss"))

        store = {}
        mapping = {}
        for row, (text, source) in enumerate(rows):
            doc_id = f"cybersec:{row}"
            store[doc_id] = Document(page_content=text, metadata={"source": source, "id": doc_id})
            mapping[row] = doc_id
        with (index_path / "index.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(store), mapping), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return index_path

    def test_ranges_sidecar_is_published_with_contiguous_spans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._write_real_index(
                root,
                [
                    ("a0", "01_web/alpha.md"),
                    ("a1", "01_web/alpha.md"),
                    ("b0", "02_beta/beta.md"),
                    ("c0", "01_web/gamma.md"),
                ],
            )

            build_cosine_files(root, "cybersec", "bge-m3")

            sidecar = index_path / "docs.cos.ranges.json"
            self.assertTrue(sidecar.is_file())
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["rows"], 4)
            self.assertEqual(
                {name: list(span) for name, span in payload["ranges"].items()},
                {
                    "01_web/alpha.md": [0, 2],
                    "02_beta/beta.md": [2, 3],
                    "01_web/gamma.md": [3, 4],
                },
            )
            # fingerprint must describe the published document file
            docs_path = index_path / "docs.cos.jsonl"
            self.assertEqual(payload["source"]["size"], docs_path.stat().st_size)
            self.assertEqual(payload["source"]["mtime_ns"], docs_path.stat().st_mtime_ns)

    def test_ranges_sidecar_is_rewritten_when_docs_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._write_real_index(
                root, [("a0", "01_web/alpha.md"), ("b0", "02_beta/beta.md")]
            )
            build_cosine_files(root, "cybersec", "bge-m3")
            first = json.loads((index_path / "docs.cos.ranges.json").read_text(encoding="utf-8"))

            # replace the source index with a different document set
            index_path = self._write_real_index(
                root, [("a0", "03_gamma/one.md"), ("a1", "03_gamma/one.md"), ("a2", "03_gamma/one.md")]
            )
            build_cosine_files(root, "cybersec", "bge-m3")
            second = json.loads((index_path / "docs.cos.ranges.json").read_text(encoding="utf-8"))

            self.assertEqual(first["ranges"], {"01_web/alpha.md": [0, 1], "02_beta/beta.md": [1, 2]})
            self.assertEqual(second["ranges"], {"03_gamma/one.md": [0, 3]})
            self.assertNotEqual(first["source"], second["source"])

    def test_stale_sidecar_is_refreshed_when_artifacts_are_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._write_real_index(
                root, [("a0", "01_web/alpha.md"), ("b0", "02_beta/beta.md")]
            )
            build_cosine_files(root, "cybersec", "bge-m3")
            sidecar = index_path / "docs.cos.ranges.json"
            stale = {"rows": 1, "source": {"size": 0, "mtime_ns": 0}, "ranges": {"x": [0, 1]}}
            sidecar.write_text(json.dumps(stale), encoding="utf-8")

            # artifacts are already current, so conversion is skipped entirely
            build_cosine_files(root, "cybersec", "bge-m3")

            refreshed = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(refreshed["ranges"], {"01_web/alpha.md": [0, 1], "02_beta/beta.md": [1, 2]})
            docs_path = index_path / "docs.cos.jsonl"
            self.assertEqual(refreshed["source"]["size"], docs_path.stat().st_size)

    def test_fresh_sidecar_is_left_alone(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._write_real_index(root, [("a0", "01_web/alpha.md")])
            build_cosine_files(root, "cybersec", "bge-m3")
            sidecar = index_path / "docs.cos.ranges.json"
            before = sidecar.read_text(encoding="utf-8")

            build_cosine_files(root, "cybersec", "bge-m3")

            self.assertEqual(sidecar.read_text(encoding="utf-8"), before)

    def test_current_manifest_skips_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = self._make_index_pair(root)
            self._write_stale_artifacts(index_path)
            source_index = index_path / "index.faiss"
            source_pickle = index_path / "index.pkl"
            (index_path / "vectors.cos.json").write_text(
                json.dumps(
                    {
                        "format": "cosine-v2",
                        "source": {
                            "index.faiss": {
                                "size": source_index.stat().st_size,
                                "mtime_ns": source_index.stat().st_mtime_ns,
                            },
                            "index.pkl": {
                                "size": source_pickle.stat().st_size,
                                "mtime_ns": source_pickle.stat().st_mtime_ns,
                            },
                        },
                        "rows": 2,
                        "dimension": 4,
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(faiss, "read_index") as read_index:
                result = build_cosine_files(root, "cybersec", "bge-m3")

            read_index.assert_not_called()
            self.assertEqual(result, index_path / "vectors.cos.f32")


class QuantizationTests(unittest.TestCase):
    def _np(self):
        import numpy as np

        return np

    def test_round_trip_preserves_inner_product_closely(self):
        np = self._np()
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(64, 128)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        query = rng.normal(size=128).astype("float32")
        query /= np.linalg.norm(query)

        quantized, scales = quantize_int8(vectors, np)
        exact = vectors @ query
        approx = (quantized.astype("float32") @ query) * scales

        self.assertEqual(quantized.dtype, np.int8)
        self.assertLess(float(np.abs(exact - approx).max()), 0.02)

    def test_quantized_dtype_and_range(self):
        np = self._np()
        vectors = np.asarray([[-1.0, 0.5, 0.0], [0.25, -0.25, 0.125]], dtype="float32")
        quantized, scales = quantize_int8(vectors, np)
        self.assertEqual(quantized.shape, (2, 3))
        self.assertTrue((np.abs(quantized.astype("int32")) <= 127).all())
        self.assertTrue((scales > 0).all())

    def test_all_zero_row_does_not_divide_by_zero(self):
        np = self._np()
        vectors = np.zeros((1, 4), dtype="float32")
        quantized, scales = quantize_int8(vectors, np)
        self.assertEqual(int(quantized[0].max()), 0)
        self.assertTrue(float(np.isfinite(scales[0])))

    def test_row_order_is_preserved(self):
        np = self._np()
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype="float32")
        quantized, _ = quantize_int8(vectors, np)
        self.assertGreater(int(quantized[0][0]), 0)
        self.assertGreater(int(quantized[1][1]), 0)
        self.assertLess(int(quantized[2][0]), 0)


if __name__ == "__main__":
    unittest.main()
