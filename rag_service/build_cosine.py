"""Build read-only, memmap-friendly cosine retrieval artifacts.

The source LangChain store consists of ``index.faiss`` and ``index.pkl``. This
command produces a versioned sidecar set without modifying either source file:

- ``vectors.cos.f32``: L2-normalized float32 vectors;
- ``vectors.cos.int8`` + ``vectors.cos.scales.f32``: optional per-row symmetric
  int8 quantization of the same vectors (4x smaller resident footprint);
- ``docs.cos.jsonl``: one UTF-8 JSON document per vector row;
- ``docs.cos.offsets.u64``: byte offsets for random document access;
- ``vectors.cos.json``: source/artifact manifest.

The builder reuses existing artifact files only when an existing manifest proves
them current for the source index; otherwise it regenerates them from the source
index.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
from pathlib import Path
from uuid import uuid4

from rag_service.relevance import identifier_tokens


_ARTIFACT_FORMAT = "cosine-v2"
_POSTINGS_FORMAT = "postings-v1"
_POSTINGS_STRIDE = 256
"""Sparse mark interval. Must stay below the reader's forward-scan bound, or a
token between two marks becomes unreachable: marks every 4096 entries with a
512-line scan silently failed to resolve almost every token."""
_QUANTIZATION_NONE = "none"
_QUANTIZATION_INT8 = "int8"
_U64 = struct.Struct("<Q")


def quantize_int8(vectors, np):
    """Return ``(int8_vectors, scales)`` using per-row symmetric quantization.

    A single global scale would waste most of the int8 range: normalized
    embedding rows differ in how concentrated they are. Scaling each row by its
    own maximum keeps the relative error uniform across rows, and the score is
    recovered exactly as ``dot(int8_row, query) * scale``.
    """
    maxima = np.abs(vectors).max(axis=1)
    scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype(np.float32)
    quantized = np.rint(vectors / scales[:, None]).clip(-127, 127).astype(np.int8)
    return quantized, scales


def _file_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_current(
    manifest_path: Path,
    source_index: Path,
    source_pickle: Path,
    artifacts: tuple[Path, Path, Path],
    index_path: Path,
    force: bool = False,
) -> bool:
    if force:
        return False
    if not manifest_path.is_file() or not all(path.is_file() for path in artifacts):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        manifest.get("format") != _ARTIFACT_FORMAT
        or manifest.get("source")
        != {"index.faiss": _file_state(source_index), "index.pkl": _file_state(source_pickle)}
    ):
        return False
    # A manifest that declares quantization is only current if the quantized
    # files it promises are actually present; otherwise the backend would fall
    # back to the larger matrix while reporting a current build.
    if manifest.get("quantization") == _QUANTIZATION_INT8:
        return (index_path / "vectors.cos.int8").is_file() and (
            index_path / "vectors.cos.scales.f32"
        ).is_file()
    return True


def _ranges_of(sources: list[str]) -> dict[str, list[int]]:
    """Group row indexes into contiguous ``{source: [start, end]}`` spans."""
    ranges: dict[str, list[int]] = {}
    current: str | None = None
    span_start = 0
    for row, source in enumerate(sources):
        if source != current:
            if current is not None:
                ranges[current] = [span_start, row]
            current = source
            span_start = row
    if current is not None:
        ranges[current] = [span_start, len(sources)]
    return ranges


def _write_ranges_sidecar(docs_path: Path, rows: int, ranges: dict[str, list[int]]) -> None:
    sidecar = docs_path.with_name("docs.cos.ranges.json")
    temporary = _temp_path(sidecar)
    try:
        temporary.write_text(
            json.dumps(
                {"rows": rows, "source": _file_state(docs_path), "ranges": ranges},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, sidecar)
    finally:
        _cleanup([temporary])


def _ensure_ranges_sidecar(docs_path: Path, index_path: Path) -> None:
    """Publish the source→row sidecar unless the current one already matches.

    Artifacts can be current while the sidecar is missing or stale (it is
    written by an earlier step of this function, or by the retrieval backend on
    first browse). Without this, a caller would pay a full document stream on
    its first filtered query.
    """
    sidecar = docs_path.with_name("docs.cos.ranges.json")
    fingerprint = _file_state(docs_path)
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if payload.get("source") == fingerprint and payload.get("ranges"):
                return
        except (OSError, ValueError):
            pass
    sources: list[str] = []
    with docs_path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
            except ValueError:
                continue
            sources.append(
                str((payload.get("metadata") or {}).get("source") or "").replace("\\", "/")
            )
    if not sources:
        return
    _write_ranges_sidecar(docs_path, len(sources), _ranges_of(sources))
    print(f"{index_path.name}: refreshed docs.cos.ranges.json ({len(sources)} rows)")


def _ensure_postings(docs_path: Path, index_path: Path) -> None:
    """Build the postings index unless the current one matches the docs file."""
    postings_path = index_path / "docs.cos.postings"
    index_file = index_path / "docs.cos.postings.idx.json"
    if postings_path.is_file() and index_file.is_file():
        try:
            payload = json.loads(index_file.read_text(encoding="utf-8"))
            if (
                payload.get("format") == _POSTINGS_FORMAT
                and payload.get("source") == _file_state(docs_path)
                and payload.get("stride") == _POSTINGS_STRIDE
                and payload.get("max_df") == POSTINGS_MAX_DF
            ):
                return
        except (OSError, ValueError):
            pass
    _build_postings(docs_path, index_path)


def _ensure_quantized(
    index_path: Path,
    vectors_path: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
    rows: int,
    dimension: int,
) -> bool:
    """Generate the int8 artifacts from the existing float32 matrix.

    Upgrading an already-built sidecar set must not re-read the multi-GB FAISS
    index or rewrite the float32 matrix: the quantized files are derived from
    that matrix, so streaming it in blocks is both cheaper and lower-risk. Writes
    go to temp files and replace at the end, so a failure leaves the working
    float32 artifacts untouched.
    """
    import numpy as np

    int8_path = index_path / "vectors.cos.int8"
    scales_path = index_path / "vectors.cos.scales.f32"
    if (
        manifest.get("quantization") == _QUANTIZATION_INT8
        and int8_path.is_file()
        and scales_path.is_file()
        and int8_path.stat().st_size == rows * dimension
        and scales_path.stat().st_size == rows * np.dtype("float32").itemsize
    ):
        return False
    # The float32 matrix is the input; never memmap it with dimensions it does
    # not actually have. Quantization is optional, so an inconsistent artifact
    # degrades to the float32 path with a warning instead of failing the build —
    # the retrieval backend validates these files independently.
    expected = rows * dimension * np.dtype("float32").itemsize
    if not vectors_path.is_file() or vectors_path.stat().st_size != expected:
        print(
            f"{index_path.name}: skipping quantization, {vectors_path.name} is not "
            f"{rows} x {dimension} float32"
        )
        return False

    temp_int8 = _temp_path(int8_path)
    temp_scales = _temp_path(scales_path)
    temporary = [temp_int8, temp_scales]
    print(f"{index_path.name}: quantizing {rows} x {dimension} vectors to int8")
    try:
        vectors = np.memmap(
            vectors_path, dtype="float32", mode="r", shape=(rows, dimension)
        )
        batch = 20_000
        with temp_int8.open("wb") as int8_handle, temp_scales.open("wb") as scales_handle:
            for start in range(0, rows, batch):
                end = min(start + batch, rows)
                block = np.ascontiguousarray(vectors[start:end], dtype="float32")
                quantized, scales = quantize_int8(block, np)
                int8_handle.write(quantized.tobytes(order="C"))
                scales_handle.write(scales.tobytes(order="C"))
                del block, quantized, scales
                if start % 200_000 == 0:
                    print(f"  quantized {end} / {rows}")
        del vectors
        os.replace(temp_int8, int8_path)
        os.replace(temp_scales, scales_path)
    finally:
        _cleanup(temporary)

    manifest["quantization"] = _QUANTIZATION_INT8
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["int8_bytes"] = int8_path.stat().st_size
    artifacts["scales_bytes"] = scales_path.stat().st_size
    temp_manifest = _temp_path(manifest_path)
    try:
        temp_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
    finally:
        _cleanup([temp_manifest])
    print(f"{index_path.name}: wrote int8 artifacts")
    return True


POSTINGS_MAX_DF = 500
"""Document-frequency ceiling for indexed identifiers.

A token appearing in more chunks than this is not discriminating (it is
effectively a common word), so storing its row list would cost space and return
hundreds of rows. Measured on the 1M-chunk corpus: df<=500 keeps 2.6M tokens /
4.3M postings, about 30 MB.
"""


def _build_postings(docs_path: Path, index_path: Path) -> None:
    """Write ``docs.cos.postings`` + ``docs.cos.postings.idx.json``.

    Format: one ``token<TAB>row,row,...`` line per token, sorted by token, plus
    a sparse byte-offset index every ``_POSTINGS_STRIDE`` tokens. A query needs
    the rows of a handful of tokens, so an on-disk sorted file with binary
    search keeps resident memory at a few hundred KB instead of the ~1 GB a
    fully materialized dict of 2.6M lists would need.

    Two streaming passes over the already-written document file, each dropping
    tokens as soon as they exceed the frequency ceiling, so peak memory is
    bounded by the surviving rare tokens rather than the whole vocabulary.
    """
    from collections import defaultdict

    counts: dict[str, int] = {}
    rejected: set[str] = set()
    with docs_path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                text = json.loads(raw.decode("utf-8", "ignore")).get("text") or ""
            except ValueError:
                continue
            for token in identifier_tokens(text):
                if token in rejected:
                    continue
                total = counts.get(token, 0) + 1
                if total > POSTINGS_MAX_DF:
                    del counts[token]
                    rejected.add(token)
                    continue
                counts[token] = total

    rows_by_token: dict[str, list[int]] = defaultdict(list)
    with docs_path.open("rb") as handle:
        for row, raw in enumerate(handle):
            if not raw.strip():
                continue
            try:
                text = json.loads(raw.decode("utf-8", "ignore")).get("text") or ""
            except ValueError:
                continue
            for token in identifier_tokens(text):
                if token in counts:
                    rows_by_token[token].append(row)

    postings_path = index_path / "docs.cos.postings"
    index_file = index_path / "docs.cos.postings.idx.json"
    temporary = postings_path.with_name(f".{postings_path.name}.{uuid4().hex}.tmp")
    marks: list[list] = []
    try:
        with temporary.open("wb") as handle:
            for position, token in enumerate(sorted(rows_by_token)):
                if position % _POSTINGS_STRIDE == 0:
                    marks.append([token, handle.tell()])
                rows = rows_by_token[token]
                handle.write(
                    f"{token}\t{','.join(str(row) for row in rows)}\n".encode("utf-8")
                )
        os.replace(temporary, postings_path)
    finally:
        _cleanup([temporary])

    index_payload = {
        "format": _POSTINGS_FORMAT,
        "source": _file_state(docs_path),
        "tokens": len(rows_by_token),
        "postings": sum(len(rows) for rows in rows_by_token.values()),
        "max_df": POSTINGS_MAX_DF,
        "stride": _POSTINGS_STRIDE,
        "marks": marks,
    }
    temporary_index = index_file.with_name(f".{index_file.name}.{uuid4().hex}.tmp")
    try:
        temporary_index.write_text(json.dumps(index_payload), encoding="utf-8")
        os.replace(temporary_index, index_file)
    finally:
        _cleanup([temporary_index])
    print(
        f"{index_path.name}: postings {index_payload['tokens']:,} tokens / "
        f"{index_payload['postings']:,} rows"
    )


def _ensure_sq8(
    index_path: Path,
    vectors_path: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
    rows: int,
    dimension: int,
) -> None:
    """Build a faiss scalar-quantized (int8) index for the full scan.

    The numpy path had to dequantize every block into float32 before the dot
    product, which measured 3.5-7.7 s per query; faiss performs the int8 dot
    product in SIMD without materializing float32, measuring 0.9-1.2 s for the
    same top-10 (8-10/10 overlap on real queries). It also keeps resident memory
    at ~1 GB, the reason int8 was chosen in the first place.

    Index is built incrementally so peak memory stays bounded by one chunk
    rather than the whole matrix.
    """
    import faiss
    import numpy as np

    sq8_path = index_path / "vectors.cos.sq8"
    if manifest.get("sq8_bytes") and sq8_path.is_file():
        return
    expected = rows * dimension * np.dtype("float32").itemsize
    if not vectors_path.is_file() or vectors_path.stat().st_size != expected:
        print(
            f"{index_path.name}: skipping SQ8, {vectors_path.name} is not "
            f"{rows} x {dimension} float32"
        )
        return

    print(f"{index_path.name}: building SQ8 index for {rows} x {dimension} vectors")
    vectors = np.memmap(vectors_path, dtype="float32", mode="r", shape=(rows, dimension))
    index = faiss.IndexScalarQuantizer(
        dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    # Training only needs a representative sample; striding avoids a second
    # full copy of the matrix.
    sample_size = min(rows, 100_000)
    stride = max(1, rows // sample_size)
    sample = np.ascontiguousarray(vectors[::stride][:sample_size])
    index.train(sample)
    del sample
    chunk = 50_000
    for start in range(0, rows, chunk):
        index.add(np.ascontiguousarray(vectors[start : start + chunk]))
    del vectors

    temporary = sq8_path.with_name(f".{sq8_path.name}.{uuid4().hex}.tmp")
    try:
        faiss.write_index(index, str(temporary))
        os.replace(temporary, sq8_path)
    finally:
        _cleanup([temporary])

    manifest["sq8_bytes"] = sq8_path.stat().st_size
    temp_manifest = _temp_path(manifest_path)
    try:
        temp_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
    finally:
        _cleanup([temp_manifest])
    print(f"{index_path.name}: wrote SQ8 index")


def build_cosine_files(
    kb_root: Path, knowledge_base: str, embedding_model: str, force: bool = False
) -> Path:
    """Build and atomically publish standalone retrieval artifacts for one KB.

    ``vectors.cos.*``/``docs.cos.*`` are only reused when an existing manifest
    proves they were derived from exactly the current ``index.faiss`` and
    ``index.pkl``. Otherwise they are converted from the source index again:
    adopting pre-existing files by size alone would silently pair a new index
    with stale vectors.
    """
    vector_name = embedding_model.replace(":", "_").replace("/", "__")
    index_path = kb_root / knowledge_base / "vector_store" / vector_name
    source_index = index_path / "index.faiss"
    source_pickle = index_path / "index.pkl"
    if not source_index.is_file() or source_index.stat().st_size <= 45:
        raise FileNotFoundError(f"source index not found: {source_index}")
    if not source_pickle.is_file() or source_pickle.stat().st_size == 0:
        raise FileNotFoundError(f"source metadata not found: {source_pickle}")

    vectors_path = index_path / "vectors.cos.f32"
    int8_path = index_path / "vectors.cos.int8"
    scales_path = index_path / "vectors.cos.scales.f32"
    docs_path = index_path / "docs.cos.jsonl"
    offsets_path = index_path / "docs.cos.offsets.u64"
    manifest_path = index_path / "vectors.cos.json"
    artifacts = (vectors_path, docs_path, offsets_path)
    if _is_current(
        manifest_path, source_index, source_pickle, artifacts, index_path, force
    ):
        print(f"{knowledge_base}/{vector_name}: standalone artifacts are already current")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _ensure_ranges_sidecar(docs_path, vectors_path.parent)
        _ensure_postings(docs_path, index_path)
        _ensure_quantized(
            index_path,
            vectors_path,
            manifest_path,
            manifest,
            int(manifest["rows"]),
            int(manifest["dimension"]),
        )
        _ensure_sq8(
            index_path,
            vectors_path,
            manifest_path,
            json.loads(manifest_path.read_text(encoding="utf-8")),
            int(manifest["rows"]),
            int(manifest["dimension"]),
        )
        return vectors_path

    import faiss
    import numpy as np
    import pickle

    source_state = {"index.faiss": _file_state(source_index), "index.pkl": _file_state(source_pickle)}
    temp_vectors = _temp_path(vectors_path)
    temp_int8 = _temp_path(int8_path)
    temp_scales = _temp_path(scales_path)
    temp_docs = _temp_path(docs_path)
    temp_offsets = _temp_path(offsets_path)
    temp_manifest = _temp_path(manifest_path)
    temporary = [
        temp_vectors,
        temp_int8,
        temp_scales,
        temp_docs,
        temp_offsets,
        temp_manifest,
    ]

    try:
        source = faiss.read_index(str(source_index))
        total, dimension = source.ntotal, source.d
        if total <= 0 or dimension <= 0:
            raise ValueError(f"source index has invalid shape: rows={total}, dimension={dimension}")

        with source_pickle.open("rb") as handle:
            docstore, index_to_docstore_id = pickle.load(handle)

        ranges: dict[str, list[int]] = {}
        current_source: str | None = None
        span_start = 0

        with temp_vectors.open("wb") as vectors_handle, temp_int8.open(
            "wb"
        ) as int8_handle, temp_scales.open("wb") as scales_handle, temp_docs.open(
            "wb"
        ) as docs_handle, temp_offsets.open("wb") as offsets_handle:
            batch_size = 20_000
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                vectors = np.ascontiguousarray(
                    source.reconstruct_n(start, end - start), dtype="float32"
                )
                faiss.normalize_L2(vectors)
                vectors_handle.write(vectors.tobytes(order="C"))
                quantized, scales = quantize_int8(vectors, np)
                int8_handle.write(quantized.tobytes(order="C"))
                scales_handle.write(scales.tobytes(order="C"))
                del vectors, quantized, scales

                for row in range(start, end):
                    document = docstore.search(index_to_docstore_id[row])
                    payload = {
                        "text": document.page_content if document is not None else "",
                        "metadata": dict(getattr(document, "metadata", {}) or {}),
                    }
                    source_path = str(
                        payload["metadata"].get("source") or ""
                    ).replace("\\", "/")
                    if source_path != current_source:
                        if current_source is not None:
                            ranges[current_source] = [span_start, row]
                        current_source = source_path
                        span_start = row
                    offsets_handle.write(_U64.pack(docs_handle.tell()))
                    docs_handle.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                        + b"\n"
                    )
                print(f"{knowledge_base}/{vector_name}: converted {end} / {total}")
            if current_source is not None:
                ranges[current_source] = [span_start, total]
        del source

        if source_state != {
            "index.faiss": _file_state(source_index),
            "index.pkl": _file_state(source_pickle),
        }:
            raise RuntimeError("source index changed while standalone artifacts were being built")

        manifest = {
            "format": _ARTIFACT_FORMAT,
            "source": source_state,
            "rows": total,
            "dimension": dimension,
            "quantization": _QUANTIZATION_INT8,
            "artifacts": {
                "vectors_bytes": temp_vectors.stat().st_size,
                "int8_bytes": temp_int8.stat().st_size,
                "scales_bytes": temp_scales.stat().st_size,
                "docs_bytes": temp_docs.stat().st_size,
                "offsets_bytes": temp_offsets.stat().st_size,
            },
        }
        temp_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

        for temporary_path, final_path in (
            (temp_vectors, vectors_path),
            (temp_int8, int8_path),
            (temp_scales, scales_path),
            (temp_docs, docs_path),
            (temp_offsets, offsets_path),
            (temp_manifest, manifest_path),
        ):
            os.replace(temporary_path, final_path)

        # Publish the source→row span sidecar used by filter-aware search and by
        # browse listings. It is fingerprinted against the published docs file,
        # exactly as the backend expects.
        _write_ranges_sidecar(docs_path, total, ranges)
        _build_postings(docs_path, index_path)
        _ensure_sq8(
            index_path,
            vectors_path,
            manifest_path,
            json.loads(manifest_path.read_text(encoding="utf-8")),
            total,
            dimension,
        )

        print(f"{knowledge_base}/{vector_name}: wrote {vectors_path.parent}")
        return vectors_path
    finally:
        _cleanup(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone cosine retrieval artifacts")
    parser.add_argument("--kb-root", required=True)
    parser.add_argument("--knowledge-base", default=None, help="build a single KB only")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate even when the manifest says the artifacts are current "
        "(needed after the artifact set itself changes, e.g. enabling quantization)",
    )
    args = parser.parse_args()

    kb_root = Path(args.kb_root)
    metadata_path = kb_root / "info.db"
    if not metadata_path.is_file():
        raise SystemExit(f"metadata database not found: {metadata_path}")

    connection = sqlite3.connect(metadata_path)
    try:
        rows = connection.execute("SELECT kb_name, embed_model FROM knowledge_base").fetchall()
    finally:
        connection.close()

    for kb_name, embed_model in rows:
        if args.knowledge_base and kb_name != args.knowledge_base:
            continue
        model = args.embedding_model or embed_model
        if not model:
            print(f"{kb_name}: no embedding model recorded, skipping")
            continue
        build_cosine_files(kb_root, kb_name, model, force=args.force)


if __name__ == "__main__":
    main()
