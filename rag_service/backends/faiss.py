from __future__ import annotations

import bisect
import contextlib
import json
import re
import sqlite3
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Tuple

import numpy as np

from rag_service.config import RagConfig
from rag_service.embeddings import OllamaEmbeddingClient
from rag_service.errors import RagEmbeddingError, RagIndexNotReadyError
from rag_service.models import RetrievalRequest, SearchResult
from rag_service.relevance import (
    LexicalReranker,
    QueryTerms,
    extract_environment_facts,
    extract_segments,
    fusion_score,
    identifier_tokens,
    image_references,
    past_event_flags,
    source_origin,
    is_near_duplicate,
    low_information_reasons,
    parse_provenance_frontmatter,
    screenshot_placeholders,
    snippet_window,
    strip_markdown_images,
    strip_provenance_frontmatter,
    text_signature,
    tokenize_query,
)

from .base import RetrievalBackend

_ARTIFACT_FORMAT = "cosine-v2"
_OFFSET_STRUCT = struct.Struct("<Q")
_MAX_CANDIDATES = 250

_LOGGED_PATHS: set = set()


def faiss_available() -> bool:
    """True when the faiss module can be imported in this interpreter.

    Without it the scan silently falls back to a numpy dequantization loop,
    measured 2.7-3.7 s per query versus 0.15 s through faiss — a 20x difference
    that nothing previously reported, so a tuning run under the wrong
    interpreter looked like a slow service rather than a degraded one.
    """
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


def dense_path(store: Dict[str, Any]) -> str:
    """Which scan implementation the store will actually use: sq8 or numpy."""
    return "sq8" if store.get("sq8") is not None else "numpy"


def _sq8_state(store: Dict[str, Any] | None) -> str:
    """Why the scan fell back, or ``loaded`` when it did not.

    The distinction matters because the remedies differ: an absent artifact is
    fixed by build_cosine, while a present-but-unreadable one is not (build_cosine
    reports the artifacts as current and exits without touching it) — the usual
    cause is a corrupt file or a process that could not map it for lack of
    memory, which is exactly when the numpy fallback appears.

    This is published in the health payload, so it must also be truthful on the
    happy path: the question only has an answer when there was a fallback. The
    earlier version returned ``unreadable`` for any file that simply existed,
    which made a healthy service advertise ``dense_path=sq8`` and
    ``sq8_state=unreadable`` at the same time.
    """
    if not store:
        return "unknown"
    if store.get("sq8") is not None:
        return "loaded"
    docs_path = store.get("docs_path")
    if docs_path is None:
        return "unknown"
    try:
        sq8_path = Path(docs_path).with_name("vectors.cos.sq8")
    except TypeError:
        return "unknown"
    if not sq8_path.is_file() or sq8_path.stat().st_size == 0:
        return "missing"
    return "unreadable"


def _path_detail(path: str, sq8_state: str | None = None) -> str:
    if path == "sq8":
        return "faiss int8 scalar quantizer"
    if path == "numpy":
        if not faiss_available():
            return (
                "numpy dequantization because faiss is not importable in this interpreter — "
                "expect roughly 20x slower scans; run under the repository virtualenv "
                "(.venv/Scripts/python.exe)"
            )
        if sq8_state == "unreadable":
            return (
                "numpy dequantization because the sq8 artifact exists but could not be "
                "loaded (corrupt, or this process lacked the memory to map it) — rebuild "
                "it with `python -m rag_service.build_cosine --force`, and check available "
                "memory; re-running without --force reports the artifacts as current and "
                "changes nothing"
            )
        return (
            "numpy dequantization because the sq8 artifact is missing — rebuild with "
            "`python -m rag_service.build_cosine`"
        )
    return "unknown"


def describe_dense_path(store: Dict[str, Any]) -> str | None:
    """One-line runtime status for a loaded store, returned once per scan path.

    Returns ``None`` afterwards: a long-running server should state its
    capability at startup, not repeat it on every query.
    """
    path = dense_path(store)
    if path in _LOGGED_PATHS:
        return None
    _LOGGED_PATHS.add(path)
    return (
        f"dense_path={path} rows={int(store['vectors'].shape[0])}: "
        f"{_path_detail(path, _sq8_state(store))}"
    )


def _postings_state(docs_path: Path) -> str:
    """Why identifier recall is unavailable: ``absent`` or ``unusable``.

    The distinction matters because the file can exist and still be rejected:
    the sidecar records a ``size+mtime_ns`` fingerprint of the document file, so
    copying a knowledge base (or cloning a repository that ships one) always
    invalidates it, and the reader then reports postings as missing although
    nothing is missing. Same remedy either way, but "missing" sends the operator
    looking for a file that is right there.
    """
    postings_path = docs_path.with_name("docs.cos.postings")
    index_file = docs_path.with_name("docs.cos.postings.idx.json")
    if not postings_path.is_file() or not index_file.is_file():
        return "missing"
    return "unusable"


def describe_status(status: Dict[str, Any]) -> str:
    """Render a :meth:`FaissBackend.status` result for startup logging.

    Marks the path as reported so a later per-query ``describe_dense_path`` does
    not print the same capability twice in one process.
    """
    path = str(status.get("dense_path") or "unknown")
    _LOGGED_PATHS.add(path)
    rows = status.get("rows")
    where = f" rows={rows}" if isinstance(rows, int) else ""
    extra = ""
    if not status.get("postings", True):
        state = str(status.get("postings_state") or "missing")
        extra = (
            f" postings={state} (identifier recall disabled; "
            "`python -m rag_service.build_cosine` regenerates the sidecar)"
        )
    return (
        f"dense_path={path}{where}: "
        f"{_path_detail(path, status.get('sq8_state'))}{extra}"
    )


def _image_availability(
    images: list[Dict[str, Any]], document_path: Path | None
) -> list[Dict[str, Any]]:
    """Say how each referenced image can actually be obtained.

    A markdown writeup points at `images/foo.png` relative to itself, and the
    importer deliberately excludes binaries and `images/` directories, so those
    references resolve to nothing: measured over this corpus, 0 image files and 0
    `images` directories exist under `content/`, against 20,464 text files. A
    remote URL is fetchable as-is.

    Reporting only a count left the caller to discover that, which is why the
    reviewer read it as a missing visual channel. The honest answer is that the
    evidence is not in the corpus at all and has to come from the upstream
    source, so each entry states which case it is.
    """
    if not images:
        return images
    out = []
    for item in images:
        src = str(item.get("src") or "")
        entry = dict(item)
        if src.startswith(("http://", "https://")):
            entry["available"] = "remote"
        else:
            candidate = (document_path.parent / src).resolve() if document_path else None
            if candidate is not None and candidate.is_file():
                entry["available"] = "local"
                entry["path"] = str(candidate)
            else:
                # Not an error: the corpus is text-only by design.
                entry["available"] = "not-imported"
        out.append(entry)
    return out


def _annotate_evidence(
    metadata: Dict[str, Any], content: str, document_path: Path | None = None
) -> None:
    """Attach the per-chunk signals every result path must carry.

    Extracted because three of the four result builders (single-chunk retrieval,
    source paging, and the lexical fallback) had drifted to reporting only the
    screenshot count. That mattered most for `filters.chunk_id`, which exists to
    verify a citation -- precisely where "this document quotes a past-event flag"
    needs to be visible.
    """
    facts = extract_environment_facts(content)
    if facts:
        metadata["facts"] = facts
    origin = source_origin(metadata.get("source") or metadata.get("source_path"))
    if origin:
        # Where the document came from, so a caller can weigh a maintained
        # handbook against a blog mirror instead of seeing them as equals.
        metadata["origin"] = origin
    flags = past_event_flags(content)
    if flags:
        # Writeups quote the flag they captured: useful for reproducing the
        # original challenge, misleading for a variant. Mark; the caller decides.
        metadata["flags"] = flags
    screenshots = screenshot_placeholders(content)
    if screenshots:
        # Strip-images keeps an alt-text placeholder, not the evidence, so the
        # caller has to fetch the original -- and the addresses let a multimodal
        # one actually read it.
        metadata["has_screenshots"] = screenshots
        images = _image_availability(image_references(content), document_path)
        if images:
            metadata["image_refs"] = images


def paginate_entries(entries, cursor, page_size):
    """Stable, stateless paging over a sorted source list.

    ``cursor`` is the first path of the *next* page (the value returned in a
    previous page's ``next_cursor``). Returns (page, next_cursor).
    """
    start = 0
    if cursor:
        start = bisect.bisect_left(entries, cursor)
    end = start + max(page_size, 1)
    page = entries[start:end]
    next_cursor = entries[end] if end < len(entries) else None
    return page, next_cursor


class FaissBackend(RetrievalBackend):
    """Read-only cosine retrieval over standalone memmap artifacts.

    ``build_cosine`` converts a LangChain FAISS store into normalized vectors
    and random-access documents. The original FAISS files remain untouched;
    this backend imports neither ``chatchat`` nor any Agent implementation.
    """

    name = "faiss"

    def __init__(self, config: RagConfig):
        self.config = config
        self._lock = threading.Lock()
        self._stores: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._embedders: Dict[str, OllamaEmbeddingClient] = {}
        self._query_cache: "OrderedDict[Tuple[str, str], List[float]]" = {}
        self._query_cache_max = 128
        # Provider failures are remembered briefly: a dead provider otherwise
        # costs a full connect/retry cycle on every call (measured 9-12 s in
        # degraded mode), and a fan-out of subagents pays it once per query.
        self._embed_failures: Dict[Tuple[str, str], Tuple[float, BaseException]] = {}
        self._embed_locks: Dict[Tuple[str, str], threading.Lock] = {}
        # ``search()`` returns a list for the stable backend interface, so an
        # empty lexical fallback needs a side channel to tell the facade that
        # the provider failed. Thread-local state keeps concurrent agent calls
        # from overwriting each other's degraded marker.
        self._search_state = threading.local()
        # In-flight readers, so a release never unmaps a store under one.
        self._readers = 0
        self._readers_lock = threading.Lock()

    def search_state(self) -> Dict[str, Any]:
        """Return per-call operational state for the service facade."""
        return {
            "degraded": getattr(self._search_state, "degraded", None),
        }

    def _reset_search_state(self) -> None:
        self._search_state.degraded = None

    @contextlib.contextmanager
    def _reader_lease(self):
        """Pin stores against release for the duration of a read.

        The count is taken *before* the store is loaded, so a concurrent eviction
        or close() cannot unmap a store this reader is about to use. Release
        reads the count without a lock: the count only ever prevents an unmap,
        so a stale read is the safe direction.
        """
        with self._readers_lock:
            self._readers += 1
        try:
            yield
        finally:
            with self._readers_lock:
                self._readers -= 1

    def close(self) -> None:
        """Release open memory maps, provider clients, and cached stores."""
        with self._lock:
            for store in self._stores.values():
                self._release_store(store)
            self._stores.clear()
            embedders = list(self._embedders.values())
            self._embedders.clear()
            self._query_cache.clear()
            self._embed_failures.clear()
            self._embed_locks.clear()
        for embedder in embedders:
            close = getattr(embedder, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> FaissBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def search(self, request: RetrievalRequest) -> List[SearchResult]:
        with self._reader_lease():
            return self._search_leased(request)

    def _search_leased(self, request: RetrievalRequest) -> List[SearchResult]:
        # The degraded marker is per call, not per thread. Without this reset it
        # sticks for the life of the worker thread, so every query after a
        # transient provider outage is labelled lexical-only even once the dense
        # path succeeds -- and callers are told to distrust correct cosine
        # scores.
        self._reset_search_state()
        chunk_id = str((request.filters or {}).get("chunk_id") or "").strip()
        if chunk_id and request.query.strip():
            # Silently ignoring it would return unrelated ranked results while
            # the caller believes it fetched one exact chunk — the same silent
            # no-op class as a misspelled exclusion.
            raise ValueError(
                "filters.chunk_id fetches one exact chunk and cannot be combined with a "
                "query; call with query='' to fetch the chunk, or drop chunk_id to search"
            )
        if not request.query.strip():
            return self._browse(request)
        embedding_model = self._resolve_embedding_model(request.knowledge_base)
        store = self._load_store(request.knowledge_base, embedding_model)
        path_note = describe_dense_path(store)
        if path_note:
            print(path_note, file=sys.stderr)
        try:
            query = self._query_embedding(embedding_model, request.query)
        except RagEmbeddingError:
            # The provider is down. Query mode needs embeddings, but the corpus
            # is still readable; preserve the failure state even when lexical
            # fallback finds no candidate, so an empty answer is not mistaken
            # for proof that the corpus lacks the topic.
            if not self.config.lexical_fallback:
                raise
            self._search_state.degraded = "lexical-only"
            return self._lexical_fallback(request, store)
        query_vector = np.asarray(query, dtype="float32")
        if query_vector.ndim != 1 or query_vector.size != store["dimension"]:
            raise RagIndexNotReadyError(
                f"query embedding dimension {query_vector.size} does not match index "
                f"dimension {store['dimension']}"
            )
        norm = np.linalg.norm(query_vector)
        if not np.isfinite(norm) or norm == 0:
            raise RagIndexNotReadyError("query embedding has zero or non-finite norm")
        query_vector /= norm

        lexical_weight = (
            request.lexical_weight
            if request.lexical_weight is not None
            else self.config.lexical_weight
        )
        terms = tokenize_query(request.query) if lexical_weight > 0 else None

        candidate_rows = None
        if request.filters:
            candidate_rows = self._filtered_rows(request, store)
        if candidate_rows is not None:
            if candidate_rows.size == 0:
                return []
            scores, indices = _top_rows_in_subset(
                store,
                query_vector,
                candidate_rows,
                max(request.top_k, self.config.filtered_candidate_limit, _MAX_CANDIDATES),
            )
        else:
            pool = self.config.candidate_pool
            if request.filters:
                pool = max(pool, self.config.filtered_candidate_limit)
            candidate_count = min(store["vectors"].shape[0], max(request.top_k, pool))
            scores, indices = _flat_ip_search(store, query_vector, candidate_count)
        if scores.size == 0:
            return []
        dense_by_row: Dict[int, float] = {
            int(row): float(score) for row, score in zip(indices, scores)
        }

        rows = list(dense_by_row)
        recall_rows = self._path_recall_rows(store, terms)
        recall_rows.extend(self._identifier_recall_rows(store, request))
        if recall_rows:
            extra = np.asarray(
                [row for row in recall_rows if row not in dense_by_row], dtype=np.int64
            )
            if extra.size:
                extra_scores = _score_rows(store, query_vector, extra)
                for row, score in zip(extra, extra_scores):
                    dense_by_row[int(row)] = float(score)
                    rows.append(int(row))
        documents = [self._read_document(store, row) for row in rows]
        dense_scores = [dense_by_row[row] for row in rows]

        ranked: List[tuple[float, float, float, int, Dict[str, Any]]] = []
        if terms is not None:
            reranker = LexicalReranker(
                terms,
                [
                    (document["text"], _first(document["metadata"], "source", "source_path") or "")
                    for document in documents
                ],
            )
            for dense_value, lexical, row, document in zip(
                dense_scores, reranker.scores, rows, documents
            ):
                dense = max(0.0, float(dense_value))
                ranked.append(
                    (
                        fusion_score(dense, lexical, lexical_weight),
                        dense,
                        lexical,
                        row,
                        document,
                    )
                )
            ranked.sort(key=lambda item: item[0], reverse=True)
        else:
            for dense_value, row, document in zip(dense_scores, rows, documents):
                dense = max(0.0, float(dense_value))
                ranked.append((dense, dense, 0.0, row, document))

        merge_neighbors = (
            request.merge_neighbors
            if request.merge_neighbors is not None
            else self.config.merge_neighbors
        )
        merge_limit = self.config.merge_neighbor_limit
        skip_low_info = self.config.low_info_filter
        filter_attachments = self.config.filter_flag_attachments

        results: List[SearchResult] = []
        seen_sources: set[str] = set()
        kept_signatures: list[list[int]] = []
        skipped_low_info = 0
        threshold = (
            request.score_threshold
            if request.score_threshold is not None
            else self.config.default_score_threshold
        )
        for score, dense, lexical, row, document in ranked:
            if score < threshold:
                continue
            if skip_low_info:
                if low_information_reasons(document["text"]):
                    skipped_low_info += 1
                    continue
            metadata = dict(document["metadata"])
            source = _first(metadata, "source", "source_path", "file_name", "filename")
            if filter_attachments and source and _is_flag_attachment_source(source):
                continue
            if not _matches_filters(source, request.filters):
                continue
            source_key = source or f"row:{row}"
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            if len(document["text"]) >= 80:
                signature = text_signature(document["text"])
                if kept_signatures and any(
                    is_near_duplicate(signature, kept) for kept in kept_signatures
                ):
                    continue
                kept_signatures.append(signature)
            chunk_id = _first(metadata, "id", "chunk_id")
            citation_kb = request.knowledge_base or self.config.default_knowledge_base
            content = document["text"]
            merged = 1
            merged_rows = [row]
            if merge_neighbors and merge_limit > 0 and request.include_content:
                content, merged, merged_rows = _merge_adjacent_rows(
                    store, row, source, merge_limit, content
                )
            # facts / flags / screenshot signals are attached by the shared
            # helper, so every result path reports the same set.
            _annotate_evidence(metadata, content, self._document_path(request, source))
            if request.extract:
                # The caller asked for the material itself, so hand back whole
                # fenced segments with their language rather than a query-centred
                # window it would have to re-assemble.
                segments = extract_segments(content, request.extract)
                metadata["segments"] = segments
                metadata["segment_kind"] = request.extract
                if not segments:
                    metadata["segments_empty"] = True
            strip_images = (
                request.strip_images
                if request.strip_images is not None
                else self.config.strip_images
            )
            snippet_chars = (
                request.snippet_chars
                if request.snippet_chars is not None
                else self.config.snippet_chars
            )
            if self.config.strip_provenance:
                provenance = parse_provenance_frontmatter(content)
                if provenance:
                    metadata["provenance"] = provenance
                content = strip_provenance_frontmatter(content)
            if strip_images:
                content = strip_markdown_images(content)
            # `extract` asks for whole segments; windowing the content first would
            # cut the fenced block the caller is after.
            if request.extract:
                snippet_chars = 0
            if snippet_chars and snippet_chars > 0 and len(content) > snippet_chars:
                content = snippet_window(content, request.query, snippet_chars)
                metadata["truncated"] = True
            content, was_truncated = _bounded_content(
                content,
                self.config.max_content_chars,
                query=request.query,
                truncated=bool(metadata.get("truncated")),
            )
            if was_truncated:
                metadata["truncated"] = True
            if lexical_weight > 0:
                metadata["dense_score"] = round(dense, 6)
                metadata["lexical_score"] = round(lexical, 6)
            if merged > 1:
                metadata["merged_chunks"] = merged
                metadata["merged_chunk_ids"] = [
                    f"{citation_kb}:{merged_row}" for merged_row in merged_rows
                ]
                metadata["chunk_id_range"] = (
                    f"{citation_kb}:{min(merged_rows)}-{citation_kb}:{max(merged_rows)}"
                )
            results.append(
                SearchResult(
                    content=content if request.include_content else None,
                    score=score,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
            )
            if len(results) >= request.top_k:
                break
        return results

    def _browse(self, request: RetrievalRequest) -> List[SearchResult]:
        """Query-free navigation: one chunk, full document, source listing, or corpus index."""
        filters = request.filters or {}
        source = str(filters.get("source") or "").strip()
        prefix = str(filters.get("source_prefix") or "").strip().replace("\\", "/")
        category = str(filters.get("category") or "").strip()
        chunk_id = str(filters.get("chunk_id") or "").strip()
        if chunk_id:
            others = sorted(key for key in filters if key != "chunk_id")
            if others:
                raise ValueError(
                    "filters.chunk_id fetches one exact chunk and ignores other filters; "
                    f"remove {', '.join(others)} or drop chunk_id"
                )
            if request.cursor:
                raise ValueError(
                    "cursor applies to source listings; filters.chunk_id returns exactly one "
                    "chunk and cannot be paginated"
                )
            return self._browse_chunk(request, chunk_id)
        if source:
            if not _matches_filters(source, filters):
                return []
            return self._browse_document(request, source)
        knowledge_base = request.knowledge_base or self.config.default_knowledge_base
        ranges = self._browse_ranges(knowledge_base)
        listing_filters = (
            "category",
            "source_prefix",
            "year",
            "exclude_source_prefix",
        )
        if any(filters.get(key) for key in listing_filters):
            entries = sorted(
                path for path in ranges if _matches_filters(path, filters)
            )
            if not entries:
                return []
            page_size = request.limit or request.top_k
            page, next_cursor = paginate_entries(entries, request.cursor, page_size)
            lines = [f"- {entry}" for entry in page]
            content = "\n".join(lines)
            if len(content) > 6000:
                content = content[:6000] + "\n… (listing page too large; lower limit or narrow source_prefix)"
            return [
                SearchResult(
                    content=content,
                    score=None,
                    source=(str(filters.get("source_prefix") or "") or str(filters.get("category") or "")),
                    chunk_id=None,
                    metadata={
                        "documents": len(entries),
                        "returned": len(page),
                        "limit": page_size,
                        "truncated": bool(next_cursor) or len(content) >= 6000,
                        "next_cursor": next_cursor,
                    },
                )
            ]
        # corpus index: categories with counts and samples
        categories = self.config.content_categories(knowledge_base)
        counts: Dict[str, int] = {}
        first_samples: Dict[str, List[str]] = {}
        for path in ranges:
            prefix = path.split("/", 1)[0] if "/" in path else path
            if not categories or prefix in categories:
                counts[prefix] = counts.get(prefix, 0) + 1
                samples = first_samples.setdefault(prefix, [])
                if len(samples) < 3:
                    samples.append(path.rsplit("/", 1)[-1])
        ordered = [name for name in categories if name in counts] + sorted(
            name for name in counts if name not in categories
        )
        blocks = []
        for name in ordered:
            blocks.append(f"{name}  ({counts[name]} documents)")
            for sample in first_samples.get(name, []):
                blocks.append(f"  - {sample}")
        content = (
            "Knowledge base index (query='' browse). Categories are source path prefixes:\n\n"
            + "\n".join(blocks)
        )
        return [
            SearchResult(
                content=content,
                score=None,
                source=None,
                chunk_id=None,
                metadata={"browse": True, "categories": ordered},
            )
        ]

    def _browse_ranges(self, knowledge_base: str) -> Dict[str, Tuple[int, int]]:
        embedding_model = self._resolve_embedding_model(knowledge_base)
        index_path = self.config.vector_store_path(knowledge_base, embedding_model)
        docs_path = index_path / "docs.cos.jsonl"
        offsets_path = index_path / "docs.cos.offsets.u64"
        manifest_path = index_path / "vectors.cos.json"
        for required in (docs_path, offsets_path, manifest_path):
            if not required.is_file():
                raise RagIndexNotReadyError(
                    f"knowledge base '{knowledge_base}' is missing standalone cosine files; "
                    "run `python -m rag_service.build_cosine`"
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = int(manifest.get("rows") or 0)
        if rows <= 0 or offsets_path.stat().st_size != rows * _OFFSET_STRUCT.size:
            raise RagIndexNotReadyError(
                f"knowledge base '{knowledge_base}' browse index is invalid"
            )
        # Query mode refuses a sidecar whose source fingerprint no longer matches
        # (see _load_store). Browse read the same manifest but never compared it,
        # so after a rebuild that skipped build_cosine every browse path kept
        # serving documents from the previous index while queries correctly
        # refused -- stale evidence in exactly the mode agents use to verify a
        # citation. Same check, same error.
        try:
            _read_current_manifest(
                manifest_path, index_path / "index.faiss", index_path / "index.pkl"
            )
        except Exception as exc:
            raise RagIndexNotReadyError(
                f"knowledge base '{knowledge_base}' standalone files are stale or unreadable; "
                "re-run `python -m rag_service.build_cosine`"
            ) from exc
        return _load_or_build_source_ranges(docs_path, rows)

    def _document_path(
        self, request: RetrievalRequest, source: str | None
    ) -> Path | None:
        """Absolute path of a cited document, for resolving its relative assets.

        Markdown image references are relative to the document, so without this
        the addresses in `image_refs` do not identify a file. `build_cosine`
        mirrors the content tree under `<kb_root>/<kb>/content`, which is where
        the source paths point.
        """
        if not source:
            return None
        knowledge_base = request.knowledge_base or self.config.default_knowledge_base
        candidate = (
            self.config.knowledge_base_root / knowledge_base / "content" / source
        )
        return candidate if candidate.is_file() else None

    def _browse_chunk(self, request: RetrievalRequest, chunk_id: str) -> List[SearchResult]:
        """Return exactly the cited chunk, so a citation can be verified.

        ``chunk_id`` is ``<knowledge_base>:<row>``; the row address is stable
        within one build, which is what verifying a citation needs. It changes
        when the index is rebuilt, so the stable pair to quote is
        ``source + chunk_id`` (or the quoted text itself).
        """
        knowledge_base = request.knowledge_base or self.config.default_knowledge_base
        chunk_kb, separator, row_text = chunk_id.rpartition(":")
        if not separator or chunk_kb != knowledge_base:
            raise ValueError(
                f"chunk_id knowledge_base must match the request ({knowledge_base!r}); "
                "copy the complete '<kb>:<row>' citation"
            )
        if not row_text.isdigit():
            raise ValueError(
                f"chunk_id must look like '<knowledge_base>:<row>' (got {chunk_id!r}); "
                "copy it from a previous result"
            )
        row = int(row_text)
        embedding_model = self._resolve_embedding_model(knowledge_base)
        store = self._load_store(knowledge_base, embedding_model)
        if row < 0 or row >= int(store["offsets"].size):
            return []
        document = self._read_document(store, row)
        metadata = dict(document["metadata"])
        source = _first(metadata, "source", "source_path", "file_name", "filename")
        content = document["text"]
        provenance = parse_provenance_frontmatter(content)
        if provenance:
            metadata["provenance"] = provenance
        content = strip_provenance_frontmatter(content)
        strip_images = request.strip_images if request.strip_images is not None else self.config.strip_images
        _annotate_evidence(metadata, content, self._document_path(request, source))
        if strip_images:
            content = strip_markdown_images(content)
        metadata.update({"browse": True, "chunk": True})
        snippet_chars = (
            request.snippet_chars
            if request.snippet_chars is not None
            else self.config.snippet_chars
        )
        if snippet_chars and snippet_chars > 0 and len(content) > snippet_chars:
            content = snippet_window(content, request.query, snippet_chars)
            metadata["truncated"] = True
        content, was_truncated = _bounded_content(
            content,
            self.config.max_content_chars,
            query=request.query,
            truncated=bool(metadata.get("truncated")),
        )
        if was_truncated:
            metadata["truncated"] = True
        return [
            SearchResult(
                content=content if request.include_content else None,
                score=None,
                source=source,
                chunk_id=_first(metadata, "id") or chunk_id,
                metadata=metadata,
            )
        ]

    def _browse_document(self, request: RetrievalRequest, source: str) -> List[SearchResult]:
        """Return a bounded page of chunks from one exact source path.

        ``snippet_chars=0`` disables query-centered snippet extraction; it does
        not disable the hard page cap. ``limit`` controls chunks per page and
        ``cursor=source:<row>`` continues at the next source row.
        """
        knowledge_base = request.knowledge_base or self.config.default_knowledge_base
        ranges = self._browse_ranges(knowledge_base)
        normalized_source = source.replace("\\", "/")
        span = ranges.get(normalized_source)
        if span is None:
            return []
        index_path = self.config.vector_store_path(
            knowledge_base, self._resolve_embedding_model(knowledge_base)
        )
        docs_path = index_path / "docs.cos.jsonl"
        offsets_path = index_path / "docs.cos.offsets.u64"
        offsets = np.fromfile(offsets_path, dtype="<u8")
        start_row, end_row = span
        page_start = start_row
        if request.cursor:
            if not request.cursor.startswith("source:"):
                raise ValueError(
                    "cursor for filters.source must come from the same source listing "
                    "and have the form source:<row>"
                )
            try:
                page_start = int(request.cursor.split(":", 1)[1])
            except (TypeError, ValueError):
                raise ValueError("cursor for filters.source is invalid; use next_cursor") from None
            if page_start < start_row or page_start > end_row:
                raise ValueError("cursor for filters.source is outside this document")
        page_size = request.limit or request.top_k
        page_end = min(end_row, page_start + page_size)
        if page_start >= page_end:
            return []
        documents = [
            self._read_document(
                {"docs_path": docs_path, "offsets": offsets, "documents": {}}, row
            )
            for row in range(page_start, page_end)
        ]
        texts = [document["text"] for document in documents]
        content = _join_merged_texts(texts) if texts else ""
        metadata: Dict[str, Any] = {
            "browse": True,
            "source_page": True,
            "chunks": end_row - start_row,
            "returned": page_end - page_start,
            "limit": page_size,
            "merged_chunks": page_end - page_start,
            "chunk_ids": [
                _first(document["metadata"], "id", "chunk_id") or f"{knowledge_base}:{row}"
                for row, document in zip(range(page_start, page_end), documents)
            ],
            "next_cursor": f"source:{page_end}" if page_end < end_row else None,
        }
        if page_end - page_start > 1:
            metadata["chunk_id_range"] = (
                f"{knowledge_base}:{page_start}-{knowledge_base}:{page_end - 1}"
            )
        provenance = parse_provenance_frontmatter(content)
        if provenance:
            metadata["provenance"] = provenance
        content = strip_provenance_frontmatter(content)
        strip_images = (
            request.strip_images
            if request.strip_images is not None
            else self.config.strip_images
        )
        _annotate_evidence(metadata, content, self._document_path(request, source))
        if strip_images:
            content = strip_markdown_images(content)
        max_chars = request.snippet_chars
        if max_chars is None:
            max_chars = self.config.snippet_chars
        if max_chars and max_chars > 0 and len(content) > max_chars:
            content = snippet_window(content, source.rsplit("/", 1)[-1], max_chars)
            metadata["truncated"] = True
        content, was_truncated = _bounded_content(
            content,
            self.config.max_content_chars,
            query=request.query,
            truncated=bool(metadata.get("truncated")),
        )
        if was_truncated:
            metadata["truncated"] = True
        if not request.include_content:
            content = None
        return [
            SearchResult(
                content=content,
                score=None,
                source=source,
                chunk_id=None,
                metadata=metadata,
            )
        ]

    def _filtered_rows(
        self, request: RetrievalRequest, store: Dict[str, Any]
    ) -> "np.ndarray | None":
        """Rows matching the request's path filters, or None when they don't apply.

        Returns None for filters that are not path facts (``year``), which are
        still applied to the documents of the ranked candidates.
        """
        if not _has_path_filters(request.filters or {}):
            return None
        rows = int(store["offsets"].size)
        ranges = _load_or_build_source_ranges(store["docs_path"], rows)
        if not ranges:
            return None
        return _path_prefix_rows(ranges, request.filters or {})

    def _load_store(self, knowledge_base: str, embedding_model: str) -> Dict[str, Any]:
        index_path = self.config.vector_store_path(knowledge_base, embedding_model)
        source_index = index_path / "index.faiss"
        source_pickle = index_path / "index.pkl"
        vectors_path = index_path / "vectors.cos.f32"
        docs_path = index_path / "docs.cos.jsonl"
        offsets_path = index_path / "docs.cos.offsets.u64"
        manifest_path = index_path / "vectors.cos.json"

        if not all(path.is_file() for path in (vectors_path, docs_path, offsets_path, manifest_path)):
            raise RagIndexNotReadyError(
                f"knowledge base '{knowledge_base}' is missing standalone cosine files; "
                "run `python -m rag_service.build_cosine --kb-root <kb_root>`"
            )
        # The source pair is only needed to *derive* the artifacts, and to prove
        # they are still current. A set published with `--prebuilt` ships
        # without it (Git cannot preserve the mtime the fingerprint records), so
        # require the files only when the manifest actually references them.
        if not _is_prebuilt_manifest(manifest_path):
            if not source_index.is_file() or source_index.stat().st_size <= 45:
                raise RagIndexNotReadyError(
                    f"knowledge base '{knowledge_base}' has no source FAISS index"
                )
            if not source_pickle.is_file() or source_pickle.stat().st_size == 0:
                raise RagIndexNotReadyError(
                    f"knowledge base '{knowledge_base}' has no source FAISS metadata"
                )

        try:
            manifest = _read_current_manifest(manifest_path, source_index, source_pickle)
        except Exception as exc:
            raise RagIndexNotReadyError(
                f"knowledge base '{knowledge_base}' standalone files are stale or unreadable; "
                "re-run `python -m rag_service.build_cosine`"
            ) from exc

        cache_key = (str(index_path), str(manifest["source"]), str(manifest["artifacts"]))
        with self._lock:
            existing = self._stores.get(cache_key)
            if existing is not None:
                return existing
            # A rebuilt index produces a new key, so the previous store (and its
            # open 4 GB memmap plus faiss index) would otherwise stay resident
            # forever in a long-lived process. Bound the cache and release the
            # evicted store's native resources.
            while len(self._stores) >= self.config.store_cache_limit:
                evicted_key = next(iter(self._stores))
                self._release_store(self._stores.pop(evicted_key))
            try:
                store = self._open_store(
                    vectors_path,
                    docs_path,
                    offsets_path,
                    manifest,
                    self.config.document_cache_limit,
                )
            except Exception as exc:
                raise RagIndexNotReadyError(
                    f"knowledge base '{knowledge_base}' standalone files are stale or unreadable; "
                    "re-run `python -m rag_service.build_cosine`"
                ) from exc
            self._stores[cache_key] = store
            return store

    def _release_store(self, store: Dict[str, Any] | None) -> None:
        """Drop a store's native resources (memmaps, faiss index).

        Skipped while a search is in flight: closing the mapping unmaps the file
        under a reader that is still scanning it, which is an access violation
        (process death), not a catchable exception. The reader's own reference
        finalises the mapping when it finishes. Readers take a lease before they
        load a store, so a reader either sees a live store or loads a fresh one.
        """
        if not store:
            return
        if self._readers > 0:
            return
        targets = [store.get("vectors")]
        quantized = store.get("quantized")
        if quantized:
            targets.append(quantized.get("vectors"))
        for target in targets:
            if target is not None and hasattr(target, "_mmap"):
                try:
                    target._mmap.close()
                except Exception:
                    pass
        store["vectors"] = None
        store["quantized"] = None
        store["sq8"] = None
        store["documents"] = {}

    @staticmethod
    def _open_store(
        vectors_path: Path,
        docs_path: Path,
        offsets_path: Path,
        manifest: Dict[str, Any],
        document_cache_limit: int,
    ) -> Dict[str, Any]:
        rows = manifest["rows"]
        dimension = manifest["dimension"]
        if not isinstance(rows, int) or rows <= 0 or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("manifest has invalid matrix shape")
        if vectors_path.stat().st_size != rows * dimension * np.dtype("float32").itemsize:
            raise ValueError("vector file size does not match manifest")
        if offsets_path.stat().st_size != rows * _OFFSET_STRUCT.size:
            raise ValueError("offset file size does not match manifest")
        if docs_path.stat().st_size == 0:
            raise ValueError("document file is empty")

        vectors = np.memmap(vectors_path, dtype="float32", mode="r", shape=(rows, dimension))
        offsets = np.fromfile(offsets_path, dtype="<u8")
        if offsets.size != rows or offsets[0] != 0:
            raise ValueError("offset file is invalid")
        if np.any(offsets[1:] <= offsets[:-1]) or offsets[-1] >= docs_path.stat().st_size:
            raise ValueError("document offsets are invalid")

        # Prefer the quantized matrix when it is present and consistent: it is
        # 4x smaller, so a resident service keeps ~1 GB instead of ~4 GB. The
        # float32 file stays authoritative and is used for any region the
        # quantized copy does not cover.
        quantized = _open_quantized(
            vectors_path.with_name("vectors.cos.int8"),
            vectors_path.with_name("vectors.cos.scales.f32"),
            rows,
            dimension,
            manifest,
            np,
        )
        sq8 = _open_sq8(vectors_path.with_name("vectors.cos.sq8"), rows)

        return {
            "vectors": vectors,
            "quantized": quantized,
            "sq8": sq8,
            "docs_path": docs_path,
            "offsets": offsets,
            "dimension": dimension,
            "documents": {},
            "document_cache_limit": document_cache_limit,
        }

    def status(self, knowledge_base: str | None = None) -> Dict[str, Any]:
        with self._reader_lease():
            return self._status_leased(knowledge_base)

    def _status_leased(self, knowledge_base: str | None = None) -> Dict[str, Any]:
        """Report runtime capabilities so callers can see degraded operation.

        Exposed through health: a service silently running the numpy scan (no
        faiss in the interpreter) or without a usable sq8 artifact looks
        20x slower than it is, and that was previously invisible.
        """
        name = knowledge_base or self.config.default_knowledge_base
        status: Dict[str, Any] = {
            "faiss_available": faiss_available(),
            "lexical_fallback": self.config.lexical_fallback,
        }
        try:
            store = self._load_store(name, self._resolve_embedding_model(name))
            status["dense_path"] = dense_path(store)
            status["rows"] = int(store["vectors"].shape[0])
            postings = _open_postings(store["docs_path"])
            status["postings"] = postings is not None
            if postings is None:
                status["postings_state"] = _postings_state(store["docs_path"])
            # Why the numpy scan is in use: an absent artifact and an unreadable
            # one need different remedies, and build_cosine only fixes the first.
            status["sq8_state"] = _sq8_state(store)
        except Exception as exc:
            status["dense_path"] = "unavailable"
            status["error"] = type(exc).__name__
        return status

    def known_sources(self, knowledge_base: str) -> set:
        """Indexed source paths, for validating filters against the corpus.

        Used to detect exclusions that match nothing (a typo silently drops no
        documents). Returns an empty set when the index is unavailable, so a
        caller can skip validation rather than fail a query over it.
        """
        with self._reader_lease():
            return self._known_sources_leased(knowledge_base)

    def _known_sources_leased(self, knowledge_base: str) -> set:
        """Body of :meth:`known_sources`, under a reader lease."""
        try:
            embedding_model = self._resolve_embedding_model(knowledge_base)
            store = self._load_store(knowledge_base, embedding_model)
            rows = int(store["offsets"].size)
            return set(_load_or_build_source_ranges(store["docs_path"], rows))
        except Exception:
            return set()

    def _lexical_fallback(
        self, request: RetrievalRequest, store: Dict[str, Any]
    ) -> List[SearchResult]:
        """Rank by lexical evidence alone when embeddings are unavailable.

        No dense signal exists, so this deliberately narrows what it claims:
        candidates come from the identifier postings index and from source-path
        matches, and their score is the IDF-weighted lexical value in ``[0, 1]``
        — *not* a cosine score. Results carry ``metadata.degraded`` so a caller
        can see the ranking is not the normal one, and the default score
        threshold is not applied (it was calibrated for cosine scores, so
        reusing it here would filter out everything useful).
        """
        from rag_service.relevance import LexicalReranker

        terms = tokenize_query(request.query)
        rows: List[int] = []
        seen: set[int] = set()
        for row in self._identifier_recall_rows(store, request):
            if row not in seen:
                seen.add(row)
                rows.append(row)
        for row in self._path_recall_rows(store, terms):
            if row not in seen:
                seen.add(row)
                rows.append(row)
        if not rows:
            return []

        documents = [self._read_document(store, row) for row in rows]
        reranker = LexicalReranker(
            terms,
            [
                (document["text"], _first(document["metadata"], "source", "source_path") or "")
                for document in documents
            ],
        )
        ranked = sorted(
            zip(reranker.scores, rows, documents), key=lambda item: -item[0]
        )
        results: List[SearchResult] = []
        seen_sources: set[str] = set()
        kept_signatures: list[list[int]] = []
        for score, row, document in ranked:
            if score <= 0:
                continue
            # The same content guards as the ranking path. Without them the
            # fallback surfaced low-information rows the main path suppresses —
            # and the postings index does reach those rows: version-like tokens
            # ("1.13") appear inside the large binary/attachment dumps, so a
            # degraded lookup could hand back a raw debug dump as "evidence".
            if self.config.low_info_filter and low_information_reasons(document["text"]):
                continue
            metadata = dict(document["metadata"])
            source = _first(metadata, "source", "source_path", "file_name", "filename")
            if self.config.filter_flag_attachments and source and _is_flag_attachment_source(source):
                continue
            if not _matches_filters(source, request.filters):
                continue
            source_key = source or f"row:{row}"
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            merge_neighbors = (
                request.merge_neighbors
                if request.merge_neighbors is not None
                else self.config.merge_neighbors
            )
            citation_kb = request.knowledge_base or self.config.default_knowledge_base
            content = document["text"]
            merged = 1
            merged_rows = [row]
            if merge_neighbors and self.config.merge_neighbor_limit > 0:
                content, merged, merged_rows = _merge_adjacent_rows(
                    store, row, source, self.config.merge_neighbor_limit, content
                )
            provenance = parse_provenance_frontmatter(content)
            if provenance:
                metadata["provenance"] = provenance
            content = strip_provenance_frontmatter(content)
            strip_images = (
                request.strip_images
                if request.strip_images is not None
                else self.config.strip_images
            )
            _annotate_evidence(metadata, content, self._document_path(request, source))
            if strip_images:
                content = strip_markdown_images(content)
            snippet_chars = (
                request.snippet_chars
                if request.snippet_chars is not None
                else self.config.snippet_chars
            )
            if snippet_chars and snippet_chars > 0 and len(content) > snippet_chars:
                content = snippet_window(content, request.query, snippet_chars)
                metadata["truncated"] = True
            content, was_truncated = _bounded_content(
                content,
                self.config.max_content_chars,
                query=request.query,
                truncated=bool(metadata.get("truncated")),
            )
            if was_truncated:
                metadata["truncated"] = True
            if merged > 1:
                metadata["merged_chunks"] = merged
                metadata["merged_chunk_ids"] = [
                    f"{citation_kb}:{merged_row}" for merged_row in merged_rows
                ]
                metadata["chunk_id_range"] = (
                    f"{citation_kb}:{min(merged_rows)}-{citation_kb}:{max(merged_rows)}"
                )
            metadata["degraded"] = "lexical-only"
            metadata["lexical_score"] = round(float(score), 6)
            results.append(
                SearchResult(
                    content=content if request.include_content else None,
                    score=score,
                    source=source,
                    chunk_id=_first(metadata, "id", "chunk_id"),
                    metadata=metadata,
                )
            )
            if len(results) >= request.top_k:
                break
        return results

    def _path_recall_rows(
        self, store: Dict[str, Any], terms: "QueryTerms | None"
    ) -> List[int]:
        """Rows force-included because their source path matches the query.

        Dense top-N alone can miss the document that is *about* the query: a
        generic passage scores marginally higher, and no reranker can fix an
        absent candidate. Path matching is the cheap recall guarantee — corpus
        file names are topic labels (``873-pentesting-rsync.md``), so a token
        found there is strong evidence. The first chunks of each matching
        document are added, since the introduction states the topic.
        """
        if terms is None or terms.is_empty or self.config.path_recall_limit <= 0:
            return []
        rows = int(store["offsets"].size)
        ranges = _load_or_build_source_ranges(store["docs_path"], rows)
        if not ranges:
            return []
        matched = _paths_matching_tokens(
            ranges, terms, self.config.path_recall_limit, self.config.path_recall_rows_per_path
        )
        return matched

    def _identifier_recall_rows(
        self, store: Dict[str, Any], request: RetrievalRequest
    ) -> List[int]:
        """Rows containing an identifier the query spells out exactly.

        Dense retrieval cannot be relied on for these: the term is rare, so the
        document stating it can sit anywhere in the ranking — the reported case
        was rank 628 of 1,016,721, one slot outside the rerank pool, and the
        path signal missed it because that file's *name* carries a different
        number. Looking the token up in the offline postings index bypasses
        dense ranking entirely.
        """
        limit = self.config.identifier_recall_limit
        if limit <= 0:
            return []
        tokens = sorted(identifier_tokens(request.query))
        if not tokens:
            return []
        postings = _open_postings(store["docs_path"])
        if not postings:
            return []
        total_rows = int(store["offsets"].size)
        rows: List[int] = []
        seen: set[int] = set()
        for token in tokens:
            for row in _lookup_postings(postings, token, limit):
                if 0 <= row < total_rows and row not in seen:
                    seen.add(row)
                    rows.append(row)
        return rows

    @staticmethod
    def _read_document(store: Dict[str, Any], row: int) -> Dict[str, Any]:
        if row < 0 or row >= len(store["offsets"]):
            raise ValueError(f"document row is out of range: {row}")
        cached = store["documents"].get(row)
        if cached is not None:
            return cached
        with store["docs_path"].open("rb") as handle:
            handle.seek(int(store["offsets"][row]))
            line = handle.readline()
        payload = json.loads(line.decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(f"document row {row} is invalid")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"document metadata for row {row} is invalid")
        documents = store["documents"]
        cache_limit = store.get("document_cache_limit")
        if cache_limit and len(documents) >= cache_limit:
            # Long-lived servers must not grow without bound: evict the oldest
            # entries (dicts preserve insertion order) instead of caching every
            # row ever read.
            for key in list(documents)[: max(1, cache_limit // 8)]:
                del documents[key]
        documents[row] = payload
        return payload

    def _query_embedding(self, model: str, query: str) -> List[float]:
        """Return the query vector, embedding once per distinct query.

        Three behaviours matter here:

        * **single-flight** — identical queries arriving together (a subagent
          fan-out asks the same question) produce one provider call;
        * **failure memoisation** — a down provider is remembered for
          ``embedding_failure_ttl`` instead of paying a connect-and-retry cycle
          per call;
        * **only real failures are cached** — ``BaseException`` (KeyboardInterrupt,
          SystemExit) must propagate untouched. Caching it replayed a phantom
          interrupt on unrelated later queries without even contacting the
          provider.
        """
        cache_key = (model, query)
        with self._lock:
            cached = self._query_cache.pop(cache_key, None)
            if cached is not None:
                self._query_cache[cache_key] = cached
                return cached
            remembered = self._recall_failure(cache_key)
            if remembered is not None:
                self._replay_failure(remembered)
            lock = self._embed_locks.get(cache_key)
            if lock is None:
                if len(self._embed_locks) >= self._query_cache_max:
                    self._embed_locks.clear()
                lock = threading.Lock()
                self._embed_locks[cache_key] = lock

        with lock:
            # Both caches must be re-checked here: a thread that was waiting on
            # the lock has to see a failure the winner just recorded, otherwise
            # every waiter re-hits the dead provider.
            with self._lock:
                cached = self._query_cache.get(cache_key)
                if cached is not None:
                    return cached
                remembered = self._recall_failure(cache_key)
            if remembered is not None:
                self._replay_failure(remembered)
            try:
                vector = self._embedder(model).embed_query(query)
            except Exception as exc:
                with self._lock:
                    self._embed_failures[cache_key] = (time.monotonic(), exc)
                raise
            with self._lock:
                self._query_cache[cache_key] = vector
                while len(self._query_cache) > self._query_cache_max:
                    self._query_cache.pop(next(iter(self._query_cache)))
            return vector

    def _recall_failure(self, cache_key: Tuple[str, str]) -> BaseException | None:
        """Return a remembered failure still inside its TTL, else None."""
        entry = self._embed_failures.get(cache_key)
        if entry is None:
            return None
        raised_at, error = entry
        if time.monotonic() - raised_at < self.config.embedding_failure_ttl:
            return error
        del self._embed_failures[cache_key]
        return None

    @staticmethod
    def _replay_failure(error: BaseException) -> "NoReturn":
        """Raise a fresh exception describing a remembered failure.

        The stored instance must not be re-raised directly: one object shared by
        concurrent callers accumulates tracebacks and is shared mutable state.
        A fresh instance per caller keeps each traceback its own.
        """
        if isinstance(error, RagEmbeddingError):
            raise RagEmbeddingError(str(error)) from None
        raise RagEmbeddingError(
            f"embedding provider failing ({type(error).__name__}): {error}"
        ) from None

    def _embedder(self, model: str) -> OllamaEmbeddingClient:
        with self._lock:
            if model not in self._embedders:
                self._embedders[model] = OllamaEmbeddingClient(
                    base_url=self.config.ollama_base_url,
                    model=model,
                    timeout=self.config.embedding_timeout,
                    keep_alive=self.config.embedding_keep_alive,
                )
            return self._embedders[model]

    def _resolve_embedding_model(self, knowledge_base: str) -> str:
        configured = self.config.embedding_model
        if configured:
            self._assert_metadata_model(knowledge_base, configured)
            return configured
        model = self._read_metadata_model(knowledge_base)
        if not model:
            raise RagIndexNotReadyError(
                f"knowledge base '{knowledge_base}' has no embedding metadata"
            )
        return model

    def _read_metadata_model(self, knowledge_base: str) -> str | None:
        if not self.config.metadata_path.is_file():
            raise RagIndexNotReadyError(
                f"knowledge base metadata is missing: {self.config.metadata_path}"
            )
        connection = sqlite3.connect(self.config.metadata_path)
        try:
            row = connection.execute(
                "SELECT embed_model FROM knowledge_base WHERE lower(kb_name) = lower(?)",
                (knowledge_base,),
            ).fetchone()
        finally:
            connection.close()
        return str(row[0]) if row and row[0] else None

    def _assert_metadata_model(self, knowledge_base: str, embedding_model: str) -> None:
        metadata_model = self._read_metadata_model(knowledge_base)
        if metadata_model is None:
            raise RagIndexNotReadyError(f"knowledge base not found: {knowledge_base}")
        if metadata_model != embedding_model:
            raise RagIndexNotReadyError(
                f"embedding model mismatch for '{knowledge_base}': "
                f"configured '{embedding_model}', metadata '{metadata_model}'"
            )


def _is_prebuilt_manifest(manifest_path: Path) -> bool:
    """True when the manifest declares a self-contained, source-less artifact set.

    Read leniently: an unreadable or malformed manifest returns False so the
    caller keeps its existing, stricter error path.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return manifest.get("source") is None and "source" in manifest


def _read_current_manifest(
    manifest_path: Path,
    source_index: Path,
    source_pickle: Path,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != _ARTIFACT_FORMAT:
        raise ValueError(f"unsupported artifact format: {manifest.get('format')!r}")
    if not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("manifest artifacts are missing")
    if manifest.get("source") is None:
        # Published with `build_cosine --prebuilt`: the set is self-contained and
        # was never tied to a source index it ships. There is consequently
        # nothing for it to be stale against.
        #
        # If a source index is present anyway, someone put one there on purpose;
        # refuse rather than serve artifacts of unknown provenance against it.
        if source_index.is_file() or source_pickle.is_file():
            raise ValueError("a source index is present but the artifacts are prebuilt")
        return manifest
    expected_source = {
        "index.faiss": _file_state(source_index),
        "index.pkl": _file_state(source_pickle),
    }
    if manifest.get("source") != expected_source:
        raise ValueError("manifest source does not match current files")
    return manifest


def _file_state(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


_CACHE_ENTRIES = 4
"""Fingerprint-keyed module caches keep at most this many generations.

Keys include the document-file fingerprint, so a rebuilt index adds an entry
instead of overwriting one. Without a bound, a long-lived process that survives
several rebuilds retains every generation — each source-range map holds ~20k
entries (several MB) and nothing evicted it, not even ``close()``.
"""


def _cache_put(cache: Dict[Any, Any], key: Any, value: Any) -> None:
    """Insert into a bounded module cache, evicting oldest entries.

    Module caches are shared by every backend instance in the process, so two
    threads can reach the eviction loop together. Guarding each step keeps a
    concurrent ``pop`` from making ``next(iter(...))`` raise ``StopIteration``.
    """
    cache[key] = value
    while len(cache) > _CACHE_ENTRIES:
        try:
            oldest = next(iter(cache))
        except StopIteration:
            return
        cache.pop(oldest, None)


_RANGES_CACHE: Dict[Tuple[str, int, int, int], Dict[str, Tuple[int, int]]] = {}


def _load_or_build_source_ranges(
    docs_path: Path, rows: int
) -> Dict[str, Tuple[int, int]]:
    """Return ``{source: (start_row, end_row)}``, building the sidecar once.

    Rows are stored in document order so every source occupies one contiguous
    span. ``build_cosine`` writes the sidecar while it converts the index, so
    the normal path is a single JSON read; the streaming fallback exists only
    for artifacts produced before that. The cache key includes the document
    file fingerprint so a rebuilt index is never served stale spans.
    """
    sidecar = docs_path.with_name("docs.cos.ranges.json")
    fingerprint = _file_state(docs_path)
    cache_key = (str(docs_path), rows, fingerprint["size"], fingerprint["mtime_ns"])
    cached = _RANGES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                payload.get("rows") == rows
                and payload.get("source") == fingerprint
                and isinstance(payload.get("ranges"), dict)
            ):
                ranges = {
                    source: (int(span[0]), int(span[1]))
                    for source, span in payload["ranges"].items()
                    if isinstance(span, list) and len(span) == 2
                }
                if ranges:
                    _cache_put(_RANGES_CACHE, cache_key, ranges)
                    return ranges
        except (OSError, ValueError):
            pass
    ranges: Dict[str, Tuple[int, int]] = {}
    start_row = 0
    current_source: str | None = None
    with docs_path.open("rb") as handle:
        for row, raw in enumerate(handle):
            if row >= rows:
                break
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
            except ValueError:
                continue
            source = str((payload.get("metadata") or {}).get("source") or "").replace("\\", "/")
            if source != current_source:
                if current_source is not None:
                    ranges[current_source] = (start_row, row)
                current_source = source
                start_row = row
    if current_source is not None:
        ranges[current_source] = (start_row, rows)
    if ranges:
        try:
            sidecar.write_text(
                json.dumps(
                    {
                        "rows": rows,
                        "source": {
                            "size": docs_path.stat().st_size,
                            "mtime_ns": docs_path.stat().st_mtime_ns,
                        },
                        "ranges": ranges,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        _cache_put(_RANGES_CACHE, cache_key, ranges)
    return ranges


def _paths_matching_tokens(
    ranges: Dict[str, Tuple[int, int]],
    terms: "QueryTerms",
    limit: int,
    rows_per_path: int,
) -> List[int]:
    """Leading rows of the ``limit`` best source paths matching ``terms``.

    Ranking uses the rarity of each matched token *among the matching paths*
    (a local IDF): a path matching ``rsync`` outranks one matching only the
    generic ``漏洞``. Ties fall back to path order, which is the corpus order.
    """
    tokens = terms.tokens()
    matched: List[Tuple[str, List[str]]] = []
    for source in ranges:
        lowered = source.replace("\\", "/").lower()
        hits = [token for token in tokens if token in lowered]
        if hits:
            matched.append((source, hits))
    if not matched:
        return []
    frequencies: Dict[str, int] = {}
    for _, hits in matched:
        for token in hits:
            frequencies[token] = frequencies.get(token, 0) + 1
    scored: List[Tuple[float, str]] = []
    for source, hits in matched:
        score = sum(1.0 / frequencies[token] for token in hits)
        scored.append((score, source))
    scored.sort(key=lambda item: (-item[0], item[1]))
    rows: List[int] = []
    for _, source in scored[:limit]:
        start, end = ranges[source]
        # The opening chunks carry the topic; a long document's middle does not
        # need to be force-added.
        rows.extend(range(start, min(end, start + rows_per_path)))
    return rows


_POSTINGS_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


def _open_postings(docs_path: Path) -> Dict[str, Any] | None:
    """Load the sparse postings index for a document file, or None.

    Only the sparse marks are held in memory (a few hundred entries); rows are
    read from disk on demand, so this does not grow with corpus size.

    An index that declares zero entries is returned as-is even when the document
    fingerprint no longer matches. It holds no offsets, so there is nothing that
    could have gone stale -- it means the corpus has no identifier tokens (no
    CVEs, version numbers, and so on), which is normal for a small fixture. The
    fingerprint check exists to stop stale *rows* being read, and an empty index
    has none. Without this the same knowledge base reported
    ``postings=unusable`` after being copied or cloned, and ``postings=ok``
    before, because copying rewrites mtime.
    """
    postings_path = docs_path.with_name("docs.cos.postings")
    index_file = docs_path.with_name("docs.cos.postings.idx.json")
    if not postings_path.is_file() or not index_file.is_file():
        return None
    fingerprint = _file_state(docs_path)
    cache_key = (str(docs_path), fingerprint["size"], fingerprint["mtime_ns"])
    cached = _POSTINGS_CACHE.get(cache_key)
    if cached is not None:
        return cached or None
    try:
        payload = json.loads(index_file.read_text(encoding="utf-8"))
        if payload.get("format") != "postings-v1":
            _cache_put(_POSTINGS_CACHE, cache_key, {})
            return None
        raw_marks = payload["marks"]
        if not raw_marks:
            postings = {"path": postings_path, "marks": [], "max_df": int(payload.get("max_df") or 0)}
            _cache_put(_POSTINGS_CACHE, cache_key, postings)
            return postings
        if payload.get("source") != fingerprint:
            _cache_put(_POSTINGS_CACHE, cache_key, {})
            return None
        marks = [(str(name), int(offset)) for name, offset in raw_marks]
    except (OSError, ValueError, KeyError, TypeError):
        _cache_put(_POSTINGS_CACHE, cache_key, {})
        return None
    postings = {"path": postings_path, "marks": marks, "max_df": int(payload.get("max_df") or 0)}
    _cache_put(_POSTINGS_CACHE, cache_key, postings)
    return postings


def _lookup_postings(postings: Dict[str, Any], token: str, limit: int) -> List[int]:
    """Return up to ``limit`` rows containing ``token`` via binary search.

    The file is sorted by token, so a sparse mark locates the neighbourhood and
    a short forward scan finds the exact line.
    """
    marks = postings["marks"]
    if not marks:
        return []
    lo, hi = 0, len(marks) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if marks[mid][0] <= token:
            lo = mid
        else:
            hi = mid - 1
    with postings["path"].open("rb") as handle:
        handle.seek(marks[lo][1])
        for _ in range(_POSTINGS_SCAN_LINES):
            line = handle.readline()
            if not line:
                return []
            name, tab, payload = line.partition(b"\t")
            if not tab:
                continue
            current = name.decode("utf-8", "ignore")
            if current > token:
                return []
            if current == token:
                rows = payload.strip().decode("ascii", "ignore").split(",")
                return [int(row) for row in rows[:limit] if row]
    return []


def _open_sq8(sq8_path: Path, rows: int):
    """Load the faiss int8 index, or None when absent/unusable.

    Degrades to the numpy paths on any problem: a broken auxiliary index must
    not take retrieval down.
    """
    if not sq8_path.is_file() or sq8_path.stat().st_size == 0:
        return None
    try:
        import faiss

        index = faiss.read_index(str(sq8_path))
        if index.ntotal != rows or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            return None
        index.nprobe = getattr(index, "nprobe", 1)
        return index
    except Exception:
        return None


def _open_quantized(
    int8_path: Path,
    scales_path: Path,
    rows: int,
    dimension: int,
    manifest: Dict[str, Any],
    np,
) -> Dict[str, Any] | None:
    """Open the int8 matrix + per-row scales, or None when unusable.

    Returning None rather than raising keeps the float32 matrix a working
    fallback: a partial or corrupt quantization must degrade to the larger
    artifact, not break retrieval.
    """
    if manifest.get("quantization") != "int8":
        return None
    if not int8_path.is_file() or not scales_path.is_file():
        return None
    try:
        if int8_path.stat().st_size != rows * dimension:
            return None
        if scales_path.stat().st_size != rows * np.dtype("float32").itemsize:
            return None
        return {
            "vectors": np.memmap(
                int8_path, dtype=np.int8, mode="r", shape=(rows, dimension)
            ),
            "scales": np.fromfile(scales_path, dtype="<f4"),
        }
    except (OSError, ValueError):
        return None


_QUANT_BLOCK = 131_072
_RECONSTRUCT_CHUNK = 16_384
"""Rows decoded per faiss reconstruct call (16k x 1024 x 4B = ~67 MB)."""
_POSTINGS_SCAN_LINES = 640
"""Forward scan bound after a sparse mark; must exceed the builder's stride."""


def _full_scores(store: Dict[str, Any], query: np.ndarray) -> np.ndarray:
    """Score every row, reading the quantized matrix when available.

    Dequantization is done per block: a single upcast of the int8 matrix to
    float32 would allocate the 4 GB the quantization exists to avoid.
    """
    quantized = store.get("quantized")
    if not quantized:
        return store["vectors"] @ query
    vectors = quantized["vectors"]
    scales = quantized["scales"]
    rows = vectors.shape[0]
    scores = np.empty(rows, dtype=np.float32)
    for start in range(0, rows, _QUANT_BLOCK):
        end = min(start + _QUANT_BLOCK, rows)
        block = vectors[start:end].astype(np.float32) @ query
        block *= scales[start:end]
        scores[start:end] = block
    return scores


def _score_rows(
    store: Dict[str, Any], query: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Score an explicit row subset.

    Uses the *same* quantized representation as the full scan. Previously the
    subset path decoded the independent per-row int8 file while the full scan
    used faiss SQ8, so the two produced slightly different scores and a near-tie
    could flip whenever a filter was present — even a filter that excluded
    nothing. Now the scorer is chosen by what is *available* (faiss or not), not
    by whether a filter was passed.
    """
    if rows.size == 0:
        return np.empty(0, dtype=np.float32)
    sq8 = store.get("sq8")
    if sq8 is not None:
        try:
            return _score_rows_sq8(sq8, query, rows)
        except Exception:
            pass  # a reconstruct failure must not break retrieval
    return _score_rows_int8(store, query, rows)


def _score_rows_sq8(index: Any, query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Decode the requested rows from the faiss index and dot them.

    ``reconstruct_n`` yields exactly the vectors ``search`` uses internally
    (verified to 6e-8), so subset scores match full-scan scores. Rows are
    grouped into contiguous runs and decoded in bounded chunks: a naive
    reconstruct spanning first..last row would allocate gigabytes when the
    subset's extremes are far apart.
    """
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    ordered_scores = np.empty(ordered.size, dtype=np.float32)
    start = 0
    while start < ordered.size:
        run_end = start + 1
        while run_end < ordered.size and ordered[run_end] == ordered[run_end - 1] + 1:
            run_end += 1
        first = int(ordered[start])
        run_length = run_end - start
        for offset in range(0, run_length, _RECONSTRUCT_CHUNK):
            take = min(_RECONSTRUCT_CHUNK, run_length - offset)
            decoded = index.reconstruct_n(first + offset, take)
            ordered_scores[start + offset : start + offset + take] = decoded @ query
        start = run_end
    scores = np.empty(rows.size, dtype=np.float32)
    scores[order] = ordered_scores
    return scores


def _score_rows_int8(
    store: Dict[str, Any], query: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Fallback scorer for interpreters without faiss (no sq8 index present)."""
    quantized = store.get("quantized")
    if not quantized:
        return store["vectors"][rows] @ query
    vectors = quantized["vectors"]
    scales = quantized["scales"]
    scores = np.empty(rows.size, dtype=np.float32)
    for start in range(0, rows.size, _QUANT_BLOCK):
        block_rows = rows[start : start + _QUANT_BLOCK]
        block = vectors[block_rows].astype(np.float32) @ query
        block *= scales[block_rows]
        scores[start : start + block_rows.size] = block
    return scores


def _top_from_scores(
    scores: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    best = min(limit, scores.size)
    if best <= 0:
        return np.empty(0, dtype="float32"), np.empty(0, dtype="int64")
    positions = np.argpartition(-scores, best - 1)[:best]
    positions = positions[np.argsort(-scores[positions], kind="stable")]
    return scores[positions], positions.astype("int64", copy=False)


def _flat_ip_search(
    store: Dict[str, Any], query: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted top-k inner-product matches without materializing documents.

    Uses the faiss int8 index when available: it computes the quantized dot
    product directly, whereas the numpy path must first expand every block to
    float32 (measured 3.5-7.7 s vs 0.9-1.2 s per query on 1M x 1024).
    """
    sq8 = store.get("sq8")
    if sq8 is not None and top_k > 0:
        scores, rows = sq8.search(np.ascontiguousarray(query.reshape(1, -1)), top_k)
        return (
            np.asarray(scores[0], dtype=np.float32),
            np.asarray(rows[0], dtype=np.int64),
        )
    scores = _full_scores(store, query)
    return _top_from_scores(scores, top_k)


def _top_rows_in_subset(
    store: Dict[str, Any],
    query: np.ndarray,
    rows: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best ``limit`` rows by inner product, scoring only ``rows``."""
    if rows.size == 0:
        return np.empty(0, dtype="float32"), np.empty(0, dtype="int64")
    scores = _score_rows(store, query, rows)
    values, positions = _top_from_scores(scores, limit)
    return values, rows[positions].astype("int64", copy=False)


def _under(source: str, prefix: str) -> bool:
    """True when ``source`` is ``prefix`` itself or lives below it."""
    return source == prefix or source.startswith(f"{prefix}/")


def _normalize_prefixes(value: Any) -> List[str]:
    if not value:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).replace("\\", "/").strip("/") for item in items if str(item).strip()]


def _has_path_filters(filters: Dict[str, Any]) -> bool:
    """True when a filter can be resolved against source→row spans.

    ``year`` belongs here: the documented contract is "a 4-digit year anywhere
    in the source path", which is a path fact like the others. Leaving it out
    applied the filter only to the global candidate pool, so ``year=2014``
    searched 600 rows out of the 1452 that actually match and returned 3 hits.
    """
    return any(
        filters.get(key)
        for key in ("category", "source_prefix", "source", "exclude_source_prefix", "year")
    )


def _path_prefix_rows(
    ranges: Dict[str, Tuple[int, int]],
    filters: Dict[str, Any],
) -> np.ndarray:
    """Rows whose source satisfies the path filters in ``filters``.

    ``category``/``source_prefix``/``source``/``exclude_source_prefix`` are all
    path facts, so they are resolved against the source spans *before* scoring
    rather than tested after a global top-N was picked. Testing them afterwards
    made narrow filters return empty: when a query's global neighbourhood lies
    in another folder, the few matching rows outside the candidate pool can
    never surface, even though the filter itself is correct.
    """
    category = str(filters.get("category") or "").replace("\\", "/").strip("/")
    prefix = str(filters.get("source_prefix") or "").replace("\\", "/").strip()
    exact = str(filters.get("source") or "").replace("\\", "/").strip()
    excludes = _normalize_prefixes(filters.get("exclude_source_prefix"))
    year = filters.get("year")
    year_pattern = (
        re.compile(rf"(?:^|[^0-9]){int(year)}(?:[^0-9]|$)") if year is not None else None
    )
    rows: List[int] = []
    for source, (start, end) in ranges.items():
        if category and not _under(source, category):
            continue
        # Literal prefix, matching _matches_filters: the same rule must decide
        # row selection and the final re-check, or a document could be scored
        # and then dropped (or never scored at all).
        if prefix and not source.startswith(prefix):
            continue
        if exact and source != exact:
            continue
        if excludes and any(source.startswith(item) for item in excludes):
            continue
        if year_pattern is not None and not year_pattern.search(source):
            continue
        rows.extend(range(start, end))
    return np.asarray(rows, dtype=np.int64)


def _matches_filters(
    source: str | None,
    filters: Dict[str, Any],
) -> bool:
    """Match only normalized, path-derived fields; never execute user input.

    Prefix semantics (matching the documented contract):

    - ``category`` is a *path segment*: the first directory of the source, so
      ``13_xianzhi`` cannot match ``13_xianzhi_notes``;
    - ``source_prefix``/``exclude_source_prefix`` are *literal string prefixes*,
      so ``13_xianzhi/17`` does exclude ``13_xianzhi/17572-...md``. Segment
      matching here was silently wrong: an exclusion meant to remove noise would
      no-op on a partial segment, and it failed in the unsafe direction
      (excluding less than asked). Add a trailing slash when you mean a whole
      directory and want to avoid ``2014`` matching ``20140``.
    """
    if not filters:
        return True
    source_value = (source or "").replace("\\", "/")
    category = filters.get("category")
    if category is not None:
        category = str(category).replace("\\", "/").strip("/")
        if not (source_value == category or source_value.startswith(f"{category}/")):
            return False
    source_prefix = filters.get("source_prefix")
    if source_prefix is not None:
        source_prefix = str(source_prefix).replace("\\", "/").strip()
        if not source_value.startswith(source_prefix):
            return False
    exclude_prefixes = _normalize_prefixes(filters.get("exclude_source_prefix"))
    if exclude_prefixes:
        if any(source_value.startswith(prefix) for prefix in exclude_prefixes):
            return False
    requested_source = filters.get("source")
    if requested_source is not None:
        requested_source = str(requested_source).replace("\\", "/")
        if source_value != requested_source:
            return False
    if "year" in filters:
        year = str(filters["year"])
        if not re.search(rf"(?:^|[^0-9]){year}(?:[^0-9]|$)", source_value):
            return False
    return True
def _first(metadata: Dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if metadata.get(key):
            return str(metadata[key])
    return None


def _is_flag_attachment_source(source: str) -> bool:
    """True for imported raw attachment files such as ``flag.txt``.

    Writeups whose file name merely mentions flags (``flag-lottery.md``,
    ``flagcheck67.md``) are content and are not affected.
    """
    normalized = source.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    name = basename.lower()
    if not name.startswith("flag"):
        return False
    if name.endswith(".md") or name.endswith(".markdown"):
        return False
    if name in {"flag"} or name.startswith("flag."):
        return True
    return False


def _merge_adjacent_rows(
    store: Dict[str, Any],
    row: int,
    source: str,
    limit: int,
    main_text: str,
) -> tuple[str, int, list[int]]:
    """Join same-source rows and return text, count, and exact row anchors."""
    total_rows = len(store["offsets"])

    def collect(start: int, step: int) -> list[tuple[int, str]]:
        gathered: list[tuple[int, str]] = []
        cursor = start
        while 0 <= cursor < total_rows and len(gathered) < limit:
            document = FaissBackend._read_document(store, cursor)
            metadata = dict(document["metadata"])
            candidate_source = _first(metadata, "source", "source_path", "file_name", "filename")
            if candidate_source != source:
                break
            gathered.append((cursor, document["text"]))
            cursor += step
        return gathered

    backward = collect(row - 1, -1)
    forward = collect(row + 1, 1)
    if not backward and not forward:
        return main_text, 1, [row]
    backward = list(reversed(backward))
    parts = backward + [(row, main_text)] + forward
    return _join_merged_texts([text for _, text in parts]), len(parts), [item[0] for item in parts]

def _bounded_content(
    content: str,
    max_chars: int,
    *,
    query: str = "",
    truncated: bool = False,
) -> tuple[str, bool]:
    """Apply an unconditional, exact output cap after transformations."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content, truncated
    # Leave room for snippet_window's line-boundary look-around so the final
    # string cannot exceed the configured cap. Keep the query anchor centered
    # when an earlier, larger snippet window was requested.
    window_size = max(1, max_chars - 160)
    window = snippet_window(content, query, window_size)
    if len(window) > max_chars:
        marker = "…"
        if max_chars <= len(marker):
            window = marker[:max_chars]
        else:
            window = window[: max_chars - len(marker)].rstrip() + marker
    return window, True


def _join_merged_texts(texts: Sequence[str]) -> str:
    """Concatenate chunks with paragraph separators, removing overlap at seams.

    Two mechanisms remove splitter artifacts:

    - line run: when the trailing lines of the joined text equal the leading
      lines of the next chunk (headings, repeated paragraphs), the duplicate
      run is dropped; short single-line matches (`` ``` ``, list markers) are
      kept unless the line is substantial (>= 25 chars);
    - exact character overlap of at least 12 chars (splitter overlap window
      cut mid-line) is trimmed.
    """
    if not texts:
        return ""
    parts: list[str] = [texts[0]]
    raw = texts[0]
    for text in texts[1:]:
        if not text:
            continue
        remainder = _trim_seam_overlap(raw, text)
        if remainder:
            parts.append(remainder)
            raw += remainder
    return "\n\n".join(parts)


def _trim_seam_overlap(left: str, text: str) -> str:
    """Return ``text`` with any duplicated seam content removed."""
    left_lines = left.split("\n")
    right_lines = text.split("\n")
    run = 0
    limit = min(len(left_lines), len(right_lines), 10)
    for size in range(limit, 0, -1):
        if right_lines[:size] == left_lines[-size:]:
            run = size
            break
    if run:
        removed_chars = sum(len(line) for line in right_lines[:run])
        if run >= 2 or removed_chars >= 25:
            remainder = "\n".join(right_lines[run:])
            if remainder.strip():
                return "\n" + remainder
            return ""
    tail = left[-300:]
    match_limit = min(len(tail), len(text), 300)
    cut = 0
    for size in range(match_limit, 11, -1):
        if tail[-size:] == text[:size]:
            cut = size
            break
    return text[cut:]
