from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class RagConfig:
    """Runtime policy for the standalone retrieval service.

    The service is intentionally read-only. Index construction and document
    management remain outside the agent-facing surface.
    """

    knowledge_base_root: Path
    default_knowledge_base: str = "cybersec"
    default_top_k: int = 5
    default_score_threshold: float = 0.45
    max_top_k: int = 50
    candidate_pool: int = 600
    """Dense candidates reranked for one unfiltered query.

    The cosine product already scores every row, so this only bounds how many
    documents are read back and lexically scored. At 250 the specific writeup
    could sit just outside the pool and never be seen by the reranker.
    """
    path_recall_limit: int = 400
    """Rows force-included from sources whose path matches the query tokens."""
    path_recall_rows_per_path: int = 2
    """Leading chunks taken per path match; the intro carries the topic."""
    identifier_recall_limit: int = 40
    """Rows force-included per exact identifier token found in the postings.

    The path signal cannot cover a body-only identifier: the reported failure
    was ``CVE-2021-3490``, whose document states the number in its text but
    spells a different one in its filename, and whose dense rank (628) fell just
    outside the rerank pool. 0 disables the lookup.
    """
    lexical_fallback: bool = True
    """Rank by lexical evidence when the embedding provider is unreachable.

    Query mode needs embeddings, but the corpus stays readable without them. In
    a CTF the difference between "no evidence" and "unranked lexical evidence"
    matters, so the service degrades instead of failing. Fallback results are
    marked ``metadata.degraded``.
    """
    document_cache_limit: int = 2048
    """Bound on cached document rows, so a long-lived server cannot grow."""
    store_cache_limit: int = 3
    """Retained index generations per process.

    Keys include the manifest fingerprint, so a rebuild creates a new entry
    rather than replacing one. Each store holds an open memmap over the full
    vector matrix (plus the faiss index), so an unbounded cache leaks several GB
    per rebuild into a long-lived service.
    """
    embedding_failure_ttl: float = 30.0
    """Seconds to remember an embedding failure.

    A dead provider costs a full connect/retry cycle per call (measured 9-12 s
    in degraded mode). Memoising the failure briefly turns "every query pays"
    into "one query per window pays", while still retrying soon enough to
    recover on its own.
    """
    filtered_candidate_limit: int = 1200
    """Candidate pool examined when filters are present.

    The dense matmul already scores every row, so the pool is only bounded by
    how many documents are read back. Filtering a 250-row pool out of a
    million-row index made narrow filters (a year, one archive) return empty
    even when matching documents existed.
    """
    max_query_length: int = 8_000
    allowed_knowledge_bases: FrozenSet[str] = frozenset()
    embedding_model: Optional[str] = None
    backend_name: str = "faiss"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_timeout: float = 120.0
    embedding_keep_alive: str = "30m"
    lexical_weight: float = 0.35
    merge_neighbors: bool = True
    merge_neighbor_limit: int = 2
    low_info_filter: bool = True
    strip_images: bool = True
    strip_provenance: bool = True
    snippet_chars: int = 800
    max_content_chars: int = 8_000
    """Hard cap for any one returned content page/result, including snippet_chars=0."""
    filter_flag_attachments: bool = True
    low_score_warn: float = 0.55

    @property
    def metadata_path(self) -> Path:
        return self.knowledge_base_root / "info.db"

    def vector_store_path(self, knowledge_base: str, embedding_model: str) -> Path:
        if not self.is_allowed_knowledge_base(knowledge_base):
            raise ValueError(f"knowledge base is not allowed: {self._explain_disallowed(knowledge_base)}")
        if not embedding_model or Path(embedding_model).name != embedding_model:
            raise ValueError("embedding_model must be a simple model name")
        vector_name = embedding_model.replace(":", "_").replace("/", "__")
        return self.knowledge_base_root / knowledge_base / "vector_store" / vector_name

    def __post_init__(self) -> None:
        object.__setattr__(self, "knowledge_base_root", Path(self.knowledge_base_root).resolve())
        if self.default_top_k < 1 or self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must be within 1..max_top_k")
        if not 0 <= self.default_score_threshold <= 2:
            raise ValueError("default_score_threshold must be within 0..2")
        if self.max_top_k < 1 or self.max_top_k > 500:
            raise ValueError("max_top_k must be within 1..500")
        if self.filtered_candidate_limit < 1 or self.filtered_candidate_limit > 100_000:
            raise ValueError("filtered_candidate_limit must be within 1..100000")
        if self.candidate_pool < 1 or self.candidate_pool > 20_000:
            raise ValueError("candidate_pool must be within 1..20000")
        if self.path_recall_limit < 0 or self.path_recall_limit > 20_000:
            raise ValueError("path_recall_limit must be within 0..20000")
        if self.path_recall_rows_per_path < 1 or self.path_recall_rows_per_path > 50:
            raise ValueError("path_recall_rows_per_path must be within 1..50")
        if self.identifier_recall_limit < 0 or self.identifier_recall_limit > 5_000:
            raise ValueError("identifier_recall_limit must be within 0..5000")
        if self.document_cache_limit < 64:
            raise ValueError("document_cache_limit must be at least 64")
        if self.embedding_failure_ttl < 0 or self.embedding_failure_ttl > 3600:
            raise ValueError("embedding_failure_ttl must be within 0..3600")
        if self.store_cache_limit < 1 or self.store_cache_limit > 32:
            raise ValueError("store_cache_limit must be within 1..32")
        if not 0 <= self.lexical_weight <= 1:
            raise ValueError("lexical_weight must be within 0..1")
        if self.merge_neighbor_limit < 0 or self.merge_neighbor_limit > 10:
            raise ValueError("merge_neighbor_limit must be within 0..10")
        if self.max_content_chars < 256 or self.max_content_chars > 100_000:
            raise ValueError("max_content_chars must be within 256..100000")
        if self.snippet_chars < 0 or self.snippet_chars > 50_000:
            raise ValueError("snippet_chars must be within 0..50000")
        if not 0 <= self.low_score_warn <= 1:
            raise ValueError("low_score_warn must be within 0..1")

    def is_allowed_knowledge_base(self, name: str) -> bool:
        if not name or name != Path(name).name or name in {".", ".."}:
            return False
        if not all(char.isalnum() or char in "_-" for char in name):
            return False
        return not self.allowed_knowledge_bases or name in self.allowed_knowledge_bases

    def _explain_disallowed(self, name: str) -> str:
        """Say which variable rejected the name, not just that it was rejected.

        The agent and the retrieval service are configured by different variables
        (`AGENT_RAG_KNOWLEDGE_BASE` vs `RAG_ALLOWED_KNOWLEDGE_BASES`), so a
        mismatch surfaces here. "not allowed: cybersec" leaves the operator
        guessing which of the two is wrong.
        """
        if self.allowed_knowledge_bases:
            allowed = ", ".join(sorted(self.allowed_knowledge_bases))
            return (
                f"{name!r} is not in the allow-list ({allowed}); add it to "
                f"RAG_ALLOWED_KNOWLEDGE_BASES, or point the caller at one of those"
            )
        return (
            f"{name!r} is not a usable knowledge base name (letters, digits, "
            "'_', '-' only)"
        )

    def content_categories(self, knowledge_base: str) -> list[str]:
        """Return indexed content prefixes (first path segment) for a KB."""
        if not self.is_allowed_knowledge_base(knowledge_base):
            return []
        content_root = self.knowledge_base_root / knowledge_base / "content"
        if not content_root.is_dir():
            return []
        return sorted(entry.name for entry in content_root.iterdir() if entry.is_dir())

    @classmethod
    def from_environment(cls, root: Optional[str] = None) -> "RagConfig":
        root_value = root or os.getenv("RAG_KB_ROOT") or os.getenv("KB_ROOT_PATH")
        if not root_value:
            raise ValueError("RAG_KB_ROOT or KB_ROOT_PATH must be configured")
        allowed = frozenset(
            value.strip()
            for value in os.getenv("RAG_ALLOWED_KNOWLEDGE_BASES", "").split(",")
            if value.strip()
        )
        return cls(
            knowledge_base_root=Path(root_value),
            default_knowledge_base=os.getenv("RAG_DEFAULT_KNOWLEDGE_BASE", "cybersec"),
            default_top_k=int(os.getenv("RAG_DEFAULT_TOP_K", "5")),
            default_score_threshold=float(os.getenv("RAG_DEFAULT_SCORE_THRESHOLD", "0.45")),
            max_top_k=int(os.getenv("RAG_MAX_TOP_K", "50")),
            filtered_candidate_limit=int(os.getenv("RAG_FILTERED_CANDIDATE_LIMIT", "1200")),
            candidate_pool=int(os.getenv("RAG_CANDIDATE_POOL", "600")),
            path_recall_limit=int(os.getenv("RAG_PATH_RECALL_LIMIT", "400")),
            path_recall_rows_per_path=int(os.getenv("RAG_PATH_RECALL_ROWS_PER_PATH", "2")),
            identifier_recall_limit=int(os.getenv("RAG_IDENTIFIER_RECALL_LIMIT", "40")),
            lexical_fallback=os.getenv("RAG_LEXICAL_FALLBACK", "1").lower()
            not in {"0", "false", "no"},
            document_cache_limit=int(os.getenv("RAG_DOCUMENT_CACHE_LIMIT", "2048")),
            embedding_failure_ttl=float(os.getenv("RAG_EMBEDDING_FAILURE_TTL", "30")),
            store_cache_limit=int(os.getenv("RAG_STORE_CACHE_LIMIT", "3")),
            max_query_length=int(os.getenv("RAG_MAX_QUERY_LENGTH", "8000")),
            allowed_knowledge_bases=allowed,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL") or "bge-m3",
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            embedding_timeout=float(os.getenv("RAG_EMBEDDING_TIMEOUT", "120")),
            embedding_keep_alive=os.getenv("RAG_EMBEDDING_KEEP_ALIVE", "30m"),
            lexical_weight=float(os.getenv("RAG_LEXICAL_WEIGHT", "0.35")),
            merge_neighbors=os.getenv("RAG_MERGE_NEIGHBORS", "1").lower() not in {"0", "false", "no"},
            merge_neighbor_limit=int(os.getenv("RAG_MERGE_NEIGHBOR_LIMIT", "2")),
            low_info_filter=os.getenv("RAG_LOW_INFO_FILTER", "1").lower() not in {"0", "false", "no"},
            strip_images=os.getenv("RAG_STRIP_IMAGES", "1").lower() not in {"0", "false", "no"},
            strip_provenance=os.getenv("RAG_STRIP_PROVENANCE", "1").lower()
            not in {"0", "false", "no"},
            snippet_chars=int(os.getenv("RAG_SNIPPET_CHARS", "800")),
            max_content_chars=int(os.getenv("RAG_MAX_CONTENT_CHARS", "8000")),
            filter_flag_attachments=os.getenv("RAG_FILTER_FLAG_ATTACHMENTS", "1").lower()
            not in {"0", "false", "no"},
            low_score_warn=float(os.getenv("RAG_LOW_SCORE_WARN", "0.55")),
        )
