"""Offline retrieval evaluation over the live standalone index.

Loads a golden query set (``tests/data/retrieval_queries.jsonl``) and measures
recall@top_k for two ranking modes:

- baseline: pure dense cosine (``lexical_weight=0``);
- fusion: dense + lexical blending at the configured weight.

Usage:

    .venv\\Scripts\\python.exe -m rag_service.evaluate ^
        --queries tests/data/retrieval_queries.jsonl --top-k 5

Threshold calibration against the labelled set (records with ``"empty": true``
must return zero results; prefix-bearing records must recall their source):

    .venv\\Scripts\\python.exe -m rag_service.evaluate ^
        --queries tests/data/retrieval_queries.jsonl --threshold-scan 0.3 0.45 0.55 0.65

Environment: ``RAG_KB_ROOT``, ``RAG_ALLOWED_KNOWLEDGE_BASES``,
``RAG_EMBEDDING_MODEL``, ``RAG_LEXICAL_WEIGHT`` (fusion mode weight),
``RAG_OLLAMA_BASE_URL``. Each query is embedded once per mode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rag_service.config import RagConfig
from rag_service.models import RetrievalRequest


_QUERY_LABELS = ("positive", "negative", "mismatch")


def _normalize_paths(value: Any, field: str, line_number: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"line {line_number}: '{field}' must be a list of strings")
    return [item.strip().replace("\\", "/") for item in value]


def load_queries(path: Path) -> list[dict]:
    """Parse the golden query set.

    Each record carries a label:

    - ``positive``: relevant evidence exists in the corpus and must be recalled.
      It declares where that evidence lives, either as ``prefixes`` (source path
      prefixes, coarse) or ``sources`` (exact ground-truth source paths,
      precise), or both. ``sources`` matters when a topic is spread across
      several corpora: guessing one folder measures the guess, not recall.
    - ``negative``: out-of-corpus query that must return **zero** results;
    - ``mismatch``: query whose only neighbours are version/architecture
      mismatches. Neighbours are expected to be returned; they are informative
      but must not be applied without checking the target environment, so the
      record is reported separately instead of being counted as a leak.

    ``"empty": true`` remains accepted as the legacy spelling of ``negative``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"query set not found: {path}")
    queries: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"line {line_number}: 'query' must be a non-empty string")
            label = record.get("label")
            if label is None:
                label = "negative" if record.get("empty") else "positive"
            if label not in _QUERY_LABELS:
                raise ValueError(
                    f"line {line_number}: 'label' must be one of {', '.join(_QUERY_LABELS)}"
                )
            prefixes = _normalize_paths(record.get("prefixes"), "prefixes", line_number)
            sources = _normalize_paths(record.get("sources"), "sources", line_number)
            if label == "positive":
                if not prefixes and not sources:
                    raise ValueError(
                        f"line {line_number}: 'positive' records need 'prefixes' or 'sources'"
                    )
            elif prefixes or sources:
                raise ValueError(
                    f"line {line_number}: '{label}' records must not declare "
                    "'prefixes' or 'sources'"
                )
            queries.append(
                {
                    "query": query.strip(),
                    "prefixes": prefixes,
                    "sources": sources,
                    "label": label,
                }
            )
    if not queries:
        raise ValueError(f"query set is empty: {path}")
    return queries


def is_relevant(source: str | None, item: dict) -> bool:
    """True when a result source is declared ground truth for the record.

    Matching accepts path prefixes (coarse) and exact source paths (precise).
    """
    value = (source or "").replace("\\", "/")
    if not value:
        return False
    if any(value == prefix.rstrip("/") or value.startswith(prefix) for prefix in item["prefixes"]):
        return True
    return value in item["sources"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall on the live index")
    parser.add_argument("--queries", required=True, help="path to the golden query JSONL")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--baseline-weight", type=float, default=0.0)
    parser.add_argument(
        "--fusion-weight",
        type=float,
        action="append",
        help="fusion weight(s) to compare; repeatable. Defaults to RAG_LEXICAL_WEIGHT or 0.2",
    )
    parser.add_argument(
        "--threshold-scan",
        type=float,
        nargs="*",
        help="scan these score thresholds (fusion mode): report positive top1/MRR vs "
        "how many 'negative' (out-of-corpus) records leak; 'mismatch' records are "
        "reported separately as informative neighbours, not leaks",
    )
    args = parser.parse_args()

    from rag_service.backends.faiss import FaissBackend
    from rag_service.service import RagService

    config = RagConfig.from_environment()
    # The comparison baseline must be the configured default, not a second
    # hardcoded constant: when the two drift, the harness silently reports
    # numbers for a weight the service does not actually use.
    fusion_weights = [
        config.lexical_weight if weight is None else weight
        for weight in (args.fusion_weight or [None])
    ]

    service = RagService(config, FaissBackend(config))
    backend_status = {}
    try:
        backend_status = service.backend.status()
    except Exception:
        backend_status = {}
    print(
        "environment: python={} faiss={} dense_path={}".format(
            sys.executable,
            "yes" if backend_status.get("faiss_available") else "NO",
            backend_status.get("dense_path", "unknown"),
        )
    )
    if not backend_status.get("faiss_available"):
        # Without faiss the scan is a numpy dequantization loop, measured ~20x
        # slower (2.7-3.7 s vs 0.15 s per query). Tuning or timing under that
        # interpreter measures the fallback, not the service.
        print(
            "WARNING: faiss is not importable in this interpreter, so the numpy "
            "fallback is being measured (roughly 20x slower scans). Run under the "
            "repository virtualenv: .venv/Scripts/python.exe",
            file=sys.stderr,
        )
    if backend_status.get("dense_path") == "numpy" and backend_status.get("faiss_available"):
        print(
            "WARNING: dense_path=numpy although faiss is available — the sq8 artifact "
            "is missing; rebuild with `python -m rag_service.build_cosine`",
            file=sys.stderr,
        )
    queries = load_queries(Path(args.queries))
    positives = [item for item in queries if item["label"] == "positive"]
    negatives = [item for item in queries if item["label"] == "negative"]
    mismatches = [item for item in queries if item["label"] == "mismatch"]
    if not positives:
        raise ValueError("query set must contain at least one positive record")

    def run_mode(weight: float, threshold: float = 0.0) -> tuple[float, float, list[str]]:
        hit_count = 0
        reciprocal_rank_sum = 0.0
        top1_count = 0
        details: list[str] = []
        for item in positives:
            started = time.perf_counter()
            response = service.search(
                RetrievalRequest(
                    query=item["query"],
                    knowledge_base=config.default_knowledge_base,
                    top_k=args.top_k,
                    score_threshold=threshold,
                    lexical_weight=weight,
                    merge_neighbors=False,
                )
            )
            elapsed = time.perf_counter() - started
            hit_rank: int | None = None
            for rank, result in enumerate(response.results, start=1):
                if is_relevant(result.source, item):
                    hit_rank = rank
                    break
            if hit_rank is not None:
                hit_count += 1
                reciprocal_rank_sum += 1.0 / hit_rank
                if hit_rank == 1:
                    top1_count += 1
            details.append(
                f"  [{item['query'][:46]:<46}] top1={'ok ' if hit_rank == 1 else 'miss'}"
                f" best_rank={hit_rank if hit_rank else '-'}"
                f" top={response.results[0].source if response.results else '-'} ({elapsed:.1f}s)"
            )
        count = len(positives)
        return (top1_count / count, reciprocal_rank_sum / count, details)

    def scan_thresholds(weight: float, thresholds: list[float]) -> None:
        print(f"\n== threshold scan (fusion weight={weight}, top_k={args.top_k}) ==")
        print("    threshold | pos_top1 | pos_mrr | neg_returned | 0-result negatives")
        for threshold in thresholds:
            pos_top1, pos_mrr, _ = run_mode(weight, threshold)
            neg_returned = 0
            neg_zero = 0
            for item in negatives:
                response = service.search(
                    RetrievalRequest(
                        query=item["query"],
                        knowledge_base=config.default_knowledge_base,
                        top_k=args.top_k,
                        score_threshold=threshold,
                        lexical_weight=weight,
                        merge_neighbors=False,
                    )
                )
                if response.results:
                    neg_returned += 1
                else:
                    neg_zero += 1
            count = len(negatives)
            print(
                f"    {threshold:>9} | {pos_top1:>7.2f} | {pos_mrr:>7.3f} | "
                f"{neg_returned:>10}/{count:<1} | {neg_zero}/{count}"
            )

    print(f"\n== dense (weight={args.baseline_weight}) recall@{args.top_k} ==")
    baseline_top1, baseline_mrr, details = run_mode(args.baseline_weight)
    print(f"top1={baseline_top1:.2f} mrr={baseline_mrr:.3f}")
    for line in details:
        print(line)

    for weight in fusion_weights:
        if weight < 0 or weight > 1:
            raise ValueError(f"fusion weight must be within 0..1")
        print(f"\n== fusion (weight={weight}) recall@{args.top_k} ==")
        top1, mrr, details = run_mode(weight)
        print(f"top1={top1:.2f} mrr={mrr:.3f}")
        for line in details:
            print(line)

    if negatives and not args.threshold_scan:
        for item in negatives:
            response = service.search(
                RetrievalRequest(
                    query=item["query"],
                    knowledge_base=config.default_knowledge_base,
                    top_k=args.top_k,
                    score_threshold=config.default_score_threshold,
                    lexical_weight=fusion_weights[0],
                    merge_neighbors=False,
                )
            )
            marker = "blocked" if not response.results else "leaked"
            print(
                f"[negative:{marker}] {item['query'][:60]:<60} "
                f"top={response.results[0].source if response.results else '-'}"
                f" score={response.results[0].score if response.results else '-'}"
            )

    if mismatches:
        print("\n== mismatch probes (informative neighbours, not leak failures) ==")
        for item in mismatches:
            response = service.search(
                RetrievalRequest(
                    query=item["query"],
                    knowledge_base=config.default_knowledge_base,
                    top_k=args.top_k,
                    score_threshold=config.default_score_threshold,
                    lexical_weight=fusion_weights[0],
                    merge_neighbors=False,
                )
            )
            marker = "neighbour" if response.results else "no-neighbour"
            print(
                f"[mismatch:{marker}] {item['query'][:56]:<56} "
                f"top={response.results[0].source if response.results else '-'}"
                f" score={response.results[0].score if response.results else '-'}"
                f" — verify version/arch before use"
            )

    if args.threshold_scan:
        scan_thresholds(fusion_weights[0], args.threshold_scan)


if __name__ == "__main__":
    main()
