"""Report the in-corpus / out-of-corpus score band and the gate trade-off.

Run this before changing `RAG_DEFAULT_SCORE_THRESHOLD` or `RAG_LEXICAL_WEIGHT`.
It measures both fused components separately, so the decision rests on data:

    .venv/Scripts/python.exe scripts/rag_threshold_band.py

Findings for the current cybersec corpus (bge-m3, weight 0.35), over 16 positive
and 12 off-corpus queries: the two fused distributions overlap (in-corpus
0.487-0.831, off-corpus 0.345-0.547), so no threshold separates them cleanly.
Gating on the dense component instead is strictly worse at the same operating
point (0.45 keeps 16/16 positives but leaks 11/12 off-corpus, against 7/12 for
the fused gate). Raising the fused gate to 0.50 or 0.55 keeps leaking under
control but drops positives -- 0.50 loses the `CVE-2021-3490` case whose top1 is
0.4866 -- so 0.45 is kept as the coarser filter and `RAG_LOW_SCORE_WARN` carries
the "this result is weak" signal instead. The gate moves with `lexical_weight`,
which is why the service warns when that knob is raised.
"""
import json
import os
import statistics
import sys
from pathlib import Path

DEFAULT_GOLDEN = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "retrieval_queries.jsonl"
)

# Off-corpus probes: unrelated domains, mixed Chinese/English, deliberately
# plausible-looking rather than gibberish, which is the hard case.
OOD = [
    "Linear algebra eigenvalue decomposition tutorial",
    "how to bake sourdough bread at home",
    "北京天气预报明天有雨吗",
    "quarterly financial report revenue growth",
    "how to train a puppy not to bite",
    "Excel 数据透视表用法",
    "machine learning overfitting regularization",
    "best coffee beans for espresso",
    "photosynthesis light dependent reactions",
    "如何挑选适合自己的跑鞋",
    "求职简历 面试技巧",
]


def main() -> int:
    from rag_service.backends.faiss import FaissBackend
    from rag_service.config import RagConfig
    from rag_service.models import RetrievalRequest

    golden = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLDEN
    config = RagConfig.from_environment()
    backend = FaissBackend(config)

    records = [
        json.loads(line)
        for line in golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    positives = [r["query"] for r in records if r.get("label", "positive") == "positive" and not r.get("empty")]
    negatives = [r["query"] for r in records if r.get("empty") or r.get("label") == "negative"]

    def components(query: str):
        results = backend.search(
            RetrievalRequest(
                query=query, knowledge_base=config.default_knowledge_base, top_k=1, score_threshold=0.0
            )
        )
        if not results:
            return None
        meta = results[0].metadata or {}
        if meta.get("dense_score") is None:
            return None
        return (meta["dense_score"], meta["lexical_score"], results[0].score)

    def band(label: str, queries: list[str]):
        rows = [row for row in (components(q) for q in queries) if row]
        if not rows:
            print(f"{label:14} no data")
            return []
        dense = [r[0] for r in rows]
        fused = [r[2] for r in rows]
        print(
            f"{label:14} n={len(rows):2}  dense[{min(dense):.3f},{max(dense):.3f}] "
            f"med={statistics.median(dense):.3f}   fused[{min(fused):.3f},{max(fused):.3f}] "
            f"med={statistics.median(fused):.3f}"
        )
        return rows

    print(f"index: {config.embedding_model}  threshold={config.default_score_threshold}  "
          f"lexical_weight={config.lexical_weight}")
    print()
    inside = band("in-corpus", positives)
    outside = band("out-of-corpus", OOD + negatives)

    if inside and outside:
        print()
        for label, index in (("dense", 0), ("fused", 2)):
            for gate in (0.45, 0.50, 0.55, 0.62):
                kept = sum(1 for row in inside if row[index] >= gate)
                leaked = sum(1 for row in outside if row[index] >= gate)
                print(
                    f"  gate on {label:5} @ {gate:.2f}: positives kept {kept}/{len(inside)}, "
                    f"off-corpus leaked {leaked}/{len(outside)}"
                )

    print()
    print("gate moves with lexical_weight (threshold on the fused score):")
    for weight in (0.0, 0.2, 0.35, 0.5, 0.6, 1.0):
        if weight >= 1.0:
            print(f"  w={weight:.2f}: nothing can pass")
            continue
        need_no_lex = config.default_score_threshold / (1.0 - weight)
        need_full_lex = (config.default_score_threshold - weight) / (1.0 - weight)
        print(
            f"  w={weight:.2f}: needs dense >= {need_no_lex:6.3f} with no lexical hits, "
            f"{need_full_lex:6.3f} with full lexical hits"
        )
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
