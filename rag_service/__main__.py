"""Run the standalone RAG service, or query it once from the command line.

Serve (default, no subcommand):

    python -m rag_service --host 127.0.0.1 --port 8791

One-shot search against the local index (no HTTP server, no persistent process):

    python -m rag_service search "glibc 2.31 tcache double free safe-linking" --top-k 5
    python -m rag_service search "" --filters category=15_butian --limit 20

Browse mode (empty query) prints the same evidence block as the MCP tool,
including `next_cursor`, so a listing can be paged from a shell or from an
agent that has no MCP transport.

Environment variables:
    RAG_KB_ROOT                  knowledge base root (required)
    RAG_ALLOWED_KNOWLEDGE_BASES  comma-separated allow-list (optional)
    RAG_DEFAULT_KNOWLEDGE_BASE   default KB name (default: cybersec)
    RAG_EMBEDDING_MODEL          embedding model name (default: bge-m3)
    RAG_OLLAMA_BASE_URL          Ollama HTTP base (default: http://127.0.0.1:11434)
    RAG_API_TOKEN                optional Bearer token for the HTTP API
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_FILTER_INT_KEYS = frozenset({"year"})
_FILTER_LIST_KEYS = frozenset({"exclude_source_prefix"})
_FILTER_KEYS = frozenset(
    {"category", "source_prefix", "source", "chunk_id", "year", "exclude_source_prefix"}
)


def _parse_filter(raw: str) -> tuple[str, object]:
    """Parse one ``key=value`` filter argument into a normalized pair."""
    key, separator, value = raw.partition("=")
    key = key.strip()
    value = value.strip()
    if not separator or not key or not value:
        raise argparse.ArgumentTypeError(
            f"filter must look like key=value, e.g. category=15_butian (got {raw!r})"
        )
    if key not in _FILTER_KEYS:
        raise argparse.ArgumentTypeError(
            f"unsupported filter {key!r}; choose from {', '.join(sorted(_FILTER_KEYS))}"
        )
    if key in _FILTER_INT_KEYS:
        try:
            return key, int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"filter {key!r} must be an integer") from exc
    if key in _FILTER_LIST_KEYS:
        return key, [item.strip() for item in value.split(",") if item.strip()]
    return key, value


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=os.environ.get("RAG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RAG_PORT", "8791")))
    parser.add_argument("--kb-root", default=None, help="knowledge base root directory")


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="query text; empty browses the corpus without embedding",
    )
    parser.add_argument("--knowledge-base", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="browse listing page size")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--snippet-chars", type=int, default=None)
    parser.add_argument(
        "--cursor", default="", help="next_cursor returned by a previous listing page"
    )
    parser.add_argument(
        "--filters",
        action="append",
        default=None,
        type=_parse_filter,
        metavar="KEY=VALUE",
        help=f"read-only filter; repeatable. Keys: {', '.join(sorted(_FILTER_KEYS))}",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the raw response JSON instead of evidence text"
    )
    parser.add_argument("--kb-root", default=None, help="knowledge base root directory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_service",
        description="Standalone local RAG retrieval service and one-shot query CLI",
    )
    _add_serve_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    _add_search_arguments(subparsers.add_parser("search", help="run one query and print evidence"))
    return parser


def build_request(args: argparse.Namespace):
    """Translate parsed CLI arguments into a validated ``RetrievalRequest``."""
    from rag_service.models import RetrievalRequest

    return RetrievalRequest(
        query=args.query,
        knowledge_base=args.knowledge_base,
        top_k=args.top_k,
        limit=args.limit,
        score_threshold=args.score_threshold,
        filters=dict(args.filters or []),
        snippet_chars=args.snippet_chars,
        cursor=args.cursor or None,
    )


def run_search(args: argparse.Namespace) -> int:
    from rag_service.backends.faiss import FaissBackend
    from rag_service.config import RagConfig
    from rag_service.service import RagService

    if args.kb_root:
        os.environ["RAG_KB_ROOT"] = args.kb_root
    config = RagConfig.from_environment()
    backend = FaissBackend(config)
    try:
        response = RagService(config, backend).search(build_request(args))
    finally:
        backend.close()
    if args.json:
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(response.as_tool_text())
    # Warnings are already embedded in the evidence block for a no-match result;
    # repeating them on stderr would duplicate the same paragraph.
    if response.results:
        for warning in response.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


def run_serve(args: argparse.Namespace) -> int:
    if args.kb_root:
        os.environ["RAG_KB_ROOT"] = args.kb_root

    import uvicorn

    from rag_service.tool_server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "search":
        return run_search(args)
    return run_serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
