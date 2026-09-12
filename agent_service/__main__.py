"""Command line entrypoint: ``python -m agent_service``.

``--check`` reports provider and index reachability without opening a port,
which is the first thing to run on an air-gapped host that is already failing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agent_service.config import AgentConfig
from agent_service.server import create_app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_service",
        description="本地资料库智能问答（离线优先的 Web 对话服务）",
    )
    parser.add_argument("--host", default=None, help="监听地址，默认取 AGENT_HOST（127.0.0.1）")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认取 AGENT_PORT（8801）")
    parser.add_argument("--check", action="store_true", help="只做自检并打印状态，不启动服务")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="打印脱敏后的生效配置（不含 API Key）",
    )
    return parser


async def _run_check(config: AgentConfig) -> int:
    """Report readiness and return an exit code a script can gate on.

    0 = everything ready; 1 = chat works but retrieval does not (degraded);
    2 = no model reachable (the agent cannot answer at all).
    """
    from agent_service.llm import probe
    from agent_service.server import _rag_status

    checks = await asyncio.gather(*(probe(provider) for provider in config.providers))
    healthy = 0
    for provider, (ok, detail) in zip(config.providers, checks):
        healthy += 1 if ok else 0
        marker = "OK  " if ok else "FAIL"
        scope = "离线" if provider.offline else "联网"
        print(f"[{marker}] {provider.id:<8} ({scope}) {provider.base_url} -> {detail}")
        print(f"         模型: {', '.join(provider.models) or '(未配置)'}")

    rag = await _rag_status(config)
    if not rag.get("enabled", True):
        print("[SKIP] RAG 未启用（AGENT_RAG_ENABLED=0）")
        rag_ok = True
    elif rag.get("error"):
        print(f"[FAIL] RAG 不可用：{rag['error']}")
        rag_ok = False
    else:
        print(f"[OK  ] RAG 知识库 {rag.get('knowledge_base')}: {rag.get('status')}")
        rag_ok = True
        # Embeddings are a separate provider from the chat model; a dead one
        # leaves every query on the lexical fallback while the index looks fine.
        embedding = str(rag.get("embedding") or "")
        if embedding.startswith("ready"):
            print(f"[OK  ] RAG embedding: {embedding}")
        elif embedding:
            print(f"[WARN] RAG embedding: {embedding} -> 查询会降级为纯词法检索（分数不是余弦相似度）")

    if healthy == 0:
        print(
            "\n没有可用模型，无法回答。离线环境请先启动 Ollama：ollama serve",
            file=sys.stderr,
        )
        return 2
    if not rag_ok:
        print("\n模型可用，但检索不可用：问答会降级为纯对话。", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = AgentConfig.from_environment()
        if args.host:
            config = _replace(config, host=args.host)
        if args.port:
            config = _replace(config, port=args.port)
    except ValueError as exc:
        # Configuration is operator input. Keep startup diagnostics at the same
        # field-level, traceback-free boundary as the RAG CLI.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(config.as_public_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.check:
        return asyncio.run(_run_check(config))

    import uvicorn

    print(f"本地资料库智能问答: http://{config.host}:{config.port}", file=sys.stderr)
    for provider in config.providers:
        scope = "离线" if provider.offline else "联网"
        print(f"  [{scope}] {provider.label}: {', '.join(provider.models)}", file=sys.stderr)
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")
    return 0


def _replace(config: AgentConfig, **changes) -> AgentConfig:
    from dataclasses import replace

    return replace(config, **changes)


if __name__ == "__main__":
    raise SystemExit(main())
