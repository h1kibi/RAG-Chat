"""Build the small bundled demo index from the template documents.

The repository ships the retrieval *service*, not an index: a real knowledge
base is built from your own corpus with an external index builder. A fresh
clone therefore had nothing to query, which made "clone and run" impossible to
demonstrate. This script produces the tiny `examples/demo-kb` fixture from the
eight curated template documents so that `RAG_KB_ROOT=examples/demo-kb` works
immediately after install.

It is a fixture generator, not an indexing pipeline: the chunking rule below is
deliberately simple and is not meant to replace a real chunker for a large
corpus.

    .venv\\Scripts\\python.exe scripts/build-demo-index.py
    .venv\\Scripts\\python.exe -m rag_service.build_cosine `
        --kb-root examples/demo-kb --knowledge-base cybersec

Requires a local Ollama with the embedding model pulled (``ollama pull bge-m3``).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO_ROOT / "knowledge-base" / "cybersec"
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "demo-kb"
KNOWLEDGE_BASE = "cybersec"
CHUNK_CHARS = 500
"""Target chunk size. Paragraph boundaries are respected where possible."""


def iter_chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split document text into paragraph-aligned chunks of at most ``limit``.

    Simple on purpose: a fixture of eight short documents does not need the
    overlap window and sentence-boundary logic a production chunker uses.
    """
    blocks = [block.strip() for block in text.split("\n\n")]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if not block:
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # A single oversized paragraph is hard-split so no chunk exceeds it.
        while len(block) > limit:
            chunks.append(block[:limit])
            block = block[limit:]
        current = block
    if current:
        chunks.append(current)
    return chunks


def build_documents(template_root: Path) -> list[dict]:
    """Read the template docs and chunk them, keeping the relative source path."""
    content_root = template_root / "content"
    if not content_root.is_dir():
        raise FileNotFoundError(f"template content not found: {content_root}")
    documents: list[dict] = []
    for path in sorted(content_root.rglob("*.md")):
        relative = path.relative_to(content_root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        for chunk in iter_chunks(text):
            documents.append({"source": relative, "text": chunk})
    if not documents:
        raise ValueError(f"no documents found under {content_root}")
    return documents


def write_store(output_root: Path, documents: list[dict], embedding_model: str) -> Path:
    """Embed the chunks and write the LangChain-shaped source store.

    ``build_cosine`` consumes exactly this pair: a faiss index holding one
    vector per row, and a pickle of ``(docstore, index_to_docstore_id)``. The
    vectors are L2-normalized here as well; ``build_cosine`` normalizes again,
    which is idempotent.
    """
    import faiss
    import numpy as np
    from langchain.docstore.document import Document
    from langchain_community.docstore.in_memory import InMemoryDocstore

    from rag_service.config import RagConfig
    from rag_service.embeddings import OllamaEmbeddingClient

    config = RagConfig(knowledge_base_root=output_root, embedding_model=embedding_model)
    client = OllamaEmbeddingClient(
        base_url=config.ollama_base_url,
        model=embedding_model,
        timeout=config.embedding_timeout,
    )

    index_path = output_root / KNOWLEDGE_BASE / "vector_store" / embedding_model.replace(":", "_")
    index_path.mkdir(parents=True, exist_ok=True)

    vectors: list[list[float]] = []
    try:
        for number, document in enumerate(documents, start=1):
            vector = client.embed_query(document["text"])
            vectors.append(vector)
            print(f"  embedded {number}/{len(documents)}: {document['source']}")
    finally:
        client.close()

    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or array.shape[0] != len(documents):
        raise ValueError(f"embedding provider returned an unusable shape: {array.shape}")
    faiss.normalize_L2(array)

    index = faiss.IndexFlatIP(array.shape[1])
    index.add(array)
    faiss.write_index(index, str(index_path / "index.faiss"))

    store: dict[str, Document] = {}
    index_to_id: dict[int, str] = {}
    for row, document in enumerate(documents):
        chunk_id = f"{KNOWLEDGE_BASE}:{row}"
        store[chunk_id] = Document(
            page_content=document["text"],
            metadata={"source": document["source"], "id": chunk_id},
        )
        index_to_id[row] = chunk_id
    with (index_path / "index.pkl").open("wb") as handle:
        pickle.dump(
            (InMemoryDocstore(store), index_to_id), handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    connection = sqlite3.connect(output_root / "info.db")
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS knowledge_base (kb_name TEXT, embed_model TEXT)")
        connection.execute("DELETE FROM knowledge_base WHERE kb_name = ?", (KNOWLEDGE_BASE,))
        connection.execute(
            "INSERT INTO knowledge_base VALUES (?, ?)", (KNOWLEDGE_BASE, embedding_model)
        )
        connection.commit()
    finally:
        connection.close()

    print(f"wrote {len(documents)} rows x {array.shape[1]} dims to {index_path}")
    return index_path


def publish(output_root: Path, embedding_model: str) -> None:
    """Convert the store and publish it in the shippable, source-less form.

    The source pair is deliberately removed afterwards. Everything the service
    reads comes from the converted artifacts, and the manifest's source
    fingerprint records mtime -- which Git cannot preserve, so a committed set
    that still referenced a source index could never validate after a clone.
    ``--prebuilt`` records that decision in the manifest.
    """
    from rag_service.build_cosine import build_cosine_files

    vectors_path = build_cosine_files(
        output_root, KNOWLEDGE_BASE, embedding_model, force=True, prebuilt=True
    )
    # `build_cosine_files` returns the vectors file, not the directory.
    for name in ("index.faiss", "index.pkl"):
        path = vectors_path.parent / name
        if path.is_file():
            path.unlink()
            print(f"  removed source artifact {name} (prebuilt set)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", default="bge-m3")
    args = parser.parse_args()

    documents = build_documents(args.template_root)
    print(f"chunked {len(documents)} rows from {args.template_root}")
    write_store(args.output, documents, args.embedding_model)
    publish(args.output, args.embedding_model)
    print(json.dumps({"rows": len(documents), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
