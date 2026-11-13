"""Index builder: load data/docs/*.md -> chunk -> embed -> save numpy index.

Usable as a CLI (`uv run helpdesk-rag-ingest`) and from the app's startup hook.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from helpdesk_rag.config import Settings
from helpdesk_rag.integrations import factory
from helpdesk_rag.logging_config import configure_logging
from helpdesk_rag.ports import Embedder, VectorStore
from helpdesk_rag.rag.chunking import load_and_chunk

logger = logging.getLogger(__name__)


def build_index(settings: Settings, embedder: Embedder) -> VectorStore:
    started = time.perf_counter()
    chunks = load_and_chunk(
        settings.docs_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vectors = embedder.embed([c.embedding_text() for c in chunks])
    store = factory.build_store(settings, vectors, chunks, embedding_model=embedder.model_name)
    logger.info(
        "index built docs=%d chunks=%d dim=%d dir=%s in %.1fs",
        len({c.doc_id for c in chunks}),
        len(chunks),
        vectors.shape[1],
        settings.index_dir,
        time.perf_counter() - started,
    )
    return store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the helpdesk RAG vector index.")
    parser.add_argument("--docs-dir", type=Path, help="override DOCS_DIR")
    parser.add_argument("--index-dir", type=Path, help="override INDEX_DIR")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in {"docs_dir": args.docs_dir, "index_dir": args.index_dir}.items() if v}
    settings = Settings(**overrides)
    configure_logging(settings.log_level)

    store = build_index(settings, factory.build_embedder(settings))
    print(f"Indexed {len(store)} chunks into {settings.index_dir}")


if __name__ == "__main__":
    main()
