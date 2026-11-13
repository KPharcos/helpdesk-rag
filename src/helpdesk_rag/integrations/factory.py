"""Builds concrete adapters from Settings. The only place that maps config -> provider class."""

from __future__ import annotations

import logging

import numpy as np

from helpdesk_rag.config import Settings
from helpdesk_rag.domain import Chunk, IndexNotReadyError
from helpdesk_rag.integrations.numpy_store import NumpyVectorStore
from helpdesk_rag.ports import Embedder, LLMClient, VectorStore

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        import anthropic

        from helpdesk_rag.integrations.anthropic_llm import AnthropicLLM

        # Treat an empty value (e.g. `ANTHROPIC_API_KEY=` in .env) as "not set".
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        api_key = api_key or None
        client = anthropic.AsyncAnthropic(
            api_key=api_key,  # None -> SDK resolves ANTHROPIC_API_KEY / profile itself
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        return AnthropicLLM(
            client,
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            effort=settings.anthropic_effort,
        )

    if settings.llm_provider == "ollama":
        import ollama

        from helpdesk_rag.integrations.ollama_llm import OllamaLLM

        client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=settings.llm_timeout_seconds)
        return OllamaLLM(
            client,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def build_embedder(settings: Settings) -> Embedder:
    from helpdesk_rag.integrations.fastembed_embedder import FastEmbedEmbedder

    return FastEmbedEmbedder(settings.embedding_model, cache_dir=settings.embedding_cache_dir)


def build_store(
    settings: Settings, vectors: np.ndarray, chunks: list[Chunk], *, embedding_model: str
) -> VectorStore:
    """Create the vector store from embedded chunks and persist it to INDEX_DIR."""
    store = NumpyVectorStore(vectors, chunks, embedding_model=embedding_model)
    store.save(settings.index_dir)
    return store


def load_store(settings: Settings) -> VectorStore | None:
    """Load the persisted index, or return None if it is missing/unusable (app reports not ready)."""
    try:
        return NumpyVectorStore.load(settings.index_dir, settings.embedding_model)
    except IndexNotReadyError as exc:
        logger.warning("Index not ready: %s", exc)
        return None
