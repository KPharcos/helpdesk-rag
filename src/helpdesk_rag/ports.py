"""Swap points: the RAG core only depends on these protocols, never on a concrete provider."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from helpdesk_rag.domain import LLMReply, RetrievedChunk


class LLMClient(Protocol):
    async def generate(self, system: str, prompt: str) -> LLMReply:
        """Return the answer and whether it was cut off. Raise LLMError / LLMTimeoutError on failure."""
        ...


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents/passages. Returns an (n, dim) float32 array. CPU-bound, synchronous."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query. Returns a (dim,) float32 array."""
        ...


class VectorStore(Protocol):
    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        """Return up to k chunks ordered by descending similarity."""
        ...

    def __len__(self) -> int: ...
