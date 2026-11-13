"""Offline test doubles for the ports (no API key, no Ollama, no model download)."""

from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np

from helpdesk_rag.domain import LLMReply

DIM = 4096
_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it", "my",
    "of", "on", "or", "the", "to", "what", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors: texts sharing words get similar vectors."""

    model_name = "fake-hash-embedder"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM, dtype=np.float32)
        for token in _tokens(text):
            bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % DIM
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class FakeLLM:
    """Records calls; returns a canned answer (`truncated` flags it), raises `error`, or sleeps `delay` first."""

    def __init__(self, answer: str = "Use the self-service portal. [password-reset#1]") -> None:
        self.answer = answer
        self.error: Exception | None = None
        self.delay: float = 0.0
        self.truncated = False
        self.calls: list[tuple[str, str]] = []
        self.active = 0
        self.max_active = 0

    async def generate(self, system: str, prompt: str) -> LLMReply:
        self.calls.append((system, prompt))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return LLMReply(self.answer, truncated=self.truncated)
        finally:
            self.active -= 1
