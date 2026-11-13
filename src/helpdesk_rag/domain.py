"""Provider-neutral domain types and errors shared by every layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AnswerStatus = Literal["answered", "no_relevant_context"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable piece of a source document."""

    chunk_id: str  # stable id, e.g. "vpn-access#2"
    doc_id: str  # source file stem, e.g. "vpn-access"
    title: str  # document title (first "# " heading)
    section: str  # nearest heading above the chunk
    text: str

    def embedding_text(self) -> str:
        """Text that gets embedded: heading context improves retrieval of short chunks."""
        return f"{self.title} - {self.section}\n{self.text}"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


@dataclass(frozen=True, slots=True)
class Answer:
    answer: str
    status: AnswerStatus
    sources: list[RetrievedChunk] = field(default_factory=list)


class HelpdeskError(Exception):
    """Base class for errors the API layer knows how to translate."""


class InvalidQuestionError(HelpdeskError):
    """The question is empty after normalisation."""


class LLMError(HelpdeskError):
    """The LLM provider failed, refused, or returned an unusable response."""


class LLMTimeoutError(LLMError):
    """The LLM call (including waiting for a concurrency slot) exceeded the timeout."""


class IndexNotReadyError(HelpdeskError):
    """No usable vector index is loaded."""
