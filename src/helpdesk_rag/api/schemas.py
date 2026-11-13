"""HTTP request/response models (with OpenAPI examples)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from helpdesk_rag.domain import MAX_TOP_K, QUESTION_MAX_CHARS, Answer

SNIPPET_CHARS = 240

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=QUESTION_MAX_CHARS),
]


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "How do I reset my password?"},
                {"question": "The VPN says my certificate expired. What should I do?", "top_k": 3},
            ]
        }
    )

    question: Question = Field(description="The employee's question in natural language.")
    top_k: int | None = Field(
        default=None, ge=1, le=MAX_TOP_K, description="How many chunks to retrieve (default from TOP_K)."
    )


class Source(BaseModel):
    doc_id: str = Field(examples=["password-reset"])
    title: str = Field(examples=["Resetting Your Password"])
    chunk_id: str = Field(examples=["password-reset#2"])
    score: float = Field(description="Cosine similarity of the chunk to the question.", examples=[0.82])
    snippet: str = Field(description="Beginning of the chunk text.")


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "1. Open https://password.brightfield.example ... [password-reset#2]",
                    "status": "answered",
                    "sources": [
                        {
                            "doc_id": "password-reset",
                            "title": "Resetting Your Password",
                            "chunk_id": "password-reset#2",
                            "score": 0.76,
                            "snippet": "Use the self-service portal ...",
                        }
                    ],
                    "truncated": False,
                    "request_id": "3f2b9c0e7a7d4c9e9b8f1f7c2d1a0b9e",
                }
            ]
        }
    )

    answer: str
    status: Literal["answered", "no_relevant_context"]
    sources: list[Source]
    truncated: bool = Field(
        default=False,
        description="True when the model stopped at its output limit, so the answer may be cut off.",
    )
    request_id: str

    @classmethod
    def from_answer(cls, answer: Answer, request_id: str) -> AskResponse:
        return cls(
            answer=answer.answer,
            status=answer.status,
            sources=[
                Source(
                    doc_id=r.chunk.doc_id,
                    title=r.chunk.title,
                    chunk_id=r.chunk.chunk_id,
                    score=round(r.score, 4),
                    snippet=_snippet(r.chunk.text),
                )
                for r in answer.sources
            ],
            truncated=answer.truncated,
            request_id=request_id,
        )


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rstrip() + "…"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    index_ready: bool
    indexed_chunks: int
    llm_provider: str
    llm_model: str
    embedding_model: str


class ErrorBody(BaseModel):
    code: str = Field(examples=["validation_error"])
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
