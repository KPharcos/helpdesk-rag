"""LangGraph question-answering flow.

    START -> validate -> retrieve --(best chunk >= min score?)--> generate   -> END
                                  \\--------------(no)------------> no_context -> END

The threshold gates the question, not individual chunks: if the best chunk passes, all
`top_k` chunks go to the LLM. A chunk that holds the answer often scores a little below a
broader chunk on the same topic, and dropping it would lose the answer.

The state is a per-request TypedDict created by `ainvoke`; nodes only read the injected,
build-once dependencies (embedder, store, LLM, semaphore), so concurrent requests never
share mutable state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from helpdesk_rag.domain import (
    AnswerStatus,
    InvalidQuestionError,
    LLMTimeoutError,
    RetrievedChunk,
)
from helpdesk_rag.ports import Embedder, LLMClient, VectorStore
from helpdesk_rag.rag.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class QAState(TypedDict, total=False):
    question: str
    top_k: int
    retrieved: list[RetrievedChunk]
    answer: str
    status: AnswerStatus


def build_graph(
    *,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    min_score: float,
    llm_timeout_seconds: float,
    llm_semaphore: asyncio.Semaphore,
):
    """Compile the QA graph. Called once at startup; the compiled graph is read-only."""

    async def validate(state: QAState) -> QAState:
        question = " ".join(state.get("question", "").split())
        if not question:
            raise InvalidQuestionError("question must not be empty")
        return {"question": question}

    async def retrieve(state: QAState) -> QAState:
        # fastembed inference is CPU-bound and synchronous: run it off the event loop.
        vector = await asyncio.to_thread(embedder.embed_query, state["question"])
        hits = store.search(vector, state["top_k"])  # sorted by score, best first
        top_score = hits[0].score if hits else float("nan")
        gate_passed = bool(hits) and top_score >= min_score
        logger.info(
            "retrieved hits=%d top_score=%.3f threshold=%.3f gate=%s",
            len(hits),
            top_score,
            min_score,
            "pass" if gate_passed else "fail",
        )
        return {"retrieved": hits if gate_passed else []}

    def route(state: QAState) -> Literal["generate", "no_context"]:
        return "generate" if state.get("retrieved") else "no_context"

    async def generate(state: QAState) -> QAState:
        prompt = build_user_prompt(state["question"], state["retrieved"])

        async def call() -> str:
            async with llm_semaphore:  # caps concurrent provider calls per process
                return await llm.generate(SYSTEM_PROMPT, prompt)

        try:
            # Waiting for a semaphore slot counts toward the timeout.
            answer = await asyncio.wait_for(call(), timeout=llm_timeout_seconds)
        except TimeoutError as exc:
            raise LLMTimeoutError(f"LLM call exceeded {llm_timeout_seconds}s") from exc
        return {"answer": answer, "status": "answered"}

    async def no_context(state: QAState) -> QAState:
        logger.info("no relevant context found; LLM not called")
        return {"answer": NO_CONTEXT_ANSWER, "status": "no_relevant_context", "retrieved": []}

    builder = StateGraph(QAState)
    builder.add_node("validate", validate)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("no_context", no_context)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", "retrieve")
    builder.add_conditional_edges("retrieve", route, ["generate", "no_context"])
    builder.add_edge("generate", END)
    builder.add_edge("no_context", END)
    return builder.compile()
