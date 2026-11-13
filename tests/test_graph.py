from __future__ import annotations

import asyncio

import pytest

from helpdesk_rag.domain import InvalidQuestionError, LLMError, LLMTimeoutError
from helpdesk_rag.rag.graph import build_graph
from helpdesk_rag.service import QAService


def _graph(embedder, store, llm, *, min_score=0.3, timeout=2.0, max_concurrent=8):
    return build_graph(
        embedder=embedder,
        store=store,
        llm=llm,
        min_score=min_score,
        llm_timeout_seconds=timeout,
        llm_semaphore=asyncio.Semaphore(max_concurrent),
    )


async def test_routes_to_generate_when_relevant(embedder, store, llm) -> None:
    state = await _graph(embedder, store, llm).ainvoke(
        {"question": "  How do I   reset my password? ", "top_k": 3}
    )
    assert state["status"] == "answered"
    assert state["answer"] == llm.answer
    assert state["question"] == "How do I reset my password?"  # normalised by validate node
    assert len(state["retrieved"]) == 3  # gate passed -> all top_k chunks are kept
    assert state["retrieved"][0].score >= 0.3
    assert len(llm.calls) == 1


async def test_routes_to_no_context_below_threshold(embedder, store, llm) -> None:
    # Threshold above any achievable score -> nothing is relevant.
    state = await _graph(embedder, store, llm, min_score=1.01).ainvoke(
        {"question": "How do I reset my password?", "top_k": 3}
    )
    assert state["status"] == "no_relevant_context"
    assert state["retrieved"] == []
    assert llm.calls == []


async def test_gate_uses_best_chunk_and_keeps_weaker_ones(embedder, store, llm) -> None:
    # Threshold = best score: the gate passes, and chunks scoring below it still reach the
    # LLM (a chunk holding the answer often scores just under a broader one).
    question = "How do I reset my password?"
    hits = store.search(embedder.embed_query(question), 10)
    assert hits[-1].score < hits[0].score  # the scenario needs weaker chunks
    state = await _graph(embedder, store, llm, min_score=hits[0].score).ainvoke(
        {"question": question, "top_k": 10}
    )
    assert state["status"] == "answered"
    assert [r.chunk.chunk_id for r in state["retrieved"]] == [h.chunk.chunk_id for h in hits]
    prompt = llm.calls[0][1]
    for hit in hits:
        assert hit.chunk.chunk_id in prompt


async def test_empty_question_rejected_by_graph(embedder, store, llm) -> None:
    with pytest.raises(InvalidQuestionError):
        await _graph(embedder, store, llm).ainvoke({"question": "  ", "top_k": 3})


async def test_llm_errors_propagate(embedder, store, llm) -> None:
    llm.error = LLMError("boom")
    with pytest.raises(LLMError):
        await _graph(embedder, store, llm).ainvoke({"question": "reset password", "top_k": 3})


async def test_timeout_includes_waiting_for_a_slot(embedder, store, llm) -> None:
    llm.delay = 0.3
    graph = _graph(embedder, store, llm, timeout=0.45, max_concurrent=1)
    results = await asyncio.gather(
        graph.ainvoke({"question": "reset password", "top_k": 2}),
        graph.ainvoke({"question": "reset password", "top_k": 2}),
        return_exceptions=True,
    )
    # First call finishes in 0.3s; the second waits 0.3s for the slot + 0.3s -> exceeds 0.45s.
    assert sum(isinstance(r, dict) for r in results) == 1
    assert sum(isinstance(r, LLMTimeoutError) for r in results) == 1


async def test_semaphore_caps_concurrent_llm_calls(embedder, store, llm) -> None:
    llm.delay = 0.05
    graph = _graph(embedder, store, llm, max_concurrent=2)
    await asyncio.gather(
        *(graph.ainvoke({"question": "reset password", "top_k": 2}) for _ in range(6))
    )
    assert len(llm.calls) == 6
    assert llm.max_active == 2


async def test_concurrent_requests_do_not_mix_state(embedder, store, llm) -> None:
    service = QAService(_graph(embedder, store, llm), default_top_k=3)
    questions = ("VPN certificate expired", "print badge printer")

    def ids(answer):
        return [s.chunk.chunk_id for s in answer.sources]

    sequential = [ids(await service.ask(q)) for q in questions]
    concurrent = [ids(a) for a in await asyncio.gather(*(service.ask(q) for q in questions))]
    assert concurrent == sequential
    assert sequential[0] != sequential[1]
    assert sequential[0][0].startswith("vpn-access#")
    assert sequential[1][0].startswith("printing#")


async def test_service_uses_default_top_k(embedder, store, llm) -> None:
    answer = await QAService(_graph(embedder, store, llm, min_score=0.0), default_top_k=2).ask(
        "reset password"
    )
    assert len(answer.sources) == 2
