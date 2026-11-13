"""Real-provider smoke tests. Deselected by default; run with:

    uv run pytest -m integration

Each test skips itself when its provider is not available (no API key / Ollama not running).
They use the real fastembed model (downloaded on first use) and the shipped sample docs.
"""

from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from helpdesk_rag.api.app import create_app
from helpdesk_rag.config import Settings
from tests.conftest import DOCS_DIR

pytestmark = pytest.mark.integration

QUESTION = "How do I reset my password if I forgot it?"


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(docs_dir=DOCS_DIR, index_dir=tmp_path / "index", ingest_on_startup=True, **overrides)


def _ask_and_check(settings: Settings) -> dict:
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["index_ready"] is True
        assert health["llm_provider"] == settings.llm_provider
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"].strip()
    assert any(s["doc_id"] == "password-reset" for s in body["sources"])
    return body


def test_claude_end_to_end(tmp_path) -> None:
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        pytest.skip("ANTHROPIC_API_KEY not set")
    body = _ask_and_check(_settings(tmp_path, llm_provider="anthropic", _env_file=None))
    assert "password-reset#" in body["answer"]  # Claude follows the citation format


def test_ollama_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # prove no key is needed
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=2).json()
    except httpx.HTTPError:
        pytest.skip(f"Ollama not reachable at {base_url}")
    if not any(m.get("name") == model or m.get("model") == model for m in tags.get("models", [])):
        pytest.skip(f"Ollama model {model} not pulled (run: ollama pull {model})")

    _ask_and_check(
        _settings(
            tmp_path,
            llm_provider="ollama",
            ollama_base_url=base_url,
            ollama_model=model,
            llm_timeout_seconds=300,
            _env_file=None,
        )
    )


async def test_default_relevance_gate_with_real_embedder(tmp_path) -> None:
    """Real fastembed model + default MIN_RELEVANCE_SCORE: off-topic is gated, on-topic passes.

    No LLM is involved (a fake records calls), so this runs without a key or Ollama.
    """
    import asyncio

    from helpdesk_rag.ingest import build_index
    from helpdesk_rag.integrations.factory import build_embedder
    from helpdesk_rag.rag.graph import build_graph
    from tests.fakes import FakeLLM

    settings = Settings(_env_file=None, docs_dir=DOCS_DIR, index_dir=tmp_path / "index")
    assert settings.min_relevance_score == 0.62  # the shipped default is what is under test
    embedder = build_embedder(settings)
    store = build_index(settings, embedder)
    llm = FakeLLM()
    graph = build_graph(
        embedder=embedder,
        store=store,
        llm=llm,
        min_score=settings.min_relevance_score,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        llm_semaphore=asyncio.Semaphore(1),
    )

    off_topic = await graph.ainvoke({"question": "What is the parental leave policy?", "top_k": 4})
    assert off_topic["status"] == "no_relevant_context"
    assert llm.calls == []

    on_topic = await graph.ainvoke({"question": QUESTION, "top_k": 4})
    assert on_topic["status"] == "answered"
    assert any(r.chunk.doc_id == "password-reset" for r in on_topic["retrieved"])

    # Regression: the liquid-damage chunk scores just under the threshold (0.6065 vs 0.62)
    # while a broader laptop chunk passes it; the gate is per question, so it must still
    # reach the LLM.
    coffee_question = "I spilled coffee on my laptop, what now?"
    coffee = await graph.ainvoke({"question": coffee_question, "top_k": 4})
    assert coffee["status"] == "answered"
    coffee_ids = [r.chunk.chunk_id for r in coffee["retrieved"]]
    assert "laptop-request#4" in coffee_ids
