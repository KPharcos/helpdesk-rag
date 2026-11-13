"""HTTP-level tests: one per row of the spec's I/O & edge-case matrix, plus health/request-id."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from helpdesk_rag.api.app import create_app
from helpdesk_rag.domain import LLMError
from helpdesk_rag.ingest import build_index
from helpdesk_rag.rag.prompts import NO_CONTEXT_ANSWER


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    return body["error"]


# --- happy path -------------------------------------------------------------------------


def test_ask_happy_path(client: TestClient, llm) -> None:
    response = client.post("/ask", json={"question": "How do I reset my password?"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"answer", "status", "sources", "truncated", "request_id"}
    assert body["truncated"] is False
    assert body["status"] == "answered"
    assert body["answer"] == llm.answer
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["sources"], "expected at least one source"
    first = body["sources"][0]
    assert set(first) == {"doc_id", "title", "chunk_id", "score", "snippet"}
    assert first["doc_id"] == "password-reset"
    assert first["title"] == "Resetting Your Password"
    assert first["chunk_id"].startswith("password-reset#")
    scores = [s["score"] for s in body["sources"]]
    assert scores == sorted(scores, reverse=True)

    # The LLM got the grounded prompt with the retrieved chunk ids.
    assert len(llm.calls) == 1
    system, prompt = llm.calls[0]
    assert "ONLY the knowledge-base excerpts" in system
    assert first["chunk_id"] in prompt
    assert "How do I reset my password?" in prompt


def test_ask_respects_top_k(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "How do I reset my password?", "top_k": 1})
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"question": "VPN certificate expired"}, headers={"X-Request-ID": "abc-123"}
    )
    assert response.headers["X-Request-ID"] == "abc-123"
    assert response.json()["request_id"] == "abc-123"


def test_unsafe_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert response.headers["X-Request-ID"] != "bad id\nwith newline"
    assert len(response.headers["X-Request-ID"]) == 32


def test_request_id_appears_in_logs(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    client.post("/ask", json={"question": "How do I reset my password?"}, headers={"X-Request-ID": "log-me-1"})
    tagged = [r for r in caplog.records if getattr(r, "request_id", None) == "log-me-1"]
    assert any("retrieved" in r.getMessage() for r in tagged)
    assert any("/ask -> 200" in r.getMessage() for r in tagged)


# --- validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   \n\t "},
        {"question": "x" * 1001},
        {},
        {"top_k": 3},
        {"question": 42},
        {"question": None},
        {"question": "x", "top_k": 0},
        {"question": "x", "top_k": 50},
        {"question": "x", "top_k": "many"},
        ["question"],
    ],
    ids=[
        "empty", "whitespace", "too-long", "empty-object", "missing-question", "int-question",
        "null-question", "top_k-zero", "top_k-50", "top_k-string", "array-body",
    ],
)
def test_ask_validation_errors(client: TestClient, llm, payload) -> None:
    error = assert_error(client.post("/ask", json=payload), 422, "validation_error")
    assert "question" in error["message"] or "top_k" in error["message"] or "body" in error["message"]
    assert llm.calls == []


def test_question_of_exactly_1000_chars_is_accepted(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "password " + "x" * 991}).status_code == 200


def test_ask_non_json_body(client: TestClient) -> None:
    response = client.post(
        "/ask", content=b"how do I reset my password", headers={"Content-Type": "application/json"}
    )
    error = assert_error(response, 422, "validation_error")
    assert "JSON" in error["message"]


def test_ask_plain_text_content_type(client: TestClient) -> None:
    response = client.post("/ask", content=b"question=hi", headers={"Content-Type": "text/plain"})
    assert_error(response, 422, "validation_error")


def test_ask_missing_body(client: TestClient) -> None:
    assert_error(client.post("/ask"), 422, "validation_error")


def test_question_of_only_separator_characters_is_rejected(client: TestClient, llm) -> None:
    # Passes the schema's strip (not Unicode whitespace) but Python's split removes it, so the
    # graph's validate step rejects it: the 422 must still come in the uniform error shape.
    error = assert_error(client.post("/ask", json={"question": "\x1c\x1d"}), 422, "validation_error")
    assert "must not be empty" in error["message"]
    assert llm.calls == []


# --- no relevant context ----------------------------------------------------------------


def test_no_relevant_context_skips_llm(client: TestClient, llm) -> None:
    response = client.post("/ask", json={"question": "quantum chromodynamics lattice gauge theory"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_relevant_context"
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["sources"] == []
    assert llm.calls == []


def test_truncated_answer_is_flagged(client: TestClient, llm) -> None:
    llm.truncated = True
    response = client.post("/ask", json={"question": "How do I reset my password?"})
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["truncated"] is True


# --- LLM failures -----------------------------------------------------------------------


def test_llm_error_returns_502(client: TestClient, llm) -> None:
    llm.error = LLMError("provider exploded")
    error = assert_error(
        client.post("/ask", json={"question": "How do I reset my password?"}), 502, "llm_error"
    )
    assert "provider exploded" not in error["message"]  # internal details stay in the logs


def test_llm_timeout_returns_504(make_settings, llm, embedder) -> None:
    settings = make_settings(llm_timeout_seconds=0.05)
    store = build_index(settings, embedder)
    llm.delay = 1.0
    app = create_app(settings, llm=llm, embedder=embedder, store=store)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/ask", json={"question": "How do I reset my password?"})
    assert_error(response, 504, "llm_timeout")


def test_provider_timeout_error_also_maps_to_504(client: TestClient, llm) -> None:
    from helpdesk_rag.domain import LLMTimeoutError

    llm.error = LLMTimeoutError("sdk timeout")
    assert_error(
        client.post("/ask", json={"question": "How do I reset my password?"}), 504, "llm_timeout"
    )


def test_unexpected_exception_returns_uniform_500(client: TestClient, llm) -> None:
    llm.error = RuntimeError("bug")
    assert_error(
        client.post("/ask", json={"question": "How do I reset my password?"}), 500, "internal_error"
    )


def test_unknown_route_uses_uniform_error_shape(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.get("/ask"), 405, "method_not_allowed")


# --- index missing / health -------------------------------------------------------------


def test_index_missing(make_settings, llm, embedder) -> None:
    settings = make_settings(ingest_on_startup=False)
    app = create_app(settings, llm=llm, embedder=embedder)
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        ask = client.post("/ask", json={"question": "How do I reset my password?"})

    assert health.status_code == 200
    assert health.json()["index_ready"] is False
    assert health.json()["indexed_chunks"] == 0
    error = assert_error(ask, 503, "index_not_ready")
    assert "restart" in error["message"]
    assert "Retry-After" not in ask.headers
    assert llm.calls == []


@pytest.mark.parametrize("docs", ["missing", "empty"])
def test_failed_startup_ingest_keeps_app_up(make_settings, llm, embedder, tmp_path, docs) -> None:
    docs_dir = tmp_path / "no-docs"
    if docs == "empty":
        docs_dir.mkdir()
    settings = make_settings(ingest_on_startup=True, docs_dir=docs_dir)
    app = create_app(settings, llm=llm, embedder=embedder)
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        ask = client.post("/ask", json={"question": "How do I reset my password?"})
    assert health.status_code == 200
    assert health.json()["index_ready"] is False
    assert_error(ask, 503, "index_not_ready")


def test_ingest_on_startup_builds_missing_index(make_settings, llm, embedder) -> None:
    settings = make_settings(ingest_on_startup=True)
    app = create_app(settings, llm=llm, embedder=embedder)
    with TestClient(app) as client:
        health = client.get("/health").json()
        ask = client.post("/ask", json={"question": "How do I reset my password?"})
    assert health["index_ready"] is True
    assert health["indexed_chunks"] > 0
    assert ask.status_code == 200
    assert (settings.index_dir / "vectors.npy").is_file()


@pytest.mark.parametrize(
    ("provider", "model_field", "model"),
    [("anthropic", "anthropic_model", "claude-opus-5-5"), ("ollama", "ollama_model", "llama3.1:8b")],
)
def test_health_reports_active_provider(
    make_settings, embedder, store, provider, model_field, model, monkeypatch
) -> None:
    # Real adapters are built from config (no network call happens at construction time).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    settings = make_settings(llm_provider=provider, **{model_field: model})
    app = create_app(settings, embedder=embedder, store=store)
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "index_ready": True,
        "indexed_chunks": len(store),
        "llm_provider": provider,
        "llm_model": model,
        "embedding_model": embedder.model_name,
    }


def test_openapi_has_examples(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    ask_request = schema["components"]["schemas"]["AskRequest"]
    assert ask_request["examples"][0]["question"]
    assert "422" in schema["paths"]["/ask"]["post"]["responses"]


def test_snippet_is_truncated_and_whitespace_collapsed() -> None:
    from helpdesk_rag.api.schemas import SNIPPET_CHARS, AskResponse
    from helpdesk_rag.domain import Answer, Chunk, RetrievedChunk

    text = ("word  \n\t" * 200)[:800]
    chunk = Chunk(chunk_id="d#0", doc_id="d", title="T", section="S", text=text)
    response = AskResponse.from_answer(
        Answer(answer="a", status="answered", sources=[RetrievedChunk(chunk, 0.123456)]),
        request_id="r",
    )
    snippet = response.sources[0].snippet
    assert len(snippet) <= SNIPPET_CHARS
    assert snippet.endswith("…")
    assert "  " not in snippet and "\n" not in snippet and "\t" not in snippet
    assert snippet.startswith("word word")
    assert response.sources[0].score == 0.1235

    short = Chunk(chunk_id="d#1", doc_id="d", title="T", section="S", text="  short\ntext ")
    resp = AskResponse.from_answer(
        Answer(answer="a", status="answered", sources=[RetrievedChunk(short, 0.5)]), request_id="r"
    )
    assert resp.sources[0].snippet == "short text"


def test_access_log_escapes_control_characters_in_the_path(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client.get("/%0Aforged%20line")  # the ASGI path is percent-decoded: a raw newline
    messages = [r.getMessage() for r in caplog.records if r.name == "helpdesk_rag.access"]
    assert messages
    assert all("\n" not in m for m in messages)
    assert any("\\nforged line" in m for m in messages)


# --- startup wiring through Settings ----------------------------------------------------


def test_persisted_index_is_loaded_at_startup(make_settings, llm, embedder) -> None:
    settings = make_settings(embedding_model=embedder.model_name)
    build_index(settings, embedder)  # what `helpdesk-rag-ingest` does
    app = create_app(settings, llm=llm, embedder=embedder)  # no store=, INGEST_ON_STARTUP=false
    with TestClient(app) as client:
        health = client.get("/health").json()
        ask = client.post("/ask", json={"question": "How do I reset my password?"})
    assert health["index_ready"] is True
    assert health["indexed_chunks"] > 0
    assert ask.status_code == 200


def test_persisted_index_of_another_embedding_model_is_not_used(make_settings, llm, embedder) -> None:
    build_index(make_settings(embedding_model=embedder.model_name), embedder)
    app = create_app(make_settings(embedding_model="some-other-model"), llm=llm, embedder=embedder)
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health").json()
        ask = client.post("/ask", json={"question": "How do I reset my password?"})
    assert health["index_ready"] is False
    assert_error(ask, 503, "index_not_ready")


def test_top_k_default_comes_from_settings(make_settings, llm, embedder) -> None:
    app = create_app(make_settings(ingest_on_startup=True, top_k=2), llm=llm, embedder=embedder)
    with TestClient(app) as client:
        body = client.post("/ask", json={"question": "How do I reset my password?"}).json()
    assert len(body["sources"]) == 2


@pytest.mark.parametrize(("limit", "expect_parallel"), [(1, False), (4, True)])
def test_llm_concurrency_limit_comes_from_settings(
    make_settings, llm, embedder, limit, expect_parallel
) -> None:
    llm.delay = 0.3
    settings = make_settings(ingest_on_startup=True, max_concurrent_llm_calls=limit, llm_timeout_seconds=30)
    app = create_app(settings, llm=llm, embedder=embedder)
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(
                pool.map(
                    lambda _: client.post("/ask", json={"question": "How do I reset my password?"}),
                    range(4),
                )
            )
    assert [r.status_code for r in responses] == [200] * 4
    assert (llm.max_active > 1) is expect_parallel
    assert llm.max_active <= limit
