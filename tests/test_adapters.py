"""Provider adapters, tested against stub SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import ollama
import pytest

from helpdesk_rag.config import Settings
from helpdesk_rag.domain import LLMError, LLMTimeoutError
from helpdesk_rag.integrations.anthropic_llm import FALLBACK_BETA, AnthropicLLM
from helpdesk_rag.integrations.factory import build_llm
from helpdesk_rag.integrations.ollama_llm import OllamaLLM
from tests.conftest import REPO_ROOT

# --- Anthropic ---------------------------------------------------------------------------


class _StubBetaMessages:
    def __init__(self, result) -> None:
        self.result = result
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _anthropic(result) -> tuple[AnthropicLLM, _StubBetaMessages]:
    messages = _StubBetaMessages(result)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicLLM(client, model="claude-opus-5-5", max_tokens=16000, effort="low"), messages


def _message(*blocks, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, stop_details=stop_details)


def _text(text: str):
    return SimpleNamespace(type="text", text=text)


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


async def test_anthropic_request_shape_and_text_extraction() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig")
    llm, messages = _anthropic(_message(thinking, _text("Step 1. "), _text("Done [a#1]")))

    assert await llm.generate("SYS", "PROMPT") == "Step 1. Done [a#1]"
    kwargs = messages.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == "SYS"
    assert kwargs["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in kwargs


async def test_anthropic_refusal_is_llm_error() -> None:
    details = SimpleNamespace(type="refusal", category="cyber", explanation="...")
    llm, _ = _anthropic(_message(_text("partial"), stop_reason="refusal", stop_details=details))
    with pytest.raises(LLMError, match="declined"):
        await llm.generate("s", "p")


async def test_anthropic_empty_text_is_llm_error() -> None:
    llm, _ = _anthropic(_message())
    with pytest.raises(LLMError):
        await llm.generate("s", "p")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (anthropic.APITimeoutError(request=_REQUEST), LLMTimeoutError),
        (anthropic.APIConnectionError(request=_REQUEST), LLMError),
        (
            anthropic.RateLimitError(
                "rate limited", response=httpx2.Response(429, request=_REQUEST), body=None
            ),
            LLMError,
        ),
        (
            anthropic.InternalServerError(
                "overloaded", response=httpx2.Response(529, request=_REQUEST), body=None
            ),
            LLMError,
        ),
        (TypeError("Could not resolve authentication method."), LLMError),
        (
            anthropic.APIResponseValidationError(
                response=httpx2.Response(200, request=_REQUEST), body=None
            ),
            LLMError,
        ),
        (anthropic.APIError("unexpected", _REQUEST, body=None), LLMError),
    ],
    ids=[
        "timeout", "connection", "rate-limit", "server-error", "missing-credentials",
        "response-validation", "generic-api-error",
    ],
)
async def test_anthropic_errors_are_mapped(exc, expected) -> None:
    llm, _ = _anthropic(exc)
    with pytest.raises(expected) as info:
        await llm.generate("s", "p")
    assert type(info.value) is expected
    assert info.value.__cause__ is exc


async def test_anthropic_unrelated_type_error_is_not_masked() -> None:
    exc = TypeError("create() got an unexpected keyword argument 'fallbacks'")
    llm, _ = _anthropic(exc)
    with pytest.raises(TypeError) as info:
        await llm.generate("s", "p")
    assert info.value is exc


# --- Ollama ------------------------------------------------------------------------------


class _StubOllama:
    def __init__(self, result) -> None:
        self.result = result
        self.kwargs: dict | None = None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _ollama(result) -> tuple[OllamaLLM, _StubOllama]:
    client = _StubOllama(result)
    return OllamaLLM(client, model="llama3.1:8b", num_ctx=8192, temperature=0.1), client


async def test_ollama_success() -> None:
    llm, client = _ollama(SimpleNamespace(message=SimpleNamespace(content="  Answer [x#1] ")))
    assert await llm.generate("SYS", "PROMPT") == "Answer [x#1]"
    assert client.kwargs["model"] == "llama3.1:8b"
    assert client.kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert client.kwargs["messages"][1] == {"role": "user", "content": "PROMPT"}
    assert client.kwargs["options"]["num_ctx"] == 8192


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout("slow"), LLMTimeoutError),
        (ollama.ResponseError("model 'llama3.1:8b' not found", 404), LLMError),
        (ConnectionError("Failed to connect to Ollama"), LLMError),
    ],
    ids=["timeout", "model-missing", "not-running"],
)
async def test_ollama_errors_are_mapped(exc, expected) -> None:
    llm, _ = _ollama(exc)
    with pytest.raises(expected) as info:
        await llm.generate("s", "p")
    assert type(info.value) is expected


async def test_ollama_empty_answer_is_error() -> None:
    llm, _ = _ollama(SimpleNamespace(message=SimpleNamespace(content="")))
    with pytest.raises(LLMError):
        await llm.generate("s", "p")


# --- factory / config switch -------------------------------------------------------------


@pytest.mark.parametrize(("provider", "cls"), [("anthropic", AnthropicLLM), ("ollama", OllamaLLM)])
def test_provider_selected_by_env(monkeypatch, provider, cls) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == provider
    assert isinstance(build_llm(settings), cls)


def test_empty_api_key_falls_back_to_sdk_env_resolution(monkeypatch) -> None:
    # `ANTHROPIC_API_KEY=` in .env must not be passed as "" (that would shadow the env/profile).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="")
    llm = build_llm(settings)
    assert llm._client.api_key == "env-key"


def test_invalid_provider_rejected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_chunk_overlap_must_be_small(monkeypatch) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=150)


def test_log_level_is_validated_and_normalised() -> None:
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValueError):
        Settings(_env_file=None, log_level="verbose")


def test_build_llm_passes_settings_to_the_anthropic_client() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key="k",
        llm_timeout_seconds=7.0,
        llm_max_retries=5,
    )
    client = build_llm(settings)._client
    assert client.timeout == 7.0
    assert client.max_retries == 5


def test_build_llm_passes_settings_to_the_ollama_client() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_base_url="http://ollama:11434",
        llm_timeout_seconds=7.0,
    )
    http_client = build_llm(settings)._client._client  # ollama.AsyncClient -> httpx.AsyncClient
    assert str(http_client.base_url).rstrip("/") == "http://ollama:11434"
    assert http_client.timeout.read == 7.0


def test_env_example_matches_code_defaults() -> None:
    # The documented values must not drift from the defaults the code ships with.
    example = Settings(_env_file=REPO_ROOT / ".env.example")
    defaults = Settings(_env_file=None)
    assert example.model_dump(exclude={"anthropic_api_key"}) == defaults.model_dump(
        exclude={"anthropic_api_key"}
    )
