from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from helpdesk_rag.api.app import create_app
from helpdesk_rag.config import Settings
from helpdesk_rag.ingest import build_index
from helpdesk_rag.integrations.numpy_store import NumpyVectorStore
from tests.fakes import FakeEmbedder, FakeLLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "data" / "docs"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never let a developer's .env or shell env leak into the offline tests."""
    monkeypatch.chdir(tmp_path)
    for var in ("LLM_PROVIDER", "MIN_RELEVANCE_SCORE", "INDEX_DIR", "DOCS_DIR", "INGEST_ON_STARTUP"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    def _make(**overrides) -> Settings:
        values = {
            "docs_dir": DOCS_DIR,
            "index_dir": tmp_path / "index",
            "min_relevance_score": 0.3,  # calibrated for FakeEmbedder's bag-of-words vectors
            "llm_timeout_seconds": 2.0,
            "log_level": "DEBUG",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return _make


@pytest.fixture
def settings(make_settings) -> Settings:
    return make_settings()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(settings: Settings, embedder: FakeEmbedder) -> NumpyVectorStore:
    return build_index(settings, embedder)


@pytest.fixture
def client(settings, llm, embedder, store) -> Iterator[TestClient]:
    app = create_app(settings, llm=llm, embedder=embedder, store=store)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
