from __future__ import annotations

import pytest

from helpdesk_rag import ingest
from helpdesk_rag.integrations import factory
from helpdesk_rag.integrations.numpy_store import NumpyVectorStore
from tests.conftest import DOCS_DIR
from tests.fakes import FakeEmbedder


def test_cli_main_writes_index_to_given_dir(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(factory, "build_embedder", lambda settings: FakeEmbedder())
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vpn.md").write_text((DOCS_DIR / "vpn-access.md").read_text(encoding="utf-8"), encoding="utf-8")
    index_dir = tmp_path / "custom-index"

    ingest.main(["--docs-dir", str(docs), "--index-dir", str(index_dir)])

    store = NumpyVectorStore.load(index_dir, expected_embedding_model=FakeEmbedder.model_name)
    assert len(store) > 0
    assert {r.chunk.doc_id for r in store.search(FakeEmbedder().embed_query("vpn"), 50)} == {"vpn"}
    assert f"into {index_dir}" in capsys.readouterr().out


def test_cli_main_fails_without_docs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(factory, "build_embedder", lambda settings: FakeEmbedder())
    with pytest.raises(FileNotFoundError):
        ingest.main(["--docs-dir", str(tmp_path), "--index-dir", str(tmp_path / "idx")])
