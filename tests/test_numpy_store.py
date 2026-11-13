from __future__ import annotations

import json

import numpy as np
import pytest

from helpdesk_rag.domain import Chunk, IndexNotReadyError
from helpdesk_rag.integrations.numpy_store import META_FILE, NumpyVectorStore


def _chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"doc#{i}", doc_id="doc", title="Doc", section="S", text=f"text {i}")


def _store() -> NumpyVectorStore:
    vectors = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    return NumpyVectorStore(vectors, [_chunk(i) for i in range(3)], embedding_model="m")


def test_search_orders_by_cosine() -> None:
    hits = _store().search(np.array([1.0, 0.1, 0.0]), k=3)
    assert [h.chunk.chunk_id for h in hits] == ["doc#0", "doc#2", "doc#1"]
    assert hits[0].score == pytest.approx(0.995, abs=1e-3)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_k_larger_than_index() -> None:
    assert len(_store().search(np.array([0.0, 1.0, 0.0]), k=10)) == 3


def test_search_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError):
        _store().search(np.array([1.0, 0.0]), k=1)


def test_round_trip(tmp_path) -> None:
    store = _store()
    store.save(tmp_path)
    loaded = NumpyVectorStore.load(tmp_path, expected_embedding_model="m")

    assert len(loaded) == 3
    query = np.array([0.2, 1.0, 0.0])
    assert [(h.chunk, round(h.score, 6)) for h in loaded.search(query, 3)] == [
        (h.chunk, round(h.score, 6)) for h in store.search(query, 3)
    ]
    assert not list(tmp_path.glob(".*.tmp"))  # atomic write leaves no temp files
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chunks.json", "vectors.npy"]


def test_load_missing_index(tmp_path) -> None:
    with pytest.raises(IndexNotReadyError):
        NumpyVectorStore.load(tmp_path / "nothing", expected_embedding_model="m")


def test_load_rejects_other_embedding_model(tmp_path) -> None:
    _store().save(tmp_path)
    with pytest.raises(IndexNotReadyError, match="re-run ingest"):
        NumpyVectorStore.load(tmp_path, expected_embedding_model="other-model")


def test_load_rejects_corrupt_metadata(tmp_path) -> None:
    _store().save(tmp_path)
    (tmp_path / META_FILE).write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(IndexNotReadyError):
        NumpyVectorStore.load(tmp_path, expected_embedding_model="m")


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((2, 3), dtype=np.float32),  # fewer vectors than chunks
        np.ones((3, 4), dtype=np.float32),  # dim differs from metadata
        np.ones(9, dtype=np.float32),  # not 2-D
    ],
    ids=["count-mismatch", "dim-mismatch", "not-2d"],
)
def test_load_rejects_mismatched_index(tmp_path, vectors) -> None:
    _store().save(tmp_path)
    np.save(tmp_path / "vectors.npy", vectors)
    with pytest.raises(IndexNotReadyError, match="re-run ingest"):
        NumpyVectorStore.load(tmp_path, expected_embedding_model="m")


def test_store_is_read_only() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store._vectors[0, 0] = 5.0


def test_load_rejects_empty_vectors_file(tmp_path) -> None:
    _store().save(tmp_path)
    (tmp_path / "vectors.npy").write_bytes(b"")  # np.load raises EOFError on an empty file
    with pytest.raises(IndexNotReadyError, match="unreadable"):
        NumpyVectorStore.load(tmp_path, expected_embedding_model="m")
