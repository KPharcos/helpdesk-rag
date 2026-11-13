"""In-memory cosine-similarity vector store, persisted as `.npy` + JSON metadata.

Good enough for a few thousand chunks (brute-force matrix-vector product). Read-only after
load, so it is safe to share between concurrent requests.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from helpdesk_rag.domain import Chunk, IndexNotReadyError, RetrievedChunk

VECTORS_FILE = "vectors.npy"
META_FILE = "chunks.json"
FORMAT_VERSION = 1


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class NumpyVectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], embedding_model: str) -> None:
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("vectors must be (n_chunks, dim) and match the number of chunks")
        self._vectors = _normalize(vectors)
        self._vectors.setflags(write=False)
        self._chunks = list(chunks)
        self.embedding_model = embedding_model

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self._chunks or k <= 0:
            return []
        query = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        if query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} != index dim {self.dim}")
        scores = self._vectors @ query
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    # --- persistence --------------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        """Replace each file atomically (unique temp file + rename).

        The two files are replaced one after the other, so the pair is not swapped as a unit;
        the app only loads the index at startup (restart after re-ingesting).
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "dim": self.dim,
            "chunks": [asdict(c) for c in self._chunks],
        }
        # Unique temp names so concurrent writers (CLI + startup ingest, several workers)
        # never write into each other's temp files.
        suffix = f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp_vec = index_dir / f".{VECTORS_FILE}.{suffix}"
        tmp_meta = index_dir / f".{META_FILE}.{suffix}"
        with tmp_vec.open("wb") as fh:
            np.save(fh, self._vectors)
        tmp_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(tmp_vec, index_dir / VECTORS_FILE)
        os.replace(tmp_meta, index_dir / META_FILE)

    @classmethod
    def exists(cls, index_dir: Path) -> bool:
        return (index_dir / VECTORS_FILE).is_file() and (index_dir / META_FILE).is_file()

    @classmethod
    def load(cls, index_dir: Path, expected_embedding_model: str) -> NumpyVectorStore:
        if not cls.exists(index_dir):
            raise IndexNotReadyError(f"No index found in {index_dir}")
        try:
            meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
            vectors = np.load(index_dir / VECTORS_FILE, allow_pickle=False)
            chunks = [Chunk(**c) for c in meta["chunks"]]
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:  # EOFError: empty .npy
            raise IndexNotReadyError(f"Index in {index_dir} is unreadable: {exc}") from exc
        if meta.get("format_version") != FORMAT_VERSION:
            raise IndexNotReadyError("Index format version mismatch; re-run ingest")
        if meta.get("embedding_model") != expected_embedding_model:
            raise IndexNotReadyError(
                f"Index was built with {meta.get('embedding_model')!r} but "
                f"{expected_embedding_model!r} is configured; re-run ingest"
            )
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] != meta.get("dim"):
            raise IndexNotReadyError(
                f"Index in {index_dir} is inconsistent (vectors {vectors.shape}, "
                f"{len(chunks)} chunks, dim {meta.get('dim')}); re-run ingest"
            )
        try:
            return cls(vectors, chunks, embedding_model=meta["embedding_model"])
        except ValueError as exc:
            raise IndexNotReadyError(f"Index in {index_dir} is invalid: {exc}; re-run ingest") from exc
