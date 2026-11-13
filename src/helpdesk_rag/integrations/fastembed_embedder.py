"""Local CPU embeddings via fastembed (ONNX). Used in both cloud and local mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        # Imported lazily so the default (offline) test suite never loads onnxruntime.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
