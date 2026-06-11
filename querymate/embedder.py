"""Embedding backends behind one tiny protocol.

``FastEmbedder`` is the real one (fastembed / bge-small-en-v1.5, local ONNX —
no API key). ``FakeEmbedder`` is a deterministic token-hash embedder so tests
and CI never download the ~100MB model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

DIM = 384  # bge-small-en-v1.5 output dimension; FakeEmbedder matches it.

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Lazy-loads the fastembed model on first use (first call downloads it)."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding  # import here: heavy

            self._model = TextEmbedding(self._model_name)
        return [list(map(float, v)) for v in self._model.embed(texts)]


class FakeEmbedder:
    """Deterministic bag-of-hashed-tokens vectors. Shared tokens → similarity,
    which is all the retriever tests need."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * DIM
            for tok in re.findall(r"[a-z0-9_]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out
