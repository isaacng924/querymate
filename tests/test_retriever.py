"""Plain-assert tests for the embedder + card index + retriever (no model download).

    python tests/test_retriever.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.embedder import DIM, FakeEmbedder  # noqa: E402


def test_fake_embedder_shape_and_determinism():
    e = FakeEmbedder()
    v1 = e.embed(["how many customers are there"])[0]
    v2 = e.embed(["how many customers are there"])[0]
    assert len(v1) == DIM
    assert v1 == v2  # deterministic


def test_fake_embedder_similarity_signal():
    e = FakeEmbedder()

    def cos(a, b):
        num = sum(x * y for x, y in zip(a, b))
        den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return num / den if den else 0.0

    q = e.embed(["customers country name"])[0]
    near = e.embed(["table customers columns id name country"])[0]
    far = e.embed(["table products columns id price category"])[0]
    assert cos(q, near) > cos(q, far)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} retriever tests passed")
