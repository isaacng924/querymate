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


import tempfile  # noqa: E402

from querymate.card_index import CardIndex  # noqa: E402
from querymate.schema_cards import SchemaCard  # noqa: E402

_CARDS = [
    SchemaCard("shop", "customers", "Table: customers\ncolumns id name country", []),
    SchemaCard("shop", "products", "Table: products\ncolumns id price category", []),
    SchemaCard("shop", "orders",
               "Table: orders\ncolumns id customer_id order_date", ["customers"]),
    SchemaCard("other", "customers", "Table: customers\ncolumns id name country", []),
]


def _build_index(path: str) -> CardIndex:
    idx = CardIndex(path, embedder=FakeEmbedder())
    idx.add_cards(_CARDS)
    return idx


def test_index_roundtrip_and_db_filter():
    with tempfile.TemporaryDirectory() as d:
        idx = _build_index(os.path.join(d, "ix.sqlite"))
        hits = idx.query("customers name country", db_id="shop", k=2)
        assert [h.table for h in hits][0] == "customers"
        assert all(h.db_id == "shop" for h in hits)
        idx.close()


def test_index_get_cards_by_table():
    with tempfile.TemporaryDirectory() as d:
        idx = _build_index(os.path.join(d, "ix.sqlite"))
        got = idx.get_cards("shop", ["orders", "customers"])
        assert sorted(c.table for c in got) == ["customers", "orders"]
        idx.close()


from querymate.retriever import Retriever  # noqa: E402


def test_retrieve_expands_fk_neighbors():
    with tempfile.TemporaryDirectory() as d:
        idx = _build_index(os.path.join(d, "ix.sqlite"))
        r = Retriever(idx)
        # 'orders' is the closest card; its FK neighbour 'customers' must ride along.
        cards, schema = r.retrieve("orders by order date and customer", db_id="shop", k=1)
        tables = [c.table for c in cards]
        assert "orders" in tables and "customers" in tables
        assert "Table: orders" in schema and "Table: customers" in schema
        idx.close()


def test_retrieve_no_duplicate_cards():
    with tempfile.TemporaryDirectory() as d:
        idx = _build_index(os.path.join(d, "ix.sqlite"))
        r = Retriever(idx)
        cards, _ = r.retrieve("customers and their orders", db_id="shop", k=3)
        tables = [c.table for c in cards]
        assert len(tables) == len(set(tables))
        idx.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} retriever tests passed")
