"""Plain-assert tests for graph wiring + LLM-free node paths.

    python tests/test_graph.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.card_index import CardIndex            # noqa: E402
from querymate.embedder import FakeEmbedder           # noqa: E402
from querymate.graph import build_graph               # noqa: E402
from querymate.nodes import (                         # noqa: E402
    retrieve_node, route_after_critic, set_retriever,
)
from querymate.retriever import Retriever             # noqa: E402
from querymate.schema_cards import SchemaCard         # noqa: E402


def test_graph_has_phase1_nodes():
    nodes = set(build_graph().get_graph().nodes)
    assert {"retrieve", "plan", "write_sql", "execute", "critic"} <= nodes


def test_route_after_critic():
    assert route_after_critic({"widen_now": True}) == "widen"
    assert route_after_critic({"widen_now": False}) == "rewrite"
    assert route_after_critic({}) == "rewrite"


def _tmp_db(dirpath: str) -> str:
    db = os.path.join(dirpath, "shop.sqlite")
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);")
    con.commit()
    con.close()
    return db


def test_retrieve_node_full_schema_fallback():
    with tempfile.TemporaryDirectory() as d:
        db = _tmp_db(d)
        out = retrieve_node({"question": "q", "db_path": db, "use_retrieval": False})
    assert "CREATE TABLE customers" in out["schema"]
    assert out["card_tables"] == []


def test_retrieve_node_uses_injected_retriever():
    with tempfile.TemporaryDirectory() as d:
        db = _tmp_db(d)
        idx = CardIndex(os.path.join(d, "ix.sqlite"), embedder=FakeEmbedder())
        idx.add_cards([SchemaCard("shop", "customers",
                                  "Table: customers\ncolumns id name", [])])
        set_retriever(Retriever(idx))
        try:
            out = retrieve_node({
                "question": "customers name", "db_path": db,
                "db_id": "shop", "retrieval_k": 3,
            })
        finally:
            set_retriever(None)
        idx.close()
    assert out["card_tables"] == ["customers"]
    assert "Table: customers" in out["schema"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} graph tests passed")
