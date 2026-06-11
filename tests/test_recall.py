"""Plain-assert tests for gold-table extraction + recall@k.

    python tests/test_recall.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.recall import gold_tables, recall_at_k  # noqa: E402


def test_gold_tables_simple():
    assert gold_tables("SELECT COUNT(*) FROM customers") == {"customers"}


def test_gold_tables_joins_and_aliases():
    sql = ("SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
           "WHERE o.id IN (SELECT order_id FROM order_items)")
    assert gold_tables(sql) == {"customers", "orders", "order_items"}


def test_gold_tables_ignores_ctes():
    sql = ("WITH big AS (SELECT customer_id FROM orders GROUP BY customer_id) "
           "SELECT * FROM big JOIN customers ON customers.id = big.customer_id")
    assert gold_tables(sql) == {"orders", "customers"}


def test_gold_tables_unparseable_is_empty():
    assert gold_tables("NOT SQL AT ALL ((((") == set()


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["a", "b", "c"]) == 1.0
    assert recall_at_k({"a", "b"}, ["a", "c"]) == 0.5
    assert recall_at_k({"a"}, []) == 0.0
    assert recall_at_k(set(), ["a"]) is None  # nothing to recall → excluded


def test_recall_case_insensitive():
    assert recall_at_k({"Customers"}, ["customers"]) == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} recall tests passed")
