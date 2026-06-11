"""Plain-assert tests for schema-card extraction.

    python tests/test_schema_cards.py
"""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.schema_cards import SchemaCard, extract_cards  # noqa: E402

DDL = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL
);
"""


def _mkdb(dirpath: str) -> str:
    db = os.path.join(dirpath, "shop.sqlite")
    con = sqlite3.connect(db)
    con.executescript(DDL)
    con.commit()
    con.close()
    return db


def test_one_card_per_table_with_fk_neighbors():
    with tempfile.TemporaryDirectory() as d:
        cards = extract_cards(_mkdb(d), db_id="shop")
    by_table = {c.table: c for c in cards}
    assert set(by_table) == {"customers", "orders"}
    assert by_table["orders"].fk_neighbors == ["customers"]
    assert by_table["customers"].fk_neighbors == []
    assert by_table["orders"].db_id == "shop"


def test_card_text_contains_ddl():
    with tempfile.TemporaryDirectory() as d:
        cards = extract_cards(_mkdb(d), db_id="shop")
    text = next(c for c in cards if c.table == "orders").text
    assert "Table: orders" in text
    assert "customer_id" in text and "REFERENCES customers" in text


def test_bird_descriptions_merged_when_present():
    with tempfile.TemporaryDirectory() as d:
        db = _mkdb(d)
        desc_dir = os.path.join(d, "database_description")
        os.makedirs(desc_dir)
        with open(os.path.join(desc_dir, "customers.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["original_column_name", "column_name", "column_description",
                        "data_format", "value_description"])
            w.writerow(["country", "country", "customer home country", "text", ""])
        cards = extract_cards(db, db_id="shop")
    text = next(c for c in cards if c.table == "customers").text
    assert "customer home country" in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} schema-card tests passed")
