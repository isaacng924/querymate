"""Build the deterministic demo SQLite DB used by the CLI and the sample eval.

    python scripts/make_demo_db.py

Schema: a tiny store (customers, products, orders, order_items). The data is
chosen so the sample questions have clear answers (e.g. Alice is the unique
"most orders" winner; total revenue = 209.0).
"""

from __future__ import annotations

import os
import sqlite3

DB = os.environ.get("QUERYMATE_DEMO_DB_PATH", "data/demo_store.sqlite")

DDL = """
CREATE TABLE customers (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price    REAL NOT NULL
);
CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL
);
"""

CUSTOMERS = [
    (1, "Alice", "United Kingdom"),
    (2, "Bob", "United States"),
    (3, "Carol", "United Kingdom"),
    (4, "Dan", "Canada"),
]
PRODUCTS = [
    (1, "The Pragmatic Programmer", "Books", 30.0),
    (2, "Clean Code", "Books", 28.0),
    (3, "Wireless Mouse", "Electronics", 20.0),
    (4, "USB-C Cable", "Electronics", 9.0),
    (5, "Coffee Mug", "Home", 12.0),
]
ORDERS = [
    (1, 1, "2026-01-05"),
    (2, 1, "2026-01-09"),
    (3, 1, "2026-02-01"),
    (4, 2, "2026-02-03"),
    (5, 3, "2026-02-10"),
]
ORDER_ITEMS = [
    (1, 1, 1),
    (1, 3, 2),
    (2, 2, 1),
    (3, 5, 3),
    (4, 4, 5),
    (5, 1, 1),
]  # total revenue = 30 + 40 + 28 + 36 + 45 + 30 = 209.0


def main() -> None:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    try:
        con.executescript(DDL)
        con.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
        con.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
        con.executemany("INSERT INTO orders VALUES (?,?,?)", ORDERS)
        con.executemany("INSERT INTO order_items VALUES (?,?,?)", ORDER_ITEMS)
        con.commit()
    finally:
        con.close()
    print(f"built demo DB → {DB}")


if __name__ == "__main__":
    main()
