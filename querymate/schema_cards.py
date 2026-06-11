"""Per-table schema cards — the retrieval unit for Phase-1 RAG.

A card = one table's DDL plus its FK neighbours, enriched with BIRD's
``database_description`` CSV column descriptions when they sit next to the DB
(``<db_dir>/database_description/<table>.csv``). BIRD CSVs use messy encodings,
so reads are cp1252 with errors ignored.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass, field


@dataclass
class SchemaCard:
    db_id: str
    table: str
    text: str
    fk_neighbors: list[str] = field(default_factory=list)


def _descriptions(db_path: str, table: str) -> list[str]:
    path = os.path.join(os.path.dirname(db_path), "database_description", f"{table}.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="cp1252", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            col = (row.get("original_column_name") or "").strip()
            desc = (row.get("column_description") or "").strip()
            val = (row.get("value_description") or "").strip()
            if col and (desc or val):
                out.append(f"  {col}: {desc}" + (f" (values: {val})" if val else ""))
    return out


def extract_cards(db_path: str, *, db_id: str) -> list[SchemaCard]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        cards = []
        for name, ddl in tables:
            fks = con.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
            neighbors = sorted({fk[2] for fk in fks})  # fk[2] = referenced table
            lines = [f"Table: {name}", ddl.strip()]
            desc = _descriptions(db_path, name)
            if desc:
                lines.append("Column notes:")
                lines.extend(desc)
            if neighbors:
                lines.append(f"Joins to: {', '.join(neighbors)}")
            cards.append(
                SchemaCard(db_id=db_id, table=name, text="\n".join(lines),
                           fk_neighbors=neighbors)
            )
        return cards
    finally:
        con.close()
