"""sqlite-vec store for schema cards.

Two tables sharing rowids: ``vec_cards`` (vec0 virtual table, ``db_id`` as a
partition key so KNN filters per database) and ``cards`` (metadata). Zero-infra:
the index is one SQLite file next to the data.
"""

from __future__ import annotations

import json
import sqlite3
import struct

import sqlite_vec

from .embedder import DIM, Embedder
from .schema_cards import SchemaCard


def _f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class CardIndex:
    def __init__(self, path: str, *, embedder: Embedder) -> None:
        self.embedder = embedder
        self.con = sqlite3.connect(path)
        self.con.enable_load_extension(True)
        sqlite_vec.load(self.con)
        self.con.enable_load_extension(False)
        self.con.executescript(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_cards USING vec0(
                db_id TEXT partition key,
                embedding FLOAT[{DIM}]
            );
            CREATE TABLE IF NOT EXISTS cards (
                rowid INTEGER PRIMARY KEY,
                db_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                card_text TEXT NOT NULL,
                fk_neighbors TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS cards_db ON cards(db_id, table_name);
            """
        )

    def add_cards(self, cards: list[SchemaCard]) -> None:
        vecs = self.embedder.embed([c.text for c in cards])
        cur = self.con.cursor()
        try:
            for card, vec in zip(cards, vecs):
                cur.execute(
                    "INSERT INTO cards(db_id, table_name, card_text, fk_neighbors) "
                    "VALUES (?, ?, ?, ?)",
                    (card.db_id, card.table, card.text, json.dumps(card.fk_neighbors)),
                )
                cur.execute(
                    "INSERT INTO vec_cards(rowid, db_id, embedding) VALUES (?, ?, ?)",
                    (cur.lastrowid, card.db_id, _f32(vec)),
                )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def _row_to_card(self, row) -> SchemaCard:
        db_id, table, text, fks = row
        return SchemaCard(db_id, table, text, json.loads(fks) if fks else [])

    def query(self, question: str, *, db_id: str, k: int) -> list[SchemaCard]:
        qvec = self.embedder.embed([question])[0]
        rows = self.con.execute(
            "SELECT c.db_id, c.table_name, c.card_text, c.fk_neighbors "
            "FROM vec_cards v JOIN cards c ON c.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND v.k = ? AND v.db_id = ? "
            "ORDER BY v.distance",
            (_f32(qvec), k, db_id),
        ).fetchall()
        return [self._row_to_card(r) for r in rows]

    def get_cards(self, db_id: str, tables: list[str]) -> list[SchemaCard]:
        if not tables:
            return []
        ph = ",".join("?" * len(tables))
        rows = self.con.execute(
            "SELECT db_id, table_name, card_text, fk_neighbors FROM cards "
            f"WHERE db_id = ? AND table_name IN ({ph})",
            (db_id, *tables),
        ).fetchall()
        return [self._row_to_card(r) for r in rows]

    def has_db(self, db_id: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM cards WHERE db_id = ? LIMIT 1", (db_id,)
        ).fetchone() is not None

    def close(self) -> None:
        self.con.close()
