"""Build the schema-card index.

    python scripts/ingest_schemas.py --demo
    python scripts/ingest_schemas.py --db-root data/bird/dev_databases
    python scripts/ingest_schemas.py --db data/demo_store.sqlite --db-id demo_store

BIRD layout (``<root>/<db_id>/<db_id>.sqlite``) and flat layouts
(``<root>/<name>.sqlite``) are both supported. Re-ingesting a db_id replaces
its cards. First run downloads the fastembed model (~100MB, one-time).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.schema_cards import extract_cards
from querymate.settings import settings


def discover(db_root: str) -> list[tuple[str, str]]:
    """Yield (db_id, db_path) for BIRD-style and flat layouts."""
    found = []
    for p in sorted(glob.glob(os.path.join(db_root, "*", "*.sqlite"))):
        db_id = os.path.basename(os.path.dirname(p))
        if os.path.splitext(os.path.basename(p))[0] == db_id:
            found.append((db_id, p))
    for p in sorted(glob.glob(os.path.join(db_root, "*.sqlite"))):
        found.append((os.path.splitext(os.path.basename(p))[0], p))
    return found


def ingest(pairs: list[tuple[str, str]], index_path: str) -> None:
    idx = CardIndex(index_path, embedder=FastEmbedder())
    try:
        for i, (db_id, db_path) in enumerate(pairs, 1):
            idx.con.execute("DELETE FROM vec_cards WHERE rowid IN "
                            "(SELECT rowid FROM cards WHERE db_id = ?)", (db_id,))
            idx.con.execute("DELETE FROM cards WHERE db_id = ?", (db_id,))
            cards = extract_cards(db_path, db_id=db_id)
            idx.add_cards(cards)
            print(f"[{i}/{len(pairs)}] {db_id}: {len(cards)} cards")
    finally:
        idx.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the schema-card index")
    ap.add_argument("--db-root", help="directory of SQLite DBs (BIRD or flat layout)")
    ap.add_argument("--db", help="one SQLite file")
    ap.add_argument("--db-id", help="db_id for --db (default: filename stem)")
    ap.add_argument("--demo", action="store_true", help="ingest the demo store DB")
    ap.add_argument("--index", default=settings.schema_index_path)
    args = ap.parse_args()

    if args.demo:
        pairs = [("demo_store", settings.demo_db_path)]
    elif args.db:
        pairs = [(args.db_id or os.path.splitext(os.path.basename(args.db))[0], args.db)]
    elif args.db_root:
        pairs = discover(args.db_root)
    else:
        ap.error("one of --demo / --db / --db-root is required")
    if not pairs:
        sys.exit(f"no .sqlite files found under '{args.db_root}'")

    os.makedirs(os.path.dirname(args.index) or ".", exist_ok=True)
    ingest(pairs, args.index)
    print(f"index → {args.index}")


if __name__ == "__main__":
    main()
