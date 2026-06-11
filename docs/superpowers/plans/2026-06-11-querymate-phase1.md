# QueryMate Phase 1 — Retrieval + Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat full-schema prompt with schema-card RAG (sqlite-vec + fastembed), add an advisory Haiku planner, Haiku/Sonnet/Opus routing with cost logging, retrieval-aware repair, and a bucketed BIRD eval with recall@k.

**Architecture:** New graph `START → retrieve → plan → write_sql → execute → (critic → write_sql | critic → retrieve-with-widened-k)`. The flat `schema` string stays — built from retrieved cards (RAG arm) or full DDL (full-schema arm), so both eval arms share one writer path. All retrieval components sit behind small protocols so tests run with a deterministic fake embedder (no model download).

**Tech Stack:** Python 3.13/uv, LangGraph, sqlglot, Anthropic SDK, sqlite-vec (≥0.1.6, partition-key KNN), fastembed (`BAAI/bge-small-en-v1.5`, 384-dim).

**Spec:** `docs/superpowers/specs/2026-06-11-querymate-phase1-design.md`

**Conventions (match Phase 0):**
- Plain-assert test files, runnable standalone: `uv run python tests/test_x.py` (each ends with the `if __name__ == "__main__"` runner copied from `tests/test_compare.py:41-46`).
- All commands run from the repo root `~/Documents/GitHub/querymate`.
- LLM-dependent paths stay OUT of the no-key test suite.

## File structure

| File | Responsibility |
|---|---|
| `querymate/embedder.py` (new) | `Embedder` protocol, `FastEmbedder` (fastembed), `FakeEmbedder` (tests), `DIM = 384` |
| `querymate/schema_cards.py` (new) | Extract per-table `SchemaCard`s from a SQLite DB + BIRD description CSVs |
| `querymate/card_index.py` (new) | sqlite-vec store: create/insert/KNN-query cards |
| `querymate/retriever.py` (new) | top-k by db_id + FK 1-hop expansion → (cards, schema string) |
| `querymate/router.py` (new) | pure routing fns: `pick_model`, `should_widen` |
| `querymate/llm.py` (modify) | cost logging on every call, `evidence` param, `plan()` call |
| `querymate/settings.py` (modify) | `fast_model`, `retrieval_k`, `PRICES` |
| `querymate/state.py` (modify) | new state fields (`cost_log` with `operator.add` reducer) |
| `querymate/nodes.py` (modify) | `retrieve_node`, `plan_node`, router-driven `write_sql_node`, widen-aware `critic_node`, `route_after_critic` |
| `querymate/graph.py` (modify) | new edges incl. critic→retrieve widen path |
| `querymate/cli.py` (modify) | auto-build demo index, `--no-rag` |
| `scripts/ingest_schemas.py` (new) | walk DB root → cards → embed → index |
| `scripts/fetch_bird.py` (new) | download/unzip BIRD dev (manual fallback) |
| `evals/recall.py` (new) | gold-table extraction + recall@k |
| `evals/run_recall.py` (new) | retrieval-only eval over full dev (no LLM) |
| `evals/make_subset.py` (new) | stratified-300 + smoke-30 subsets |
| `evals/run_bird.py` (modify) | `--arm`, difficulty buckets, evidence, cost aggregation |
| `evals/make_chart.py` (new) | bucketed EX chart |
| `tests/test_schema_cards.py`, `tests/test_retriever.py`, `tests/test_router.py`, `tests/test_recall.py` (new) | plain-assert suites |

---

### Task 1: Dependencies + settings

**Files:**
- Modify: `pyproject.toml`
- Modify: `querymate/settings.py`

- [ ] **Step 1.1: Add deps to `pyproject.toml`**

Replace the `dependencies` list and `[project.optional-dependencies]` with:

```toml
dependencies = [
    "anthropic>=0.49",
    "langgraph>=0.2.50",
    "sqlglot>=25.0",
    "pydantic-settings>=2.0",
    "sqlite-vec>=0.1.6",
    "fastembed>=0.5",
]

[project.optional-dependencies]
# Phase 3 observability. Install with: uv sync --extra trace
trace = ["langfuse>=2.0"]
# Eval chart + BIRD download helpers. Install with: uv sync --extra eval
eval = ["matplotlib>=3.9"]
```

- [ ] **Step 1.2: Sync**

Run: `uv sync --extra eval`
Expected: resolves and installs `sqlite-vec`, `fastembed`, `matplotlib` without error.

- [ ] **Step 1.3: Extend `querymate/settings.py`**

Insert below the existing `escalate_model` line (`settings.py:19`):

```python
    fast_model: str = "claude-haiku-4-5"      # simple lookups (router tier 1)
    planner_model: str = "claude-haiku-4-5"   # advisory plan call

    # --- Retrieval (Phase 1) ---------------------------------------------
    retrieval_k: int = 5            # top-k schema cards per question
    schema_index_path: str = "data/schema_index.sqlite"
```

And at module level (below the `Settings` class, above `settings = Settings()`):

```python
# USD per MTok (input, output). Cache reads are billed at 0.1x input; cache
# writes at 1.25x input. Update when Anthropic prices change.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
}
```

- [ ] **Step 1.4: Verify imports still work**

Run: `uv run python -c "from querymate.settings import settings, PRICES; print(settings.retrieval_k, PRICES[settings.writer_model])"`
Expected: `5 (3.0, 15.0)`

- [ ] **Step 1.5: Run existing suite (must stay green)**

Run: `uv run python tests/test_validator.py && uv run python tests/test_executor.py && uv run python tests/test_compare.py`
Expected: all pass (25 tests total).

- [ ] **Step 1.6: Commit**

```bash
git add pyproject.toml uv.lock querymate/settings.py
git commit -m "feat: Phase 1 deps (sqlite-vec, fastembed) + routing/retrieval settings"
```

---

### Task 2: Embedder protocol + fake

**Files:**
- Create: `querymate/embedder.py`
- Test: `tests/test_retriever.py` (started here — fake-embedder tests live with retriever tests)

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_retriever.py`:

```python
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
```

- [ ] **Step 2.2: Run to verify it fails**

Run: `uv run python tests/test_retriever.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'querymate.embedder'`

- [ ] **Step 2.3: Implement `querymate/embedder.py`**

```python
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
```

- [ ] **Step 2.4: Run to verify it passes**

Run: `uv run python tests/test_retriever.py`
Expected: `2 retriever tests passed`

- [ ] **Step 2.5: Commit**

```bash
git add querymate/embedder.py tests/test_retriever.py
git commit -m "feat: embedder protocol with fastembed backend + deterministic fake"
```

---

### Task 3: Schema cards

**Files:**
- Create: `querymate/schema_cards.py`
- Test: `tests/test_schema_cards.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_schema_cards.py`:

```python
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
```

- [ ] **Step 3.2: Run to verify it fails**

Run: `uv run python tests/test_schema_cards.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'querymate.schema_cards'`

- [ ] **Step 3.3: Implement `querymate/schema_cards.py`**

```python
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
```

- [ ] **Step 3.4: Run to verify it passes**

Run: `uv run python tests/test_schema_cards.py`
Expected: `3 schema-card tests passed`

- [ ] **Step 3.5: Commit**

```bash
git add querymate/schema_cards.py tests/test_schema_cards.py
git commit -m "feat: per-table schema cards with FK edges + BIRD description merge"
```

---

### Task 4: sqlite-vec card index

**Files:**
- Create: `querymate/card_index.py`
- Test: `tests/test_retriever.py` (extend)

- [ ] **Step 4.1: Write the failing test**

Append to `tests/test_retriever.py`, below the existing tests (keep the `__main__` runner at the bottom of the file):

```python
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
```

- [ ] **Step 4.2: Run to verify it fails**

Run: `uv run python tests/test_retriever.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'querymate.card_index'`

- [ ] **Step 4.3: Implement `querymate/card_index.py`**

```python
"""sqlite-vec store for schema cards.

Two tables sharing rowids: ``vec_cards`` (vec0 virtual table, ``db_id`` as a
partition key so KNN filters per database) and ``cards`` (metadata). Zero-infra:
the index is one SQLite file next to the data.
"""

from __future__ import annotations

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
                fk_neighbors TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS cards_db ON cards(db_id, table_name);
            """
        )

    def add_cards(self, cards: list[SchemaCard]) -> None:
        vecs = self.embedder.embed([c.text for c in cards])
        cur = self.con.cursor()
        for card, vec in zip(cards, vecs):
            cur.execute(
                "INSERT INTO cards(db_id, table_name, card_text, fk_neighbors) "
                "VALUES (?, ?, ?, ?)",
                (card.db_id, card.table, card.text, ",".join(card.fk_neighbors)),
            )
            cur.execute(
                "INSERT INTO vec_cards(rowid, db_id, embedding) VALUES (?, ?, ?)",
                (cur.lastrowid, card.db_id, _f32(vec)),
            )
        self.con.commit()

    def _row_to_card(self, row) -> SchemaCard:
        db_id, table, text, fks = row
        return SchemaCard(db_id, table, text, fks.split(",") if fks else [])

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
```

- [ ] **Step 4.4: Run to verify it passes**

Run: `uv run python tests/test_retriever.py`
Expected: `4 retriever tests passed`

Note: if this fails with `AttributeError ... enable_load_extension`, the Python build lacks SQLite extension support — use a uv-managed interpreter (`uv python install 3.13 && uv sync`). Surface this in the README troubleshooting note in Task 12.

- [ ] **Step 4.5: Commit**

```bash
git add querymate/card_index.py tests/test_retriever.py
git commit -m "feat: sqlite-vec card index with per-db partition-key KNN"
```

---

### Task 5: Retriever (top-k + FK 1-hop + schema string)

**Files:**
- Create: `querymate/retriever.py`
- Test: `tests/test_retriever.py` (extend)

- [ ] **Step 5.1: Write the failing test**

Append to `tests/test_retriever.py` (above the `__main__` runner):

```python
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
```

- [ ] **Step 5.2: Run to verify it fails**

Run: `uv run python tests/test_retriever.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'querymate.retriever'`

- [ ] **Step 5.3: Implement `querymate/retriever.py`**

```python
"""Schema-card retrieval: KNN top-k per db_id, then FK 1-hop expansion so join
paths survive, then the writer's schema string built from the selected cards."""

from __future__ import annotations

from .card_index import CardIndex
from .schema_cards import SchemaCard


class Retriever:
    def __init__(self, index: CardIndex) -> None:
        self.index = index

    def retrieve(
        self, question: str, *, db_id: str, k: int
    ) -> tuple[list[SchemaCard], str]:
        hits = self.index.query(question, db_id=db_id, k=k)
        seen = {c.table for c in hits}
        neighbor_tables = [
            t for c in hits for t in c.fk_neighbors if t not in seen and not seen.add(t)
        ]
        cards = hits + self.index.get_cards(db_id, neighbor_tables)
        schema = "\n\n".join(c.text for c in cards)
        return cards, schema
```

- [ ] **Step 5.4: Run to verify it passes**

Run: `uv run python tests/test_retriever.py`
Expected: `6 retriever tests passed`

- [ ] **Step 5.5: Commit**

```bash
git add querymate/retriever.py tests/test_retriever.py
git commit -m "feat: retriever with FK 1-hop expansion and schema-string build"
```

---

### Task 6: Router + widen decision (pure functions)

**Files:**
- Create: `querymate/router.py`
- Test: `tests/test_router.py`

- [ ] **Step 6.1: Write the failing test**

Create `tests/test_router.py`:

```python
"""Plain-assert tests for model routing + retrieval-widening decisions.

    python tests/test_router.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.router import pick_model, should_widen  # noqa: E402
from querymate.settings import settings  # noqa: E402


def test_simple_plan_routes_to_fast_model():
    plan = {"tables": ["customers"], "join_count": 0, "aggregations": [], "filters": []}
    assert pick_model(plan, attempts=0, max_attempts=3) == settings.fast_model


def test_joins_route_to_writer_model():
    plan = {"tables": ["a", "b"], "join_count": 1, "aggregations": [], "filters": []}
    assert pick_model(plan, attempts=0, max_attempts=3) == settings.writer_model


def test_aggregations_route_to_writer_model():
    plan = {"tables": ["a"], "join_count": 0, "aggregations": ["COUNT"], "filters": []}
    assert pick_model(plan, attempts=0, max_attempts=3) == settings.writer_model


def test_no_plan_routes_to_writer_model():
    assert pick_model(None, attempts=0, max_attempts=3) == settings.writer_model


def test_final_attempt_escalates_even_with_simple_plan():
    plan = {"tables": ["a"], "join_count": 0, "aggregations": [], "filters": []}
    assert pick_model(plan, attempts=2, max_attempts=3) == settings.escalate_model


def test_zero_max_attempts_never_escalates():
    assert pick_model(None, attempts=0, max_attempts=0) == settings.writer_model


def test_widen_on_unknown_table_once():
    assert should_widen("unknown_table", widened=False, use_retrieval=True)
    assert should_widen("unknown_column", widened=False, use_retrieval=True)


def test_no_widen_when_already_widened_or_no_retrieval_or_other_error():
    assert not should_widen("unknown_table", widened=True, use_retrieval=True)
    assert not should_widen("unknown_table", widened=False, use_retrieval=False)
    assert not should_widen("syntax", widened=False, use_retrieval=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} router tests passed")
```

- [ ] **Step 6.2: Run to verify it fails**

Run: `uv run python tests/test_router.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'querymate.router'`

- [ ] **Step 6.3: Implement `querymate/router.py`**

```python
"""Pure routing decisions — kept free of I/O so they unit-test trivially.

``pick_model``: Haiku for no-join/no-aggregation questions, Sonnet otherwise,
Opus on the final allowed attempt (Phase-0 escalation semantics preserved).
``should_widen``: the critic widens retrieval (k x 2, once) when the DB says a
table/column doesn't exist — the schema the writer saw was probably incomplete.
"""

from __future__ import annotations

from typing import Optional

from .settings import settings

_WIDEN_KINDS = {"unknown_table", "unknown_column"}


def pick_model(plan_info: Optional[dict], *, attempts: int, max_attempts: int) -> str:
    if max_attempts > 0 and attempts >= max(max_attempts - 1, 0):
        return settings.escalate_model
    if plan_info is None:
        return settings.writer_model  # planner failed → safe default
    if not plan_info.get("join_count", 0) and not plan_info.get("aggregations"):
        return settings.fast_model
    return settings.writer_model


def should_widen(err_kind: str, *, widened: bool, use_retrieval: bool) -> bool:
    return use_retrieval and not widened and err_kind in _WIDEN_KINDS
```

Note the call-signature difference from Phase 0's inline escalation (`nodes.py:14-19`): `pick_model(plan_info, attempts=…, max_attempts=…)` — Task 9 replaces that inline logic with this function.

- [ ] **Step 6.4: Run to verify it passes**

Run: `uv run python tests/test_router.py`
Expected: `8 router tests passed`

- [ ] **Step 6.5: Commit**

```bash
git add querymate/router.py tests/test_router.py
git commit -m "feat: pure model-routing + retrieval-widening decision functions"
```

---

### Task 7: Cost accounting + evidence in `llm.py`

**Files:**
- Modify: `querymate/settings.py` (add `call_cost`)
- Modify: `querymate/llm.py`
- Test: `tests/test_router.py` (extend — cost lives with routing tests)

- [ ] **Step 7.1: Write the failing test**

Append to `tests/test_router.py` (above the `__main__` runner):

```python
from querymate.settings import call_cost  # noqa: E402


def test_call_cost_sonnet():
    # 1M in @ $3 + 1M out @ $15
    assert abs(call_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 0, 0) - 18.0) < 1e-9


def test_call_cost_cache_rates():
    # cache read 0.1x input rate; cache write 1.25x input rate (Haiku: $1/MTok in)
    c = call_cost("claude-haiku-4-5", 0, 0, 1_000_000, 1_000_000)
    assert abs(c - (0.1 + 1.25)) < 1e-9


def test_call_cost_unknown_model_is_zero():
    assert call_cost("not-a-model", 1000, 1000, 0, 0) == 0.0
```

- [ ] **Step 7.2: Run to verify it fails**

Run: `uv run python tests/test_router.py`
Expected: FAIL with `ImportError: cannot import name 'call_cost'`

- [ ] **Step 7.3: Add `call_cost` to `querymate/settings.py`**

Below the `PRICES` dict:

```python
def call_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD for one API call. Unknown models cost 0 (logged, not billed-estimated)."""
    if model not in PRICES:
        return 0.0
    in_rate, out_rate = PRICES[model]
    return (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_read_tokens * in_rate * 0.1
        + cache_write_tokens * in_rate * 1.25
    ) / 1_000_000
```

- [ ] **Step 7.4: Run to verify it passes**

Run: `uv run python tests/test_router.py`
Expected: `11 router tests passed`

- [ ] **Step 7.5: Add the usage helper + evidence param to `querymate/llm.py`**

Add imports at the top of `llm.py` (`time` to stdlib block; `call_cost` from settings):

```python
import time

from .settings import call_cost
```

Add below the `client()` function:

```python
def _usage_entry(resp, model: str, t0: float, *, purpose: str) -> dict:
    u = resp.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    return {
        "purpose": purpose,                    # writer | critic | planner
        "model": model,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "cost_usd": call_cost(model, u.input_tokens, u.output_tokens,
                              cache_read, cache_write),
        "latency_s": round(time.monotonic() - t0, 3),
    }
```

- [ ] **Step 7.6: Change `write_sql` to accept `evidence` and return `(sql, entry)`**

In `write_sql` (`llm.py:55`): add the keyword parameter `evidence: Optional[str] = None` after `repair_hint`, and insert into the user-content build (after the `parts = [f"Question: {question}"]` line):

```python
    if evidence:
        parts.append(f"Hint (domain evidence): {evidence}")
```

Wrap the API call with timing and change the return to a tuple — replace from `resp = client().messages.create(` to the end of the function with:

```python
    t0 = time.monotonic()
    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        output_config={"format": _SQL_FORMAT},
    )
    entry = _usage_entry(resp, model, t0, purpose="writer")
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        return json.loads(text)["sql"].strip(), entry
    except Exception:
        return text.strip(), entry
```

Update the return type annotation: `-> tuple[str, dict]`.

- [ ] **Step 7.7: Same for `diagnose`**

In `diagnose` (`llm.py:95`): wrap with `t0 = time.monotonic()` before the call, and replace the final `return` with:

```python
    entry = _usage_entry(resp, model, t0, purpose="critic")
    return next((b.text for b in resp.content if b.type == "text"), error), entry
```

Update annotation: `-> tuple[str, dict]`.

(Callers in `nodes.py` break until Task 9 rewires them — that's expected; the no-key test suite doesn't import `nodes`.)

- [ ] **Step 7.8: Verify imports + suite**

Run: `uv run python -c "import querymate.llm" && uv run python tests/test_router.py && uv run python tests/test_validator.py`
Expected: all pass.

- [ ] **Step 7.9: Commit**

```bash
git add querymate/settings.py querymate/llm.py tests/test_router.py
git commit -m "feat: per-call cost accounting + BIRD evidence hint in writer prompt"
```

---

### Task 8: Planner LLM call

**Files:**
- Modify: `querymate/llm.py`

- [ ] **Step 8.1: Add the plan format + `plan()` to `llm.py`**

Below `_SQL_FORMAT`:

```python
_PLAN_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "tables": {"type": "array", "items": {"type": "string"},
                       "description": "Tables the query needs."},
            "join_count": {"type": "integer",
                           "description": "Number of JOINs expected."},
            "aggregations": {"type": "array", "items": {"type": "string"},
                             "description": "Aggregate functions needed (COUNT, SUM...)."},
            "filters": {"type": "array", "items": {"type": "string"},
                        "description": "WHERE/HAVING conditions in plain words."},
        },
        "required": ["tables", "join_count", "aggregations", "filters"],
        "additionalProperties": False,
    },
}
```

At the end of `llm.py`:

```python
def plan(
    *,
    question: str,
    schema: str,
    dialect: str,
    model: str,
    evidence: Optional[str] = None,
    max_tokens: int = 500,
) -> tuple[Optional[dict], Optional[dict]]:
    """Advisory query plan. Returns (plan_info, usage_entry); (None, entry-or-None)
    on any failure — the loop must never depend on the planner."""
    parts = [f"Question: {question}"]
    if evidence:
        parts.append(f"Hint (domain evidence): {evidence}")
    parts.append(
        "Plan the SQL query: which tables, how many joins, which aggregate "
        "functions, which filters. Do not write SQL."
    )
    t0 = time.monotonic()
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": _system(schema, dialect),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": "\n".join(parts)}],
            output_config={"format": _PLAN_FORMAT},
        )
        entry = _usage_entry(resp, model, t0, purpose="planner")
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        info = json.loads(text)
        if not isinstance(info, dict) or "tables" not in info:
            return None, entry
        return info, entry
    except Exception:
        return None, None
```

- [ ] **Step 8.2: Verify import**

Run: `uv run python -c "from querymate.llm import plan, write_sql, diagnose; print('ok')"`
Expected: `ok`

- [ ] **Step 8.3: Commit**

```bash
git add querymate/llm.py
git commit -m "feat: advisory Haiku planner call with structured plan output"
```

---

### Task 9: State + nodes + graph rewire

**Files:**
- Modify: `querymate/state.py`
- Modify: `querymate/llm.py` (writer gains `plan` param)
- Modify: `querymate/nodes.py`
- Modify: `querymate/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 9.1: Write the failing test**

Create `tests/test_graph.py`:

```python
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
```

- [ ] **Step 9.2: Run to verify it fails**

Run: `uv run python tests/test_graph.py`
Expected: FAIL with `ImportError` (no `retrieve_node` / `set_retriever` yet).

- [ ] **Step 9.3: Extend `querymate/state.py`**

Replace the imports and add fields so the full file reads:

```python
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class DBError(TypedDict):
    # unsafe | parse_error | syntax | unknown_table | unknown_column | timeout | other
    kind: str
    message: str


class QueryState(TypedDict, total=False):
    """State carried through the LangGraph loop.

    Phase 1: ``schema`` is built from retrieved schema cards (RAG arm) or the
    full DDL (full-schema arm / fallback) — the writer sees one string either way.
    """

    question: str
    evidence: Optional[str]       # BIRD domain hint, when the dataset provides one
    schema: str
    dialect: str
    db_path: str
    db_id: str                    # index partition key (defaults to db filename stem)

    # retrieval (Phase 1)
    use_retrieval: bool
    retrieval_k: int
    retrieval_widened: bool       # widen happened at some point (bounds it to once)
    widen_now: bool               # critic verdict for the router edge this turn
    card_tables: list[str]        # tables whose cards the writer saw

    # planner (Phase 1)
    use_planner: bool
    plan: Optional[str]           # rendered plan text for the writer prompt
    plan_info: Optional[dict]     # structured plan for the model router

    sql: Optional[str]            # last query the writer produced
    validated_sql: Optional[str]  # post-validation (LIMIT-injected) query that ran
    attempts: int                 # repair attempts consumed
    max_attempts: int
    last_error: Optional[DBError]
    repair_hint: Optional[str]    # critic → writer

    rows: Optional[list[tuple[Any, ...]]]
    columns: Optional[list[str]]

    # cost accounting: nodes return {"cost_log": [entry]} and entries accumulate
    cost_log: Annotated[list[dict], operator.add]

    # knobs
    auto_limit: bool
    use_llm_critic: bool
```

- [ ] **Step 9.4: Writer gains the `plan` param in `querymate/llm.py`**

In `write_sql`: add keyword param `plan: Optional[str] = None` (next to `evidence`), and after the evidence append in the user-content build:

```python
    if plan:
        parts.append(f"Query plan (advisory):\n{plan}")
```

- [ ] **Step 9.5: Rewrite `querymate/nodes.py`**

Full replacement:

```python
"""LangGraph nodes: retrieve → plan → write_sql → execute → (critic loop)."""

from __future__ import annotations

import os
from typing import Optional

from . import llm
from .executor import ExecError, schema_text, validate_and_run
from .retriever import Retriever
from .router import pick_model, should_widen
from .settings import settings
from .state import QueryState

_retriever: Optional[Retriever] = None


def set_retriever(r: Optional[Retriever]) -> None:
    """Inject a retriever (tests / eval with a custom index)."""
    global _retriever
    _retriever = r


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        from .card_index import CardIndex
        from .embedder import FastEmbedder

        if not os.path.exists(settings.schema_index_path):
            raise FileNotFoundError(
                f"schema index not found at '{settings.schema_index_path}' — build "
                "it first: uv run python scripts/ingest_schemas.py --demo  (or "
                "--db-root <dir> for BIRD)"
            )
        _retriever = Retriever(
            CardIndex(settings.schema_index_path, embedder=FastEmbedder())
        )
    return _retriever


def _db_id(state: QueryState) -> str:
    return state.get("db_id") or os.path.splitext(
        os.path.basename(state["db_path"])
    )[0]


def retrieve_node(state: QueryState) -> dict:
    if not state.get("use_retrieval", True):
        return {"schema": schema_text(state["db_path"]), "card_tables": []}
    k = state.get("retrieval_k", settings.retrieval_k)
    cards, schema = get_retriever().retrieve(
        state["question"], db_id=_db_id(state), k=k
    )
    return {"schema": schema, "card_tables": [c.table for c in cards]}


def _render_plan(info: dict) -> str:
    return (
        f"tables: {', '.join(info.get('tables', [])) or '-'}\n"
        f"joins: {info.get('join_count', 0)}\n"
        f"aggregations: {', '.join(info.get('aggregations', [])) or '-'}\n"
        f"filters: {'; '.join(info.get('filters', [])) or '-'}"
    )


def plan_node(state: QueryState) -> dict:
    if not state.get("use_planner", True):
        return {"plan": None, "plan_info": None}
    info, entry = llm.plan(
        question=state["question"],
        schema=state["schema"],
        dialect=state.get("dialect", "sqlite"),
        model=settings.planner_model,
        evidence=state.get("evidence"),
    )
    out: dict = {
        "plan_info": info,
        "plan": _render_plan(info) if info else None,
    }
    if entry:
        out["cost_log"] = [entry]
    return out


def write_sql_node(state: QueryState) -> dict:
    model = pick_model(
        state.get("plan_info"),
        attempts=state.get("attempts", 0),
        max_attempts=state.get("max_attempts", settings.max_attempts),
    )
    sql, entry = llm.write_sql(
        question=state["question"],
        schema=state["schema"],
        dialect=state.get("dialect", "sqlite"),
        model=model,
        repair_hint=state.get("repair_hint"),
        prev_sql=state.get("sql"),
        evidence=state.get("evidence"),
        plan=state.get("plan"),
    )
    return {"sql": sql, "cost_log": [entry]}


def execute_node(state: QueryState) -> dict:
    try:
        validated, rows, columns = validate_and_run(
            state["sql"],
            state["db_path"],
            dialect=state.get("dialect", "sqlite"),
            auto_limit=state.get("auto_limit", settings.auto_limit),
            max_rows=settings.max_rows,
            timeout_s=settings.statement_timeout_s,
        )
        return {
            "validated_sql": validated,
            "rows": rows,
            "columns": columns,
            "last_error": None,
        }
    except ExecError as e:
        return {"last_error": e.err, "rows": None, "columns": None}


def critic_node(state: QueryState) -> dict:
    attempts = state.get("attempts", 0) + 1
    err = state.get("last_error") or {"kind": "other", "message": "unknown"}
    widen = should_widen(
        err["kind"],
        widened=state.get("retrieval_widened", False),
        use_retrieval=state.get("use_retrieval", True),
    )
    if widen:
        return {
            "attempts": attempts,
            "widen_now": True,
            "retrieval_widened": True,
            "retrieval_k": state.get("retrieval_k", settings.retrieval_k) * 2,
            "repair_hint": (
                f"{err['kind']}: {err['message']} — schema context was widened; "
                "use only tables/columns in the (new) schema."
            ),
        }
    out: dict = {"attempts": attempts, "widen_now": False}
    if state.get("use_llm_critic"):
        hint, entry = llm.diagnose(
            question=state["question"],
            schema=state["schema"],
            dialect=state.get("dialect", "sqlite"),
            sql=state.get("sql", ""),
            error=err["message"],
            model=settings.writer_model,
        )
        out["repair_hint"] = hint
        out["cost_log"] = [entry]
    else:
        out["repair_hint"] = f"{err['kind']}: {err['message']}"
    return out


def route_after_execute(state: QueryState) -> str:
    if state.get("last_error") is None:
        return "ok"
    if state.get("attempts", 0) >= state.get("max_attempts", settings.max_attempts):
        return "give_up"
    return "critic"


def route_after_critic(state: QueryState) -> str:
    return "widen" if state.get("widen_now") else "rewrite"
```

- [ ] **Step 9.6: Rewire `querymate/graph.py`**

Replace the docstring diagram and `build_graph` body:

```python
"""Assemble the Phase-1 agent loop as a LangGraph state machine.

    START → retrieve → plan → write_sql → execute ─(ok / give_up)→ END
               ▲                  ▲           │
               │                  │     (error, attempts < max)
               │                  │           ▼
               └──(widen k once)── critic ────┘
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    critic_node,
    execute_node,
    plan_node,
    retrieve_node,
    route_after_critic,
    route_after_execute,
    write_sql_node,
)
from .state import QueryState

_GRAPH = None


def build_graph():
    g = StateGraph(QueryState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("plan", plan_node)
    g.add_node("write_sql", write_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "write_sql")
    g.add_edge("write_sql", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"ok": END, "give_up": END, "critic": "critic"},
    )
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"widen": "retrieve", "rewrite": "write_sql"},
    )
    return g.compile()


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
```

- [ ] **Step 9.7: Run to verify it passes**

Run: `uv run python tests/test_graph.py`
Expected: `4 graph tests passed`

- [ ] **Step 9.8: Full suite**

Run: `for t in tests/test_*.py; do uv run python "$t" || break; done`
Expected: every file prints its pass line.

- [ ] **Step 9.9: Commit**

```bash
git add querymate/state.py querymate/llm.py querymate/nodes.py querymate/graph.py tests/test_graph.py
git commit -m "feat: retrieve/plan nodes, retrieval-aware repair edge, cost-log state"
```

---

### Task 10: Ingestion script

**Files:**
- Create: `scripts/ingest_schemas.py`

- [ ] **Step 10.1: Implement `scripts/ingest_schemas.py`**

```python
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
```

- [ ] **Step 10.2: Smoke-run on the demo DB**

Run: `uv run python scripts/make_demo_db.py && uv run python scripts/ingest_schemas.py --demo`
Expected: model downloads on first run, then `[1/1] demo_store: 4 cards` and `index → data/schema_index.sqlite`.

- [ ] **Step 10.3: Verify the index answers a retrieval query**

Run: `uv run python -c "
from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.retriever import Retriever
r = Retriever(CardIndex('data/schema_index.sqlite', embedder=FastEmbedder()))
cards, _ = r.retrieve('which customer placed the most orders', db_id='demo_store', k=2)
print([c.table for c in cards])"`
Expected: a list containing `orders` and `customers` (FK expansion may add more).

- [ ] **Step 10.4: Commit**

```bash
git add scripts/ingest_schemas.py
git commit -m "feat: schema ingestion script (BIRD + flat layouts, demo mode)"
```

---

### Task 11: CLI — auto demo index + `--no-rag`

**Files:**
- Modify: `querymate/cli.py`

- [ ] **Step 11.1: Update `querymate/cli.py`**

Replace `run_question` and `main` (keep the imports, add the new ones):

```python
from __future__ import annotations

import argparse
import os
import sys

from .executor import schema_text
from .graph import get_graph
from .settings import settings


def _ensure_demo_index() -> None:
    """Clone-and-run: build the demo index on first use (downloads the
    embedding model one time)."""
    from .card_index import CardIndex
    from .embedder import FastEmbedder

    if not os.path.exists(settings.schema_index_path):
        print("(building schema index for the demo DB — first run only)")
        os.makedirs(os.path.dirname(settings.schema_index_path) or ".", exist_ok=True)
    idx = CardIndex(settings.schema_index_path, embedder=FastEmbedder())
    try:
        if not idx.has_db("demo_store"):
            from .schema_cards import extract_cards

            idx.add_cards(extract_cards(settings.demo_db_path, db_id="demo_store"))
    finally:
        idx.close()


def run_question(
    question: str,
    *,
    db_path: str | None = None,
    dialect: str = "sqlite",
    max_attempts: int | None = None,
    use_retrieval: bool = True,
) -> dict:
    db_path = db_path or settings.demo_db_path
    db_id = os.path.splitext(os.path.basename(db_path))[0]
    if use_retrieval and db_id == "demo_store":
        _ensure_demo_index()
    state = {
        "question": question,
        "db_path": db_path,
        "db_id": db_id,
        "dialect": dialect,
        "schema": "" if use_retrieval else schema_text(db_path),
        "use_retrieval": use_retrieval,
        "use_planner": True,
        "retrieval_k": settings.retrieval_k,
        "attempts": 0,
        "max_attempts": settings.max_attempts if max_attempts is None else max_attempts,
        "auto_limit": settings.auto_limit,
        "use_llm_critic": False,
        "cost_log": [],
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate — NL→SQL copilot (Phase 1)")
    ap.add_argument("question", help="a question in plain English")
    ap.add_argument("--db", default=None, help="path to a SQLite DB (default: demo)")
    ap.add_argument("--no-rag", action="store_true",
                    help="skip retrieval; prompt with the full schema")
    args = ap.parse_args()

    out = run_question(args.question, db_path=args.db,
                       use_retrieval=not args.no_rag)

    print("\nSQL:\n" + (out.get("validated_sql") or out.get("sql") or "(none)"))
    if out.get("last_error"):
        print(
            f"\n[failed after {out.get('attempts', 0)} repair attempt(s)] "
            f"{out['last_error']}"
        )
        sys.exit(1)

    cols = out.get("columns") or []
    rows = out.get("rows") or []
    print("\nResult: " + (" | ".join(cols) if cols else "(no columns)"))
    for r in rows[:50]:
        print("  " + " | ".join(str(c) for c in r))
    cost = sum(e.get("cost_usd", 0.0) for e in out.get("cost_log", []))
    models = [e["model"].rsplit("-", 1)[0] for e in out.get("cost_log", [])
              if e.get("purpose") == "writer"]
    print(
        f"\n({len(rows)} row(s); repair attempts={out.get('attempts', 0)}; "
        f"writer={models[-1] if models else '-'}; cost=${cost:.4f})"
    )


if __name__ == "__main__":
    main()
```

Note for non-demo `--db` paths with RAG on: `get_retriever()` (Task 9) raises a clear `FileNotFoundError` telling the user to run `ingest_schemas.py` — that is the intended UX; only the demo DB auto-builds.

- [ ] **Step 11.2: Verify CLI help + no-key import path**

Run: `uv run querymate --help`
Expected: shows `--no-rag` flag, exits 0.

(Live check, only if `ANTHROPIC_API_KEY` is set in `.env` — optional here, required before Task 13's eval run: `uv run querymate "Which customer has placed the most orders? Return their name."` → expect SQL + `Alice` + a cost line.)

- [ ] **Step 11.3: Commit**

```bash
git add querymate/cli.py
git commit -m "feat: CLI auto-builds demo index, --no-rag flag, cost summary line"
```

---

### Task 12: recall@k + BIRD tooling (subset, download, retrieval-only eval)

**Files:**
- Create: `evals/recall.py`
- Create: `evals/run_recall.py`
- Create: `evals/make_subset.py`
- Create: `scripts/fetch_bird.py`
- Test: `tests/test_recall.py`

- [ ] **Step 12.1: Write the failing test**

Create `tests/test_recall.py`:

```python
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
```

- [ ] **Step 12.2: Run to verify it fails**

Run: `uv run python tests/test_recall.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.recall'`

- [ ] **Step 12.3: Implement `evals/recall.py`**

```python
"""Schema-retrieval recall@k.

Gold tables come from parsing the gold SQL with sqlglot (CTE names excluded —
they aren't schema tables). recall@k = |gold ∩ retrieved| / |gold|; questions
whose gold SQL yields no tables (or doesn't parse) return None and are
excluded from the average, so a parser hiccup can't inflate the score.
"""

from __future__ import annotations

from typing import Optional

import sqlglot
from sqlglot import exp


def gold_tables(sql: str, dialect: str = "sqlite") -> set[str]:
    try:
        stmt = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return set()
    if stmt is None:
        return set()
    ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    return {
        t.name.lower()
        for t in stmt.find_all(exp.Table)
        if t.name and t.name.lower() not in ctes
    }


def recall_at_k(gold: set[str], retrieved: list[str]) -> Optional[float]:
    gold_l = {g.lower() for g in gold}
    if not gold_l:
        return None
    got = {r.lower() for r in retrieved}
    return len(gold_l & got) / len(gold_l)
```

- [ ] **Step 12.4: Run to verify it passes**

Run: `uv run python tests/test_recall.py`
Expected: `6 recall tests passed`

- [ ] **Step 12.5: Implement `evals/run_recall.py` (retrieval-only, no LLM, free)**

```python
"""Retrieval-only recall@k eval over a BIRD-format subset. No LLM calls — this
runs on the FULL dev set for free.

    python evals/run_recall.py --subset data/bird/dev.json \
        --index data/schema_index.sqlite --ks 3 5 10
"""

from __future__ import annotations

import argparse
import json

from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.retriever import Retriever
from querymate.settings import settings

from evals.recall import gold_tables, recall_at_k


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate retrieval recall@k eval")
    ap.add_argument("--subset", default="evals/data/sample_bird_subset.json")
    ap.add_argument("--index", default=settings.schema_index_path)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--report", default="evals/recall_report.json")
    args = ap.parse_args()

    with open(args.subset) as f:
        items = json.load(f)
    retriever = Retriever(CardIndex(args.index, embedder=FastEmbedder()))

    sums = {k: 0.0 for k in args.ks}
    counts = {k: 0 for k in args.ks}
    skipped = 0
    for i, it in enumerate(items, 1):
        gold = gold_tables(it["SQL"])
        if not gold:
            skipped += 1
            continue
        for k in args.ks:
            cards, _ = retriever.retrieve(it["question"], db_id=it["db_id"], k=k)
            r = recall_at_k(gold, [c.table for c in cards])
            if r is not None:
                sums[k] += r
                counts[k] += 1
        if i % 100 == 0:
            print(f"[{i}/{len(items)}]")

    report = {
        "n": len(items),
        "skipped_unparseable_gold": skipped,
        "recall_at_k": {
            str(k): round(sums[k] / counts[k], 4) if counts[k] else None
            for k in args.ks
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== Schema-retrieval recall ===")
    for k in args.ks:
        print(f"recall@{k}: {report['recall_at_k'][str(k)]}")
    print(f"report → {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.6: Smoke-run recall on the demo subset**

Run: `uv run python evals/run_recall.py --ks 2 4`
Expected: prints `recall@2` / `recall@4` values (demo: recall@4 should be 1.0 — only 4 tables exist) and writes `evals/recall_report.json`.

- [ ] **Step 12.7: Implement `evals/make_subset.py`**

```python
"""Stratified BIRD subsets: 100 per difficulty bucket (or --per-bucket) + a
30-question smoke set, deterministic (seeded).

    python evals/make_subset.py --dev data/bird/dev.json
"""

from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser(description="Make stratified BIRD subsets")
    ap.add_argument("--dev", required=True, help="path to BIRD dev.json")
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--smoke-per-bucket", type=int, default=10)
    ap.add_argument("--out", default="evals/data/bird_stratified.json")
    ap.add_argument("--smoke-out", default="evals/data/bird_smoke.json")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    with open(args.dev) as f:
        items = json.load(f)

    buckets: dict[str, list] = {}
    for it in items:
        buckets.setdefault(it.get("difficulty", "unknown"), []).append(it)
    print({k: len(v) for k, v in sorted(buckets.items())})

    rng = random.Random(args.seed)
    full, smoke = [], []
    for name in sorted(buckets):
        pool = sorted(buckets[name], key=lambda x: x["question"])
        rng.shuffle(pool)
        full.extend(pool[: args.per_bucket])
        smoke.extend(pool[: args.smoke_per_bucket])

    for path, data in ((args.out, full), (args.smoke_out, smoke)):
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        print(f"{path}: {len(data)} items")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.8: Implement `scripts/fetch_bird.py`**

```python
"""Download + unpack the BIRD dev set into data/bird/.

    python scripts/fetch_bird.py

Tries the official OSS mirror; if the URL has moved, prints manual steps
(https://bird-bench.github.io → Dev set). Expected result:
    data/bird/dev.json
    data/bird/dev_databases/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEST = "data/bird"


def _flatten(root: str) -> None:
    """The zip nests everything under dev_20240627/ (and databases in a second
    zip). Normalise to data/bird/dev.json + data/bird/dev_databases/."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == "dev.json" and dirpath != root:
                shutil.move(os.path.join(dirpath, f), os.path.join(root, f))
            if f == "dev_databases.zip":
                with zipfile.ZipFile(os.path.join(dirpath, f)) as z:
                    z.extractall(root)
    for dirpath, dirs, _ in os.walk(root):
        for d in dirs:
            if d == "dev_databases" and dirpath != root:
                target = os.path.join(root, d)
                if not os.path.exists(target):
                    shutil.move(os.path.join(dirpath, d), target)


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    zip_path = os.path.join(DEST, "dev.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {URL} (~1.2GB)…")
        try:
            urllib.request.urlretrieve(URL, zip_path)
        except Exception as e:
            sys.exit(
                f"download failed ({e}).\nManual fallback: get the dev set from "
                "https://bird-bench.github.io and unzip so that data/bird/dev.json "
                "and data/bird/dev_databases/ exist."
            )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DEST)
    _flatten(DEST)
    ok = os.path.exists(os.path.join(DEST, "dev.json")) and os.path.isdir(
        os.path.join(DEST, "dev_databases")
    )
    print("ok: data/bird ready" if ok else
          "unzipped, but layout unexpected — arrange manually (see docstring)")


if __name__ == "__main__":
    main()
```

(Heads-up for the executor: the OSS URL has historically worked but BIRD occasionally moves hosting; the manual fallback is authoritative. Do NOT mark this step failed if the download is just slow — it's ~1.2GB.)

- [ ] **Step 12.9: Commit**

```bash
git add evals/recall.py evals/run_recall.py evals/make_subset.py scripts/fetch_bird.py tests/test_recall.py
git commit -m "feat: recall@k eval (LLM-free), stratified subset maker, BIRD fetcher"
```

---

### Task 13: `run_bird.py` upgrades + chart

**Files:**
- Modify: `evals/run_bird.py` (full replacement below)
- Create: `evals/make_chart.py`

- [ ] **Step 13.1: Replace `evals/run_bird.py`**

```python
"""BIRD execution-accuracy eval — Phase 1.

Per item, runs the graph twice (single-shot and critic-loop) and scores by EX.
Reports single-shot EX, final EX and the self-correction lift, all bucketed by
BIRD difficulty, plus cost/latency by routing tier.

Arms: --arm rag (schema-card retrieval; default) | --arm full-schema (Phase-0
behaviour). Run both on the same subset to get the retrieval lift.

    python evals/run_bird.py --subset evals/data/bird_smoke.json \
        --db-root data/bird/dev_databases --arm rag
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.executor import run_query
from querymate.graph import get_graph
from querymate.nodes import set_retriever
from querymate.retriever import Retriever
from querymate.settings import settings

from evals.compare import execution_match


def _db_path(db_id: str, root: str) -> str:
    candidates = [
        os.path.join(root, f"{db_id}.sqlite"),
        os.path.join(root, "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(root, db_id, f"{db_id}.sqlite"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"no sqlite db for '{db_id}' under '{root}' (tried {candidates})")


def _gold_rows(gold_sql: str, db_path: str) -> list:
    # Gold SQL comes from the dataset (trusted) — run it directly, read-only.
    return run_query(
        gold_sql, db_path,
        max_rows=settings.max_rows, timeout_s=settings.statement_timeout_s,
    )[0]


def _predict(it: dict, db_path: str, *, use_retrieval: bool, k: int,
             max_attempts: int) -> dict:
    state = {
        "question": it["question"],
        "evidence": it.get("evidence") or None,
        "db_path": db_path,
        "db_id": it["db_id"],
        "dialect": "sqlite",
        "use_retrieval": use_retrieval,
        "use_planner": True,
        "retrieval_k": k,
        "attempts": 0,
        "max_attempts": max_attempts,
        "auto_limit": False,  # don't truncate a large gold result set during eval
        "use_llm_critic": False,
        "cost_log": [],
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate BIRD EX eval (Phase 1)")
    ap.add_argument("--subset", default="evals/data/sample_bird_subset.json")
    ap.add_argument("--db-root", default="data")
    ap.add_argument("--arm", choices=["rag", "full-schema"], default="rag")
    ap.add_argument("--k", type=int, default=settings.retrieval_k)
    ap.add_argument("--index", default=settings.schema_index_path)
    ap.add_argument("--report", default=None,
                    help="default: evals/report_<arm>.json")
    args = ap.parse_args()
    report_path = args.report or f"evals/report_{args.arm.replace('-', '_')}.json"

    use_retrieval = args.arm == "rag"
    if use_retrieval:
        if not os.path.exists(args.index):
            raise SystemExit(
                f"schema index '{args.index}' missing — build it first:\n"
                f"  uv run python scripts/ingest_schemas.py --db-root {args.db_root} "
                f"--index {args.index}"
            )
        set_retriever(Retriever(CardIndex(args.index, embedder=FastEmbedder())))

    with open(args.subset) as f:
        items = json.load(f)

    n = len(items)
    errors = 0
    detail = []
    bucket_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "single": 0, "final": 0}
    )
    cost_total = 0.0
    tier_calls: dict[str, int] = defaultdict(int)

    for i, it in enumerate(items, 1):
        q, gold, db_id = it["question"], it["SQL"], it["db_id"]
        bucket = it.get("difficulty", "unknown")
        try:
            db_path = _db_path(db_id, args.db_root)
            gold_rows = _gold_rows(gold, db_path)
        except Exception as e:
            errors += 1
            print(f"[{i}/{n}] SKIP {db_id}: {e}")
            continue

        try:
            out = _predict(it, db_path, use_retrieval=use_retrieval, k=args.k,
                           max_attempts=settings.max_attempts)
            final_ok = out.get("last_error") is None and execution_match(
                out.get("rows"), gold_rows)
            attempts = out.get("attempts", 0)
            for e in out.get("cost_log", []):
                cost_total += e.get("cost_usd", 0.0)
                if e.get("purpose") == "writer":
                    tier_calls[e["model"]] += 1
        except Exception as e:
            final_ok, attempts, out = False, 0, {}
            print(f"   predict error (final): {e}")

        try:
            out1 = _predict(it, db_path, use_retrieval=use_retrieval, k=args.k,
                            max_attempts=0)  # no repair
            single_ok = out1.get("last_error") is None and execution_match(
                out1.get("rows"), gold_rows)
            for e in out1.get("cost_log", []):
                cost_total += e.get("cost_usd", 0.0)
                if e.get("purpose") == "writer":
                    tier_calls[e["model"]] += 1
        except Exception as e:
            single_ok = False
            print(f"   predict error (single-shot): {e}")

        b = bucket_stats[bucket]
        b["n"] += 1
        b["single"] += int(single_ok)
        b["final"] += int(final_ok)
        print(
            f"[{i}/{n}] {db_id} ({bucket}): single={'PASS' if single_ok else 'fail'} "
            f"final={'PASS' if final_ok else 'fail'} attempts={attempts}  | {q}"
        )
        detail.append({
            "question": q, "db_id": db_id, "difficulty": bucket,
            "single_ok": single_ok, "final_ok": final_ok, "attempts": attempts,
            "card_tables": out.get("card_tables", []),
        })

    scored = sum(b["n"] for b in bucket_stats.values())
    ex_single = sum(b["single"] for b in bucket_stats.values())
    ex_final = sum(b["final"] for b in bucket_stats.values())
    report = {
        "arm": args.arm,
        "k": args.k if use_retrieval else None,
        "n": n, "scored": scored, "errors": errors,
        "ex_single_shot": round(ex_single / scored, 4) if scored else None,
        "ex_final": round(ex_final / scored, 4) if scored else None,
        "self_correction_lift": round((ex_final - ex_single) / scored, 4) if scored else None,
        "buckets": {
            name: {
                "n": b["n"],
                "ex_single_shot": round(b["single"] / b["n"], 4),
                "ex_final": round(b["final"] / b["n"], 4),
            }
            for name, b in sorted(bucket_stats.items())
        },
        "cost_usd_total": round(cost_total, 4),
        "cost_usd_per_question": round(cost_total / scored, 6) if scored else None,
        "writer_calls_by_model": dict(tier_calls),
        "items": detail,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== QueryMate execution accuracy [{args.arm}] ===")
    print(f"scored: {scored}/{n}  (errors: {errors})")
    if scored:
        print(f"single-shot EX      : {report['ex_single_shot']:.3f}")
        print(f"final EX (w/ critic): {report['ex_final']:.3f}")
        print(f"self-correction lift: {report['self_correction_lift']:+.3f}")
        for name, b in report["buckets"].items():
            print(f"  {name:<12} n={b['n']:<4} single={b['ex_single_shot']:.3f} "
                  f"final={b['ex_final']:.3f}")
        print(f"cost: ${report['cost_usd_total']:.2f} total "
              f"(${report['cost_usd_per_question']:.4f}/question) "
              f"writers={report['writer_calls_by_model']}")
    print(f"report → {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 13.2: Implement `evals/make_chart.py`**

```python
"""Bucketed-EX chart from one or two run_bird reports (the portfolio artifact).

    python evals/make_chart.py evals/report_rag.json evals/report_full_schema.json
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    paths = sys.argv[1:] or ["evals/report_rag.json"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = []
    for p in paths:
        with open(p) as f:
            reports.append(json.load(f))

    buckets = sorted({b for r in reports for b in r["buckets"]})
    series = []  # (label, [ex per bucket])
    for r in reports:
        for kind in ("ex_single_shot", "ex_final"):
            label = f"{r['arm']} {'single-shot' if kind == 'ex_single_shot' else 'critic-loop'}"
            series.append((label, [r["buckets"].get(b, {}).get(kind) or 0.0
                                   for b in buckets]))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(series)
    for i, (label, vals) in enumerate(series):
        xs = [j + i * width for j in range(len(buckets))]
        bars = ax.bar(xs, vals, width=width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(buckets))])
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Execution Accuracy (EX)")
    ax.set_title("QueryMate — EX by difficulty: retrieval & self-correction lift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "evals/ex_chart.png"
    fig.savefig(out, dpi=160)
    print(f"chart → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 13.3: Smoke both on the demo subset (needs `ANTHROPIC_API_KEY` in `.env`)**

This is the FIRST live-LLM run of the project. If no key is available, stop and flag it — do not mark this step done.

Run: `uv run python scripts/ingest_schemas.py --demo && uv run python evals/run_bird.py --arm rag --db-root data && uv run python evals/run_bird.py --arm full-schema --db-root data && uv run python evals/make_chart.py evals/report_rag.json evals/report_full_schema.json`
Expected: both runs print EX lines (demo subset: expect EX ≥ 0.8 — these are easy questions; cost ≈ a few cents), `evals/ex_chart.png` written.

- [ ] **Step 13.4: Commit**

```bash
git add evals/run_bird.py evals/make_chart.py
git commit -m "feat: bucketed two-arm BIRD eval with cost accounting + EX chart"
```

---

### Task 14: README + wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 14.1: Update `README.md`**

Make these targeted edits:

1. Title: `# QueryMate — NL→SQL analytics copilot (Phase 0)` → `(Phase 1)`.
2. Replace the intro paragraph's last sentence (`This repo is **Phase 0** … come in later phases.`) with:

```markdown
This repo is at **Phase 1** of the build spec: schema-card RAG (sqlite-vec +
fastembed, fully local), an advisory planner, Haiku/Sonnet/Opus model routing
with per-call cost accounting, retrieval-aware repair, and an eval that reports
execution accuracy by difficulty bucket plus schema-retrieval recall@k.
Clarifier, explainer, LLMOps dashboards, and the CI gate come in later phases.
```

3. Replace the architecture diagram block with:

```
START → retrieve → plan → write_sql → execute ──(ok / give_up)──► END
           ▲                  ▲           │
           │                  │     (error, attempts < max)
           │                  │           ▼
           └──(widen k, once)─ critic ────┘
```

and add below the existing component bullets:

```markdown
- **retrieve** (`querymate/retriever.py`) — schema-card RAG: fastembed
  (`bge-small-en-v1.5`, local — no API key) over **sqlite-vec**, top-k per
  database + FK 1-hop expansion. When the DB reports an unknown table/column,
  the critic **widens retrieval** (k×2, once) instead of just re-prompting.
- **plan** (`querymate/llm.py:plan`) — one Haiku call sketches tables/joins/
  aggregations; advisory only. Its join/aggregation count drives **model
  routing** (`querymate/router.py`): Haiku for simple lookups, Sonnet
  otherwise, Opus on the final attempt. Every call's tokens/cost/latency land
  in the run's `cost_log`.
```

4. In **Setup**, append after the `make_demo_db.py` line:

```bash
uv run python scripts/ingest_schemas.py --demo   # build the demo schema index
```

5. Replace the **Run the execution-accuracy eval** section body with:

```markdown
```bash
# one-time: BIRD dev set (~1.2GB) + index + stratified subsets
uv run python scripts/fetch_bird.py
uv run python scripts/ingest_schemas.py --db-root data/bird/dev_databases
uv run python evals/make_subset.py --dev data/bird/dev.json

# retrieval quality — no LLM calls, free, full dev set
uv run python evals/run_recall.py --subset data/bird/dev.json --ks 3 5 10

# execution accuracy — both arms on the stratified subset, then the chart
uv run python evals/run_bird.py --subset evals/data/bird_stratified.json --db-root data/bird/dev_databases --arm rag
uv run python evals/run_bird.py --subset evals/data/bird_stratified.json --db-root data/bird/dev_databases --arm full-schema
uv run python evals/make_chart.py evals/report_rag.json evals/report_full_schema.json
```

Reports land in `evals/report_<arm>.json` (EX by difficulty bucket,
self-correction lift, cost/question by routing tier) and
`evals/recall_report.json`; the chart in `evals/ex_chart.png`.
```

6. Update **Run the tests** command list to all five files:

```bash
for t in tests/test_*.py; do uv run python "$t"; done
```

7. In **What's deliberately *not* here yet**, delete `schema-card RAG + retrieval recall@k, a planner,` (now built) so it reads: explainer + faithfulness judge, Langfuse tracing, CI regression gate, red-team suite, pgvector backend.
8. Add a **Troubleshooting** line at the end:

```markdown
> `enable_load_extension` AttributeError → your Python lacks SQLite extension
> support; use a uv-managed interpreter (`uv python install 3.13 && uv sync`).
```

- [ ] **Step 14.2: Full suite, final check**

Run: `for t in tests/test_*.py; do uv run python "$t" || break; done`
Expected: all five files pass (existing 25 + new ~21).

- [ ] **Step 14.3: Commit + push**

```bash
git add README.md
git commit -m "docs: Phase 1 README — RAG pipeline, routing, eval runbook"
git push
```

---

## Runbook — generating the Phase 1 numbers (after all tasks; needs API credits)

Not part of the code tasks — run with Isaac's go-ahead, in order, costs noted:

1. `uv run python scripts/fetch_bird.py` — ~1.2GB download, free.
2. `uv run python scripts/ingest_schemas.py --db-root data/bird/dev_databases` — local embedding of ~95 DBs, free, ~minutes.
3. `uv run python evals/make_subset.py --dev data/bird/dev.json` — instant.
4. `uv run python evals/run_recall.py --subset data/bird/dev.json --ks 3 5 10` — full 1,534, **free** (no LLM).
5. Smoke: `uv run python evals/run_bird.py --subset evals/data/bird_smoke.json --db-root data/bird/dev_databases --arm rag` — ~30 questions, ~$1.
6. Both arms on `evals/data/bird_stratified.json` (~300 × 2 runs × 2 arms ≈ $10–30, 1–2h). Watch the smoke run's `cost_usd_per_question` before committing to this.
7. `uv run python evals/make_chart.py evals/report_rag.json evals/report_full_schema.json` → `evals/ex_chart.png` — the build-in-public artifact.

## Self-review notes

- Spec coverage: retrieval stack (T2–5, 10), planner (T8–9), routing + cost (T6–7), evidence fix (T7), retrieval-aware repair (T9), CLI fallback UX (T11), recall@k free on full dev (T12), stratified buckets + arms + chart (T13), README (T14). ✔
- Phase-0 escalation semantics preserved verbatim in `pick_model` (`attempts >= max_attempts - 1`, never when `max_attempts == 0`). ✔
- Both eval arms flow through `retrieve_node` (full-schema arm = `use_retrieval: False` fallback), so writer prompting is identical across arms — retrieval lift is attributable. ✔
- `cost_log` uses a LangGraph `operator.add` reducer; every node returns a list, including the second `_predict` invocation starting fresh (`"cost_log": []` in the initial state). ✔







