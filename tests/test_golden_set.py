"""Golden-set integrity: every gold query parses, runs read-only on the demo
DB, and returns at least one row — a broken golden set must never reach the gate.

    python tests/test_golden_set.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.executor import run_query  # noqa: E402

_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evals", "data", "golden_set.json"
)
_DEMO_DB = os.path.join(os.path.dirname(__file__), "..", "data", "demo_store.sqlite")

with open(_GOLDEN_PATH) as f:
    GOLDEN = json.load(f)


def test_golden_size_and_shape():
    assert len(GOLDEN) == 40
    for it in GOLDEN:
        assert it["db_id"] == "demo_store"
        assert it["question"].strip() and it["SQL"].strip()
        assert it["category"] in {
            "simple", "filter", "aggregation", "business_term",
            "multi_step", "negation", "date", "ambiguous",
        }


def test_every_gold_query_executes_with_rows():
    assert os.path.exists(_DEMO_DB), "run scripts/make_demo_db.py first"
    empty, errors = [], []
    for it in GOLDEN:
        try:
            rows, _ = run_query(it["SQL"], _DEMO_DB)
        except Exception as e:
            errors.append(f"{it['question']!r}: {e}")
            continue
        if not rows or all(all(c is None for c in r) for r in rows):
            empty.append(it["question"])
    assert not errors, f"gold SQL failed to execute: {errors}"
    assert not empty, f"gold SQL returned no data: {empty}"


def test_every_category_represented():
    cats = {it["category"] for it in GOLDEN}
    assert {"business_term", "ambiguous", "multi_step", "negation", "date"} <= cats


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} golden-set tests passed")
