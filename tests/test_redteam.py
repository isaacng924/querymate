"""Red-team safety suite: every corpus attack must die at the trust boundary.

``must_reject`` entries must be refused by the static validator (UnsafeSQL).
``must_fail_execution`` entries pass parsing (single SELECT) but must fail at
the sandboxed executor (ExecError) — defence in depth. Prints the safety
pass-rate; the suite REQUIRES 100%.

    python tests/test_redteam.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.executor import ExecError, validate_and_run  # noqa: E402
from querymate.validator import UnsafeSQL, validate_sql  # noqa: E402

_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evals", "data", "redteam_corpus.json"
)
_DEMO_DB = os.path.join(os.path.dirname(__file__), "..", "data", "demo_store.sqlite")

with open(_CORPUS_PATH) as f:
    CORPUS = json.load(f)


def test_corpus_shape():
    assert CORPUS["must_reject"] and CORPUS["must_fail_execution"]
    for section in CORPUS.values():
        for item in section:
            assert item["category"] and "sql" in item


def test_validator_rejects_all_must_reject():
    passed, failed = 0, []
    for item in CORPUS["must_reject"]:
        try:
            validate_sql(item["sql"])
            failed.append(f"{item['category']}: {item['sql']!r}")
        except UnsafeSQL:
            passed += 1
    total = len(CORPUS["must_reject"])
    print(f"  validator safety pass-rate: {passed}/{total}")
    assert not failed, f"validator ACCEPTED unsafe SQL: {failed}"


def test_executor_blocks_all_must_fail_execution():
    assert os.path.exists(_DEMO_DB), "run scripts/make_demo_db.py first"
    passed, failed = 0, []
    for item in CORPUS["must_fail_execution"]:
        try:
            validate_and_run(item["sql"], _DEMO_DB, auto_limit=False)
            failed.append(f"{item['category']}: {item['sql']!r}")
        except ExecError:
            passed += 1
    total = len(CORPUS["must_fail_execution"])
    print(f"  executor safety pass-rate: {passed}/{total}")
    assert not failed, f"executor RAN unsafe SQL: {failed}"


def test_readonly_backstop_direct_write():
    # Even a hypothetically validated write dies on the mode=ro connection.
    from querymate.executor import run_query

    assert os.path.exists(_DEMO_DB), "run scripts/make_demo_db.py first"
    try:
        run_query("UPDATE products SET price = 0", _DEMO_DB)
        raised = False
    except ExecError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} redteam tests passed")
