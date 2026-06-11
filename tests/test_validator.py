"""Plain-assert tests for the trust boundary (no pytest required).

    python tests/test_validator.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.validator import UnsafeSQL, validate_sql  # noqa: E402


def test_select_passes_and_gets_limit():
    out = validate_sql("SELECT id FROM t", auto_limit=True, max_rows=1000)
    assert "select" in out.lower() and "limit" in out.lower(), out


def test_existing_limit_not_doubled():
    out = validate_sql("SELECT id FROM t LIMIT 5", auto_limit=True, max_rows=1000)
    assert out.lower().count("limit") == 1, out


def test_no_limit_when_disabled():
    out = validate_sql("SELECT id FROM t", auto_limit=False)
    assert "limit" not in out.lower(), out


def test_cte_select_passes():
    out = validate_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x", auto_limit=False)
    assert "with" in out.lower() and "select" in out.lower(), out


def test_union_passes():
    out = validate_sql("SELECT 1 UNION SELECT 2", auto_limit=False)
    assert "union" in out.lower(), out


def _expect_unsafe(sql, kind=None):
    try:
        validate_sql(sql)
    except UnsafeSQL as e:
        if kind:
            assert e.kind == kind, f"{sql!r} → kind {e.kind!r} (wanted {kind!r})"
        return
    raise AssertionError(f"expected UnsafeSQL for: {sql!r}")


def test_rejects_drop():
    _expect_unsafe("DROP TABLE customers", "non_select")


def test_rejects_insert():
    _expect_unsafe("INSERT INTO t VALUES (1)", "non_select")


def test_rejects_update():
    _expect_unsafe("UPDATE t SET x = 1")


def test_rejects_delete():
    _expect_unsafe("DELETE FROM t")


def test_rejects_multiple_statements():
    _expect_unsafe("SELECT 1; DROP TABLE t", "multiple_statements")


def test_rejects_pragma():
    _expect_unsafe("PRAGMA table_info(t)")


def test_rejects_attach():
    _expect_unsafe("ATTACH DATABASE 'x.db' AS y")


def test_rejects_empty():
    _expect_unsafe("", "empty")


def test_rejects_garbage():
    _expect_unsafe("this is not sql ;;;")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} validator tests passed")
