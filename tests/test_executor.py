"""Plain-assert tests for the executor + validate_and_run.

    python tests/test_executor.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.executor import ExecError, run_query, validate_and_run  # noqa: E402


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE t(id INTEGER, name TEXT);"
        "INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c');"
    )
    con.commit()
    con.close()
    return path


def test_count_query():
    db = _tmp_db()
    try:
        rows, cols = run_query("SELECT COUNT(*) FROM t", db)
        assert rows == [(3,)], rows
    finally:
        os.remove(db)


def test_select_rows():
    db = _tmp_db()
    try:
        _, rows, cols = validate_and_run("SELECT name FROM t ORDER BY id", db, auto_limit=False)
        assert rows == [("a",), ("b",), ("c",)], rows
        assert cols == ["name"], cols
    finally:
        os.remove(db)


def test_unknown_table_error():
    db = _tmp_db()
    try:
        try:
            validate_and_run("SELECT * FROM nope", db)
        except ExecError as e:
            assert e.err["kind"] == "unknown_table", e.err
            return
        raise AssertionError("expected ExecError")
    finally:
        os.remove(db)


def test_unsafe_blocked_before_execution():
    db = _tmp_db()
    try:
        try:
            validate_and_run("DROP TABLE t", db)
        except ExecError as e:
            assert e.err["kind"] == "unsafe", e.err
            # table must still be there
            rows, _ = run_query("SELECT COUNT(*) FROM t", db)
            assert rows == [(3,)], rows
            return
        raise AssertionError("expected ExecError")
    finally:
        os.remove(db)


def test_readonly_connection_rejects_writes():
    db = _tmp_db()
    try:
        # run_query opens read-only, so even a raw write statement can't mutate.
        try:
            run_query("INSERT INTO t VALUES (4,'d')", db)
        except ExecError:
            pass  # acceptable: engine refuses the write
        rows, _ = run_query("SELECT COUNT(*) FROM t", db)
        assert rows == [(3,)], rows
    finally:
        os.remove(db)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} executor tests passed")
