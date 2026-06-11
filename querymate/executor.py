"""Sandboxed execution: read-only SQLite + a wall-clock statement timeout.

The connection is opened ``mode=ro`` so the engine itself refuses writes even if
something slipped past the parser. A progress handler aborts long/looping
queries. Errors are returned structured so the critic can act on them.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from .state import DBError
from .validator import UnsafeSQL, validate_sql


class ExecError(Exception):
    def __init__(self, err: DBError) -> None:
        self.err = err
        super().__init__(err["message"])


def _connect_ro(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _classify(msg: str) -> str:
    m = msg.lower()
    if "no such table" in m:
        return "unknown_table"
    if "no such column" in m:
        return "unknown_column"
    if "syntax error" in m or "near " in m:
        return "syntax"
    if "interrupt" in m or "timeout" in m:
        return "timeout"
    return "other"


def run_query(
    sql: str,
    db_path: str,
    *,
    max_rows: int = 5000,
    timeout_s: float = 10.0,
) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Execute an already-validated query read-only. Raises ``ExecError``."""
    con = _connect_ro(db_path)
    try:
        deadline = time.monotonic() + timeout_s
        # Returning non-zero from the handler aborts the statement → OperationalError.
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [tuple(r) for r in cur.fetchmany(max_rows)]
        return rows, columns
    except sqlite3.OperationalError as e:
        raise ExecError({"kind": _classify(str(e)), "message": str(e)})
    except sqlite3.Error as e:
        raise ExecError({"kind": "other", "message": str(e)})
    finally:
        con.close()


def validate_and_run(
    sql: str,
    db_path: str,
    *,
    dialect: str = "sqlite",
    auto_limit: bool = True,
    max_rows: int = 5000,
    timeout_s: float = 10.0,
) -> tuple[str, list[tuple[Any, ...]], list[str]]:
    try:
        validated = validate_sql(
            sql, dialect=dialect, auto_limit=auto_limit, max_rows=max_rows
        )
    except UnsafeSQL as e:
        raise ExecError({"kind": "unsafe", "message": f"{e.kind}: {e}"})
    rows, columns = run_query(
        validated, db_path, max_rows=max_rows, timeout_s=timeout_s
    )
    return validated, rows, columns


def schema_text(db_path: str) -> str:
    """Phase-0 schema context: the DB's CREATE TABLE statements.

    Phase 1 replaces this with retrieved schema cards (RAG over pgvector).
    """
    con = _connect_ro(db_path)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
        return "\n\n".join(r[0] for r in rows)
    finally:
        con.close()
