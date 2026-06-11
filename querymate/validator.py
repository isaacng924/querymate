"""The trust boundary: static SQL validation with sqlglot.

A "malicious" string can't do damage if it never reaches the engine as anything
other than a single read-only SELECT. This module is the parse-level half of
that guarantee (the read-only SQLite connection in ``executor.py`` is the other
half). It is the most security-critical code in QueryMate and the most heavily
unit-tested.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

# Allowed top-level statement shapes: a SELECT, a set operation (UNION/INTERSECT/
# EXCEPT), or a parenthesised subquery. Anything else (INSERT/UPDATE/DELETE/DROP/
# CREATE/ALTER/PRAGMA/ATTACH/...) is rejected.
_READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Subquery)

# Node types that must not appear anywhere in the tree (defence in depth — the
# top-level check already excludes them as roots, this catches nested oddities
# and the sqlglot ``Command`` catch-all for statements it can't classify).
_FORBIDDEN = tuple(
    t
    for t in (
        getattr(exp, name, None)
        for name in (
            "Insert", "Update", "Delete", "Drop", "Create", "Alter",
            "Command", "Pragma", "Attach", "Set", "Use",
        )
    )
    if t is not None
)


class UnsafeSQL(Exception):
    def __init__(self, kind: str, message: str = "") -> None:
        self.kind = kind
        super().__init__(message or kind)


def validate_sql(
    sql: str,
    *,
    dialect: str = "sqlite",
    auto_limit: bool = True,
    max_rows: int = 5000,
) -> str:
    """Return a safe, single read-only query string, or raise ``UnsafeSQL``.

    When ``auto_limit`` is set and the top statement is a LIMIT-less SELECT, a
    ``LIMIT max_rows`` is appended.
    """
    raw = (sql or "").strip()
    if not raw:
        raise UnsafeSQL("empty", "empty SQL")

    try:
        statements = [s for s in sqlglot.parse(raw, read=dialect) if s is not None]
    except Exception as e:  # sqlglot.ParseError and friends
        raise UnsafeSQL("parse_error", str(e))

    if len(statements) != 1:
        raise UnsafeSQL(
            "multiple_statements", f"expected exactly 1 statement, got {len(statements)}"
        )

    stmt = statements[0]
    if not isinstance(stmt, _READ_ONLY_ROOTS):
        raise UnsafeSQL(
            "non_select",
            f"only read-only SELECT queries are allowed (got {type(stmt).__name__})",
        )

    for item in stmt.walk():
        # sqlglot ≥30 yields bare nodes; older versions yield (node, parent, key).
        node = item[0] if isinstance(item, tuple) else item
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQL("non_select", f"forbidden node: {type(node).__name__}")

    if auto_limit and isinstance(stmt, exp.Select) and stmt.args.get("limit") is None:
        stmt = stmt.limit(max_rows)

    return stmt.sql(dialect=dialect)
