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
