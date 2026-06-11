"""Execution-accuracy comparison.

Order-insensitive multiset match on returned rows — a local approximation of the
BIRD/Spider Execution Accuracy (EX) metric. For the real benchmark, defer to the
official BIRD/Spider eval scripts; this is the fast local signal used in Phase 0.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional


def _norm_cell(c: Any) -> tuple:
    if c is None:
        return ("none",)
    if isinstance(c, bool):
        return ("num", float(int(c)))
    if isinstance(c, int):
        return ("num", float(c))
    if isinstance(c, (float, Decimal)):
        return ("num", round(float(c), 6))
    return ("str", str(c))


def _norm_rows(rows) -> list:
    return sorted(tuple(_norm_cell(c) for c in row) for row in rows)


def execution_match(
    pred_rows: Optional[list], gold_rows: Optional[list]
) -> bool:
    if pred_rows is None or gold_rows is None:
        return False
    return _norm_rows(pred_rows) == _norm_rows(gold_rows)
