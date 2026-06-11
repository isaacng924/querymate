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
