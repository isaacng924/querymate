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
