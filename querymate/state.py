from __future__ import annotations

from typing import Any, Optional, TypedDict


class DBError(TypedDict):
    # unsafe | parse_error | syntax | unknown_table | unknown_column | timeout | other
    kind: str
    message: str


class QueryState(TypedDict, total=False):
    """State carried through the LangGraph loop.

    Phase 0 uses a flat ``schema`` string (the demo DB's CREATE statements).
    Phase 1 replaces it with retrieved schema cards (RAG).
    """

    question: str
    schema: str
    dialect: str
    db_path: str

    sql: Optional[str]            # last query the writer produced
    validated_sql: Optional[str]  # post-validation (LIMIT-injected) query that ran
    attempts: int                 # repair attempts consumed
    max_attempts: int
    last_error: Optional[DBError]
    repair_hint: Optional[str]    # critic → writer

    rows: Optional[list[tuple[Any, ...]]]
    columns: Optional[list[str]]

    # knobs
    auto_limit: bool
    use_llm_critic: bool
