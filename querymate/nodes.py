"""LangGraph nodes: write_sql → execute → (critic → write_sql) loop."""

from __future__ import annotations

from . import llm
from .executor import ExecError, validate_and_run
from .settings import settings
from .state import QueryState


def write_sql_node(state: QueryState) -> dict:
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", settings.max_attempts)
    # Escalate to the stronger model on the last allowed attempt.
    model = (
        settings.escalate_model
        if attempts >= max(max_attempts - 1, 0) and max_attempts > 0
        else settings.writer_model
    )
    sql = llm.write_sql(
        question=state["question"],
        schema=state["schema"],
        dialect=state.get("dialect", "sqlite"),
        model=model,
        repair_hint=state.get("repair_hint"),
        prev_sql=state.get("sql"),
    )
    return {"sql": sql}


def execute_node(state: QueryState) -> dict:
    try:
        validated, rows, columns = validate_and_run(
            state["sql"],
            state["db_path"],
            dialect=state.get("dialect", "sqlite"),
            auto_limit=state.get("auto_limit", settings.auto_limit),
            max_rows=settings.max_rows,
            timeout_s=settings.statement_timeout_s,
        )
        return {
            "validated_sql": validated,
            "rows": rows,
            "columns": columns,
            "last_error": None,
        }
    except ExecError as e:
        return {"last_error": e.err, "rows": None, "columns": None}


def critic_node(state: QueryState) -> dict:
    attempts = state.get("attempts", 0) + 1
    err = state.get("last_error") or {"kind": "other", "message": "unknown"}
    if state.get("use_llm_critic"):
        hint = llm.diagnose(
            question=state["question"],
            schema=state["schema"],
            dialect=state.get("dialect", "sqlite"),
            sql=state.get("sql", ""),
            error=err["message"],
            model=settings.writer_model,
        )
    else:
        hint = f"{err['kind']}: {err['message']}"
    return {"attempts": attempts, "repair_hint": hint}


def route_after_execute(state: QueryState) -> str:
    if state.get("last_error") is None:
        return "ok"
    if state.get("attempts", 0) >= state.get("max_attempts", settings.max_attempts):
        return "give_up"
    return "critic"
