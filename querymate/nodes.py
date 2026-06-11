"""LangGraph nodes: retrieve → plan → write_sql → execute → (critic loop)."""

from __future__ import annotations

import os
from typing import Optional

from . import llm
from .executor import ExecError, schema_text, validate_and_run
from .retriever import Retriever
from .router import pick_model, should_widen
from .settings import settings
from .state import QueryState

_retriever: Optional[Retriever] = None


def set_retriever(r: Optional[Retriever]) -> None:
    """Inject a retriever (tests / eval with a custom index)."""
    global _retriever
    _retriever = r


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        from .card_index import CardIndex
        from .embedder import FastEmbedder

        if not os.path.exists(settings.schema_index_path):
            raise FileNotFoundError(
                f"schema index not found at '{settings.schema_index_path}' — build "
                "it first: uv run python scripts/ingest_schemas.py --demo  (or "
                "--db-root <dir> for BIRD)"
            )
        _retriever = Retriever(
            CardIndex(settings.schema_index_path, embedder=FastEmbedder())
        )
    return _retriever


def _db_id(state: QueryState) -> str:
    return state.get("db_id") or os.path.splitext(
        os.path.basename(state["db_path"])
    )[0]


def retrieve_node(state: QueryState) -> dict:
    if not state.get("use_retrieval", True):
        return {"schema": schema_text(state["db_path"]), "card_tables": []}
    k = state.get("retrieval_k", settings.retrieval_k)
    cards, schema = get_retriever().retrieve(
        state["question"], db_id=_db_id(state), k=k
    )
    return {"schema": schema, "card_tables": [c.table for c in cards]}


def _render_plan(info: dict) -> str:
    return (
        f"tables: {', '.join(info.get('tables', [])) or '-'}\n"
        f"joins: {info.get('join_count', 0)}\n"
        f"aggregations: {', '.join(info.get('aggregations', [])) or '-'}\n"
        f"filters: {'; '.join(info.get('filters', [])) or '-'}"
    )


def plan_node(state: QueryState) -> dict:
    if not state.get("use_planner", True):
        return {"plan": None, "plan_info": None}
    info, entry = llm.plan(
        question=state["question"],
        schema=state["schema"],
        dialect=state.get("dialect", "sqlite"),
        model=settings.planner_model,
        evidence=state.get("evidence"),
    )
    out: dict = {
        "plan_info": info,
        "plan": _render_plan(info) if info else None,
    }
    if entry:
        out["cost_log"] = [entry]
    return out


def write_sql_node(state: QueryState) -> dict:
    model = pick_model(
        state.get("plan_info"),
        attempts=state.get("attempts", 0),
        max_attempts=state.get("max_attempts", settings.max_attempts),
    )
    sql, entry = llm.write_sql(
        question=state["question"],
        schema=state["schema"],
        dialect=state.get("dialect", "sqlite"),
        model=model,
        repair_hint=state.get("repair_hint"),
        prev_sql=state.get("sql"),
        evidence=state.get("evidence"),
        plan=state.get("plan"),
    )
    return {"sql": sql, "cost_log": [entry]}


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
    widen = should_widen(
        err["kind"],
        widened=state.get("retrieval_widened", False),
        use_retrieval=state.get("use_retrieval", True),
    )
    if widen:
        return {
            "attempts": attempts,
            "widen_now": True,
            "retrieval_widened": True,
            "retrieval_k": state.get("retrieval_k", settings.retrieval_k) * 2,
            "repair_hint": (
                f"{err['kind']}: {err['message']} — schema context was widened; "
                "use only tables/columns in the (new) schema."
            ),
        }
    out: dict = {"attempts": attempts, "widen_now": False}
    if state.get("use_llm_critic"):
        hint, entry = llm.diagnose(
            question=state["question"],
            schema=state["schema"],
            dialect=state.get("dialect", "sqlite"),
            sql=state.get("sql", ""),
            error=err["message"],
            model=settings.writer_model,
        )
        out["repair_hint"] = hint
        out["cost_log"] = [entry]
    else:
        out["repair_hint"] = f"{err['kind']}: {err['message']}"
    return out


def route_after_execute(state: QueryState) -> str:
    if state.get("last_error") is None:
        return "ok"
    if state.get("attempts", 0) >= state.get("max_attempts", settings.max_attempts):
        return "give_up"
    return "critic"


def route_after_critic(state: QueryState) -> str:
    return "widen" if state.get("widen_now") else "rewrite"
