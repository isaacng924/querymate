"""Anthropic SDK calls for the writer and (optional) critic.

Uses structured outputs to parse the SQL cleanly, prompt caching on the stable
schema/system prefix, and adaptive thinking on the writer. Models are passed in
by the caller (routed per ``settings``).
"""

from __future__ import annotations

import json
import time
from typing import Optional

import anthropic
from .settings import call_cost

_client: Optional[anthropic.Anthropic] = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def _usage_entry(resp, model: str, t0: float, *, purpose: str) -> dict:
    u = resp.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    return {
        "purpose": purpose,                    # writer | critic | planner
        "model": model,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "cost_usd": call_cost(model, u.input_tokens, u.output_tokens,
                              cache_read, cache_write),
        "latency_s": round(time.monotonic() - t0, 3),
    }


_SQL_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single read-only SQLite SELECT that answers the question.",
            }
        },
        "required": ["sql"],
        "additionalProperties": False,
    },
}

_PLAN_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "tables": {"type": "array", "items": {"type": "string"},
                       "description": "Tables the query needs."},
            "join_count": {"type": "integer",
                           "description": "Number of JOINs expected."},
            "aggregations": {"type": "array", "items": {"type": "string"},
                             "description": "Aggregate functions needed (COUNT, SUM...)."},
            "filters": {"type": "array", "items": {"type": "string"},
                        "description": "WHERE/HAVING conditions in plain words."},
        },
        "required": ["tables", "join_count", "aggregations", "filters"],
        "additionalProperties": False,
    },
}

_JUDGE_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "faithful": {
                "type": "boolean",
                "description": "True only if EVERY claim in the answer is "
                               "directly supported by the provided rows.",
            },
            "reason": {"type": "string",
                       "description": "One sentence justifying the verdict."},
        },
        "required": ["faithful", "reason"],
        "additionalProperties": False,
    },
}


def _system(schema: str, dialect: str) -> str:
    return (
        f"You write a single read-only {dialect} SELECT query that answers the "
        "user's question.\n"
        "Rules:\n"
        "- Output exactly ONE SELECT statement. Never INSERT/UPDATE/DELETE/DROP/"
        "ALTER/CREATE/PRAGMA/ATTACH.\n"
        "- Use only tables and columns that appear in the schema below. Never "
        "invent names.\n"
        "- Prefer explicit JOINs and qualified column names.\n\n"
        f"Database schema ({dialect}):\n{schema}"
    )


def write_sql(
    *,
    question: str,
    schema: str,
    dialect: str,
    model: str,
    evidence: Optional[str] = None,
    plan: Optional[str] = None,
    repair_hint: Optional[str] = None,
    prev_sql: Optional[str] = None,
    max_tokens: int = 1500,
) -> tuple[str, dict]:
    # cache_control on the schema/system prefix: stable across the writer/critic
    # calls for a question (and across questions on the same DB).
    system = [
        {
            "type": "text",
            "text": _system(schema, dialect),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    parts = [f"Question: {question}"]
    if evidence:
        parts.append(f"Hint (domain evidence): {evidence}")
    if plan:
        parts.append(f"Query plan (advisory):\n{plan}")
    if prev_sql:
        parts.append(f"\nYour previous query was:\n{prev_sql}")
    if repair_hint:
        parts.append(f"\nIt failed. Fix it. Diagnosis:\n{repair_hint}")

    t0 = time.monotonic()
    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        output_config={"format": _SQL_FORMAT},
    )
    entry = _usage_entry(resp, model, t0, purpose="writer")
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        return json.loads(text)["sql"].strip(), entry
    except Exception:
        return text.strip(), entry


def diagnose(
    *,
    question: str,
    schema: str,
    dialect: str,
    sql: str,
    error: str,
    model: str,
    max_tokens: int = 400,
) -> tuple[str, dict]:
    """Richer critic (optional). The Phase-0 default feeds the raw DB error back
    as the hint; set ``use_llm_critic`` to call this for a real diagnosis."""
    t0 = time.monotonic()
    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _system(schema, dialect),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nFailing query:\n{sql}\n"
                    f"Database error: {error}\n"
                    "In 1-2 sentences, diagnose the root cause and how to fix it. "
                    "Do not write SQL."
                ),
            }
        ],
    )
    entry = _usage_entry(resp, model, t0, purpose="critic")
    return next((b.text for b in resp.content if b.type == "text"), error), entry


def plan(
    *,
    question: str,
    schema: str,
    dialect: str,
    model: str,
    evidence: Optional[str] = None,
    max_tokens: int = 500,
) -> tuple[Optional[dict], Optional[dict]]:
    """Advisory query plan. Returns (plan_info, usage_entry); (None, entry-or-None)
    on any failure — the loop must never depend on the planner."""
    parts = [f"Question: {question}"]
    if evidence:
        parts.append(f"Hint (domain evidence): {evidence}")
    parts.append(
        "Plan the SQL query: which tables, how many joins, which aggregate "
        "functions, which filters. Do not write SQL."
    )
    t0 = time.monotonic()
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": _system(schema, dialect),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": "\n".join(parts)}],
            output_config={"format": _PLAN_FORMAT},
        )
        entry = _usage_entry(resp, model, t0, purpose="planner")
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        info = json.loads(text)
        if not isinstance(info, dict) or "tables" not in info:
            return None, entry
        return info, entry
    except Exception:
        return None, None


def explain(
    *,
    question: str,
    columns: list[str],
    rows: list,
    model: str,
    max_tokens: int = 300,
) -> tuple[Optional[str], Optional[dict]]:
    """1-2 sentence answer narration over the result rows. Advisory — returns
    (None, None) on any failure; the caller never depends on it."""
    sample = rows[:20]
    t0 = time.monotonic()
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": (
                    "You summarise SQL query results. Answer the user's question "
                    "in 1-2 sentences using ONLY the rows provided. State numbers "
                    "exactly as they appear. If the rows don't answer the "
                    "question, say so."
                ),
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\nColumns: {columns}\n"
                    f"Rows (first {len(sample)} of {len(rows)}): {sample}"
                ),
            }],
        )
        entry = _usage_entry(resp, model, t0, purpose="explainer")
        return next((b.text for b in resp.content if b.type == "text"), None), entry
    except Exception:
        return None, None


def judge_faithfulness(
    *,
    question: str,
    answer: str,
    columns: list[str],
    rows: list,
    model: str,
    max_tokens: int = 300,
) -> tuple[Optional[dict], Optional[dict]]:
    """LLM-as-judge: does the answer only state what the rows support?
    Returns ({faithful, reason}, usage_entry); (None, None) on failure."""
    sample = rows[:20]
    t0 = time.monotonic()
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": (
                    "You are a strict faithfulness judge. Given a question, the "
                    "result rows, and an answer, decide whether every claim in "
                    "the answer is supported by the rows. Unsupported numbers, "
                    "invented entities, or over-claims mean NOT faithful."
                ),
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\nColumns: {columns}\n"
                    f"Rows (first {len(sample)} of {len(rows)}): {sample}\n"
                    f"Answer to judge: {answer}"
                ),
            }],
            output_config={"format": _JUDGE_FORMAT},
        )
        entry = _usage_entry(resp, model, t0, purpose="judge")
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        verdict = json.loads(text)
        if not isinstance(verdict, dict) or "faithful" not in verdict:
            return None, entry
        return verdict, entry
    except Exception:
        return None, None
