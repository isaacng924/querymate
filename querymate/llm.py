"""Anthropic SDK calls for the writer and (optional) critic.

Uses structured outputs to parse the SQL cleanly, prompt caching on the stable
schema/system prefix, and adaptive thinking on the writer. Models are passed in
by the caller (routed per ``settings``).
"""

from __future__ import annotations

import json
from typing import Optional

import anthropic

_client: Optional[anthropic.Anthropic] = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


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
    repair_hint: Optional[str] = None,
    prev_sql: Optional[str] = None,
    max_tokens: int = 1500,
) -> str:
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
    if prev_sql:
        parts.append(f"\nYour previous query was:\n{prev_sql}")
    if repair_hint:
        parts.append(f"\nIt failed. Fix it. Diagnosis:\n{repair_hint}")

    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        output_config={"format": _SQL_FORMAT},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        return json.loads(text)["sql"].strip()
    except Exception:
        return text.strip()


def diagnose(
    *,
    question: str,
    schema: str,
    dialect: str,
    sql: str,
    error: str,
    model: str,
    max_tokens: int = 400,
) -> str:
    """Richer critic (optional). The Phase-0 default feeds the raw DB error back
    as the hint; set ``use_llm_critic`` to call this for a real diagnosis."""
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
    return next((b.text for b in resp.content if b.type == "text"), error)
