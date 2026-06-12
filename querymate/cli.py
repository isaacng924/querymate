from __future__ import annotations

import argparse
import os
import sys

from .executor import schema_text
from .graph import get_graph
from .settings import settings


def _ensure_demo_index() -> None:
    """Clone-and-run: build the demo index on first use (downloads the
    embedding model one time)."""
    from .card_index import CardIndex
    from .embedder import FastEmbedder

    if not os.path.exists(settings.schema_index_path):
        print("(building schema index for the demo DB — first run only)")
        os.makedirs(os.path.dirname(settings.schema_index_path) or ".", exist_ok=True)
    idx = CardIndex(settings.schema_index_path, embedder=FastEmbedder())
    try:
        if not idx.has_db("demo_store"):
            from .schema_cards import extract_cards

            idx.add_cards(extract_cards(settings.demo_db_path, db_id="demo_store"))
    finally:
        idx.close()


def run_question(
    question: str,
    *,
    db_path: str | None = None,
    dialect: str = "sqlite",
    max_attempts: int | None = None,
    use_retrieval: bool = True,
) -> dict:
    db_path = db_path or settings.demo_db_path
    db_id = os.path.splitext(os.path.basename(db_path))[0]
    if use_retrieval and db_id == "demo_store":
        _ensure_demo_index()
    state = {
        "question": question,
        "db_path": db_path,
        "db_id": db_id,
        "dialect": dialect,
        "schema": "" if use_retrieval else schema_text(db_path),
        "use_retrieval": use_retrieval,
        "use_planner": True,
        "retrieval_k": settings.retrieval_k,
        "attempts": 0,
        "max_attempts": settings.max_attempts if max_attempts is None else max_attempts,
        "auto_limit": settings.auto_limit,
        "use_llm_critic": False,
        "cost_log": [],
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate — NL→SQL copilot (Phase 2)")
    ap.add_argument("question", help="a question in plain English")
    ap.add_argument("--db", default=None, help="path to a SQLite DB (default: demo)")
    ap.add_argument("--no-rag", action="store_true",
                    help="skip retrieval; prompt with the full schema")
    ap.add_argument("--no-explain", action="store_true",
                    help="skip the natural-language answer line")
    args = ap.parse_args()

    out = run_question(args.question, db_path=args.db,
                       use_retrieval=not args.no_rag)

    print("\nSQL:\n" + (out.get("validated_sql") or out.get("sql") or "(none)"))
    if out.get("last_error"):
        print(
            f"\n[failed after {out.get('attempts', 0)} repair attempt(s)] "
            f"{out['last_error']}"
        )
        sys.exit(1)

    cols = out.get("columns") or []
    rows = out.get("rows") or []
    print("\nResult: " + (" | ".join(cols) if cols else "(no columns)"))
    for r in rows[:50]:
        print("  " + " | ".join(str(c) for c in r))

    cost_entries = list(out.get("cost_log", []))
    if not args.no_explain:
        from . import llm

        answer, entry = llm.explain(
            question=args.question, columns=cols, rows=rows,
            model=settings.fast_model,
        )
        if entry:
            cost_entries.append(entry)
        if answer:
            print(f"\nAnswer: {answer}")

    cost = sum(e.get("cost_usd", 0.0) for e in cost_entries)
    models = [e["model"] for e in cost_entries if e.get("purpose") == "writer"]
    print(
        f"\n({len(rows)} row(s); repair attempts={out.get('attempts', 0)}; "
        f"writer={models[-1] if models else '-'}; cost=${cost:.4f})"
    )


if __name__ == "__main__":
    main()
