from __future__ import annotations

import argparse
import sys

from .executor import schema_text
from .graph import get_graph
from .settings import settings


def run_question(
    question: str,
    *,
    db_path: str | None = None,
    dialect: str = "sqlite",
    max_attempts: int | None = None,
) -> dict:
    db_path = db_path or settings.demo_db_path
    state = {
        "question": question,
        "db_path": db_path,
        "dialect": dialect,
        "schema": schema_text(db_path),
        "attempts": 0,
        "max_attempts": settings.max_attempts if max_attempts is None else max_attempts,
        "auto_limit": settings.auto_limit,
        "use_llm_critic": False,
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate — NL→SQL copilot (Phase 0)")
    ap.add_argument("question", help="a question in plain English")
    ap.add_argument("--db", default=None, help="path to a SQLite DB (default: demo)")
    args = ap.parse_args()

    out = run_question(args.question, db_path=args.db)

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
    print(f"\n({len(rows)} row(s); repair attempts={out.get('attempts', 0)})")


if __name__ == "__main__":
    main()
