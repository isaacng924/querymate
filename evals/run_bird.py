"""BIRD-subset execution-accuracy eval.

Runs each NL question through the QueryMate graph and scores the result against
the gold query by Execution Accuracy (EX). Reports two numbers — single-shot EX
(no critic loop) and final EX (with the loop) — and their difference, the
**self-correction lift**: the headline portfolio number.

Input is a JSON list of BIRD-format items: {"db_id", "question", "SQL", ...}.
Defaults to the bundled demo subset; point ``--subset`` / ``--db-root`` at the
real BIRD dev set to get a benchmark number.

    python evals/run_bird.py
    python evals/run_bird.py --subset bird/dev.json --db-root bird/dev_databases
"""

from __future__ import annotations

import argparse
import json
import os

from querymate.executor import run_query, schema_text
from querymate.graph import get_graph
from querymate.settings import settings

from evals.compare import execution_match


def _db_path(db_id: str, root: str) -> str:
    candidates = [
        os.path.join(root, f"{db_id}.sqlite"),
        os.path.join(root, "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(root, db_id, f"{db_id}.sqlite"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"no sqlite db for '{db_id}' under '{root}' (tried {candidates})")


def _gold_rows(gold_sql: str, db_path: str) -> list:
    # Gold SQL comes from the dataset (trusted) — run it directly, read-only.
    return run_query(
        gold_sql, db_path,
        max_rows=settings.max_rows, timeout_s=settings.statement_timeout_s,
    )[0]


def _predict(question: str, schema: str, db_path: str, max_attempts: int) -> dict:
    state = {
        "question": question, "db_path": db_path, "dialect": "sqlite",
        "schema": schema, "attempts": 0, "max_attempts": max_attempts,
        "auto_limit": False,  # don't truncate a large gold result set during eval
        "use_llm_critic": False,
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate BIRD-subset EX eval")
    ap.add_argument("--subset", default="evals/data/sample_bird_subset.json")
    ap.add_argument("--db-root", default="data")
    ap.add_argument("--report", default="evals/report.json")
    args = ap.parse_args()

    with open(args.subset) as f:
        items = json.load(f)

    n = len(items)
    ex_final = ex_single = errors = 0
    detail = []

    for i, it in enumerate(items, 1):
        q, gold, db_id = it["question"], it["SQL"], it["db_id"]
        try:
            db_path = _db_path(db_id, args.db_root)
            schema = schema_text(db_path)
            gold_rows = _gold_rows(gold, db_path)
        except Exception as e:
            errors += 1
            print(f"[{i}/{n}] SKIP {db_id}: {e}")
            continue

        try:
            out = _predict(q, schema, db_path, settings.max_attempts)
            final_ok = out.get("last_error") is None and execution_match(out.get("rows"), gold_rows)
            attempts = out.get("attempts", 0)
        except Exception as e:
            final_ok, attempts = False, 0
            print(f"   predict error (final): {e}")

        try:
            out1 = _predict(q, schema, db_path, 0)  # max_attempts=0 → no repair
            single_ok = out1.get("last_error") is None and execution_match(out1.get("rows"), gold_rows)
        except Exception as e:
            single_ok = False
            print(f"   predict error (single-shot): {e}")

        ex_final += int(final_ok)
        ex_single += int(single_ok)
        print(
            f"[{i}/{n}] {db_id}: single={'PASS' if single_ok else 'fail'} "
            f"final={'PASS' if final_ok else 'fail'} attempts={attempts}  | {q}"
        )
        detail.append(
            {"question": q, "db_id": db_id, "single_ok": single_ok,
             "final_ok": final_ok, "attempts": attempts}
        )

    scored = n - errors
    report = {
        "n": n, "scored": scored, "errors": errors,
        "ex_single_shot": round(ex_single / scored, 4) if scored else None,
        "ex_final": round(ex_final / scored, 4) if scored else None,
        "self_correction_lift": round((ex_final - ex_single) / scored, 4) if scored else None,
        "items": detail,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== QueryMate execution accuracy ===")
    print(f"scored: {scored}/{n}  (errors: {errors})")
    if scored:
        print(f"single-shot EX      : {report['ex_single_shot']:.3f}")
        print(f"final EX (w/ critic): {report['ex_final']:.3f}")
        print(f"self-correction lift: {report['self_correction_lift']:+.3f}")
    print(f"report → {args.report}")


if __name__ == "__main__":
    main()
