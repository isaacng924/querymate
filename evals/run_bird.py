"""BIRD execution-accuracy eval — Phase 1.

Per item, runs the graph twice (single-shot and critic-loop) and scores by EX.
Reports single-shot EX, final EX and the self-correction lift, all bucketed by
BIRD difficulty, plus cost/latency by routing tier.

Arms: --arm rag (schema-card retrieval; default) | --arm full-schema (Phase-0
behaviour). Run both on the same subset to get the retrieval lift.

    python evals/run_bird.py --subset evals/data/bird_smoke.json \
        --db-root data/bird/dev_databases --arm rag
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.executor import run_query
from querymate.graph import get_graph
from querymate.nodes import set_retriever
from querymate.retriever import Retriever
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


def _predict(it: dict, db_path: str, *, use_retrieval: bool, k: int,
             max_attempts: int) -> dict:
    state = {
        "question": it["question"],
        "evidence": it.get("evidence") or None,
        "db_path": db_path,
        "db_id": it["db_id"],
        "dialect": "sqlite",
        "use_retrieval": use_retrieval,
        "use_planner": True,
        "retrieval_k": k,
        "attempts": 0,
        "max_attempts": max_attempts,
        "auto_limit": False,  # don't truncate a large gold result set during eval
        "use_llm_critic": False,
        "cost_log": [],
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate BIRD EX eval (Phase 1)")
    ap.add_argument("--subset", default="evals/data/sample_bird_subset.json")
    ap.add_argument("--db-root", default="data")
    ap.add_argument("--arm", choices=["rag", "full-schema"], default="rag")
    ap.add_argument("--k", type=int, default=settings.retrieval_k)
    ap.add_argument("--index", default=settings.schema_index_path)
    ap.add_argument("--report", default=None,
                    help="default: evals/report_<arm>.json")
    args = ap.parse_args()
    report_path = args.report or f"evals/report_{args.arm.replace('-', '_')}.json"

    use_retrieval = args.arm == "rag"
    if use_retrieval:
        if not os.path.exists(args.index):
            raise SystemExit(
                f"schema index '{args.index}' missing — build it first:\n"
                f"  uv run python scripts/ingest_schemas.py --db-root {args.db_root} "
                f"--index {args.index}"
            )
        set_retriever(Retriever(CardIndex(args.index, embedder=FastEmbedder())))

    if not os.path.exists(args.subset):
        raise SystemExit(f"subset file '{args.subset}' not found")
    with open(args.subset) as f:
        items = json.load(f)

    n = len(items)
    errors = 0
    detail = []
    bucket_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "single": 0, "final": 0}
    )
    cost_total = 0.0
    tier_calls: dict[str, int] = defaultdict(int)

    for i, it in enumerate(items, 1):
        q, gold, db_id = it["question"], it["SQL"], it["db_id"]
        bucket = it.get("difficulty", "unknown")
        try:
            db_path = _db_path(db_id, args.db_root)
            gold_rows = _gold_rows(gold, db_path)
        except Exception as e:
            errors += 1
            print(f"[{i}/{n}] SKIP {db_id}: {e}")
            continue

        try:
            out = _predict(it, db_path, use_retrieval=use_retrieval, k=args.k,
                           max_attempts=settings.max_attempts)
            final_ok = out.get("last_error") is None and execution_match(
                out.get("rows"), gold_rows)
            attempts = out.get("attempts", 0)
            for entry in out.get("cost_log", []):
                cost_total += entry.get("cost_usd", 0.0)
                if entry.get("purpose") == "writer":
                    tier_calls[entry["model"]] += 1
        except Exception as e:
            final_ok, attempts, out = False, 0, {}
            print(f"   predict error (final): {e}")

        try:
            out1 = _predict(it, db_path, use_retrieval=use_retrieval, k=args.k,
                            max_attempts=0)  # no repair
            single_ok = out1.get("last_error") is None and execution_match(
                out1.get("rows"), gold_rows)
            for entry in out1.get("cost_log", []):
                cost_total += entry.get("cost_usd", 0.0)
                if entry.get("purpose") == "writer":
                    tier_calls[entry["model"]] += 1
        except Exception as e:
            single_ok = False
            print(f"   predict error (single-shot): {e}")

        b = bucket_stats[bucket]
        b["n"] += 1
        b["single"] += int(single_ok)
        b["final"] += int(final_ok)
        print(
            f"[{i}/{n}] {db_id} ({bucket}): single={'PASS' if single_ok else 'fail'} "
            f"final={'PASS' if final_ok else 'fail'} attempts={attempts}  | {q}"
        )
        detail.append({
            "question": q, "db_id": db_id, "difficulty": bucket,
            "single_ok": single_ok, "final_ok": final_ok, "attempts": attempts,
            "card_tables": out.get("card_tables", []),
        })

    scored = sum(b["n"] for b in bucket_stats.values())
    ex_single = sum(b["single"] for b in bucket_stats.values())
    ex_final = sum(b["final"] for b in bucket_stats.values())
    report = {
        "arm": args.arm,
        "k": args.k if use_retrieval else None,
        "n": n, "scored": scored, "errors": errors,
        "ex_single_shot": round(ex_single / scored, 4) if scored else None,
        "ex_final": round(ex_final / scored, 4) if scored else None,
        "self_correction_lift": round((ex_final - ex_single) / scored, 4) if scored else None,
        "buckets": {
            name: {
                "n": b["n"],
                "ex_single_shot": round(b["single"] / b["n"], 4),
                "ex_final": round(b["final"] / b["n"], 4),
            }
            for name, b in sorted(bucket_stats.items())
        },
        "cost_usd_total": round(cost_total, 4),
        "cost_usd_per_question": round(cost_total / scored, 6) if scored else None,
        "writer_calls_by_model": dict(tier_calls),
        "items": detail,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== QueryMate execution accuracy [{args.arm}] ===")
    print(f"scored: {scored}/{n}  (errors: {errors})")
    if scored:
        print(f"single-shot EX      : {report['ex_single_shot']:.3f}")
        print(f"final EX (w/ critic): {report['ex_final']:.3f}")
        print(f"self-correction lift: {report['self_correction_lift']:+.3f}")
        for name, b in report["buckets"].items():
            print(f"  {name:<12} n={b['n']:<4} single={b['ex_single_shot']:.3f} "
                  f"final={b['ex_final']:.3f}")
        print(f"cost: ${report['cost_usd_total']:.2f} total "
              f"(${report['cost_usd_per_question']:.4f}/question) "
              f"writers={report['writer_calls_by_model']}")
    print(f"report → {report_path}")


if __name__ == "__main__":
    main()
