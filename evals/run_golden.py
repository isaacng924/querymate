"""Golden-set eval + regression gate.

Runs the 40-question golden set through the graph (RAG arm), scores EX, and —
unless --no-judge — narrates each answer (explainer) and judges its
faithfulness. Compares against the committed baseline with --gate.

    python evals/run_golden.py                       # report only
    python evals/run_golden.py --gate                # exit 1 on regression
    python evals/run_golden.py --update-baseline     # refresh the baseline
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

from querymate import llm
from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.executor import run_query
from querymate.graph import get_graph
from querymate.nodes import set_retriever
from querymate.retriever import Retriever
from querymate.settings import settings

from evals.compare import execution_match

# Gate thresholds (the deploy-gate policy lives with the gate).
EX_DROP_MAX = 0.02      # absolute percentage points
COST_RISE_MAX = 0.15    # relative


def gate_failures(report: dict, baseline: dict) -> list[str]:
    """Pure comparison — returns human-readable failure reasons (empty = pass).
    Faithfulness is reported but NOT gated; safety is enforced in tier 1."""
    fails = []
    if report["ex"] < baseline["ex"] - EX_DROP_MAX:
        fails.append(
            f"EX regression: {report['ex']:.3f} < baseline {baseline['ex']:.3f} "
            f"- {EX_DROP_MAX:.2f} allowance"
        )
    b_cost = baseline.get("cost_usd_per_question")
    r_cost = report.get("cost_usd_per_question")
    if b_cost and r_cost and r_cost > b_cost * (1 + COST_RISE_MAX):
        fails.append(
            f"cost/question rose >{COST_RISE_MAX:.0%}: ${r_cost:.4f} vs "
            f"baseline ${b_cost:.4f}"
        )
    return fails


def _predict(it: dict, db_path: str) -> dict:
    state = {
        "question": it["question"],
        "evidence": it.get("evidence") or None,
        "db_path": db_path,
        "db_id": it["db_id"],
        "dialect": "sqlite",
        "use_retrieval": True,
        "use_planner": True,
        "retrieval_k": settings.retrieval_k,
        "attempts": 0,
        "max_attempts": settings.max_attempts,
        "auto_limit": False,
        "use_llm_critic": False,
        "cost_log": [],
    }
    return get_graph().invoke(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate golden-set eval + gate")
    ap.add_argument("--golden", default="evals/data/golden_set.json")
    ap.add_argument("--db", default=settings.demo_db_path)
    ap.add_argument("--index", default=settings.schema_index_path)
    ap.add_argument("--baseline", default="evals/baseline.json")
    ap.add_argument("--report", default="evals/golden_report.json")
    ap.add_argument("--no-judge", action="store_true",
                    help="EX only — skip explainer + faithfulness judge")
    ap.add_argument("--gate", action="store_true",
                    help="compare vs baseline; exit 1 on regression")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set — the golden runner makes live LLM "
            "calls (writer/planner per question). Set it in .env or the env."
        )
    if args.gate and args.update_baseline:
        raise SystemExit(
            "--gate with --update-baseline would compare the run against the "
            "baseline it just wrote (always passes) — run them separately."
        )
    if not os.path.exists(args.index):
        raise SystemExit(
            f"schema index '{args.index}' missing — build it first:\n"
            "  uv run python scripts/ingest_schemas.py --demo"
        )
    set_retriever(Retriever(CardIndex(args.index, embedder=FastEmbedder())))

    with open(args.golden) as f:
        items = json.load(f)

    n = len(items)
    ex_ok = 0
    judged = faithful = 0
    cost_total = 0.0
    cost_by_purpose: dict[str, float] = {}
    by_category: dict[str, dict[str, int]] = {}
    detail = []

    for i, it in enumerate(items, 1):
        gold_rows = run_query(
            it["SQL"], args.db,
            max_rows=settings.max_rows, timeout_s=settings.statement_timeout_s,
        )[0]
        entries = []
        try:
            out = _predict(it, args.db)
            entries.extend(out.get("cost_log", []))
            ok = out.get("last_error") is None and execution_match(
                out.get("rows"), gold_rows)
        except Exception as e:
            out, ok = {}, False
            print(f"   predict error: {e}")

        verdict = None
        if ok and not args.no_judge:
            answer, e1 = llm.explain(
                question=it["question"], columns=out.get("columns") or [],
                rows=out.get("rows") or [], model=settings.fast_model,
            )
            if e1:
                entries.append(e1)
            if answer:
                verdict, e2 = llm.judge_faithfulness(
                    question=it["question"], answer=answer,
                    columns=out.get("columns") or [], rows=out.get("rows") or [],
                    model=settings.writer_model,
                )
                if e2:
                    entries.append(e2)
                if verdict is not None:
                    judged += 1
                    faithful += int(bool(verdict.get("faithful")))

        for e in entries:
            cost_total += e.get("cost_usd", 0.0)
            p = e.get("purpose", "other")
            cost_by_purpose[p] = cost_by_purpose.get(p, 0.0) + e.get("cost_usd", 0.0)

        cat = it.get("category", "uncategorised")
        c = by_category.setdefault(cat, {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += int(ok)
        ex_ok += int(ok)
        print(f"[{i}/{n}] {'PASS' if ok else 'fail'} ({cat}) | {it['question']}")
        detail.append({
            "question": it["question"], "category": cat, "ok": ok,
            "faithful": None if verdict is None else bool(verdict.get("faithful")),
        })

    report = {
        "n": n,
        "ex": round(ex_ok / n, 4),
        "faithfulness_rate": round(faithful / judged, 4) if judged else None,
        "judged": judged,
        "cost_usd_total": round(cost_total, 4),
        "cost_usd_per_question": round(cost_total / n, 6) if n else None,
        "cost_by_purpose": {k: round(v, 4) for k, v in sorted(cost_by_purpose.items())},
        "by_category": {
            k: {"n": v["n"], "ex": round(v["ok"] / v["n"], 4)}
            for k, v in sorted(by_category.items())
        },
        "items": detail,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Golden set ===")
    print(f"EX: {report['ex']:.3f}  faithfulness: {report['faithfulness_rate']}  "
          f"cost/question: ${report['cost_usd_per_question']:.4f}")
    for k, v in report["by_category"].items():
        print(f"  {k:<14} n={v['n']:<3} ex={v['ex']:.3f}")
    print(f"report → {args.report}")

    if args.update_baseline:
        baseline = {
            "ex": report["ex"],
            "faithfulness_rate": report["faithfulness_rate"],
            "cost_usd_per_question": report["cost_usd_per_question"],
            "n": n,
            "updated": datetime.date.today().isoformat(),
        }
        with open(args.baseline, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"baseline updated → {args.baseline}")

    if args.gate:
        if not os.path.exists(args.baseline):
            raise SystemExit(
                f"no baseline at '{args.baseline}' — run with --update-baseline "
                "once (and commit the file) before gating."
            )
        with open(args.baseline) as f:
            baseline = json.load(f)
        fails = gate_failures(report, baseline)
        if fails:
            print("\nGATE FAILED:")
            for r in fails:
                print(f"  ✗ {r}")
            sys.exit(1)
        print(f"\nGATE PASSED (baseline {baseline.get('updated', '?')}, "
              f"ex {baseline['ex']:.3f})")


if __name__ == "__main__":
    main()
