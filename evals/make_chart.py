"""Bucketed-EX chart from one or two run_bird reports (the portfolio artifact).

    python evals/make_chart.py evals/report_rag.json evals/report_full_schema.json
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    paths = sys.argv[1:] or ["evals/report_rag.json"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = []
    for p in paths:
        with open(p) as f:
            reports.append(json.load(f))

    buckets = sorted({b for r in reports for b in r.get("buckets", {})})
    if not buckets:
        raise SystemExit("no buckets in the given report(s) — nothing to chart")
    series = []  # (label, [ex per bucket])
    for r in reports:
        for kind in ("ex_single_shot", "ex_final"):
            label = f"{r['arm']} {'single-shot' if kind == 'ex_single_shot' else 'critic-loop'}"
            series.append((label, [r["buckets"].get(b, {}).get(kind) or 0.0
                                   for b in buckets]))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(series)
    for i, (label, vals) in enumerate(series):
        xs = [j + i * width for j in range(len(buckets))]
        bars = ax.bar(xs, vals, width=width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(buckets))])
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Execution Accuracy (EX)")
    ax.set_title("QueryMate — EX by difficulty: retrieval & self-correction lift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "evals/ex_chart.png"
    fig.savefig(out, dpi=160)
    print(f"chart → {out}")


if __name__ == "__main__":
    main()
