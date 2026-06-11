"""Retrieval-only recall@k eval over a BIRD-format subset. No LLM calls — this
runs on the FULL dev set for free.

    python evals/run_recall.py --subset data/bird/dev.json \
        --index data/schema_index.sqlite --ks 3 5 10
"""

from __future__ import annotations

import argparse
import json

from querymate.card_index import CardIndex
from querymate.embedder import FastEmbedder
from querymate.retriever import Retriever
from querymate.settings import settings

from evals.recall import gold_tables, recall_at_k


def main() -> None:
    ap = argparse.ArgumentParser(description="QueryMate retrieval recall@k eval")
    ap.add_argument("--subset", default="evals/data/sample_bird_subset.json")
    ap.add_argument("--index", default=settings.schema_index_path)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--report", default="evals/recall_report.json")
    args = ap.parse_args()

    with open(args.subset) as f:
        items = json.load(f)
    retriever = Retriever(CardIndex(args.index, embedder=FastEmbedder()))

    sums = {k: 0.0 for k in args.ks}
    counts = {k: 0 for k in args.ks}
    skipped = 0
    for i, it in enumerate(items, 1):
        gold = gold_tables(it["SQL"])
        if not gold:
            skipped += 1
            continue
        for k in args.ks:
            cards, _ = retriever.retrieve(it["question"], db_id=it["db_id"], k=k)
            r = recall_at_k(gold, [c.table for c in cards])
            if r is not None:
                sums[k] += r
                counts[k] += 1
        if i % 100 == 0:
            print(f"[{i}/{len(items)}]")

    report = {
        "n": len(items),
        "skipped_unparseable_gold": skipped,
        "recall_at_k": {
            str(k): round(sums[k] / counts[k], 4) if counts[k] else None
            for k in args.ks
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== Schema-retrieval recall ===")
    for k in args.ks:
        print(f"recall@{k}: {report['recall_at_k'][str(k)]}")
    print(f"report → {args.report}")


if __name__ == "__main__":
    main()
