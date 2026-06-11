"""Stratified BIRD subsets: 100 per difficulty bucket (or --per-bucket) + a
30-question smoke set, deterministic (seeded).

    python evals/make_subset.py --dev data/bird/dev.json
"""

from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser(description="Make stratified BIRD subsets")
    ap.add_argument("--dev", required=True, help="path to BIRD dev.json")
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--smoke-per-bucket", type=int, default=10)
    ap.add_argument("--out", default="evals/data/bird_stratified.json")
    ap.add_argument("--smoke-out", default="evals/data/bird_smoke.json")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    with open(args.dev) as f:
        items = json.load(f)

    buckets: dict[str, list] = {}
    for it in items:
        buckets.setdefault(it.get("difficulty", "unknown"), []).append(it)
    print({k: len(v) for k, v in sorted(buckets.items())})

    rng = random.Random(args.seed)
    full, smoke = [], []
    for name in sorted(buckets):
        pool = sorted(buckets[name], key=lambda x: x["question"])
        rng.shuffle(pool)
        full.extend(pool[: args.per_bucket])
        smoke.extend(pool[: args.smoke_per_bucket])

    for path, data in ((args.out, full), (args.smoke_out, smoke)):
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        print(f"{path}: {len(data)} items")


if __name__ == "__main__":
    main()
