# QueryMate — NL→SQL analytics copilot (Phase 2)

[![ci](https://github.com/isaacng924/querymate/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacng924/querymate/actions/workflows/ci.yml)

Plain-English question → schema-grounded SQL → sandboxed execution → a
self-correcting critic loop → answer. This repo is at **Phase 2** of the build spec: everything from Phase 1
(schema-card RAG, advisory planner, model routing with cost accounting,
retrieval-aware repair, bucketed BIRD eval) plus an enforced eval harness — a
red-team safety suite and golden-set integrity checks gating every PR for
free, an answer narration with an LLM-as-judge faithfulness metric, and an
on-demand regression gate that blocks on EX or cost regressions against a
committed baseline. Langfuse dashboards, the chat UI, and the clarifier come
in later phases.

> The differentiator isn't "an LLM writes SQL" — it's the **eval harness**:
> execution accuracy on a public benchmark (BIRD/Spider), a measurable
> self-correction lift, and a hard read-only trust boundary.

## Architecture

```
START → retrieve → plan → write_sql → execute ──(ok / give_up)──► END
           ▲                  ▲           │
           │                  │     (error, attempts < max)
           │                  │           ▼
           └──(widen k, once)─ critic ────┘
```

- **write_sql** (`querymate/llm.py`) — Anthropic SDK, structured output, adaptive
  thinking, prompt caching on the schema prefix. Routed: Sonnet writes/repairs,
  escalates to Opus on the final attempt (see *Model routing*).
- **execute** (`querymate/validator.py` + `executor.py`) — the **trust boundary**:
  `sqlglot` rejects anything that isn't a single read-only SELECT, then the query
  runs on a `mode=ro` SQLite connection with a wall-clock timeout and a row cap.
- **critic** (`querymate/nodes.py`) — feeds the structured DB error back as a
  repair hint and loops (set `use_llm_critic` for an LLM diagnosis instead).
- **retrieve** (`querymate/retriever.py`) — schema-card RAG: fastembed
  (`bge-small-en-v1.5`, local — no API key) over **sqlite-vec**, top-k per
  database + FK 1-hop expansion. When the DB reports an unknown table/column,
  the critic **widens retrieval** (k×2, once) instead of just re-prompting.
- **plan** (`querymate/llm.py:plan`) — one Haiku call sketches tables/joins/
  aggregations; advisory only. Its join/aggregation count drives **model
  routing** (`querymate/router.py`): Haiku for simple lookups, Sonnet
  otherwise, Opus on the final attempt. Every call's tokens/cost/latency land
  in the run's `cost_log`.

## Setup

```bash
cd querymate
uv sync                      # create venv + install deps
cp .env.example .env         # add your ANTHROPIC_API_KEY (needed for the LLM path)
uv run python scripts/make_demo_db.py
uv run python scripts/ingest_schemas.py --demo   # build the demo schema index
```

## Run the tests (no API key needed)

The trust boundary and the eval comparator are pure-logic and fully tested:

```bash
for t in tests/test_*.py; do uv run python "$t"; done
```

## Ask a question (needs ANTHROPIC_API_KEY)

```bash
uv run querymate "Which customer has placed the most orders? Return their name."
```

The CLI ends with an `Answer:` line — a 1–2 sentence narration of the result
rows (skip with `--no-explain`).

## Run the execution-accuracy eval (needs ANTHROPIC_API_KEY)

````bash
# one-time: BIRD dev set (~1.2GB) + index + stratified subsets
uv run python scripts/fetch_bird.py
uv run python scripts/ingest_schemas.py --db-root data/bird/dev_databases
uv run python evals/make_subset.py --dev data/bird/dev.json

# retrieval quality — no LLM calls, free, full dev set
uv run python evals/run_recall.py --subset data/bird/dev.json --ks 3 5 10

# execution accuracy — both arms on the stratified subset, then the chart
uv run python evals/run_bird.py --subset evals/data/bird_stratified.json --db-root data/bird/dev_databases --arm rag
uv run python evals/run_bird.py --subset evals/data/bird_stratified.json --db-root data/bird/dev_databases --arm full-schema
uv run python evals/make_chart.py evals/report_rag.json evals/report_full_schema.json
````

Reports land in `evals/report_<arm>.json` (EX by difficulty bucket,
self-correction lift, cost/question by routing tier) and
`evals/recall_report.json`; the chart in `evals/ex_chart.png`.

## Safety & regression gate

Tier 1 runs free on every push (`.github/workflows/ci.yml`): all plain-assert
suites including the **red-team corpus** (`evals/data/redteam_corpus.json` —
DML/DDL, stacked statements, comment smuggling, CTE-wrapped writes, PRAGMA/
ATTACH, `load_extension` abuse) at a required **100% block rate**, plus golden-
set integrity and gate-threshold unit tests.

Tier 2 is the paid gate — manual trigger only (`eval-gate` workflow, or locally):

```bash
uv run python evals/run_golden.py                    # 40-question golden set: EX + faithfulness + cost
uv run python evals/run_golden.py --update-baseline  # refresh evals/baseline.json (commit it)
uv run python evals/run_golden.py --gate             # exit 1 if EX drops >2pp or cost/question rises >15%
```

The judge (`querymate/llm.py:judge_faithfulness`) scores whether the answer
narration only states what the rows support; faithfulness is reported per run
but only EX and cost gate the merge — safety gates at 100% in tier 1.

## Model routing

Per the build spec, QueryMate routes models for cost (`querymate/settings.py`):
a fast model (Sonnet 4.6) writes and repairs; the loop escalates to Opus 4.8 on
the final attempt. Set `QUERYMATE_WRITER_MODEL=claude-opus-4-8` (and
`QUERYMATE_ESCALATE_MODEL`) to pay for max quality everywhere.

## What's deliberately *not* here yet

Langfuse tracing (`querymate/trace.py` is the seam), the chat UI + clarifier, and a pgvector retriever backend.

## Layout

```
querymate/   validator · executor · llm · nodes · graph · cli · settings · state · trace · embedder · schema_cards · card_index · retriever · router
evals/       compare (EX) · recall · run_bird (harness) · run_recall · make_subset · make_chart · run_golden · data/ (golden_set, redteam_corpus)
scripts/     make_demo_db.py · ingest_schemas.py · fetch_bird.py
tests/       validator · executor · compare · schema_cards · retriever · router · recall · graph · redteam · golden_set · gate   (plain-assert)
```

## Troubleshooting

> `enable_load_extension` AttributeError → your Python lacks SQLite extension
> support; use a uv-managed interpreter (`uv python install 3.13 && uv sync`).
