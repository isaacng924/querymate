# QueryMate — NL→SQL analytics copilot (Phase 0)

Plain-English question → schema-grounded SQL → sandboxed execution → a
self-correcting critic loop → answer. This repo is **Phase 0** of the build spec:
the core agent loop plus an **objective execution-accuracy eval** with a
single-shot-vs-self-correction lift. Schema-RAG, planner, clarifier, explainer,
LLMOps, and the CI gate come in later phases.

> The differentiator isn't "an LLM writes SQL" — it's the **eval harness**:
> execution accuracy on a public benchmark (BIRD/Spider), a measurable
> self-correction lift, and a hard read-only trust boundary.

## Architecture (this phase)

```
START → write_sql ─► execute ──(ok / give_up)──► END
                        │
                  (error, attempts < max)
                        ▼
                     critic ──► write_sql   (loop, bounded by max_attempts)
```

- **write_sql** (`querymate/llm.py`) — Anthropic SDK, structured output, adaptive
  thinking, prompt caching on the schema prefix. Routed: Sonnet writes/repairs,
  escalates to Opus on the final attempt (see *Model routing*).
- **execute** (`querymate/validator.py` + `executor.py`) — the **trust boundary**:
  `sqlglot` rejects anything that isn't a single read-only SELECT, then the query
  runs on a `mode=ro` SQLite connection with a wall-clock timeout and a row cap.
- **critic** (`querymate/nodes.py`) — feeds the structured DB error back as a
  repair hint and loops (set `use_llm_critic` for an LLM diagnosis instead).

## Setup

```bash
cd querymate
uv sync                      # create venv + install deps
cp .env.example .env         # add your ANTHROPIC_API_KEY (needed for the LLM path)
uv run python scripts/make_demo_db.py
```

## Run the tests (no API key needed)

The trust boundary and the eval comparator are pure-logic and fully tested:

```bash
uv run python tests/test_validator.py
uv run python tests/test_executor.py
uv run python tests/test_compare.py
```

## Ask a question (needs ANTHROPIC_API_KEY)

```bash
uv run querymate "Which customer has placed the most orders? Return their name."
```

## Run the execution-accuracy eval (needs ANTHROPIC_API_KEY)

```bash
uv run python evals/run_bird.py
```

Prints **single-shot EX**, **final EX**, and the **self-correction lift**, and
writes `evals/report.json`. Out of the box it runs the bundled demo subset
(`evals/data/sample_bird_subset.json` against the demo DB). Point it at the real
benchmark to get a credible number:

```bash
uv run python evals/run_bird.py --subset path/to/bird/dev.json --db-root path/to/bird/dev_databases
```

(The local comparator is an order-insensitive multiset match — a Phase-0
approximation of BIRD/Spider EX. For an official number, run the BIRD/Spider
eval scripts; this is the fast local signal.)

## Model routing

Per the build spec, QueryMate routes models for cost (`querymate/settings.py`):
a fast model (Sonnet 4.6) writes and repairs; the loop escalates to Opus 4.8 on
the final attempt. Set `QUERYMATE_WRITER_MODEL=claude-opus-4-8` (and
`QUERYMATE_ESCALATE_MODEL`) to pay for max quality everywhere.

## What's deliberately *not* here yet

Phase 1+ (see the vault build spec): schema-card RAG + retrieval recall@k, a
planner, an explainer with a faithfulness judge, Langfuse tracing
(`querymate/trace.py` is the seam), a CI regression gate, and a red-team suite.

## Layout

```
querymate/   validator · executor · llm · nodes · graph · cli · settings · state · trace
evals/       compare (EX) · run_bird (harness) · data/sample_bird_subset.json
scripts/     make_demo_db.py
tests/       test_validator · test_executor · test_compare   (plain-assert)
```
