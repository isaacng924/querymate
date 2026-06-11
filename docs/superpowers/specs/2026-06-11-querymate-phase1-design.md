# QueryMate Phase 1 — Retrieval + Planning: Design

**Date:** 2026-06-11
**Status:** Approved
**Builds on:** Phase 0 (writer → executor → critic loop, sqlglot trust boundary, BIRD-subset EX eval)
**Source spec:** vault `03 - Resources/QueryMate — NL-to-SQL Build Spec.md` (Phase 1 section)

## Goal

Replace the flat full-schema prompt with schema-card RAG, add a planner, measure
retrieval recall@k, run a credible bucketed-EX number on BIRD dev, and add
Haiku/Sonnet/Opus model routing with per-call cost logging.

## Decisions (settled in brainstorming)

| Decision | Choice | Why |
|---|---|---|
| Vector store | **sqlite-vec** | Zero infra; keeps repo clone-and-run. Retriever interface leaves pgvector as a Phase 3 backend. |
| Embeddings | **fastembed** (`bge-small-en-v1.5`, local ONNX) | No second API key; tiny volume (~1–2k cards + 1 query embed/question). |
| EX eval scale | **Stratified 300** (100 × simple/moderate/challenging) + 30-question smoke set | Credible bucketed EX at ~$10–30; full 1,534 stays a flag for later. recall@k runs on the full 1,534 (free — no LLM). |
| Planner | **Structured plan node** (one Haiku call) | Plan grounds the writer prompt and its complexity drives routing. |
| Integration | **In-graph nodes with retrieval-aware repair** | Critic routes `unknown_table`/`unknown_column` errors back to retrieval with widened k — agentic repair, recall@k stays deterministic/measurable. |

## Architecture

```
START → retrieve → plan → write_sql → execute ──ok/give_up──► END
            ▲                  ▲          │ (error)
            │                  └─ critic ◄┘
            └──(unknown_table/column, ≤1 widen: k×2)──┘
```

New `QueryState` fields: `cards`, `retrieval_k`, `retrieval_widened`, `plan`,
`cost_log`. The existing flat `schema` string stays and is **built from the
retrieved cards** — the eval's full-schema arm bypasses retrieval and fills it
the Phase 0 way, so the two arms share one code path through the writer.

## Components

### Schema cards + ingestion
- A **schema card** = one table: name, columns + types, PK/FK edges, plus
  BIRD `database_description` CSV column descriptions when available.
- `querymate/schema_cards.py` — extraction via `sqlite_master` / PRAGMA.
- `scripts/ingest_schemas.py` — walk a DB root → cards → fastembed →
  **sqlite-vec** index at `data/schema_index.sqlite`.

### Retriever (`querymate/retriever.py`)
- `retrieve(question, db_id, k) → list[SchemaCard]`: embed question → top-k
  vector search filtered by `db_id` → **FK 1-hop expansion** (include tables
  joined to hits so join paths survive) → build `schema` string.
- Embedder behind a small protocol (`querymate/embedder.py`); tests inject a
  deterministic fake — no model download in tests/CI.

### Planner (`querymate/nodes.py` + `llm.py`)
- One **Haiku** call, structured output: `{tables, join_count, aggregations, filters}`.
- Plan text injected into the writer prompt. **Advisory, never fatal** —
  malformed/failed plan → writer proceeds with cards only, failure logged.

### Model routing (`querymate/router.py`)
- Pure function over the plan: no joins + no aggregation → **Haiku 4.5**;
  otherwise → **Sonnet 4.6**; final attempt → **Opus 4.8** (existing escalate
  behaviour, unchanged). **No plan (planner failed) → Sonnet** — the safe default.
- Price table in `settings.py`; every LLM call appends
  `{model, tokens_in, tokens_out, cost, latency}` to `cost_log`.

### Phase 0 gap fix
- BIRD items carry an `evidence` hint; the writer prompt now includes it
  (standard practice — EX isn't credible without it). Eval passes it through.

## Eval plan (`evals/`)

- `evals/make_subset.py` — stratified 300 by BIRD difficulty + 30-question smoke set.
- **recall@k** — extract gold tables from gold SQL via sqlglot; recall@k =
  fraction of gold tables in retrieved top-k. Run on **full 1,534** at k=3/5/10
  (no LLM calls, free).
- **EX** — stratified 300, two arms (`--arm rag|full-schema`) × single-shot/final:
  bucketed EX, retrieval lift, self-correction lift, cost/question by routing tier.
  `report.json` extended accordingly.
- `evals/make_chart.py` — bucketed EX chart (the build-in-public artifact).
- BIRD dev download: `scripts/fetch_bird.py` best-effort (`gdown`) + README
  manual-download fallback (BIRD hosts on Google Drive).

## Error handling

- **No schema index**: CLI builds one on the fly for the demo DB (clone-and-run
  preserved); the eval errors loudly with the exact ingest command.
- **Retrieval widening** bounded at 1 per question; total attempt ceiling unchanged.
- **Planner/embedder failures** degrade gracefully (skip plan / clear error);
  the loop never crashes on an advisory component.

## Testing

Plain-assert style matching the existing suite: `test_schema_cards`,
`test_retriever` (fake embedder), `test_router`, `test_recall`. The existing
25 tests keep passing untouched. LLM-dependent paths stay out of the
no-key test suite, as in Phase 0.

## Out of scope (later phases)

Langfuse wiring (Phase 3), DeepEval/Promptfoo + CI regression gate (Phase 2),
chat UI (Phase 3), pgvector backend (Phase 3).

## New dependencies

`sqlite-vec`, `fastembed` (+ `gdown` as a script-only extra). All pip-installable, zero infra.

## Deliverables

1. Bucketed EX chart: single-shot vs final × RAG vs full-schema, by difficulty.
2. recall@k curve (k=3/5/10) on full BIRD dev.
3. Cost/question table by routing tier.
4. The retrieval-aware repair loop — the writeup's headline mechanism.
