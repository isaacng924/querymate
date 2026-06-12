# QueryMate Phase 2 — Eval Harness → CI: Design

**Date:** 2026-06-12
**Status:** Approved
**Builds on:** Phase 1 (schema-card RAG, planner, routing, cost accounting, two-arm BIRD eval)
**Source spec:** vault `03 - Resources/QueryMate — NL-to-SQL Build Spec.md` (Phase 2 section)

## Goal

A regression gate that makes eval rigor enforceable: red-team safety at 100% on
every PR for free, plus an on-demand golden-set gate (EX + faithfulness + cost
vs a committed baseline) that blocks on regression.

## Decisions (settled in brainstorming)

| Decision | Choice | Why |
|---|---|---|
| Per-PR CI spend | **LLM-free only** | No standing API spend; deterministic checks gate every PR. The LLM gate exists as a manual-trigger workflow — real gate, zero accidental cost. |
| Tooling | **Native Python, no DeepEval/Promptfoo** | Repo stays pure plain-assert/uv; red-team is deterministic against the trust boundary; judge is a direct Anthropic SDK call. "Built a custom eval harness + LLM-as-judge" is the claim. |
| Faithfulness dependency | **Minimal explainer now** | Judge needs a narration to judge. ~40 lines following llm.py patterns; CLI gains an `Answer:` line; full product surface stays Phase 3. |
| Architecture | **Two-tier harness + committed baseline** | Tier 1 free/every-PR; tier 2 on-demand vs `evals/baseline.json` with explicit thresholds. |

## Components

### Red-team / safety suite (tier 1)
- `evals/data/redteam_corpus.json` — ~25–30 attack strings, categorised:
  DML/DDL, multi-statement stacking, comment smuggling, CTE-wrapped writes,
  PRAGMA/ATTACH/VACUUM, `load_extension()` abuse, garbage/empty.
- `tests/test_redteam.py` — every corpus entry must raise `UnsafeSQL` from
  `validate_sql`; prints safety pass-rate; asserts 100%. Plus an executor
  backstop case (read-only connection refuses a write directly).

### Golden set
- `evals/data/golden_set.json` — ~40 hand-written Q→SQL on the demo DB,
  BIRD-compatible (`db_id`, `question`, `SQL`) + `category`:
  business-term, ambiguous, multi-step, negation, aggregation+filter.
- `tests/test_golden_set.py` (tier 1) — every gold query parses, executes
  read-only on the demo DB, returns deterministic rows.

### Explainer + faithfulness judge (`querymate/llm.py`)
- `explain(question, columns, rows, model)` → 1–2 sentence answer;
  purpose `"explainer"`; Haiku (`settings.fast_model`); rows truncated (≤20)
  before prompting. **Plain post-execution call, not a graph node.**
- `judge_faithfulness(question, answer, columns, rows)` → structured
  `{faithful: bool, reason: str}`; purpose `"judge"`; Sonnet
  (`settings.writer_model`) — judge tier ≥ explainer tier.
- CLI prints `Answer: …` after the result rows; explainer cost joins the
  cost summary line. Explainer/judge failures degrade gracefully
  (skip + log), matching the planner's advisory contract.

### Golden runner + gate (tier 2)
- `evals/run_golden.py` — golden set through the graph (RAG arm): EX per item;
  on success → explain → judge. Aggregates **EX**, **faithfulness rate**
  (faithful / answered), **cost/question** (broken out by purpose) →
  `evals/golden_report.json`.
- Gate logic is a pure importable function
  `gate_failures(report, baseline) -> list[str]` (`tests/test_gate.py` covers it):
  - fail if EX drops **> 2 percentage points** vs baseline
  - fail if cost/question rises **> 15%** vs baseline
  - safety is NOT in the baseline — tier 1 enforces 100% on every PR
- Flags: `--gate` (exit 1 + printed reasons), `--update-baseline`
  (deliberate refresh), `--no-judge` (EX-only, cheaper run).
- Missing baseline → clear error: run `--update-baseline` first.
- `evals/baseline.json` is **committed**; `evals/golden_report.json` gitignored.

### GitHub Actions
- `.github/workflows/ci.yml` — on push + PR: `uv sync` → run all plain-assert
  suites. FakeEmbedder only — no model download, no API key, ~1 min.
- `.github/workflows/eval-gate.yml` — `workflow_dispatch` only: build demo DB →
  ingest demo → `run_golden.py --gate`; `ANTHROPIC_API_KEY` from repo secret;
  uploads `golden_report.json` as an artifact. Never auto-runs.

## Error handling

- Explainer/judge: advisory contract — failures skip the metric for that item
  and log; never crash the run. Judge unavailable → faithfulness reported as
  `null`, gate compares only metrics present in BOTH report and baseline.
- Gate with no baseline → explicit error + instruction, exit ≠ 0.
- Red-team corpus load failure → test suite fails loudly (it IS the gate).

## Testing

New tier-1 plain-assert suites: `test_redteam` (safety 100%),
`test_golden_set` (gold integrity), `test_gate` (threshold logic incl.
boundary cases: exactly-2pp drop passes, >2pp fails; cost +15% passes,
>+15% fails; missing-metric handling). Existing 55 tests untouched.

## Out of scope (Phase 3+)

Langfuse tracing + prompt registry, visual dashboard (golden_report.json +
existing chart satisfy the Phase 2 deliverable), full BIRD stratified gate
(stays the manual runbook), chat UI, pgvector backend.

## Deliverables

1. Green tier-1 gate on every PR: 55 existing tests + redteam + golden
   integrity + gate-logic tests, zero spend.
2. `evals/run_golden.py --gate` — the on-demand regression gate with
   committed baseline and explicit thresholds.
3. CLI answers: `Answer:` narration line via the minimal explainer.
4. First baseline (`evals/baseline.json`) — produced once the API key exists
   (same session as the Phase 1 runbook).
