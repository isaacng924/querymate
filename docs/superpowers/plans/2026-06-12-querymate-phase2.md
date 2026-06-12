# QueryMate Phase 2 — Eval Harness → CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two-tier regression gate — free deterministic safety/integrity checks on every PR, plus an on-demand golden-set gate (EX + faithfulness + cost vs a committed baseline) — with a minimal explainer + LLM-as-judge.

**Architecture:** Tier 1 is pure plain-assert (red-team corpus vs the sqlglot trust boundary, golden-set integrity, gate-logic unit tests) and runs in GitHub Actions on every push. Tier 2 (`evals/run_golden.py --gate`) runs the 40-question golden set through the Phase 1 graph, adds explainer+judge, and compares against `evals/baseline.json`; it only runs via `workflow_dispatch` or locally. No new frameworks — everything follows existing llm.py / plain-assert patterns.

**Tech Stack:** Python 3.13/uv, sqlglot, Anthropic SDK (existing), GitHub Actions (astral-sh/setup-uv).

**Spec:** `docs/superpowers/specs/2026-06-12-querymate-phase2-design.md`

**Conventions (match Phase 1):** plain-assert test files runnable standalone with the `__main__` runner block LAST; all commands from repo root; LLM-call code has no no-key tests.

## File structure

| File | Responsibility |
|---|---|
| `evals/data/redteam_corpus.json` (new) | attack corpus: `must_reject` (validator) + `must_fail_execution` (executor backstop) |
| `tests/test_redteam.py` (new) | fires corpus; safety pass-rate printed; asserts 100% |
| `evals/data/golden_set.json` (new) | 40 Q→SQL on demo DB with `category` |
| `tests/test_golden_set.py` (new) | every gold query parses + executes read-only + rows non-empty |
| `querymate/llm.py` (modify) | `explain()` + `judge_faithfulness()` (+ `_JUDGE_FORMAT`) |
| `querymate/cli.py` (modify) | `Answer:` line, `--no-explain`, cost merge |
| `evals/run_golden.py` (new) | golden runner; `gate_failures()` pure fn; `--gate` / `--update-baseline` / `--no-judge` |
| `tests/test_gate.py` (new) | gate threshold boundary tests |
| `.github/workflows/ci.yml` (new) | tier 1 on push/PR |
| `.github/workflows/eval-gate.yml` (new) | tier 2 on workflow_dispatch |
| `.gitignore` (modify) | ignore `evals/golden_report.json` |
| `README.md` (modify) | Phase 2 sections + CI badge |

**Demo-DB facts used by gold queries** (from `scripts/make_demo_db.py` — verify there if a test disagrees): customers Alice/Bob/Carol/Dan (Dan has NO orders); 5 products; 5 orders (Alice 3, Bob 1, Carol 1); total revenue 209.0; revenue/customer Alice 134, Bob 45, Carol 30; units/product P1=2,P2=1,P3=2,P4=5,P5=3; category revenue Books 88, Electronics 85, Home 36; Feb-2026 orders = 3, revenue 111; total units 13; 'Home' is the only category with no January orders; Carol is the only UK customer with no January order.

---

### Task 1: Red-team corpus + safety suite

**Files:**
- Create: `evals/data/redteam_corpus.json`
- Test: `tests/test_redteam.py`

- [ ] **Step 1.1: Create `evals/data/redteam_corpus.json`**

```json
{
  "must_reject": [
    {"category": "dml", "sql": "INSERT INTO customers VALUES (99, 'Mallory', 'XX')"},
    {"category": "dml", "sql": "UPDATE products SET price = 0"},
    {"category": "dml", "sql": "DELETE FROM orders"},
    {"category": "dml", "sql": "REPLACE INTO products VALUES (1, 'x', 'y', 0)"},
    {"category": "ddl", "sql": "DROP TABLE customers"},
    {"category": "ddl", "sql": "CREATE TABLE pwn (x INT)"},
    {"category": "ddl", "sql": "ALTER TABLE customers ADD COLUMN pwned INT"},
    {"category": "ddl", "sql": "CREATE TRIGGER t AFTER INSERT ON orders BEGIN DELETE FROM orders; END"},
    {"category": "ddl", "sql": "DROP INDEX some_index"},
    {"category": "multi_statement", "sql": "SELECT 1; DROP TABLE customers"},
    {"category": "multi_statement", "sql": "SELECT * FROM customers; DELETE FROM customers"},
    {"category": "multi_statement", "sql": "; DROP TABLE orders"},
    {"category": "comment_smuggle", "sql": "SELECT 1; -- innocent\nDROP TABLE customers"},
    {"category": "comment_smuggle", "sql": "SELECT /* harmless */ 1; DROP TABLE products"},
    {"category": "pragma_attach", "sql": "PRAGMA writable_schema = 1"},
    {"category": "pragma_attach", "sql": "ATTACH DATABASE '/tmp/evil.db' AS evil"},
    {"category": "pragma_attach", "sql": "VACUUM"},
    {"category": "pragma_attach", "sql": "ANALYZE"},
    {"category": "cte_write", "sql": "WITH x AS (SELECT 1) INSERT INTO customers VALUES (99, 'M', 'XX')"},
    {"category": "cte_write", "sql": "WITH x AS (SELECT 1) DELETE FROM orders"},
    {"category": "garbage", "sql": ""},
    {"category": "garbage", "sql": "   "},
    {"category": "garbage", "sql": "DROP"},
    {"category": "garbage", "sql": "NOT SQL AT ALL (((("}
  ],
  "must_fail_execution": [
    {"category": "function_abuse", "sql": "SELECT load_extension('/tmp/evil.so')"},
    {"category": "function_abuse", "sql": "SELECT writefile('/tmp/pwn', 'x')"},
    {"category": "function_abuse", "sql": "SELECT readfile('/etc/passwd')"}
  ]
}
```

- [ ] **Step 1.2: Write the failing test** — create `tests/test_redteam.py`:

```python
"""Red-team safety suite: every corpus attack must die at the trust boundary.

``must_reject`` entries must be refused by the static validator (UnsafeSQL).
``must_fail_execution`` entries pass parsing (single SELECT) but must fail at
the sandboxed executor (ExecError) — defence in depth. Prints the safety
pass-rate; the suite REQUIRES 100%.

    python tests/test_redteam.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.executor import ExecError, validate_and_run  # noqa: E402
from querymate.validator import UnsafeSQL, validate_sql  # noqa: E402

_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evals", "data", "redteam_corpus.json"
)
_DEMO_DB = os.path.join(os.path.dirname(__file__), "..", "data", "demo_store.sqlite")

with open(_CORPUS_PATH) as f:
    CORPUS = json.load(f)


def test_corpus_shape():
    assert CORPUS["must_reject"] and CORPUS["must_fail_execution"]
    for section in CORPUS.values():
        for item in section:
            assert item["category"] and "sql" in item


def test_validator_rejects_all_must_reject():
    passed, failed = 0, []
    for item in CORPUS["must_reject"]:
        try:
            validate_sql(item["sql"])
            failed.append(f"{item['category']}: {item['sql']!r}")
        except UnsafeSQL:
            passed += 1
    total = len(CORPUS["must_reject"])
    print(f"  validator safety pass-rate: {passed}/{total}")
    assert not failed, f"validator ACCEPTED unsafe SQL: {failed}"


def test_executor_blocks_all_must_fail_execution():
    assert os.path.exists(_DEMO_DB), "run scripts/make_demo_db.py first"
    passed, failed = 0, []
    for item in CORPUS["must_fail_execution"]:
        try:
            validate_and_run(item["sql"], _DEMO_DB, auto_limit=False)
            failed.append(f"{item['category']}: {item['sql']!r}")
        except ExecError:
            passed += 1
    total = len(CORPUS["must_fail_execution"])
    print(f"  executor safety pass-rate: {passed}/{total}")
    assert not failed, f"executor RAN unsafe SQL: {failed}"


def test_readonly_backstop_direct_write():
    # Even a hypothetically validated write dies on the mode=ro connection.
    from querymate.executor import run_query

    assert os.path.exists(_DEMO_DB), "run scripts/make_demo_db.py first"
    try:
        run_query("UPDATE products SET price = 0", _DEMO_DB)
        raised = False
    except ExecError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} redteam tests passed")
```

- [ ] **Step 1.3: Run to verify it fails**

Run: `uv run python tests/test_redteam.py`
Expected: FAIL — `FileNotFoundError` if Step 1.1 skipped, else passes; ensure the corpus file is what makes it pass (write the test first if strict TDD ordering matters: the corpus is data, the test is the assertion).

- [ ] **Step 1.4: Run to verify it passes**

Run: `uv run python scripts/make_demo_db.py && uv run python tests/test_redteam.py`
Expected: `4 redteam tests passed` with both pass-rates at 100%.

**If any `must_reject` entry is NOT rejected:** that is a real validator hole. Do not silently move the entry to `must_fail_execution` — STOP and report DONE_WITH_CONCERNS with the entry, unless the executor also provably blocks it AND the controller approves the reclassification.

- [ ] **Step 1.5: Commit**

```bash
git add evals/data/redteam_corpus.json tests/test_redteam.py
git commit -m "feat: red-team safety suite — corpus vs trust boundary at 100%"
```

---

### Task 2: Golden set + integrity test

**Files:**
- Create: `evals/data/golden_set.json`
- Test: `tests/test_golden_set.py`

- [ ] **Step 2.1: Create `evals/data/golden_set.json`** (40 items; BIRD-compatible + `category`; `evidence` carries the disambiguation for ambiguous items, exactly like BIRD does):

```json
[
  {"db_id": "demo_store", "category": "simple", "question": "How many customers are there?", "SQL": "SELECT COUNT(*) FROM customers"},
  {"db_id": "demo_store", "category": "simple", "question": "How many products do we sell?", "SQL": "SELECT COUNT(*) FROM products"},
  {"db_id": "demo_store", "category": "simple", "question": "How many orders have been placed?", "SQL": "SELECT COUNT(*) FROM orders"},
  {"db_id": "demo_store", "category": "simple", "question": "How many customers are from the United Kingdom?", "SQL": "SELECT COUNT(*) FROM customers WHERE country = 'United Kingdom'"},
  {"db_id": "demo_store", "category": "filter", "question": "List the names of customers from Canada.", "SQL": "SELECT name FROM customers WHERE country = 'Canada'"},
  {"db_id": "demo_store", "category": "filter", "question": "List the product names in the Books category.", "SQL": "SELECT name FROM products WHERE category = 'Books'"},
  {"db_id": "demo_store", "category": "filter", "question": "Which products cost more than 25? Return their names.", "SQL": "SELECT name FROM products WHERE price > 25"},
  {"db_id": "demo_store", "category": "aggregation", "question": "What is the average product price?", "SQL": "SELECT AVG(price) FROM products"},
  {"db_id": "demo_store", "category": "aggregation", "question": "What is the price of the cheapest product?", "SQL": "SELECT MIN(price) FROM products"},
  {"db_id": "demo_store", "category": "aggregation", "question": "What is the name of the most expensive product?", "SQL": "SELECT name FROM products ORDER BY price DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "business_term", "question": "What is the total revenue across all orders?", "evidence": "revenue = quantity times product price", "SQL": "SELECT SUM(oi.quantity * p.price) FROM order_items oi JOIN products p ON p.id = oi.product_id"},
  {"db_id": "demo_store", "category": "business_term", "question": "What is our average order value?", "evidence": "average order value = total revenue divided by number of orders; revenue = quantity times price", "SQL": "SELECT SUM(oi.quantity * p.price) * 1.0 / COUNT(DISTINCT oi.order_id) FROM order_items oi JOIN products p ON p.id = oi.product_id"},
  {"db_id": "demo_store", "category": "business_term", "question": "Show revenue by product category, highest first.", "evidence": "revenue = quantity times product price", "SQL": "SELECT p.category, SUM(oi.quantity * p.price) AS rev FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.category ORDER BY rev DESC"},
  {"db_id": "demo_store", "category": "business_term", "question": "What is our best-selling product?", "evidence": "best-selling = most units sold", "SQL": "SELECT p.name FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.id ORDER BY SUM(oi.quantity) DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "business_term", "question": "Which product generated the most revenue?", "evidence": "revenue = quantity times product price", "SQL": "SELECT p.name FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.id ORDER BY SUM(oi.quantity * p.price) DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Which customer has placed the most orders? Return their name.", "SQL": "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY COUNT(*) DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Which customer generated the most revenue? Return their name.", "evidence": "revenue = quantity times product price", "SQL": "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id GROUP BY c.id ORDER BY SUM(oi.quantity * p.price) DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Show each customer's total revenue, highest first.", "evidence": "revenue = quantity times product price; customers without orders are excluded", "SQL": "SELECT c.name, SUM(oi.quantity * p.price) AS rev FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id GROUP BY c.id ORDER BY rev DESC"},
  {"db_id": "demo_store", "category": "multi_step", "question": "How many orders did Alice place?", "SQL": "SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.name = 'Alice'"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Which products did Bob order? Return the product names.", "SQL": "SELECT DISTINCT p.name FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE c.name = 'Bob'"},
  {"db_id": "demo_store", "category": "negation", "question": "Which customers have not placed any orders? Return their names.", "SQL": "SELECT name FROM customers WHERE id NOT IN (SELECT customer_id FROM orders)"},
  {"db_id": "demo_store", "category": "negation", "question": "Which product categories had no orders in January 2026?", "SQL": "SELECT DISTINCT category FROM products WHERE category NOT IN (SELECT p.category FROM orders o JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE o.order_date LIKE '2026-01%')"},
  {"db_id": "demo_store", "category": "negation", "question": "Which customers from the United Kingdom did not place an order in January 2026? Return their names.", "SQL": "SELECT name FROM customers WHERE country = 'United Kingdom' AND id NOT IN (SELECT customer_id FROM orders WHERE order_date LIKE '2026-01%')"},
  {"db_id": "demo_store", "category": "date", "question": "How many orders were placed in February 2026?", "SQL": "SELECT COUNT(*) FROM orders WHERE order_date LIKE '2026-02%'"},
  {"db_id": "demo_store", "category": "date", "question": "What was the total revenue in February 2026?", "evidence": "revenue = quantity times product price", "SQL": "SELECT SUM(oi.quantity * p.price) FROM orders o JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE o.order_date LIKE '2026-02%'"},
  {"db_id": "demo_store", "category": "date", "question": "On what date did Alice place her first order?", "SQL": "SELECT MIN(o.order_date) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.name = 'Alice'"},
  {"db_id": "demo_store", "category": "date", "question": "What is the date of the most recent order?", "SQL": "SELECT MAX(order_date) FROM orders"},
  {"db_id": "demo_store", "category": "ambiguous", "question": "Who are our top 2 customers? Return their names.", "evidence": "top customers = highest total revenue; revenue = quantity times product price", "SQL": "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id GROUP BY c.id ORDER BY SUM(oi.quantity * p.price) DESC LIMIT 2"},
  {"db_id": "demo_store", "category": "ambiguous", "question": "What is our most popular product?", "evidence": "most popular = most units sold", "SQL": "SELECT p.name FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.id ORDER BY SUM(oi.quantity) DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "ambiguous", "question": "Which country is our biggest market?", "evidence": "biggest market = country with the highest total revenue; revenue = quantity times product price", "SQL": "SELECT c.country FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id GROUP BY c.country ORDER BY SUM(oi.quantity * p.price) DESC LIMIT 1"},
  {"db_id": "demo_store", "category": "aggregation", "question": "How many units of Electronics products have been sold in total?", "SQL": "SELECT SUM(oi.quantity) FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.category = 'Electronics'"},
  {"db_id": "demo_store", "category": "multi_step", "question": "How many distinct products has Alice ordered?", "SQL": "SELECT COUNT(DISTINCT oi.product_id) FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE c.name = 'Alice'"},
  {"db_id": "demo_store", "category": "aggregation", "question": "What is the average number of units per order?", "evidence": "units = sum of item quantities in the order", "SQL": "SELECT SUM(oi.quantity) * 1.0 / COUNT(DISTINCT oi.order_id) FROM order_items oi"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Which customers bought 'The Pragmatic Programmer'? Return their names.", "SQL": "SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE p.name = 'The Pragmatic Programmer'"},
  {"db_id": "demo_store", "category": "multi_step", "question": "How many units of the 'USB-C Cable' were sold?", "SQL": "SELECT SUM(oi.quantity) FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.name = 'USB-C Cable'"},
  {"db_id": "demo_store", "category": "business_term", "question": "What was the revenue from the Books category?", "evidence": "revenue = quantity times product price", "SQL": "SELECT SUM(oi.quantity * p.price) FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.category = 'Books'"},
  {"db_id": "demo_store", "category": "simple", "question": "How many different countries do our customers come from?", "SQL": "SELECT COUNT(DISTINCT country) FROM customers"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Show every customer and how many orders they placed, including customers with none.", "SQL": "SELECT c.name, COUNT(o.id) FROM customers c LEFT JOIN orders o ON o.customer_id = c.id GROUP BY c.id"},
  {"db_id": "demo_store", "category": "aggregation", "question": "How many units have been sold across all orders?", "SQL": "SELECT SUM(quantity) FROM order_items"},
  {"db_id": "demo_store", "category": "multi_step", "question": "Which customers placed more than one order? Return their names.", "SQL": "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id HAVING COUNT(*) > 1"}
]
```

- [ ] **Step 2.2: Write the failing test** — create `tests/test_golden_set.py`:

```python
"""Golden-set integrity: every gold query parses, runs read-only on the demo
DB, and returns at least one row — a broken golden set must never reach the gate.

    python tests/test_golden_set.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.executor import run_query  # noqa: E402

_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evals", "data", "golden_set.json"
)
_DEMO_DB = os.path.join(os.path.dirname(__file__), "..", "data", "demo_store.sqlite")

with open(_GOLDEN_PATH) as f:
    GOLDEN = json.load(f)


def test_golden_size_and_shape():
    assert len(GOLDEN) == 40
    for it in GOLDEN:
        assert it["db_id"] == "demo_store"
        assert it["question"].strip() and it["SQL"].strip()
        assert it["category"] in {
            "simple", "filter", "aggregation", "business_term",
            "multi_step", "negation", "date", "ambiguous",
        }


def test_every_gold_query_executes_with_rows():
    assert os.path.exists(_DEMO_DB), "run scripts/make_demo_db.py first"
    empty, errors = [], []
    for it in GOLDEN:
        try:
            rows, _ = run_query(it["SQL"], _DEMO_DB)
        except Exception as e:
            errors.append(f"{it['question']!r}: {e}")
            continue
        if not rows or all(all(c is None for c in r) for r in rows):
            empty.append(it["question"])
    assert not errors, f"gold SQL failed to execute: {errors}"
    assert not empty, f"gold SQL returned no data: {empty}"


def test_every_category_represented():
    cats = {it["category"] for it in GOLDEN}
    assert {"business_term", "ambiguous", "multi_step", "negation", "date"} <= cats


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} golden-set tests passed")
```

- [ ] **Step 2.3: Run to verify**

Run: `uv run python tests/test_golden_set.py`
Expected: `3 golden-set tests passed`. If any gold query errors or returns no rows, FIX THE GOLD SQL (the demo-DB facts table at the top of this plan is the ground truth) — do not weaken the test.

- [ ] **Step 2.4: Commit**

```bash
git add evals/data/golden_set.json tests/test_golden_set.py
git commit -m "feat: 40-question golden set with tier-1 integrity tests"
```

---

### Task 3: Explainer + faithfulness judge

**Files:**
- Modify: `querymate/llm.py` (append below `plan()`; `_JUDGE_FORMAT` goes next to `_PLAN_FORMAT`)

No no-key tests (repo convention for LLM-call code). Both functions follow the
advisory contract: any failure returns `(None, None)`.

- [ ] **Step 3.1: Add `_JUDGE_FORMAT`** directly below `_PLAN_FORMAT` in `llm.py`:

```python
_JUDGE_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "faithful": {
                "type": "boolean",
                "description": "True only if EVERY claim in the answer is "
                               "directly supported by the provided rows.",
            },
            "reason": {"type": "string",
                       "description": "One sentence justifying the verdict."},
        },
        "required": ["faithful", "reason"],
        "additionalProperties": False,
    },
}
```

- [ ] **Step 3.2: Append `explain()` and `judge_faithfulness()` at the end of `llm.py`:**

```python
def explain(
    *,
    question: str,
    columns: list[str],
    rows: list,
    model: str,
    max_tokens: int = 300,
) -> tuple[Optional[str], Optional[dict]]:
    """1-2 sentence answer narration over the result rows. Advisory — returns
    (None, None) on any failure; the caller never depends on it."""
    sample = rows[:20]
    t0 = time.monotonic()
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": (
                    "You summarise SQL query results. Answer the user's question "
                    "in 1-2 sentences using ONLY the rows provided. State numbers "
                    "exactly as they appear. If the rows don't answer the "
                    "question, say so."
                ),
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\nColumns: {columns}\n"
                    f"Rows (first {len(sample)} of {len(rows)}): {sample}"
                ),
            }],
        )
        entry = _usage_entry(resp, model, t0, purpose="explainer")
        return next((b.text for b in resp.content if b.type == "text"), None), entry
    except Exception:
        return None, None


def judge_faithfulness(
    *,
    question: str,
    answer: str,
    columns: list[str],
    rows: list,
    model: str,
    max_tokens: int = 300,
) -> tuple[Optional[dict], Optional[dict]]:
    """LLM-as-judge: does the answer only state what the rows support?
    Returns ({faithful, reason}, usage_entry); (None, None) on failure."""
    sample = rows[:20]
    t0 = time.monotonic()
    try:
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": (
                    "You are a strict faithfulness judge. Given a question, the "
                    "result rows, and an answer, decide whether every claim in "
                    "the answer is supported by the rows. Unsupported numbers, "
                    "invented entities, or over-claims mean NOT faithful."
                ),
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\nColumns: {columns}\n"
                    f"Rows (first {len(sample)} of {len(rows)}): {sample}\n"
                    f"Answer to judge: {answer}"
                ),
            }],
            output_config={"format": _JUDGE_FORMAT},
        )
        entry = _usage_entry(resp, model, t0, purpose="judge")
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        verdict = json.loads(text)
        if not isinstance(verdict, dict) or "faithful" not in verdict:
            return None, entry
        return verdict, entry
    except Exception:
        return None, None
```

- [ ] **Step 3.3: Verify imports + suite untouched**

Run: `uv run python -c "from querymate.llm import explain, judge_faithfulness; print('ok')" && uv run python tests/test_router.py`
Expected: `ok` and `11 router tests passed`.

- [ ] **Step 3.4: Commit**

```bash
git add querymate/llm.py
git commit -m "feat: minimal explainer + LLM-as-judge faithfulness call"
```

---

### Task 4: CLI answer line

**Files:**
- Modify: `querymate/cli.py`

- [ ] **Step 4.1: Add the `--no-explain` flag** in `main()` after the `--no-rag` argument:

```python
    ap.add_argument("--no-explain", action="store_true",
                    help="skip the natural-language answer line")
```

- [ ] **Step 4.2: Add the answer call.** In `main()`, replace the block from `cols = out.get("columns") or []` to the final `print(...)` with:

```python
    cols = out.get("columns") or []
    rows = out.get("rows") or []
    print("\nResult: " + (" | ".join(cols) if cols else "(no columns)"))
    for r in rows[:50]:
        print("  " + " | ".join(str(c) for c in r))

    cost_entries = list(out.get("cost_log", []))
    if not args.no_explain:
        from . import llm

        answer, entry = llm.explain(
            question=args.question, columns=cols, rows=rows,
            model=settings.fast_model,
        )
        if entry:
            cost_entries.append(entry)
        if answer:
            print(f"\nAnswer: {answer}")

    cost = sum(e.get("cost_usd", 0.0) for e in cost_entries)
    models = [e["model"].rsplit("-", 1)[0] for e in cost_entries
              if e.get("purpose") == "writer"]
    print(
        f"\n({len(rows)} row(s); repair attempts={out.get('attempts', 0)}; "
        f"writer={models[-1] if models else '-'}; cost=${cost:.4f})"
    )
```

- [ ] **Step 4.3: Verify**

Run: `uv run querymate --help && uv run python tests/test_graph.py`
Expected: help shows `--no-explain`; `4 graph tests passed`.

- [ ] **Step 4.4: Commit**

```bash
git add querymate/cli.py
git commit -m "feat: CLI Answer narration line via explainer (--no-explain to skip)"
```

---

### Task 5: Golden runner + gate

**Files:**
- Create: `evals/run_golden.py`
- Test: `tests/test_gate.py`
- Modify: `.gitignore` (add `evals/golden_report.json`)

- [ ] **Step 5.1: Write the failing test** — create `tests/test_gate.py`:

```python
"""Plain-assert tests for the regression-gate threshold logic.

    python tests/test_gate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.run_golden import gate_failures  # noqa: E402

BASE = {"ex": 0.80, "cost_usd_per_question": 0.010}


def test_no_regression_passes():
    assert gate_failures({"ex": 0.80, "cost_usd_per_question": 0.010}, BASE) == []
    assert gate_failures({"ex": 0.85, "cost_usd_per_question": 0.005}, BASE) == []


def test_ex_drop_boundary():
    # exactly 2pp down passes; more fails
    assert gate_failures({"ex": 0.78, "cost_usd_per_question": 0.010}, BASE) == []
    fails = gate_failures({"ex": 0.779, "cost_usd_per_question": 0.010}, BASE)
    assert len(fails) == 1 and "EX" in fails[0]


def test_cost_rise_boundary():
    # exactly +15% passes; more fails
    assert gate_failures({"ex": 0.80, "cost_usd_per_question": 0.0115}, BASE) == []
    fails = gate_failures({"ex": 0.80, "cost_usd_per_question": 0.0116}, BASE)
    assert len(fails) == 1 and "cost" in fails[0]


def test_both_fail_reported():
    fails = gate_failures({"ex": 0.5, "cost_usd_per_question": 1.0}, BASE)
    assert len(fails) == 2


def test_missing_cost_skips_cost_check():
    assert gate_failures({"ex": 0.80}, BASE) == []
    assert gate_failures({"ex": 0.80, "cost_usd_per_question": 1.0},
                         {"ex": 0.80}) == []


def test_faithfulness_not_gated():
    assert gate_failures(
        {"ex": 0.80, "cost_usd_per_question": 0.010, "faithfulness_rate": 0.1},
        {**BASE, "faithfulness_rate": 1.0},
    ) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} gate tests passed")
```

- [ ] **Step 5.2: Run to verify it fails**

Run: `uv run python tests/test_gate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.run_golden'`

- [ ] **Step 5.3: Create `evals/run_golden.py`:**

```python
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
```

- [ ] **Step 5.4: Run gate tests**

Run: `uv run python tests/test_gate.py`
Expected: `6 gate tests passed`

- [ ] **Step 5.5: Add to `.gitignore`:** append the line `evals/golden_report.json`

- [ ] **Step 5.6: Full suite**

Run: `for t in tests/test_*.py; do uv run python "$t" || break; done`
Expected: 11 suites all pass (the 8 Phase 1 suites + redteam + golden_set + gate).

- [ ] **Step 5.7: Commit**

```bash
git add evals/run_golden.py tests/test_gate.py .gitignore
git commit -m "feat: golden-set runner with baseline regression gate (EX/cost thresholds)"
```

---

### Task 6: GitHub Actions workflows

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/eval-gate.yml`

- [ ] **Step 6.1: Create `.github/workflows/ci.yml`** (tier 1 — free, every push/PR; FakeEmbedder only, no model download, no key):

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.13"
      - run: uv sync
      - run: uv run python scripts/make_demo_db.py
      - name: plain-assert suites (free tier — no LLM, no model download)
        run: |
          set -e
          for t in tests/test_*.py; do uv run python "$t"; done
```

- [ ] **Step 6.2: Create `.github/workflows/eval-gate.yml`** (tier 2 — manual trigger only; needs the `ANTHROPIC_API_KEY` repo secret):

```yaml
name: eval-gate

on:
  workflow_dispatch:

jobs:
  golden-gate:
    runs-on: ubuntu-latest
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.13"
      - run: uv sync
      - run: uv run python scripts/make_demo_db.py
      - run: uv run python scripts/ingest_schemas.py --demo
      - name: golden-set regression gate (EX / cost vs committed baseline)
        run: uv run python evals/run_golden.py --gate
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: golden-report
          path: evals/golden_report.json
```

- [ ] **Step 6.3: Validate YAML locally**

Run: `uv run python -c "import yaml,glob; [yaml.safe_load(open(p)) for p in glob.glob('.github/workflows/*.yml')]; print('yaml ok')"`
Expected: `yaml ok` (pyyaml is already a transitive dependency).

- [ ] **Step 6.4: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/eval-gate.yml
git commit -m "ci: free tier-1 gate on every push + manual golden-set eval gate"
```

---

### Task 7: README + wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 7.1: README edits** (read the file first; it's the Phase 1 version):

1. Title → `# QueryMate — NL→SQL analytics copilot (Phase 2)`, and add a CI badge line directly under it:

```markdown
[![ci](https://github.com/isaacng924/querymate/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacng924/querymate/actions/workflows/ci.yml)
```

2. Intro paragraph: replace the sentence starting `This repo is at **Phase 1**` through `...come in later phases.` with:

```markdown
This repo is at **Phase 2** of the build spec: everything from Phase 1
(schema-card RAG, advisory planner, model routing with cost accounting,
retrieval-aware repair, bucketed BIRD eval) plus an enforced eval harness — a
red-team safety suite and golden-set integrity checks gating every PR for
free, an answer narration with an LLM-as-judge faithfulness metric, and an
on-demand regression gate that blocks on EX or cost regressions against a
committed baseline. Langfuse dashboards, the chat UI, and the clarifier come
in later phases.
```

3. Add a new section after "## Run the execution-accuracy eval …":

````markdown
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
````

4. In the CLI section ("## Ask a question…"), note the answer line — append:

```markdown
The CLI ends with an `Answer:` line — a 1–2 sentence narration of the result
rows (skip with `--no-explain`).
```

5. Layout block updates: `evals/` line gains `run_golden · data/ (golden_set, redteam_corpus)`; `tests/` line gains `redteam · golden_set · gate`.
6. "What's deliberately *not* here yet": remove "a CI regression gate, a red-team suite" so it reads: Langfuse tracing (`querymate/trace.py` is the seam), the chat UI + clarifier, and a pgvector retriever backend.

- [ ] **Step 7.2: Full suite, final check**

Run: `for t in tests/test_*.py; do uv run python "$t" || break; done`
Expected: all 11 suites pass.

- [ ] **Step 7.3: Commit**

```bash
git add README.md
git commit -m "docs: Phase 2 README — safety suite, regression gate, CI badge"
```

---

## Runbook — activating the gate (needs API key; not part of the code tasks)

1. Add `ANTHROPIC_API_KEY` to the repo's GitHub secrets (Settings → Secrets → Actions) and to local `.env`.
2. `uv run python evals/run_golden.py --update-baseline` (~40 questions ≈ $1–2 with judge) → commit `evals/baseline.json`.
3. Trigger the `eval-gate` workflow once from the Actions tab to verify the green path.
4. From then on: tier 1 gates every PR free; run the gate manually after prompt/agent changes.

## Self-review notes

- Spec coverage: red-team (T1), golden set + integrity (T2), explainer + judge (T3), CLI answer (T4), runner + gate + baseline (T5), workflows (T6), README (T7), runbook for baseline creation. ✔
- Gold SQL verified against the demo-DB facts table (Dan orderless → negation non-empty; 'Home' the only Jan-unordered category; Carol the only UK customer without a January order). The tier-1 integrity test enforces this mechanically. ✔
- `gate_failures` boundary semantics match tests: strict `<` / `>` so exactly-2pp and exactly-+15% pass. ✔
- All new tier-1 tests are key-free and model-download-free (red-team + golden integrity hit only sqlglot/sqlite; gate tests are pure logic). ✔




