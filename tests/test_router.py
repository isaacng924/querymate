"""Plain-assert tests for model routing + retrieval-widening decisions.

    python tests/test_router.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from querymate.router import pick_model, should_widen  # noqa: E402
from querymate.settings import settings  # noqa: E402


def test_simple_plan_routes_to_fast_model():
    plan = {"tables": ["customers"], "join_count": 0, "aggregations": [], "filters": []}
    assert pick_model(plan, attempts=0, max_attempts=3) == settings.fast_model


def test_joins_route_to_writer_model():
    plan = {"tables": ["a", "b"], "join_count": 1, "aggregations": [], "filters": []}
    assert pick_model(plan, attempts=0, max_attempts=3) == settings.writer_model


def test_aggregations_route_to_writer_model():
    plan = {"tables": ["a"], "join_count": 0, "aggregations": ["COUNT"], "filters": []}
    assert pick_model(plan, attempts=0, max_attempts=3) == settings.writer_model


def test_no_plan_routes_to_writer_model():
    assert pick_model(None, attempts=0, max_attempts=3) == settings.writer_model


def test_final_attempt_escalates_even_with_simple_plan():
    plan = {"tables": ["a"], "join_count": 0, "aggregations": [], "filters": []}
    assert pick_model(plan, attempts=2, max_attempts=3) == settings.escalate_model


def test_zero_max_attempts_never_escalates():
    assert pick_model(None, attempts=0, max_attempts=0) == settings.writer_model


def test_widen_on_unknown_table_once():
    assert should_widen("unknown_table", widened=False, use_retrieval=True)
    assert should_widen("unknown_column", widened=False, use_retrieval=True)


def test_no_widen_when_already_widened_or_no_retrieval_or_other_error():
    assert not should_widen("unknown_table", widened=True, use_retrieval=True)
    assert not should_widen("unknown_table", widened=False, use_retrieval=False)
    assert not should_widen("syntax", widened=False, use_retrieval=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} router tests passed")
