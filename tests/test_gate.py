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
