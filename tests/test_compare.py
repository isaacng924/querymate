"""Plain-assert tests for execution-accuracy comparison.

    python tests/test_compare.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.compare import execution_match  # noqa: E402


def test_same_rows_different_order_match():
    assert execution_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])


def test_different_multiset_no_match():
    assert not execution_match([(1,), (1,)], [(1,)])


def test_float_rounding_match():
    assert execution_match([(209.0000001,)], [(209.0,)])


def test_int_float_equiv_match():
    assert execution_match([(3,)], [(3.0,)])


def test_none_handling():
    assert execution_match([(None, 1)], [(None, 1)])
    assert not execution_match([(None,)], [(0,)])


def test_none_pred_no_match():
    assert not execution_match(None, [(1,)])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} compare tests passed")
