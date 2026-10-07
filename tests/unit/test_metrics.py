"""Unit tests for evaluation metrics."""

from src.eval.metrics import step_level_f1, step_match_rate


GT = [
    {"action": "goto", "target": "Page", "value": "https://a.com", "expected": None},
    {"action": "click", "target": "Login", "value": None, "expected": None},
    {"action": "assert_text", "target": None, "value": None, "expected": "OK"},
]

PRED = [
    {"action": "goto", "target": "Page", "value": "https://a.com", "expected": None},
    {"action": "click", "target": "Login", "value": None, "expected": None},
    {"action": "assert_text", "target": None, "value": None, "expected": "OK"},
]


def test_perfect_step_match():
    assert step_match_rate(PRED, GT) == 1.0


def test_partial_step_match():
    partial = PRED[:2] + [{"action": "click", "target": "Wrong", "value": None, "expected": None}]
    assert step_match_rate(partial, GT) == 2 / 3


def test_step_f1_perfect():
    m = step_level_f1(PRED, GT)
    assert m["f1"] == 1.0
