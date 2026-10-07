"""Evaluation metrics: step match rate, step-level F1."""

from __future__ import annotations

from typing import Any


def _step_key(step: dict[str, Any]) -> tuple:
    return (
        step.get("action"),
        (step.get("target") or "").lower().strip(),
        (step.get("value") or "").lower().strip(),
        (step.get("expected") or "").lower().strip(),
    )


def step_match_rate(predicted: list[dict], ground_truth: list[dict]) -> float:
    """
    Fraction of ground-truth steps matched in prediction (order-aware prefix match).
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    matched = 0
    for i, gt in enumerate(ground_truth):
        if i >= len(predicted):
            break
        if _step_key(predicted[i]) == _step_key(gt):
            matched += 1
        else:
            break

    return matched / len(ground_truth)


def step_level_f1(predicted: list[dict], ground_truth: list[dict]) -> dict[str, float]:
    """Set-based step F1 treating each step as a tuple key."""
    pred_set = {_step_key(s) for s in predicted}
    gt_set = {_step_key(s) for s in ground_truth}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
