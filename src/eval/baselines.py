"""Baseline runners A/B/C/D for validation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.eval.metrics import step_level_f1, step_match_rate
from src.pipeline.specsnap_pipeline import SpecSnapPipeline


@dataclass
class EvalRecord:
    sample_id: str
    baseline: str
    step_match: float
    f1: float
    schema_valid: bool
    grounding_valid: bool
    latency_ms: float


def run_baseline_on_manifest(
    manifest_path: str | Path,
    baseline: str,
    pipeline: SpecSnapPipeline | None = None,
    limit: int | None = None,
) -> list[EvalRecord]:
    """Run one baseline over manifest rows with ground_truth_steps."""
    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit)

    pipe = pipeline or SpecSnapPipeline(use_mock_llm=True)
    records: list[EvalRecord] = []

    for _, row in df.iterrows():
        import json

        gt_steps = json.loads(row["ground_truth_steps"]) if isinstance(row["ground_truth_steps"], str) else row["ground_truth_steps"]

        result = pipe.generate(
            image_path=row["image_path"],
            objective=row["nl_objective"],
            expected_outcome=row["nl_expected_outcome"],
            baseline=baseline,  # type: ignore[arg-type]
        )

        pred_steps = result.test_case["steps"] if result.test_case else []
        sm = step_match_rate(pred_steps, gt_steps)
        f1 = step_level_f1(pred_steps, gt_steps)["f1"]

        records.append(
            EvalRecord(
                sample_id=row["sample_id"],
                baseline=baseline,
                step_match=sm,
                f1=f1,
                schema_valid=result.schema_valid,
                grounding_valid=result.grounding_valid,
                latency_ms=result.latency_ms,
            )
        )

    return records


def summarize(records: list[EvalRecord]) -> dict:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "mean_step_match": sum(r.step_match for r in records) / len(records),
        "mean_f1": sum(r.f1 for r in records) / len(records),
        "schema_pass_rate": sum(r.schema_valid for r in records) / len(records),
        "grounding_pass_rate": sum(r.grounding_valid for r in records) / len(records),
        "mean_latency_ms": sum(r.latency_ms for r in records) / len(records),
    }
