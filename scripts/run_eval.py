"""Run baselines A/B/C/D on validation split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.baselines import run_baseline_on_manifest, summarize
from src.pipeline.specsnap_pipeline import SpecSnapPipeline

MANIFEST = ROOT / "data" / "splits" / "val.csv"
FALLBACK_MANIFEST = ROOT / "data" / "manifest.csv"
OUTPUT = ROOT / "results" / "eval_results.json"


def main():
    manifest = MANIFEST if MANIFEST.exists() else FALLBACK_MANIFEST
    if not manifest.exists():
        print("No manifest found. Run scripts/build_manifest.py first.")
        sys.exit(1)

    pipeline = SpecSnapPipeline(use_mock_llm=True)
    all_results = {}

    for baseline in ("A", "B", "C", "D"):
        if baseline == "D":
            from src.vision.blip2_caption import BLIP2Captioner
            if not BLIP2Captioner.available():
                print("Skipping Baseline D (torch/transformers not installed)")
                all_results["D"] = {"summary": {"skipped": True}, "records": []}
                continue
        print(f"Running Baseline {baseline}...")
        records = run_baseline_on_manifest(manifest, baseline, pipeline=pipeline)
        all_results[baseline] = {
            "summary": summarize(records),
            "records": [r.__dict__ for r in records],
        }
        print(f"  step_match={all_results[baseline]['summary'].get('mean_step_match', 0):.2%}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults: {OUTPUT}")


if __name__ == "__main__":
    main()
