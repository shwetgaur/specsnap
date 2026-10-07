"""End-to-end Sauce Demo demo: screenshot → SpecSnap pipeline → JSON output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.specsnap_pipeline import SpecSnapPipeline

SCREENSHOT = ROOT / "data" / "manual_web" / "saucedemo" / "login_screen.png"
OUTPUT = ROOT / "results" / "saucedemo_generated.json"

OBJECTIVE = "Verify login with valid credentials"
EXPECTED = "Products inventory page is visible after successful login"


def main():
    if not SCREENSHOT.exists():
        print(f"Screenshot missing. Run: python scripts/capture_saucedemo.py")
        sys.exit(1)

    pipeline = SpecSnapPipeline(use_mock_llm=False)
    if not pipeline.llm.available:
        print("GROQ_API_KEY not set — using mock LLM for demo")
        pipeline = SpecSnapPipeline(use_mock_llm=True)

    print(f"Input:  {SCREENSHOT}")
    print(f"Objective: {OBJECTIVE}")
    print("Running SpecSnap pipeline (Baseline C)...")

    result = pipeline.generate(
        image_path=SCREENSHOT,
        objective=OBJECTIVE,
        expected_outcome=EXPECTED,
        baseline="C",
        source_screenshot=str(SCREENSHOT.relative_to(ROOT)),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_case": result.test_case,
        "validation": {
            "schema_valid": result.schema_valid,
            "grounding_valid": result.grounding_valid,
            "errors": result.errors,
        },
        "latency_ms": result.latency_ms,
        "clip_matches": result.clip_matches,
        "ocr_region_count": len(result.ocr_regions),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nLatency: {result.latency_ms:.0f} ms")
    print(f"Schema valid:    {result.schema_valid}")
    print(f"Grounding valid: {result.grounding_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
    print(f"\nOutput: {OUTPUT}")
    if result.test_case:
        print(json.dumps(result.test_case, indent=2))


if __name__ == "__main__":
    main()
