"""Build pilot manifest.csv from available corpus sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.manifest import steps_to_json_column

SAUCEDEMO_GT = [
    {"step_id": "s1", "action": "goto", "target": "Login page", "value": "https://www.saucedemo.com/", "expected": None},
    {"step_id": "s2", "action": "fill", "target": "Username", "value": "standard_user", "expected": None},
    {"step_id": "s3", "action": "fill", "target": "Password", "value": "secret_sauce", "expected": None},
    {"step_id": "s4", "action": "click", "target": "Login", "value": None, "expected": None},
    {"step_id": "s5", "action": "assert_text", "target": None, "value": None, "expected": "Products"},
]


def seed_manual_web() -> list[dict]:
    """Seed manual_web records (expand to 50+ for CA-3)."""
    img = ROOT / "data" / "manual_web" / "saucedemo" / "login_screen.png"
    if not img.exists():
        return []
    return [
        {
            "sample_id": "manual_web_saucedemo_login_001",
            "image_path": str(img.relative_to(ROOT)).replace("\\", "/"),
            "nl_objective": "Verify login with valid credentials",
            "nl_expected_outcome": "Products inventory page is visible",
            "ground_truth_steps": steps_to_json_column(SAUCEDEMO_GT),
            "app_category": "login",
            "source": "manual_web",
        }
    ]


def main():
    records = seed_manual_web()
    out = ROOT / "data" / "manifest.csv"
    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} records to {out}")
    print("Expand with Rico/MobilityUI/synthetic via dataset pipelines.")


if __name__ == "__main__":
    main()
