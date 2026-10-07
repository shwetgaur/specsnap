"""Dataset manifest utilities and train/val/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
TRAIN_SIZE = 600
VAL_SIZE = 150
TEST_SIZE = 100


REQUIRED_COLUMNS = [
    "sample_id",
    "image_path",
    "nl_objective",
    "nl_expected_outcome",
    "ground_truth_steps",
    "app_category",
    "source",
]


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df


def save_splits(
    manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = SPLIT_SEED,
) -> dict[str, Path]:
    """Split manifest into train/val/test (600/150/100 for 850 records)."""
    df = load_manifest(manifest_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # First hold out test
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=df["app_category"]
    )
    # Then split train/val
    val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=seed, stratify=train_val["app_category"]
    )

    paths = {
        "train": out / "train.csv",
        "val": out / "val.csv",
        "test": out / "test.csv",
    }
    train.to_csv(paths["train"], index=False)
    val.to_csv(paths["val"], index=False)
    test.to_csv(paths["test"], index=False)
    return paths


def steps_to_json_column(steps: list[dict]) -> str:
    return json.dumps(steps, ensure_ascii=False)
