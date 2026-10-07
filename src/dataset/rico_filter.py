"""Rico dataset filtering pipeline (reproducible)."""

from __future__ import annotations

import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.ocr.tesseract_ocr import extract_text
from src.preprocess.image import load_image, mean_brightness

PHASH_HAMMING_MAX = 5
BRIGHTNESS_MIN = 40
WIDGET_MIN_PX = 32
OCR_CONF_MIN = 0.7


def widget_interactive(metadata: dict, min_px: int = WIDGET_MIN_PX) -> bool:
    """Check Rico JSON metadata for ≥1 widget ≥ min_px."""
    widgets = metadata.get("widgets", []) or metadata.get("children", [])
    for w in widgets:
        bounds = w.get("bounds") or w.get("bbox")
        if not bounds:
            continue
        if len(bounds) == 4:
            w_px = bounds[2] - bounds[0]
            h_px = bounds[3] - bounds[1]
        else:
            w_px = bounds.get("width", 0)
            h_px = bounds.get("height", 0)
        if w_px >= min_px and h_px >= min_px:
            return True
    return False


def passes_ocr(image: Image.Image) -> bool:
    regions = extract_text(image, confidence_min=OCR_CONF_MIN)
    return len(regions) >= 1


def dedup_phash(images: list[tuple[str, Image.Image]], max_dist: int = PHASH_HAMMING_MAX) -> list[str]:
    """Greedy dedup by pHash Hamming distance."""
    kept: list[str] = []
    hashes: list[imagehash.ImageHash] = []

    for sample_id, img in images:
        h = imagehash.phash(img)
        if any(h - prev <= max_dist for prev in hashes):
            continue
        kept.append(sample_id)
        hashes.append(h)
    return kept


def filter_rico_record(image_path: Path, meta_path: Path | None) -> tuple[bool, dict]:
    """Apply ordered Rico filters to one record."""
    img = load_image(image_path)
    metrics: dict = {}

    if mean_brightness(img) < BRIGHTNESS_MIN:
        metrics["reject"] = "brightness"
        return False, metrics

    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not widget_interactive(meta):
            metrics["reject"] = "interactivity"
            return False, metrics

    if not passes_ocr(img):
        metrics["reject"] = "ocr"
        return False, metrics

    metrics["pass"] = True
    return True, metrics


def stratified_sample(
    records: list[dict],
    n: int,
    category_key: str = "app_category",
    seed: int = 42,
) -> list[dict]:
    """Stratified sample by app category."""
    import pandas as pd

    df = pd.DataFrame(records)
    if len(df) <= n:
        return records

    frac = n / len(df)
    sampled = df.groupby(category_key, group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * frac)), random_state=seed),
        include_groups=False,
    )
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    return sampled.to_dict("records")
