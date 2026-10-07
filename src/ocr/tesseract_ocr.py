"""Tesseract OCR extraction with confidence filtering."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

CONFIDENCE_MIN = 0.7


@dataclass
class OCRRegion:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


def _configure_tesseract() -> None:
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found


def from_sidecar(sidecar_path: str | Path) -> list[OCRRegion]:
    """Load OCR regions from JSON sidecar (Playwright DOM text export)."""
    import json

    data = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    return [
        OCRRegion(
            text=item["text"],
            confidence=float(item.get("confidence", 0.95)),
            left=int(item.get("left", 0)),
            top=int(item.get("top", 0)),
            width=int(item.get("width", 0)),
            height=int(item.get("height", 0)),
        )
        for item in data.get("regions", [])
    ]


def extract_text(
    image: Image.Image,
    confidence_min: float = CONFIDENCE_MIN,
    sidecar_path: str | Path | None = None,
) -> list[OCRRegion]:
    """Extract on-screen text regions above confidence threshold."""
    if sidecar_path and Path(sidecar_path).exists():
        return from_sidecar(sidecar_path)

    _configure_tesseract()
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        return []

    regions: list[OCRRegion] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        conf_norm = conf / 100.0
        if conf_norm < confidence_min:
            continue
        regions.append(
            OCRRegion(
                text=text,
                confidence=conf_norm,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            )
        )
    return regions


def all_text(regions: list[OCRRegion]) -> list[str]:
    return [r.text for r in regions]


def text_corpus(regions: list[OCRRegion]) -> str:
    return " | ".join(r.text for r in regions)
