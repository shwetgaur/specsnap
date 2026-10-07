"""Grounding validator: targets must appear in OCR or CLIP context."""

from __future__ import annotations

import re
from typing import Iterable


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _fuzzy_in_corpus(target: str, corpus: str) -> bool:
    """Check if target (or significant tokens) appear in corpus."""
    t = _normalize(target)
    c = _normalize(corpus)
    if t in c:
        return True
    tokens = [tok for tok in re.split(r"[\s\-_/]+", t) if len(tok) >= 3]
    if not tokens:
        return t in c
    matched = sum(1 for tok in tokens if tok in c)
    return matched >= max(1, len(tokens) // 2)


def build_grounding_corpus(
    ocr_texts: Iterable[str],
    clip_labels: Iterable[str],
    extra: Iterable[str] | None = None,
) -> str:
    parts = list(ocr_texts) + list(clip_labels)
    if extra:
        parts.extend(extra)
    return " | ".join(parts)


def validate_grounding(
    steps: list[dict],
    ocr_texts: list[str],
    clip_labels: list[str],
    extra_context: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Verify each non-null target is grounded in OCR or CLIP context.
    Returns (all_grounded, list of violation messages).
    """
    corpus = build_grounding_corpus(ocr_texts, clip_labels, extra_context)
    violations: list[str] = []

    for step in steps:
        target = step.get("target")
        action = step.get("action", "")
        if target is None or action in ("assert_text", "assert_url", "goto"):
            continue
        if not _fuzzy_in_corpus(str(target), corpus):
            violations.append(
                f"Step {step.get('step_id')}: target '{target}' not grounded in OCR/CLIP context"
            )

    return len(violations) == 0, violations
