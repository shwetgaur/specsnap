"""SpecSnapPipeline — end-to-end multimodal test case generation."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from src.llm.groq_client import GroqLLMClient, MockLLMClient
from src.llm.prompt import render_generation_prompt
from src.ocr.tesseract_ocr import extract_text, text_corpus
from src.preprocess.image import load_image, quality_filters
from src.validate.grounding import validate_grounding
from src.validate.schema_validator import validate_and_parse
from src.vision.clip_encoder import CLIPEncoder

BaselineMode = Literal["A", "B", "C", "D"]

# Default UI element labels for CLIP retrieval on web login screens
DEFAULT_CLIP_CANDIDATES = [
    "login button",
    "username text field",
    "password text field",
    "submit button",
    "sign in form",
    "navigation menu",
    "shopping cart icon",
    "product list page",
    "checkout button",
    "settings gear icon",
    "search bar",
    "error message banner",
]


@dataclass
class PipelineResult:
    test_case: dict | None
    schema_valid: bool
    grounding_valid: bool
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    ocr_regions: list[dict] = field(default_factory=list)
    clip_matches: list[tuple[str, float]] = field(default_factory=list)
    retries: int = 0
    baseline: str = "C"


class SpecSnapPipeline:
    """
    Late-fusion pipeline:
    preprocess → CLIP → OCR → Jinja2 prompt → Groq LLM → validate + grounding
    """

    def __init__(
        self,
        clip_top_k: int | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_retries: int = 1,
        use_mock_llm: bool = False,
    ):
        self.clip_top_k = int(clip_top_k or os.environ.get("CLIP_TOP_K", 5))
        self.max_steps = int(max_steps or os.environ.get("MAX_STEPS", 8))
        self.temperature = float(temperature or os.environ.get("LLM_TEMPERATURE", 0.2))
        self.max_retries = max_retries

        self.clip = CLIPEncoder()
        groq = GroqLLMClient(temperature=self.temperature)
        self.llm = MockLLMClient(temperature=self.temperature) if (use_mock_llm or not groq.available) else groq
        self._blip2 = None

    def _get_blip2(self):
        if self._blip2 is None:
            from src.vision.blip2_caption import BLIP2Captioner

            self._blip2 = BLIP2Captioner()
        return self._blip2

    def generate(
        self,
        image_path: str | Path,
        objective: str,
        expected_outcome: str,
        baseline: BaselineMode = "C",
        source_screenshot: str | None = None,
        clip_candidates: list[str] | None = None,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        path = Path(image_path)
        image = load_image(path)
        source = source_screenshot or str(path)

        passed, qmetrics = quality_filters(image)
        if not passed:
            return PipelineResult(
                test_case=None,
                schema_valid=False,
                grounding_valid=False,
                errors=[f"Image quality filter failed: {qmetrics}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
                baseline=baseline,
            )

        sidecar = path.with_suffix(path.suffix + ".ocr.json")
        ocr_regions = extract_text(image, sidecar_path=sidecar if sidecar.exists() else None)
        ocr_texts = [r.text for r in ocr_regions]
        ocr_str = text_corpus(ocr_regions)

        clip_matches: list[tuple[str, float]] = []
        image_caption: str | None = None

        if baseline in ("C", "D"):
            query_emb = self.clip.encode_image(image)
            candidates = clip_candidates or DEFAULT_CLIP_CANDIDATES
            clip_matches = self.clip.top_k_matches(query_emb, k=self.clip_top_k, candidate_labels=candidates)

        if baseline == "D":
            image_caption = self._get_blip2().caption(image)

        # Baseline-specific prompt context
        prompt_ocr = ocr_str if baseline in ("B", "C", "D") else ""
        prompt_clip = clip_matches if baseline == "C" else []
        prompt_caption = image_caption if baseline == "D" else None

        prompt = render_generation_prompt(
            objective=objective,
            expected_outcome=expected_outcome,
            ocr_text=prompt_ocr,
            clip_matches=prompt_clip,
            image_caption=prompt_caption,
            max_steps=self.max_steps,
        )

        errors: list[str] = []
        test_case: dict | None = None
        schema_valid = False
        grounding_valid = False
        retries = 0

        for attempt in range(self.max_retries + 1):
            retries = attempt
            try:
                raw = self.llm.generate(prompt)
            except Exception as exc:
                errors = [f"LLM generation failed: {exc}"]
                continue

            raw["source_screenshot"] = source
            raw["objective"] = objective
            raw["expected_outcome"] = expected_outcome
            if "test_id" not in raw:
                raw["test_id"] = f"generated_{uuid.uuid4().hex[:8]}"

            parsed, schema_errors = validate_and_parse(raw)
            if parsed is None:
                errors = schema_errors
                prompt += "\n\nPrevious output was invalid. Fix these errors: " + "; ".join(schema_errors)
                continue

            schema_valid = True
            g_ok, g_errors = validate_grounding(
                [s.model_dump(mode="json") for s in parsed.steps],
                ocr_texts=ocr_texts if baseline in ("B", "C", "D") else [],
                clip_labels=[m[0] for m in clip_matches],
            )
            grounding_valid = g_ok
            if not g_ok:
                errors = g_errors
                prompt += "\n\nGrounding violations: " + "; ".join(g_errors)
                continue

            test_case = parsed.model_dump(mode="json")
            errors = []
            break

        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            test_case=test_case,
            schema_valid=schema_valid,
            grounding_valid=grounding_valid,
            errors=errors,
            latency_ms=latency,
            ocr_regions=[r.__dict__ for r in ocr_regions],
            clip_matches=clip_matches,
            retries=retries,
            baseline=baseline,
        )
