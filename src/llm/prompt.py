"""Jinja2 prompt templates for test case generation."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_generation_prompt(
    objective: str,
    expected_outcome: str,
    ocr_text: str,
    clip_matches: list[tuple[str, float]],
    image_caption: str | None = None,
    max_steps: int = 8,
) -> str:
    """Assemble the LLM prompt from NL + OCR + CLIP context."""
    template = _env.get_template("generate_test.j2")
    return template.render(
        objective=objective,
        expected_outcome=expected_outcome,
        ocr_text=ocr_text,
        clip_matches=clip_matches,
        image_caption=image_caption,
        max_steps=max_steps,
        allowed_actions=["click", "fill", "select", "assert_text", "assert_url", "goto"],
    )
