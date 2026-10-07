"""Groq Llama 3.1 8B client for test case generation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.2


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class GroqLLMClient:
    """Generate test cases via Groq API."""

    def __init__(
        self,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        model: str = MODEL,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.temperature = temperature
        self.model = model
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, max_tokens: int = 2048) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("GROQ_API_KEY not set")

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You output only valid JSON test case objects. No explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return _extract_json(content)


class MockLLMClient:
    """Deterministic fallback when Groq is unavailable (local dev / CI)."""

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        self.temperature = temperature

    @property
    def available(self) -> bool:
        return True

    def generate(self, prompt: str, max_tokens: int = 2048) -> dict[str, Any]:
        """Heuristic JSON from prompt OCR section for Sauce Demo login."""
        ocr_lower = prompt.lower()
        steps: list[dict] = []

        if "saucedemo" in prompt.lower() or "login" in prompt.lower():
            steps = [
                {
                    "step_id": "s1",
                    "action": "goto",
                    "target": "Login page",
                    "value": "https://www.saucedemo.com/",
                    "expected": None,
                },
                {
                    "step_id": "s2",
                    "action": "fill",
                    "target": "Username",
                    "value": "standard_user",
                    "expected": None,
                },
                {
                    "step_id": "s3",
                    "action": "fill",
                    "target": "Password",
                    "value": "secret_sauce",
                    "expected": None,
                },
                {
                    "step_id": "s4",
                    "action": "click",
                    "target": "Login",
                    "value": None,
                    "expected": None,
                },
                {
                    "step_id": "s5",
                    "action": "assert_text",
                    "target": None,
                    "value": None,
                    "expected": "Products",
                },
            ]
        else:
            # Generic fallback from OCR tokens
            targets = []
            for token in ["login", "username", "password", "submit", "sign in", "continue"]:
                if token in ocr_lower:
                    targets.append(token.title())
            if not targets:
                targets = ["Submit button"]

            steps.append(
                {
                    "step_id": "s1",
                    "action": "click",
                    "target": targets[0],
                    "value": None,
                    "expected": None,
                }
            )
            steps.append(
                {
                    "step_id": "s2",
                    "action": "assert_text",
                    "target": None,
                    "value": None,
                    "expected": "Success",
                }
            )

        obj_match = re.search(r"## Objective\s*\n(.+?)\n", prompt)
        exp_match = re.search(r"## Expected Outcome\s*\n(.+?)\n", prompt)
        objective = obj_match.group(1).strip() if obj_match else "Verify UI flow"
        expected = exp_match.group(1).strip() if exp_match else "Expected state reached"

        return {
            "test_id": "generated_001",
            "source_screenshot": "uploaded.png",
            "objective": objective,
            "expected_outcome": expected,
            "steps": steps[:8],
        }
