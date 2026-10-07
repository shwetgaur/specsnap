"""Pydantic models for SpecSnap test case output."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    ASSERT_TEXT = "assert_text"
    ASSERT_URL = "assert_url"
    GOTO = "goto"


class Step(BaseModel):
    step_id: str = Field(..., pattern=r"^s[0-9]+$")
    action: Action
    target: str | None = None
    value: str | None = None
    expected: str | None = None

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, v: str) -> str:
        if not v.startswith("s"):
            raise ValueError("step_id must start with 's'")
        return v


class TestCase(BaseModel):
    test_id: str
    source_screenshot: str
    objective: str
    expected_outcome: str
    steps: list[Step] = Field(..., min_length=1, max_length=8)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
