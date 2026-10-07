"""JSON Schema validation for test case output."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from jsonschema import Draft7Validator

from src.models.test_case import TestCase

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "test_case_output.v1.json"


def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_test_case(data: dict) -> tuple[bool, list[str]]:
    """Validate dict against JSON schema. Returns (valid, errors)."""
    schema = load_schema()
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if not errors:
        return True, []
    messages = [f"{list(e.path)}: {e.message}" for e in errors]
    return False, messages


def validate_and_parse(data: dict) -> tuple[TestCase | None, list[str]]:
    """Validate and parse into Pydantic model."""
    ok, errors = validate_test_case(data)
    if not ok:
        return None, errors
    try:
        return TestCase.model_validate(data), []
    except Exception as exc:
        return None, [str(exc)]
