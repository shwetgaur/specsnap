"""Unit tests for JSON schema validation."""

from src.models.test_case import TestCase
from src.validate.schema_validator import validate_test_case


VALID = {
    "test_id": "t1",
    "source_screenshot": "a.png",
    "objective": "Login",
    "expected_outcome": "Dashboard",
    "steps": [
        {"step_id": "s1", "action": "goto", "target": "Home", "value": "https://x.com", "expected": None},
        {"step_id": "s2", "action": "click", "target": "Login", "value": None, "expected": None},
    ],
}


def test_valid_test_case():
    ok, errors = validate_test_case(VALID)
    assert ok
    assert errors == []


def test_invalid_action_rejected():
    bad = {**VALID, "steps": [{**VALID["steps"][0], "action": "tap"}]}
    ok, errors = validate_test_case(bad)
    assert not ok


def test_pydantic_parse():
    tc = TestCase.model_validate(VALID)
    assert tc.steps[0].action.value == "goto"
