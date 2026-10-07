"""Unit tests for grounding validator."""

from src.validate.grounding import validate_grounding


def test_grounded_target_passes():
    steps = [{"step_id": "s1", "action": "click", "target": "Login button", "value": None, "expected": None}]
    ok, errs = validate_grounding(steps, ocr_texts=["Login", "Username"], clip_labels=["login button"])
    assert ok
    assert errs == []


def test_ungrounded_target_fails():
    steps = [{"step_id": "s1", "action": "click", "target": "Invisible widget", "value": None, "expected": None}]
    ok, errs = validate_grounding(steps, ocr_texts=["Home"], clip_labels=["menu"])
    assert not ok
    assert len(errs) == 1


def test_goto_and_assert_skipped():
    steps = [
        {"step_id": "s1", "action": "goto", "target": "Any page", "value": "https://x.com", "expected": None},
        {"step_id": "s2", "action": "assert_text", "target": None, "value": None, "expected": "Done"},
    ]
    ok, errs = validate_grounding(steps, ocr_texts=[], clip_labels=[])
    assert ok
