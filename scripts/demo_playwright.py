"""Optional Playwright demo: execute generated Sauce Demo test steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "results" / "saucedemo_generated.json"


def run_steps(steps: list[dict]) -> bool:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        for step in steps:
            action = step["action"]
            target = step.get("target")
            value = step.get("value")
            expected = step.get("expected")

            if action == "goto":
                page.goto(value or "https://www.saucedemo.com/")
            elif action == "fill":
                page.fill(f"input[placeholder*='{target}' i], input[name*='{target}' i]", value or "")
            elif action == "click":
                page.get_by_role("button", name=target or "Login").click()
            elif action == "assert_text":
                page.get_by_text(expected or "").first.wait_for(timeout=5000)
            elif action == "assert_url":
                assert expected in page.url
            print(f"  OK  {step['step_id']}: {action}")

        browser.close()
    return True


def main():
    if not GENERATED.exists():
        print(f"Run scripts/e2e_saucedemo.py first. Missing: {GENERATED}")
        sys.exit(1)

    data = json.loads(GENERATED.read_text(encoding="utf-8"))
    test_case = data.get("test_case")
    if not test_case:
        print("No test_case in generated output")
        sys.exit(1)

    print("Executing generated test steps on Sauce Demo...")
    try:
        run_steps(test_case["steps"])
        print("Demo PASSED")
    except Exception as exc:
        print(f"Demo FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
