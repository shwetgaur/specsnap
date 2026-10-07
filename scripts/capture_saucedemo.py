"""Capture Sauce Demo login screenshot at 1280x720 for manual_web corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "data" / "manual_web" / "saucedemo" / "login_screen.png"
OCR_SIDEcar = OUTPUT.with_suffix(".png.ocr.json")


def export_dom_text(page) -> dict:
    """Export visible text as OCR sidecar for pipeline grounding."""
    regions = []
    for el in page.locator("input, button, label, h1, h2, h3, h4, span, div").all():
        try:
            text = (el.inner_text() or el.get_attribute("placeholder") or "").strip()
            if not text or len(text) > 80:
                continue
            box = el.bounding_box()
            if not box:
                continue
            regions.append(
                {
                    "text": text,
                    "confidence": 0.95,
                    "left": int(box["x"]),
                    "top": int(box["y"]),
                    "width": int(box["width"]),
                    "height": int(box["height"]),
                }
            )
        except Exception:
            continue
    # Dedupe by text
    seen = set()
    unique = []
    for r in regions:
        key = r["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return {"regions": unique}


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install playwright: pip install playwright && playwright install chromium")
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto("https://www.saucedemo.com/", wait_until="networkidle")
        page.screenshot(path=str(OUTPUT), full_page=False)
        sidecar = export_dom_text(page)
        OCR_SIDEcar.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        browser.close()

    print(f"Saved: {OUTPUT}")
    print(f"OCR sidecar: {OCR_SIDEcar} ({len(sidecar['regions'])} regions)")


if __name__ == "__main__":
    main()
