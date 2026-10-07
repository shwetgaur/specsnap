"""Create a Sauce Demo-like sample screenshot + OCR sidecar (offline fallback)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manual_web" / "saucedemo" / "login_screen.png"
SIDECAR = OUT.with_suffix(".png.ocr.json")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), color=(80, 81, 81))
    draw = ImageDraw.Draw(img)

    try:
        font_lg = ImageFont.truetype("arial.ttf", 36)
        font_md = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font_lg = ImageFont.load_default()
        font_md = font_lg

    draw.rectangle([440, 180, 840, 520], fill=(255, 255, 255))
    draw.text((500, 210), "Swag Labs", fill=(60, 60, 60), font=font_lg)

    fields = [
        ("Username", 500, 280, 760, 50),
        ("Password", 500, 360, 760, 50),
    ]
    regions = []
    for label, x, y, fw, fh in fields:
        draw.rectangle([x, y, x + fw, y + fh], outline=(180, 180, 180), fill=(250, 250, 250))
        draw.text((x + 10, y + 12), label, fill=(120, 120, 120), font=font_md)
        regions.append({"text": label, "confidence": 0.95, "left": x, "top": y, "width": fw, "height": fh})

    btn = (500, 440, 760, 490)
    draw.rectangle(btn, fill=(50, 120, 50))
    draw.text((580, 452), "Login", fill=(255, 255, 255), font=font_md)
    regions.append({"text": "Login", "confidence": 0.95, "left": btn[0], "top": btn[1], "width": btn[2] - btn[0], "height": btn[3] - btn[1]})

    img.save(OUT)
    SIDECAR.write_text(json.dumps({"regions": regions}, indent=2), encoding="utf-8")
    print(f"Created {OUT}")
    print(f"Created {SIDECAR}")


if __name__ == "__main__":
    main()
