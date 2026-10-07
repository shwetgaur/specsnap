"""Unit tests for image preprocessing."""

from PIL import Image

from src.preprocess.image import letterbox_image, mean_brightness, quality_filters


def test_letterbox_is_square():
    img = Image.new("RGB", (800, 600), color=(128, 128, 128))
    out = letterbox_image(img, size=512)
    assert out.size == (512, 512)


def test_brightness_solid_gray():
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    assert mean_brightness(img) == 100.0


def test_quality_filters_pass_bright_sharp():
    img = Image.new("RGB", (200, 200), color=(200, 200, 200))
    # Add noise for laplacian variance
    pixels = img.load()
    for x in range(200):
        for y in range(200):
            v = 200 if (x + y) % 2 == 0 else 180
            pixels[x, y] = (v, v, v)
    passed, metrics = quality_filters(img)
    assert passed
    assert metrics["brightness"] >= 40
