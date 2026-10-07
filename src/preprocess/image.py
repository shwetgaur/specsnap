"""Image preprocessing: letterbox, normalization, quality filters."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_SIZE = 512
BRIGHTNESS_MIN = 40
BLUR_LAPLACIAN_MIN = 100


def load_image(path: str | Path) -> Image.Image:
    """Load a PNG/JPEG screenshot as RGB PIL Image."""
    img = Image.open(path).convert("RGB")
    return img


def letterbox_image(
    image: Image.Image,
    size: int = TARGET_SIZE,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize with aspect ratio preserved, pad to square."""
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def to_numpy_normalized(image: Image.Image) -> np.ndarray:
    """Letterboxed CHW float32 array with ImageNet normalization."""
    arr = np.array(letterbox_image(image), dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return np.transpose(arr, (2, 0, 1))


def to_tensor(image: Image.Image):
    """Letterboxed ImageNet-normalized tensor [3, 512, 512] (requires torch)."""
    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torch/torchvision required for to_tensor()") from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(letterbox_image(image))


def mean_brightness(image: Image.Image) -> float:
    """Mean pixel intensity (0–255)."""
    arr = np.array(image.convert("L"), dtype=np.float32)
    return float(arr.mean())


def laplacian_variance(image: Image.Image) -> float:
    """Blur proxy via Laplacian variance."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_filters(
    image: Image.Image,
    brightness_min: float = BRIGHTNESS_MIN,
    blur_min: float = BLUR_LAPLACIAN_MIN,
) -> tuple[bool, dict[str, float]]:
    """Return (passes, metrics) for brightness and blur filters."""
    brightness = mean_brightness(image)
    blur = laplacian_variance(image)
    metrics = {"brightness": brightness, "laplacian_var": blur}
    passed = brightness >= brightness_min and blur >= blur_min
    return passed, metrics
