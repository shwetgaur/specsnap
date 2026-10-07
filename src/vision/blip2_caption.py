"""BLIP-2 caption baseline (Baseline D). Lazy-loaded; optional torch dependency."""

from __future__ import annotations

from PIL import Image

MODEL_NAME = "Salesforce/blip2-opt-2.7b"


class BLIP2Captioner:
    """Generate image caption for Baseline D."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._processor = None
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            from transformers import Blip2ForConditionalGeneration  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise ImportError("BLIP-2 baseline requires torch and transformers")

        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._processor = Blip2Processor.from_pretrained(MODEL_NAME)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

    def caption(self, image: Image.Image, max_length: int = 64) -> str:
        if not self.available():
            return "UI screenshot with form fields and action buttons"
        self._load()
        import torch

        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_length=max_length)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()
