"""CLIP ViT-B/32 encoder and top-k retrieval with optional mock fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess.image import letterbox_image

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512

_TORCH_AVAILABLE = False
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore


class CLIPEncoder:
    """Frozen CLIP ViT-B/32 image/text encoder with retrieval."""

    def __init__(self, device: str | None = None, use_mock: bool | None = None):
        self.use_mock = use_mock if use_mock is not None else not _TORCH_AVAILABLE
        self.device = device or ("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

        if not self.use_mock:
            self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

        self._index_embeddings: np.ndarray | None = None
        self._index_labels: list[str] = []

    def _mock_embedding(self, seed_text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """512-dim L2-normalized image embedding."""
        if self.use_mock:
            letterboxed = letterbox_image(image)
            return self._mock_embedding(f"img:{letterboxed.tobytes()[:256]!r}")

        assert self.model is not None and self.processor is not None
        letterboxed = letterbox_image(image)
        inputs = self.processor(images=letterboxed, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()

    def encode_text(self, text: str) -> np.ndarray:
        """512-dim L2-normalized text embedding."""
        if self.use_mock:
            return self._mock_embedding(f"txt:{text}")

        assert self.model is not None and self.processor is not None
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()

    def cache_embedding(self, image: Image.Image, cache_path: str | Path) -> np.ndarray:
        emb = self.encode_image(image)
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    def load_or_encode(self, image: Image.Image, cache_path: str | Path | None) -> np.ndarray:
        if cache_path and Path(cache_path).exists():
            return np.load(cache_path)
        emb = self.encode_image(image)
        if cache_path:
            np.save(cache_path, emb)
        return emb

    def build_index(self, embeddings: np.ndarray, labels: list[str]) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._index_embeddings = embeddings / np.maximum(norms, 1e-8)
        self._index_labels = labels

    def top_k_matches(
        self,
        query_emb: np.ndarray,
        k: int = 5,
        candidate_labels: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if candidate_labels:
            text_embs = np.stack([self.encode_text(lbl) for lbl in candidate_labels])
            sims = text_embs @ query_emb
            order = np.argsort(-sims)[:k]
            return [(candidate_labels[i], float(sims[i])) for i in order]

        if self._index_embeddings is None:
            return []

        sims = self._index_embeddings @ query_emb
        order = np.argsort(-sims)[:k]
        return [(self._index_labels[i], float(sims[i])) for i in order]
