"""Cache CLIP embeddings for manifest records."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess.image import load_image
from src.vision.clip_encoder import CLIPEncoder

MANIFEST = ROOT / "data" / "manifest.csv"
EMBED_DIR = ROOT / "data" / "embeddings"


def main():
    if not MANIFEST.exists():
        print("manifest.csv not found")
        sys.exit(1)

    df = pd.read_csv(MANIFEST)
    clip = CLIPEncoder()
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sid = row["sample_id"]
        cache = EMBED_DIR / f"{sid}.npy"
        if cache.exists():
            continue
        img_path = ROOT / row["image_path"]
        if not img_path.exists():
            continue
        image = load_image(img_path)
        clip.cache_embedding(image, cache)

    print(f"Embeddings cached in {EMBED_DIR}")


if __name__ == "__main__":
    main()
