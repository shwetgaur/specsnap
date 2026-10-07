"""FastAPI application for SpecSnap."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import __version__
from src.llm.groq_client import MockLLMClient
from src.pipeline.specsnap_pipeline import SpecSnapPipeline

app = FastAPI(
    title="SpecSnap",
    description="Multimodal AI for Automated Test Case Generation",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
_pipeline: SpecSnapPipeline | None = None


def get_pipeline() -> SpecSnapPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SpecSnapPipeline()
    return _pipeline


@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "ok",
        "version": __version__,
        "llm": "mock" if isinstance(pipe.llm, MockLLMClient) else "groq",
    }


@app.post("/api/v1/generate")
async def generate(
    image: UploadFile = File(...),
    objective: str = Form(...),
    expected_outcome: str = Form(...),
    baseline: str = Form(default="C"),
):
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = get_pipeline().generate(
            image_path=tmp_path,
            objective=objective,
            expected_outcome=expected_outcome,
            baseline=baseline if baseline in ("A", "B", "C", "D") else "C",
            source_screenshot=image.filename or "uploaded.png",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return JSONResponse(
        {
            "test_case": result.test_case,
            "validation": {
                "schema_valid": result.schema_valid,
                "grounding_valid": result.grounding_valid,
                "errors": result.errors,
                "retries": result.retries,
            },
            "metadata": {
                "latency_ms": round(result.latency_ms, 2),
                "baseline": result.baseline,
                "ocr_region_count": len(result.ocr_regions),
                "clip_matches": [{"label": m[0], "score": m[1]} for m in result.clip_matches],
            },
        }
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
