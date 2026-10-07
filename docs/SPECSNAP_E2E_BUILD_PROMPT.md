# SpecSnap — End-to-End Build Prompt

**Project:** SpecSnap — Multimodal AI for Automated Test Case Generation  
**Course:** B.Tech AIML · Multimodal AI Mini Project · Group 32 · SIT Pune  
**Team:** Sehajdeep Singh Sikka (23070126119), Shwet Gaur (23070126126)  
**Guide:** Prof. Nivedita Mishra · **SDG:** 9

## Standalone Scope

SpecSnap **generates** structured JSON test steps from UI screenshots + natural language.  
It does **not** depend on DS-1 (`agentic-webapp-test-executor`). Optional Playwright demo runs inside this repo only.

## Input / Output

| Input | Output |
|-------|--------|
| PNG screenshot + `nl_objective` + `nl_expected_outcome` | JSON test steps (`click`, `fill`, `select`, `assert_text`, `assert_url`, `goto`) |

## Pipeline

```
letterbox 512×512 → CLIP ViT-B/32 → Tesseract OCR →
Jinja2 prompt (NL + OCR + top-5 CLIP) → Groq Llama 3.1 8B (temp=0.2, max 8 steps) →
jsonschema + grounding check → TestCase JSON
```

## Dataset (850 records)

| Source | Count |
|--------|-------|
| Rico | 500 |
| MobilityUI | 200 |
| Manual web | 50 → 150 (CA-3) |
| Synthetic | 100 |

Split (seed=42): train 600 / val 150 / test 100 (sealed in `data/test_holdout/`).

## Evaluation Baselines

| Baseline | Input | Target |
|----------|-------|--------|
| A | NL only | ~60% step match |
| B | NL + OCR | ~73% |
| C | NL + CLIP top-5 | ≥90% (production) |
| D | NL + BLIP-2 caption | ~83% |

## Definition of Done

See project README checklist. Key deliverables: manifest, CLIP cache, Groq generation, grounding validator, baselines, FastAPI + UI, EDA notebook, unit tests.

## First Session Goal

Scaffold repo + **single Sauce Demo screenshot end-to-end**:

```bash
python scripts/capture_saucedemo.py
python scripts/e2e_saucedemo.py
python scripts/demo_playwright.py   # optional
uvicorn src.api.main:app --reload
```
