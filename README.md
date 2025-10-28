# author.ai

author.ai now ships with a five-stage verification pipeline layered on top of the original regex-based extractor.  You can still run `author-extract` for legacy JSON, or upgrade to the full Stage A–E flow via `author-verify`.

## Pipeline Overview
```
[Stage A] Candidate Extraction --> [Stage B] Hybrid Retrieval --> [Stage C] Judge --> [Stage D] Scoring --> [Stage E] Report
```
- **Stage A — Candidate extraction** converts regex claims into the new strict `Claim` contract.
- **Stage B — Hybrid retrieval** indexes user-provided sources (BM25 + dense + numeric + tables).
- **Stage C — Judge** evaluates each claim using evidence-only JSON prompts (stubbed deterministically for tests).
- **Stage D — Scoring** combines deterministic features with configurable weights + temperature calibration.
- **Stage E — Report** produces an HTML report with inline highlights and evidence cards plus JSONL for downstream tooling.

Configuration lives in `author_ai/config.py`; every tolerance, weight, and batch size is tweakable.

## Features
- Detect statistics, ratios (`1 in 5`), ranges (`10–12%`), deltas (`up 2pp vs 2023`), relative changes, and per-capita rates.
- Locale-aware number parsing with unit normalisation and canonical representations.
- Hybrid retrieval stubs that run offline (BM25, dense cosine, number index, table rows).
- Deterministic judge + scoring layers with strict Pydantic validation.
- Click-powered CLIs: `author-extract` (legacy) and `author-verify` (Stage A–E).

## Getting Started

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/)

### Installation
```bash
poetry install
```

### Running the Legacy Extractor
```bash
echo "About 23.5% of UK A&E attendances in Q2 2024 breached the four-hour standard, up 2pp vs 2023." \
  | poetry run author-extract --pretty
```

### Running the Full Pipeline
1. Prepare a directory of plain-text / Markdown / JSON sources, e.g. `samples/`.
2. Provide the article to analyse via stdin or `--infile`.
```bash
poetry run author-verify --infile sample.txt --sources ./samples/ --outdir ./artifacts
```
- STDOUT: JSON lines (one per claim) combining claim + verification + score + evidence ids.
- Artifacts: `artifacts/author_ai_report.html` and `artifacts/author_ai_report.jsonl` with inline highlights and evidence cards.

## Configuration
- Defaults live in `author_ai/config.py`.
- Adjust tolerances, weights, and retrieval parameters directly or load the dataclasses and override in code.

## Testing
```bash
poetry run pytest
```
The suite covers models, each pipeline stage, scoring logic, and CLI smoke tests.

## Tooling
- `poetry run ruff check .`
- `poetry run black .`
- `poetry run pre-commit run --all-files`
