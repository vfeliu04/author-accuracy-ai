# Author AI — Report Accuracy & Credibility Checker

Author AI fact-checks a report against a set of trusted source documents. Upload source PDFs and a report PDF, and it extracts the factual claims from the report, verifies each claim against the sources with retrieval + LLM judgment, and scores the report on three metrics — then lets you interrogate the results through a grounded chat assistant.

## What it does

- **Accuracy** — every factual claim in the report is extracted, matched against the source documents (FAISS vector retrieval + Claude reranking), and labeled `SUPPORTED`, `CONTRADICTED`, or `NOT_FOUND` with a confidence band. Accuracy = supported claims ÷ total claims.
- **Credibility** — each source is scored 0–100 from its metadata (completeness, publisher authority, recency, enrichment confidence via Crossref), then aggregated into a report-level score weighted by how often each source actually supported claims.
- **Validity** — structural heuristics on the report itself: topic coverage, internal numeric consistency, methodology signals, geographic context, and source recency.
- **Recommended sources** — scholarly works fetched from the [OpenAlex](https://openalex.org) API, filtered by embedding similarity to the report's content.
- **Grounded chat** — an assistant (evidence / guidance / creative modes) that answers questions strictly from the stored claims, evidence snippets, and metrics. See [docs/chat.md](docs/chat.md).
- **Claims workspace** — a full-screen review UI showing each claim beside the report PDF and the cited source PDF, opened at the exact pages.

## How it works

```
 source PDFs ─┐
              ├─► ingestion (OCR · tables · charts) ─► chunking ─► OpenAI embeddings ─► FAISS + SQLite
 report PDF ──┘
                    │
                    ▼
        claim extraction (heuristics + Claude)
                    │
                    ▼
   per-claim retrieval ─► Claude rerank ─► Claude verdict ─► accuracy score
                    │
                    ▼
     credibility · validity · OpenAlex recommendations
                    │
                    ▼
        job result ─► React dashboard + grounded chat
```

The pipeline runs as a background job with step-by-step progress the UI polls. Full detail in [docs/architecture.md](docs/architecture.md) and [docs/metrics.md](docs/metrics.md).

**Stack:** Flask + SQLite + FAISS on the backend; Anthropic Claude for reranking, verdicts, summaries, and chat; OpenAI for text embeddings; React 18 + Vite + TypeScript on the frontend.

## Repository layout

```
backend/
  app.py                 Flask API (all HTTP endpoints, background jobs)
  author_ai/
    pipelines/           ingestion, accuracy, credibility, validity, chat
    services/            vector store, embeddings, reranker, verdict classifier,
                         OCR, tables, charts, summaries, OpenAlex recommendations
    storage/             SQLite repository
  tests/                 pytest suite
  .env.example           backend configuration template
frontend/
  src/                   React SPA (upload page, dashboard, claims workspace, chat)
  .env.example           frontend configuration template
example_sources/         sample PDFs to try the app with
docs/                    full documentation (see below)
```

## Quickstart

```bash
# Backend (Python 3.9+)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY / OPENAI_API_KEY
flask run --port 5001

# Frontend (Node 18+), in a second terminal
cd frontend
npm install
cp .env.example .env        # VITE_API_KEY must match the backend API_KEY
npm run dev                 # http://localhost:5173
```

Upload a few PDFs from `example_sources/` as sources, `World_Hunger_Fake.pdf` as the report, and run the pipeline. Full setup, optional tools (OCR, table extraction), and troubleshooting: [docs/development.md](docs/development.md).

## Documentation

| Doc | What's inside |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System design, data flow, backend layers, frontend structure |
| [docs/api.md](docs/api.md) | Every HTTP endpoint, auth, and error handling |
| [docs/metrics.md](docs/metrics.md) | The exact math behind accuracy, credibility, and validity |
| [docs/chat.md](docs/chat.md) | Chat modes, grounding, and claim commands |
| [docs/development.md](docs/development.md) | Setup, running, testing, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | Every environment variable and its default |
| [docs/history.md](docs/history.md) | How the project evolved; guide to the other branches |

## Project status

This is a working prototype built for local, single-user use:

- Each pipeline run **replaces** the previous run's data (single-report, last-run-wins design).
- Auth is a shared `X-API-Key` header — suitable for development, not production (it is disabled entirely if `API_KEY` is unset).
- The other branches in this repo are historical prototypes, not alternatives — see [docs/history.md](docs/history.md) before exploring them.

## License

Released under the [MIT License](LICENSE).
