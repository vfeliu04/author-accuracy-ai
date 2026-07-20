# Project History & Branch Guide

This repository contains four generations of the same idea — automatically checking a report's factual claims against trusted sources. Only one branch is the live product; the others are preserved history. This page explains what each branch is so you don't get lost.

## Branch guide

| Branch | What it is | State |
|---|---|---|
| `main` | The current app — the working branch (see below for its unusual past) | **Active — start here** |
| `author-ai-report-view` | The former development branch of the current app; work now continues on `main` | Superseded by `main` |
| `author_verify` | Generation 1: the "hallucination-checker" prototype (Flask + Jinja + Bootstrap) | Superseded, kept for reference |
| `author_verify_broken_code` | `author_verify` plus one experimental commit (literally titled "broken code") adding a multi-extractor `pdf_pipeline/` that never worked correctly | Quarantined experiment |
| `mvp_rag` | An unrelated, self-contained RAG demo (Flask + Qdrant + GPT-4o-mini) built on a cleared tree | Historical only |

## Timeline

### v0.1 — deterministic CLI (Oct 2025)

The project began as `author.ai v0.1`: a pure-Python, offline CLI (`author-extract` / `author-verify`) with regex claim extraction, pure-Python BM25 + hashed-cosine retrieval, a rule-based verification "judge", weighted linear scoring with temperature calibration, and an HTML report with inline claim highlights. No LLM calls at all. This code no longer exists on any branch tip, but is preserved in history at the commit just before `main` was cleared (`git show 8253ba5`).

### The fork (Oct 2025)

`main` was then deliberately emptied ("Clear main branch") and development split two ways:

- On top of the cleared tree, **`mvp_rag`** was built — a beginner-documented Flask + Qdrant + OpenAI RAG demo, unrelated to fact-checking. Abandoned after two days.
- From the pre-clear commit, **`author_verify`** restarted the fact-checking idea as the *hallucination-checker*: a Flask server-rendered app with FAISS retrieval over source PDFs, regex numeric-claim extraction, GPT-4.1 reranking/explanations, and a claim-detail UI with evidence switching. It worked, but PDF extraction and table rendering stayed flaky.

### The failed PDF pipeline experiment (Nov 2025)

`author_verify_broken_code` holds a single extra commit that bolted a second-generation PDF ingestion pipeline (document router, OCR, Camelot/Tabula/pdfplumber extractors, GPT table formatting) onto the gen-1 app. It shipped with table-ID collisions and a layout-extraction bug that disabled heading detection, was named "broken code" by its own author, and was never merged. It remains the only place that design exists.

### Generation 2 — the current app (Nov 2025 → present)

`author-ai-report-view` is a ground-up rewrite: the `backend/` Flask API with dedicated pipelines (ingestion, accuracy, credibility, validity, chat) and services, plus the `frontend/` React/Vite dashboard. Major milestones, in order:

1. **Frontend MVP and backend wiring** — upload flow, dashboard, chat UI.
2. **LLM verification loop** — claim extraction and verdicts connected end-to-end.
3. **Retrieval upgrades** — LangChain recursive text splitting; a LlamaIndex/Haystack experiment (Haystack was later removed after system errors); lightweight claim classification.
4. **Recommended sources** — OpenAlex queries from report-derived keywords with embedding-based re-ranking.
5. **Post-review hardening** (the "changes after Nico" pass) — retry with exponential backoff around LLM calls, and a per-claim `processing_mode` flag recording whether a verdict came from the LLM or the numeric heuristic.
6. **Claim quality guardrails** — heading/subheading filters, claim-ID tracking so chat can reference specific claims.
7. **The 2026 overhaul** — a five-front upgrade:
   - migration of all generation/judging LLM calls from OpenAI to **Anthropic Claude** (embeddings stayed OpenAI);
   - **asynchronous pipeline jobs** with per-step progress the UI polls;
   - **chart ingestion** — extracting figures from PDFs into synthetic evidence chunks so claims can be checked against charts;
   - an **accuracy-pipeline quality pass** — per-source retrieval, multi-evidence verdicts, temporal mismatch downgrades, compound-claim decomposition, non-numeric claim support;
   - the **dark-theme dashboard redesign** with the claims workspace (side-by-side claim/report/source review).

## Reading old code

All historical trees are reachable without switching branches:

```bash
git show 8253ba5 --stat          # v0.1 CLI, final state
git ls-tree -r author_verify --name-only | head    # gen-1 hallucination-checker
git show author_verify:hallucination-checker/src/hallcheck/verify.py
```
