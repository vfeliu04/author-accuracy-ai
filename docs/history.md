# Project History & Branch Guide

This repository contains five generations of the same idea — automatically checking a report's factual claims against trusted sources. Only one lineage is the live product; the others are preserved history. This page explains what each branch is so you don't get lost.

## Branch guide

| Branch | What it is | State |
|---|---|---|
| `main` | The current app (see below for its unusual past) | **Active — start here** |
| `v2` | The clean-slate rebuild — FastAPI backend rewrite + adapted frontend, merging into `main` (see [The v2 rebuild](#the-v2-rebuild-2026-08)) | Development branch of the current app |
| `ultra-base` | A pinned snapshot of `v2` history kept only as the diff base for a deep multi-agent code review | Review artifact, not a line of development |
| `author_verify` | Generation 1: the "hallucination-checker" prototype (Flask + Jinja + Bootstrap) | Superseded, kept for reference |
| `author_verify_broken_code` | `author_verify` plus one experimental commit (literally titled "broken code") adding a multi-extractor `pdf_pipeline/` that never worked correctly | Quarantined experiment |
| `mvp_rag` | An unrelated, self-contained RAG demo (Flask + Qdrant + GPT-4o-mini) built on a cleared tree | Historical only |

(`author-ai-report-view`, the former development branch of the v1 app, was folded into `main` and deleted; its history is all reachable from `main`.)

## Timeline

### v0.1 — deterministic CLI (Oct 2025)

The project began as `author.ai v0.1`: a pure-Python, offline CLI (`author-extract` / `author-verify`) with regex claim extraction, pure-Python BM25 + hashed-cosine retrieval, a rule-based verification "judge", weighted linear scoring with temperature calibration, and an HTML report with inline claim highlights. No LLM calls at all. This code no longer exists on any branch tip, but is preserved in history at the commit just before `main` was cleared (`git show 8253ba5`).

### The fork (Oct 2025)

`main` was then deliberately emptied ("Clear main branch") and development split two ways:

- On top of the cleared tree, **`mvp_rag`** was built — a beginner-documented Flask + Qdrant + OpenAI RAG demo, unrelated to fact-checking. Abandoned after two days.
- From the pre-clear commit, **`author_verify`** restarted the fact-checking idea as the *hallucination-checker*: a Flask server-rendered app with FAISS retrieval over source PDFs, regex numeric-claim extraction, GPT-4.1 reranking/explanations, and a claim-detail UI with evidence switching. It worked, but PDF extraction and table rendering stayed flaky.

### The failed PDF pipeline experiment (Nov 2025)

`author_verify_broken_code` holds a single extra commit that bolted a second-generation PDF ingestion pipeline (document router, OCR, Camelot/Tabula/pdfplumber extractors, GPT table formatting) onto the gen-1 app. It shipped with table-ID collisions and a layout-extraction bug that disabled heading detection, was named "broken code" by its own author, and was never merged. It remains the only place that design exists.

### Generation 2 — the v1 app (Nov 2025 → mid-2026)

`author-ai-report-view` (since folded into `main` and deleted as a branch) was a ground-up rewrite: the `backend/` Flask API with dedicated pipelines (ingestion, accuracy, credibility, validity, chat) and services, plus the `frontend/` React/Vite dashboard. Major milestones, in order:

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

## The v2 rebuild (2026-08)

The v1 app worked, but its foundations were showing: last-run-wins storage, heuristics welded to LLM calls, silent fallbacks, and no way to measure whether a "fix" actually improved anything. Rather than patch further, the backend was rewritten from a clean slate on the `v2` branch — FastAPI + Pydantic v2, every table keyed by `run_id` so no run is ever deleted, the LLM confined to language judgments it must back with quoted evidence that plain code re-verifies, and failures loud instead of papered over. The frontend was kept and adapted. The rebuild ran as seven phases over roughly two weeks (2026-08):

1. **Phase 0 — scaffold.** The `v2` branch, pinned dependencies, ruff, a FastAPI skeleton, pytest and CI wired from day one.
2. **Phase 1 — storage.** Run-scoped SQLite with atomic migrations; sqlite-vec for vectors and FTS5 for keywords, fused by reciprocal-rank hybrid search. All later stages sit on this layer.
3. **Phase 2 — ingestion.** Docling parsing with tables and figures as first-class chunks — on the same example PDFs, v1 had extracted **zero** tables; v2 got 13 table and 22 figure chunks alongside the text.
4. **Phase 3 — LLM layer + claim extraction.** A shared Anthropic client (structured outputs, vision, per-call token logging), claim extraction on the frontier model, and — for the first time in the project's history — a golden eval set to measure against. Extraction recall climbed to 0.97 on the dev set once table content fed into extraction and an exhaustive-extraction prompt rule fixed a displacement bug (it reached 1.00 with the Phase 5 extraction change). The golden set itself was adversarially audited by independent agents reading only the sources: **8 of the 32 original verdict labels were wrong** and were fixed, and a held-out set (scored only at phase boundaries, never tuned against) was added to keep the dev set honest.
5. **Phase 4 — verification.** Per-claim hybrid retrieval over the sources, structured verdicts with schema-quoted evidence that code mechanically verifies — a failed quote check downgrades the verdict to UNVERIFIABLE rather than trusting it. Bulk verification runs on the Batch API; figure evidence attaches the chart image for vision. Fittingly, the measuring stick needed measuring too: a 25-agent review found the eval scorer could cross-pair two same-valued claims and corrupt a recorded reference — fixed with best-fit pairing and a regression test.
6. **Phase 5 — scoring, jobs, API.** The three metrics (accuracy over decided claims, per-source credibility with Crossref-verified metadata, a code-weighted validity rubric), a jobs worker with startup recovery so an interrupted run is re-queued instead of stranded, and an authenticated HTTP API that is fail-closed: no key, no server — with auth and size caps enforced in pre-body middleware so an unauthenticated upload cannot even be parsed.
7. **Phase 6 — frontend rebuild.** The React app rebuilt on the v2 API with TanStack Query: run history, a compare view with per-metric deltas, the claims workspace, and grounded chat over a finished run using Anthropic prompt caching.
8. **Phase 7 — ship.** Stance-aware accuracy landed — claims are tagged `asserted` or `disavowed`, so a report that debunks a falsehood is scored on its actual position instead of being penalized for mentioning it (the largest known scoring distortion in the v1-style design). Canonical prompt hashes now guard both eval commands against scoring stale rows after a prompt edit; the golden eval runs in CI as a manually dispatched workflow; the docs were rewritten for v2; and `v2` merges into `main`.

The through-line of the rebuild: every judgment is either made by code, or made by the LLM and then checked by code — and every change to a prompt or retrieval step is scored against the golden sets before it ships.
