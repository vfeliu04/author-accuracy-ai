# Configuration Reference

All backend settings come from environment variables. `backend/app.py` loads `backend/.env` with python-dotenv (`override=False`, so variables already set in your shell win over the file). The values are parsed once into a frozen `Settings` dataclass in `backend/author_ai/config.py` and cached by `get_settings()` (an `lru_cache`), so **any change requires a backend restart**. A starter file is provided at `backend/.env.example`:

```bash
cp backend/.env.example backend/.env
# then edit backend/.env with your keys
```

Parsing rules (see the helpers at the top of `config.py`):

- **Booleans**: the string `"true"` (case-insensitive) is `True`; anything else is `False`.
- **Paths**: resolved to absolute paths at startup, and the key directories (`DATA_ROOT`, `CACHE_DIR`, `FAISS_INDEX_DIR`, `CLAIM_VECTOR_PATH`, `LOG_DIR`) are auto-created by `get_settings()`.
- **Numbers**: parsed with plain `float()` / `int()` — a malformed value crashes at startup rather than falling back to the default.

> **Gotcha:** the path defaults are *relative* (`./data`, `./logs`, …) and are resolved against the process working directory. Always start the server from `backend/` (see [development.md](development.md)), or set absolute paths in `.env`.

## API keys and auth

Never commit real values; `backend/.env` should stay untracked. Names only below.

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Claude access for chat, explanations, verdicts, rerank, claim filtering | `services/summarizer.py`, `services/verdict_classifier.py`, `services/reranker.py`, `pipelines/chat.py` |
| `OPENAI_API_KEY` | unset | OpenAI embeddings and the optional LlamaIndex section summarizer | `services/embedding.py`, `services/section_indexer.py` |
| `API_KEY` | unset | Shared secret checked against the `X-API-Key` request header | `app.py` (`require_api_key`) |

> **Gotcha:** if `API_KEY` is unset, `require_api_key` in `app.py` is a **no-op** — every endpoint is open. Set it for anything beyond local development. See [api.md](api.md) for which endpoints are guarded.

If the LLM keys are missing the code degrades rather than crashing: clients are simply not constructed and the services fall back to heuristics (e.g. a deterministic character-frequency vector instead of OpenAI embeddings, extractive summaries and rule-based verdicts instead of Claude).

## Storage paths

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `DATA_ROOT` | `./data` | Root folder for uploaded PDFs and derived data | `services/file_store.py` |
| `SQLITE_PATH` | `./data/accuracy.db` | SQLite database file | `storage/database.py`, `services/environment.py` |
| `CACHE_DIR` | `./data/cache` | Cached artifacts (wiped on each pipeline run) | `services/environment.py` |
| `FAISS_INDEX_DIR` | `./data/indexes` | Base directory for FAISS vector indexes | `services/vector_store.py`, `services/environment.py` |
| `CLAIM_VECTOR_PATH` | `./data/indexes/claims` | Per-report FAISS index of extracted claims | `pipelines/accuracy.py`, `pipelines/chat.py` |
| `SOURCE_VECTOR_PATH` | `./data/indexes/sources` | Per-report FAISS index of source chunks | `pipelines/accuracy.py`, `pipelines/chat.py` |
| `LOG_DIR` | `./logs` | Backend log file location | `services/logger.py` |

> **Gotcha:** the app is single-report, last-run-wins. `reset_environment()` in `services/environment.py` wipes the pipeline tables in `SQLITE_PATH` and clears `FAISS_INDEX_DIR` and `CACHE_DIR` at the start of each run. Do not point these at directories containing anything you want to keep. See [architecture.md](architecture.md).

## Model selection

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model for all vector search | `services/embedding.py` |
| `EXPLANATION_MODEL` | `claude-sonnet-4-6` | Verdict explanations and summaries | `services/verdict_classifier.py`, `services/summarizer.py` |
| `LLM_CHAT_MODEL` | `claude-sonnet-4-6` | Chat responses (see [chat.md](chat.md)) | `pipelines/chat.py` |
| `RERANK_MODEL` | `claude-haiku-4-5` | Evidence reranking after retrieval | `services/reranker.py` |
| `CLAIM_CLASSIFIER_MODEL` | `claude-haiku-4-5` | LLM claim-verifiability batch filter | `pipelines/accuracy.py` |

> **Gotcha:** `services/section_indexer.py` passes `EXPLANATION_MODEL` (a Claude model id) into LlamaIndex's **OpenAI** LLM wrapper together with `OPENAI_API_KEY`. With the defaults this fails and the code silently falls back to the heuristic summarizer. If you want working LlamaIndex section summaries, that wiring needs an OpenAI model — as shipped, treat it as effectively disabled.

## Retrieval and reranking

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `RETRIEVAL_TOP_K` | `8` | FAISS hits fetched per claim — but only on the fallback path with no indexed sources; normal runs retrieve per source instead | `pipelines/accuracy.py` |
| `RETRIEVAL_PER_SOURCE_CAP` | `3` | Evidence chunks fetched per source document per claim (the effective limit on normal runs) | `pipelines/accuracy.py` |
| `RETRIEVAL_SUPPORT_THRESHOLD` | `0.35` | Minimum similarity for evidence to count toward a verdict | `pipelines/accuracy.py` |

## Claim extraction and pipeline

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `CLAIM_SCORE_MIN` | `1.5` | Minimum heuristic score for a sentence to become a claim | `pipelines/accuracy.py` |
| `CLAIM_ALPHA_RATIO_MIN` | `0.15` | Minimum ratio of alphabetic characters (rejects table-noise "sentences") | `pipelines/accuracy.py` |
| `CLAIM_NON_NUMERIC_SCORE_MIN` | `0.5` | Score floor for claims without numbers | `pipelines/accuracy.py` |
| `CLAIM_QUALITY_LLM_FILTER` | `false` | Extra Claude pass that drops non-verifiable claims (`_llm_batch_filter`) | `pipelines/accuracy.py` |
| `CLAIM_DECOMPOSE_ENABLED` | `true` | Rule-based splitting of compound claims into parts | `pipelines/accuracy.py` |
| `SECTION_SUMMARY_MODE` | `lazy` | `lazy` = summarize sections during the accuracy run; `eager` = at ingestion | `pipelines/accuracy.py`, `pipelines/ingestion.py` |
| `PIPELINE_MAX_WORKERS` | `4` | Thread pool size for parallel per-claim evidence retrieval | `pipelines/accuracy.py` |

## Verdicts and confidence

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `VERDICT_MULTI_EVIDENCE_CAP` | `5` | Max evidence snippets combined into one verdict prompt | `pipelines/accuracy.py` |
| `TEMPORAL_MISMATCH_TOLERANCE_YEARS` | `2` | Year gap between claim and evidence before flagging a temporal mismatch | `pipelines/accuracy.py` |

## OCR

OCR runs when a PDF yields fewer than `PDF_OCR_MIN_TEXT_THRESHOLD` characters **or** its ASCII ratio falls below `PDF_OCR_LOW_QUALITY_RATIO` (`should_run_ocr` in `services/ocr.py`).

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `PDF_OCR_ENABLED` | `true` | Master switch for OCR | `services/ocr.py` |
| `PDF_OCR_MIN_TEXT_THRESHOLD` | `256` | Character-count trigger for OCR | `services/ocr.py` |
| `PDF_OCR_LOW_QUALITY_RATIO` | `0.6` | ASCII-ratio trigger for OCR | `services/ocr.py` |
| `OCR_MY_PDF_PATH` | `/usr/local/bin/ocrmypdf` | Path to the `ocrmypdf` binary | `services/ocr.py` |

## Table extraction

`extract_tables` in `services/table_extraction.py` dispatches on `PDF_TABLES_ENGINE`; any value other than `tabula` or `pdftables` raises a `ValueError`.

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `PDF_TABLES_ENGINE` | `tabula` | `tabula` (local Java jar) or `pdftables` (hosted API) | `services/table_extraction.py` |
| `TABULA_JAR_PATH` | `/usr/local/tabula/tabula.jar` | Location of the Tabula jar (extraction is skipped with a warning if missing) | `services/table_extraction.py` |
| `TABULA_JAI_CLASSPATH` | empty | Extra Java classpath entries prepended when invoking Tabula | `services/table_extraction.py` |
| `PDF_TABLES_API_KEY` | unset | PDFTables.com API key (engine skipped if missing) | `services/table_extraction.py` |
| `PDF_TABLES_TIMEOUT` | `30` | PDFTables HTTP timeout in seconds | `services/table_extraction.py` |

> **Note:** `PDF_TABLES_TIMEOUT` is the one setting read with a raw `os.getenv` inside `table_extraction.py` instead of going through `Settings` — it is *not* cached and is re-read on every PDFTables request. In practice a restart is still needed to change it, because `backend/.env` is only loaded into the environment once at startup.

## Chart ingestion

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `CHART_INGESTION_ENABLED` | `true` | Extract embedded chart images from PDFs | `services/charts.py` |
| `CHART_MIN_AREA` | `4096` | Minimum pixel area (w×h) for an image to be treated as a chart | `services/charts.py` |
| `CHART_MAX_FACT_POINTS` | `4` | Max data points per chart turned into searchable fact chunks | `services/charts.py` |

## Credibility

Publisher lists are comma-separated, lowercase substrings matched against source metadata (see [metrics.md](metrics.md)).

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `AUTHORITY_PUBLISHERS_TIER1` | `fao,un,world bank,imf,who,unicef,oecd` | Publishers granted the top authority score | `pipelines/credibility.py` |
| `AUTHORITY_PUBLISHERS_TIER2` | `reuters,associated press,bbc,nature,science,lancet` | Publishers granted the second-tier authority score | `pipelines/credibility.py` |

## Validity

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `VALIDITY_TOPICS` | `climate,supply,logistics,nutrition,conflict` | Comma-separated topics the coverage score checks for | `pipelines/validity.py` |
| `VALIDITY_WEIGHTS` | `coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1` | `name:weight` pairs for the validity sub-scores | `pipelines/validity.py` |

## Recommendations, OpenAlex, Crossref

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `RECOMMENDATION_SIMILARITY_THRESHOLD` | `0.18` | Minimum relevance score for a recommended source | `services/recommendations.py` |
| `RECOMMENDATION_PUBLICATION_CUTOFF_YEAR` | `2018` | Oldest publication year requested from OpenAlex | `services/recommendations.py` |
| `OPENALEX_BASE_URL` | `https://api.openalex.org` | OpenAlex API endpoint | `services/recommendations.py` |
| `OPENALEX_MAILTO` | unset | Contact email appended to OpenAlex requests (polite pool) | `services/recommendations.py` |
| `CROSSREF_MAILTO` | `dev@example.com` | Contact email in the Crossref `User-Agent` header | `services/metadata_enrichment.py` |

## Chat

See [chat.md](chat.md) for how these interact.

| Env var | Default | Controls | Main consumer |
|---|---|---|---|
| `CHAT_HISTORY_LENGTH` | `20` | Prior messages loaded into the chat context | `pipelines/chat.py` |
| `CLAIM_RELEVANCE_MIN` | `0.35` | Base similarity threshold for *source* snippets pulled into chat (applied at ×0.8, lowered further in guidance/creative modes); claims are ranked, not thresholded | `pipelines/chat.py` |
| `CLAIM_CONTEXT_LIMIT` | `5` | Sizes the claim vector search (`top_k` is at least 2× this); it does **not** cap the prompt — every ranked claim is injected | `pipelines/chat.py` |
| `SOURCE_CONTEXT_LIMIT` | `6` | Max source snippets injected into a chat prompt | `pipelines/chat.py` |

## Flask server variables

These are read by the Flask CLI itself, not by application code, and only apply when starting with `flask run`. Running `python app.py` directly ignores them and hardcodes `debug=True, port=5001` in the `__main__` block of `app.py`.

| Env var | Read by |
|---|---|
| `FLASK_APP`, `FLASK_ENV`, `FLASK_DEBUG`, `FLASK_RUN_PORT` | Flask CLI (`flask run`) |

## Declared but currently unused

The following are defined in `Settings` but **no code ever reads them** (verified by grep). Setting them has no effect:

| Env var | Default | Intended purpose |
|---|---|---|
| `RERANK_WITH_GPT` | `true` | Toggle for LLM reranking (the reranker always runs when a Claude client exists) |
| `RERANK_MAX_CANDIDATES` | `10` | Cap on rerank candidates |
| `PDF_OCR_FORCE_WHEN_SCANNED` | `true` | Force OCR on scanned PDFs |
| `CLAIM_PRIORITY_WEIGHT` | `1.5` | Claim prioritization weight |
| `CRED_TITLE_MATCH_THRESHOLD` | `0.85` | Credibility title-match threshold |
| `CRED_RECENCY_DECAY_YEARS` | `10` | Credibility recency decay window |
| `AUTHORITATIVE_PUBLISHERS` | `fao,un,world bank,imf` | Superseded by the tiered publisher lists |
| `VALIDITY_TOPIC_THRESHOLD` | `0.35` | Validity topic threshold |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.95` | Near-duplicate detection threshold |

In addition, the local `backend/.env` carries keys that nothing in the codebase consumes at all (not even `Settings`): `BACKEND_PORT`, `LOG_LEVEL`, `CLASSPATH`, `PDF_PIPELINE_LEGACY`, `ENABLE_SEMANTIC_CLAIMS`, and `VERDICT_CONFIDENCE_BANDS`. They are leftovers from earlier iterations (see [history.md](history.md)) and can be removed safely.
