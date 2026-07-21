# Chat Assistant

The chat assistant answers questions about the most recent verification report. It is implemented by `ChatService` in `backend/author_ai/pipelines/chat.py` and exposed through the `POST /api/chat` and `GET /api/chat/history` routes in `backend/app.py`. Answers are grounded in pipeline output — claims, verdicts, evidence snippets, and metric diagnostics — rather than open-ended world knowledge. See [architecture.md](architecture.md) for how the pipeline produces that data and [api.md](api.md) for full endpoint reference.

## Modes

The assistant has three modes, defined in the `CHAT_MODES` tuple. Each mode swaps the system prompt used for the LLM call (`_mode_system_prompt`) and, for guidance/creative, loosens the source-relevance threshold and appends "Suggested improvements" to the metrics block.

| Mode | Behavior |
|---|---|
| `evidence` (default) | Verification assistant. Answers strictly from Claim Findings, supporting context, and diagnostics; says so instead of speculating when evidence is missing. |
| `guidance` | Report coach. Still grounded in claims, but translates diagnostics into actionable recommendations, highlights gaps, and suggests next steps. |
| `creative` | Brainstorming assistant. Grounds statements in the verified context but may offer clearly labelled advisory ideas; instructed to never fabricate data and to mark general recommendations as "Advisory". |

### Auto mode vs. locked mode

The client sends `mode` and a boolean `mode_locked` with each request. An invalid or missing `mode` falls back to `evidence`. Independently, `_infer_mode` scans the question for keywords:

- guidance keywords: `improve`, `improvement`, `fix`, `revise`, `revamp`, `better`, `enhance`
- creative keywords: `brainstorm`, `idea`, `ideas`, `scenario`, `plan`, `future`, `summary`, `recommend`

If `mode_locked` is false and the inferred mode differs from the active one, the response includes a `suggested_mode` field — the server still answers in the *active* mode; switching is left to the frontend. With `mode_locked: true`, suggestions are suppressed entirely.

Questions containing `mode`, `modes`, `model`, `models`, `setting`, `settings`, or `assistant mode` (see `_is_mode_help_question`) get an extra "Assistant Modes" context block describing all three modes plus Auto.

> **Gotcha:** mode-help detection is substring-based, so asking about "models" in your report (e.g. statistical models) also injects the mode explanations into the answer context.

## Intent detection: report vs. small talk

`_detect_intent` decides whether a question is about the report:

1. If the lowercased question contains any report keyword (`claim`, `claims`, `credibility`, `validity`, `accuracy`, `source`, `author`, `publication`, `report`, `evidence`, `support`, `contradict`, `table`) → **report**.
2. Otherwise, if it contains a greeting/small-talk phrase (`hey`, `hello`, `hi`, `thanks`, `thank you`, `who are you`, `help`, `good morning`, `good afternoon`) → **general**.
3. Otherwise: **general** if the question is two words or fewer, else **report**.

General intent goes to `_small_talk_answer`: a short LLM call (max 512 tokens) with a friendly system prompt that steers the user back to report topics, or a canned greeting when no LLM is configured.

> **Gotcha:** keyword matching is plain substring matching, so e.g. "hi" inside "think" can nudge borderline questions toward the small-talk path. Report keywords are checked first, so any question that mentions claims, sources, or metrics always takes the report path.

## Direct claim commands

`_extract_claim_request` recognizes explicit claim references and bypasses semantic retrieval for claim selection (the referenced claims become the context directly, with no relevance scores):

| Pattern | Example | Resolves via |
|---|---|---|
| `all claims` / `list all claims` | "show me all claims" | every claim in the report |
| `claim <n>` | "claim 3" | 1-based position in the report's claim list |
| `claims <n>[, <n>] and <n>` | "claims 2 and 4" | multiple 1-based positions |
| `claim id <uuid>` | "claim id 3f2a91…" | the claim's `claim_id` |

If nothing resolves, the assistant replies "No matching claims were found for your request." without calling the LLM. Because the word "claim" is itself a report keyword, these commands always route through the report path.

> **Note:** a `_deterministic_claim_response` helper exists in `chat.py` but is currently dead code — the same resolution logic is inlined in `respond`, and matched claims still flow through the normal LLM answer path (with evidence and metrics attached) rather than a purely deterministic listing.

## How answers are grounded

For report-intent questions, `respond` assembles a single prompt from these sections, in order:

1. **Report summary** — the `summary` field from the report document's metadata, if present.
2. **Claim Findings** — the question is embedded (`embed_texts`, OpenAI `EMBEDDING_MODEL`, default `text-embedding-3-large`) and searched against the per-report **claim FAISS store** (`CLAIM_VECTOR_PATH`). `_select_claim_context` ranks all of the report's claims by cosine similarity; if no vector is available it falls back to sorting by verdict priority (SUPPORTED → CONTRADICTED → NOT_FOUND → other) and confidence. Each claim block includes its ordinal number, ID, text, verdict, confidence, explanation, and the stored evidence snippets (`_group_evidence_by_claim` reads the `claim_evidence` table).
3. **Supporting Context** — `_select_source_context` searches the per-report **source FAISS store** (`SOURCE_VECTOR_PATH`), over-fetching `SOURCE_CONTEXT_LIMIT × 3` hits, filtering by a mode-adjusted threshold (`CLAIM_RELEVANCE_MIN × 0.8` as the base; guidance multiplies by 0.8, creative by 0.7, floor 0.05), de-duplicating by source, and keeping at most `SOURCE_CONTEXT_LIMIT` snippets. If the search yields nothing, it falls back to the first evidence snippet of each selected claim.
4. **Report context** — accuracy/validity/credibility summary strings from the report document metadata (`_build_core_context`).
5. **Metric Diagnostics** — `_build_metric_context` summarizes supported/contradicted claim counts, the validity breakdown (coverage, consistency, methodology, context, recency, plus missing topics and methodology gaps), the overall credibility score, per-source credibility scores with components and usage counts, and up to five recommended external sources. In guidance/creative modes it appends improvement suggestions derived from those diagnostics. See [metrics.md](metrics.md) for what the numbers mean.
6. **Recent conversation** — up to `CHAT_HISTORY_LENGTH` prior turns from `chat_logs`, trimmed by `_trim_history` to a ~1,200-token budget (rough 4-chars-per-token estimate, newest turns kept).
7. The mode-help block (if requested), the active mode name, and the user question.

> **Note:** despite its name, `CLAIM_CONTEXT_LIMIT` does not cap how many claims enter the prompt — it only feeds into the FAISS `top_k` calculation (`max(total claims, limit × 2)`). In practice every claim in the report is included, ordered by relevance. Only *source* snippets are capped.

## LLM call and the no-LLM fallback

When the `anthropic` package is installed and `ANTHROPIC_API_KEY` is set, `_llm_answer` sends the assembled prompt as a single user message to `LLM_CHAT_MODEL` (default `claude-sonnet-4-6`, Anthropic Messages API, `max_tokens=2048`) with the mode-specific system prompt. `_format_response` post-processes the text: it strips `**` bold markers and converts `* ` bullets to `• `.

Without an Anthropic client, `ChatService` runs in "heuristic mode" (logged at startup) and `_compose_answer` produces a deterministic digest instead: bulleted claims with verdicts, supporting snippets, the metrics block, report context, and conversation context, ending with a line noting the response was generated without the LLM.

The same heuristic digest is also the runtime fallback: if the Anthropic API call raises mid-request, `_llm_answer` logs a warning and returns `_compose_answer`'s deterministic answer instead of failing the request.

## Persistence and history

Every exchange is written to the `chat_logs` table in SQLite (`backend/data/accuracy.db`) by `_finalize_response` — one row for the user turn and one for the assistant turn. `session_id` defaults to `"anonymous"` when the client omits it. The assistant row's `context_ids` JSON records the mode plus the claim IDs and source IDs used, enabling later inspection of what grounded each answer.

The `/api/chat` response payload is:

```json
{
  "answer": "…",
  "claims_used": [ … ],
  "sources_used": [ … ],
  "mode": "evidence",
  "suggested_mode": null
}
```

`GET /api/chat/history?job_id=<id>` resolves the job's report and returns `{"history": [...]}` — the full log for that report, oldest first, each entry containing `session_id`, `role`, `message`, `timestamp`, and `context_ids`. It returns 400 without a `job_id` and 404 for an unknown job.

`POST /api/chat` accepts `{question, session_id?, job_id?, mode?, mode_locked?}`. With a `job_id`, the job must exist and have status `DONE` (otherwise 400 "Report not ready"); without one, the latest job's report is used. If no completed report exists at all, it returns 400 "No completed report". Both routes require the `X-API-Key` header when the `API_KEY` env var is set — auth is a no-op otherwise (see [configuration.md](configuration.md)).

```bash
curl -s -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "Which claims were contradicted?", "mode": "evidence", "mode_locked": false}'
```

> **Gotcha:** chat history does not survive a new pipeline run. `reset_environment` in `backend/author_ai/services/environment.py` deletes all rows from `chat_logs` (along with documents, claims, evidence, and scores) and wipes the FAISS index directories before each run — the app is single-report, last-run-wins by design. See [history.md](history.md).

## Configuration

All settings are read in `backend/author_ai/config.py` (`Settings`). Names only — values come from the environment; see [configuration.md](configuration.md) for the full list.

| Env var | Default | Used for |
|---|---|---|
| `LLM_CHAT_MODEL` | `claude-sonnet-4-6` | Anthropic model for chat answers and small talk |
| `CHAT_HISTORY_LENGTH` | `20` | Max prior turns fetched before token-budget trimming |
| `CLAIM_CONTEXT_LIMIT` | `5` | Feeds the claim-store `top_k` calculation (does not cap claims in the prompt) |
| `CLAIM_RELEVANCE_MIN` | `0.35` | Base relevance threshold; source search uses 0.8 × this, further relaxed in guidance/creative modes |
| `SOURCE_CONTEXT_LIMIT` | `6` | Max supporting source snippets in the prompt |
| `ANTHROPIC_API_KEY` | unset | Enables LLM answers; unset means heuristic mode |
| `OPENAI_API_KEY` | unset | Enables real question embeddings; unset falls back to a deterministic hash embedding |
| `CLAIM_VECTOR_PATH` | `./data/indexes/claims` | Per-report claim FAISS store location |
| `SOURCE_VECTOR_PATH` | `./data/indexes/sources` | Per-report source FAISS store location |

Related reading: [development.md](development.md) for running the backend locally, [api.md](api.md) for the other endpoints the frontend pairs with chat.
