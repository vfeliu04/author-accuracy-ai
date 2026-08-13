# Chat

Run-grounded Q&A over a completed analysis: `POST /api/runs/{run_id}/chat`, implemented in `backend/authorai/chat.py` with the model call in `backend/authorai/llm.py` (`AnthropicClient.chat`). The pipeline already made every judgment — chat only surfaces the stored analysis; it never re-verifies anything. The endpoint requires the run to be `DONE` (409 otherwise); see [api.md](api.md) for the HTTP contract.

## Run-scoped context

`build_context` renders the run's entire stored analysis into one text block, using the same DB reads as `GET /api/runs/{run_id}/report` (so chat and the report page can never describe different data):

1. **Scores** — accuracy (with the correct/incorrect stance-agreement breakdown when present), coverage, credibility /100, validity /100, and the supported/contradicted/unverifiable counts, from `get_run_scores`.
2. **Claims** — every verdict row from `list_verdicts_with_evidence`: verdict, claim text, rationale, and (when a chunk was cited) the evidence quote with its source document title and page. Disavowed claims are tagged `(disavowed by the report)`.
3. **Sources** — each source's credibility tier and 0–100 total from `list_source_credibility`.

The system prompt (`CHAT_SYSTEM`) tells the model to answer **only** from this block, to say plainly when the analysis does not cover something, and never to invent claims, verdicts, or sources.

## Prompt caching

The context is static once a run is `DONE`, so it is sent as an Anthropic prompt-cached system block. The system parameter is a list of two blocks:

| Block | Content | Cached? |
|---|---|---|
| 1 | `CHAT_SYSTEM` + the rendered per-run context | Yes — `"cache_control": {"type": "ephemeral"}` |
| 2 | The mode instruction | No |

The cache breakpoint sits on block 1, and the mode instruction lives **outside** it — switching between evidence/guidance/creative changes only the small uncached suffix, so a mode switch still reads the cached prefix. Per-turn, only the mode line and the messages are new input. Cache effectiveness is visible in the logs: every call logs `cache_write=` / `cache_read=` token counts (`_log_usage` in `llm.py`).

## Modes

`mode` in the request body, validated as a Pydantic `Literal` (anything else → 422). Each mode swaps only the second system block:

| Mode | Instruction |
|---|---|
| `evidence` (default) | Answer precisely from the verdicts and quoted evidence; cite the specific claims and sources relied on |
| `guidance` | Help the author improve the report, grounding every suggestion in the contradicted/unverifiable claims and the weakest sources |
| `creative` | Brainstorm freely, but stay anchored to the report's topic and findings — never contradict the analysis |

## History: client-held, server stores nothing

There is no chat persistence and no session id. The client sends prior turns in the request (`history`, capped at 50 turns of ≤ 8,000 chars each by the request model); the server:

1. keeps only the most recent `chat_history_turns` messages (default 12; a value ≤ 0 sends no history — the code guards the `[-0:]` slice, which would otherwise keep everything);
2. drops any leading **assistant** turns left after trimming — the Anthropic API rejects a conversation that does not start with a user turn;
3. appends the current `question` as the final user message.

Restarting the backend loses nothing, because there is nothing to lose.

## Model call

One non-streaming `messages.create` per request:

| Parameter | Value |
|---|---|
| Model | `chat_model`, default `claude-sonnet-5` |
| `max_tokens` | `chat_max_tokens`, default 2048 |
| `thinking` | `{"type": "disabled"}` — grounded Q&A over a prepared context is a bounded task; disabling thinking keeps the whole budget for the visible answer and responses fast |
| `system` | The two blocks above |

The client is fail-loud: `AnthropicClient` refuses to construct without `ANTHROPIC_API_KEY`, and an empty/whitespace answer raises (surfacing as a 500) rather than returning silence. Token usage is logged per call.

## Stance in answers

`CHAT_SYSTEM` carries the verdict legend: SUPPORTED / CONTRADICTED / UNVERIFIABLE are *relative to the ingested sources only*, and a claim marked "disavowed by the report" is one the report **itself** calls false. Accuracy counts stance-verdict agreement, so a disavowed claim with a CONTRADICTED verdict means the report was right to reject it — the model is explicitly instructed not to narrate disavowed-CONTRADICTED claims as errors by the report.

## Example

```bash
curl -s -X POST http://localhost:8000/api/runs/$RUN_ID/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AUTHORAI_API_KEY" \
  -d '{
    "question": "Which claims were contradicted, and by which sources?",
    "mode": "evidence",
    "history": []
  }'
# → {"answer": "…", "mode": "evidence"}
```

## Configuration

Full list in [configuration.md](configuration.md); the chat-specific settings:

| Env var | Default | Controls |
|---|---|---|
| `AUTHORAI_CHAT_MODEL` | `claude-sonnet-5` | Model for chat answers |
| `AUTHORAI_CHAT_MAX_TOKENS` | `2048` | Response token budget |
| `AUTHORAI_CHAT_HISTORY_TURNS` | `12` | Most recent history messages kept per request |
