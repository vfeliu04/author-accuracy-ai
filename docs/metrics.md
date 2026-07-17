# Metrics: How the Scores Are Computed

Every report gets three scores — **Accuracy**, **Credibility**, and **Validity** — plus optional external source **Recommendations**. This page documents the exact formulas as implemented in `backend/author_ai/pipelines/` and `backend/author_ai/services/`. Default thresholds live in `backend/author_ai/config.py` and are overridable via environment variables (names shown in CAPS; see [configuration.md](configuration.md)). For where these pipelines run in the request flow, see [architecture.md](architecture.md); for the endpoints that expose the results, see [api.md](api.md).

## Accuracy

Implemented in `AccuracyPipeline` (`backend/author_ai/pipelines/accuracy.py`). The report PDF is ingested, factual claims are extracted, each claim is matched against uploaded source documents via FAISS retrieval, and each claim receives a verdict of `SUPPORTED`, `CONTRADICTED`, or `NOT_FOUND`.

**Final score:** `accuracy_pct = SUPPORTED claims / total claims × 100`. Both `CONTRADICTED` and `NOT_FOUND` count against the score.

### Claim extraction (`_extract_claims`)

The report body text is split into sentences with the regex `(?<=[.!?])\s+`. Each sentence then passes through filters, in order:

1. **Title/header filter** (`_is_title_or_header`) drops a sentence if any of these fire:
   - it contains the cleaned report filename (underscores/hyphens → spaces, lowercased, length > 5);
   - 3–20 words where ≥ 60% of non-stopword words are capitalized, with no verb hints and no stat pattern (`digit + %/percent/million/billion/thousand/kg/mt/usd/$`);
   - it contains a colon whose left side is 1–6 title-cased words and the sentence has no verb hints;
   - it ends with a bare number 1–20 (page-number style) with no statistical context before it.
2. **Heading-prefix filter** skips sentences starting with a section number (e.g. `3.3 Supply Chain…`) whose remainder contains no digits. Bullet/numbering prefixes are stripped from what survives.
3. **Basic shape**: must contain letters and have at least 3 words. A sentence whose *only* number is a leading heading-like token (and no units/years after it) is skipped.
4. **Numeric path** — sentences containing digits are scored by `_claim_score`:

   | Feature | Weight |
   |---|---|
   | Contains a digit (required; score is 0.0 without one) | +2.0 |
   | Unit or year present (`percent`, `%`, `million`, `billion`, `thousand`, `19xx`/`20xx`) | +2.0 |
   | Verb hint (`is/are/was/shows/reported/estimated/increased/fell/…`) | +1.0 |
   | ≥ 4 tokens | +0.5 |
   | Alphabetic-character ratio below `CLAIM_ALPHA_RATIO_MIN` (default 0.15) | −2.0 |
   | Single number at sentence end with no unit/year (list-item pattern) | −1.5 |
   | Contains `:` with no unit/year | −1.0 |

   The sentence becomes a claim only if the score ≥ `CLAIM_SCORE_MIN` (default **1.5**).
5. **Non-numeric path** — digit-free sentences are kept only if they have **≥ 2 verb-hint matches** and a secondary score ≥ `CLAIM_NON_NUMERIC_SCORE_MIN` (default 0.5): +1.0 for any verb hint, +0.5 for ≥ 2 verb matches, +0.5 for ≥ 4 tokens, +0.5 for alpha ratio ≥ 0.5.

Each kept claim records metadata: `primary_value` (first number), `units` (`percent`/`%`/`million`/`billion`, else `"count"`), `year`, and the parent section (id/title/page/summary) when the sentence can be located in a section.

### Verifiability filter (`_filter_verifiable_claims`)

Claims are dropped if they start with non-verifiable openers (`the report`, `this report`, `figure N`, `table N`, `as shown in`, `see figure/table`) or with vague openers (`data shows`, `research shows`, `it is noted`, `it was found`) while containing no digits. If `CLAIM_QUALITY_LLM_FILTER=true` (default **false**), a single batched Claude call (`CLAIM_CLASSIFIER_MODEL`, default `claude-haiku-4-5`) labels each claim `VERIFIABLE`/`NOT_VERIFIABLE` and drops the latter; on any error all claims are kept.

### Compound-claim decomposition (`_decompose_claims`)

When `CLAIM_DECOMPOSE_ENABLED=true` (default), claims are split on `, while|whereas|but|however|although|though|yet|and `. The split is applied only if at least two resulting parts contain verb hints; parts with ≥ 4 words become independent claims carrying a `decomposed_from` metadata pointer.

### Evidence retrieval (`_retrieve_evidence`)

- **Per-source search**: when source documents exist, each source's FAISS index is queried separately for the top `RETRIEVAL_PER_SOURCE_CAP` hits (default **3**), then all hits are merged and sorted by cosine similarity descending. This guarantees every source gets a chance to contribute.
- **Global fallback**: with no known sources, a single global search returns `RETRIEVAL_TOP_K` hits (default **8**).
- **Raw-chunk fallback**: if the vector index returns nothing, the first 5 chunks stored in SQLite are used with a synthetic score of `0.9 × RETRIEVAL_SUPPORT_THRESHOLD` (just below the support threshold, so they only produce a verdict via the numeric-overlap override).

Hits are enriched with their parent section, then — if there is more than one — reordered by the **LLM reranker** (`EvidenceReranker` in `backend/author_ai/services/reranker.py`). The reranker prompts Claude (`RERANK_MODEL`, default `claude-haiku-4-5`) with the claim plus numbered snippets (truncated to 600 characters) and parses `index|score` lines; scores are clamped to [0, 1] and stored as `rerank_score`, with unranked hits appended at the end. Without `ANTHROPIC_API_KEY` (or on any error) the vector-similarity order is kept.

> **Gotcha:** the support-threshold check below always uses the *vector similarity* `score` of the rerank-top hit — `rerank_score` never replaces it. Reranking changes which hit is "best", not the number compared against the threshold.

### Verdicts

The best hit's cosine similarity is compared to `RETRIEVAL_SUPPORT_THRESHOLD` (default **0.35**). A **numeric-overlap override** (`_should_override_threshold`) lets a below-threshold hit through anyway when the claim text and the best snippet share at least one identical numeric string.

If the threshold passes (or is overridden), the top `VERDICT_MULTI_EVIDENCE_CAP` hits (default **5**) go to `VerdictClassifier.classify_multi` (`backend/author_ai/services/verdict_classifier.py`), a single Claude call (`EXPLANATION_MODEL`, default `claude-sonnet-4-6`) that sees all evidence passages at once and returns JSON `{label, confidence, reason}` with label `SUPPORTED | CONTRADICTED | INCONCLUSIVE` (2 retries with backoff). With exactly one evidence passage, `classify_multi` delegates to the single-evidence `classify` path, which is the only path that formats chart-derived chunks with figure label and structured x/y/series data; the multi-evidence prompt uses plain `[n] Source, Page + snippet` blocks. Mapping:

| Classifier label | Claim verdict |
|---|---|
| `SUPPORTED` | `SUPPORTED` |
| `CONTRADICTED` | `CONTRADICTED` |
| `INCONCLUSIVE` (or anything else) | `NOT_FOUND` |

Below threshold with no override → `NOT_FOUND` directly. If the LLM is unavailable or returns unparsable output, a **heuristic fallback** compares the claim's `primary_value` (or, failing that, the first number in the claim text) with the first number in the best snippet: relative difference ≥ 15% → `CONTRADICTED` (confidence 0.65), otherwise `SUPPORTED` (0.6); missing/unparsable numbers → `INCONCLUSIVE` (0.5).

**Temporal grounding:** years (`19xx`/`20xx`) are extracted from the claim and the best snippet. If both have years and the minimum gap exceeds `TEMPORAL_MISMATCH_TOLERANCE_YEARS` (default **2**), a `SUPPORTED` verdict is downgraded to `NOT_FOUND` (classifier confidence reduced by 0.2, floored at 0.05); a `CONTRADICTED` verdict keeps its label but gets the mismatch appended to its explanation.

### Confidence

```
confidence = min(0.99, max(0.05, max(retrieval_score, llm_confidence) + evidence_count_bonus))
```

`evidence_count_bonus` applies only when the classifier said `SUPPORTED`: +0.03 for each *additional* hit whose vector score exceeds the support threshold, capped at +0.10. Bands (`_band_from_confidence`): **HIGH** ≥ 0.75, **MEDIUM** ≥ 0.45, **LOW** otherwise.

### Accuracy limitations

> - Extraction is regex/keyword-based: the naive sentence splitter mishandles abbreviations and decimals adjacent to periods, and verb-hint/header heuristics are English-only.
> - The numeric-overlap override can promote evidence that shares a number coincidentally (e.g. the same year in an unrelated sentence).
> - The heuristic verdict fallback compares only the *first* number on each side.
> - `NOT_FOUND` counts against accuracy, so verifying a report against missing or off-topic sources yields a score near 0% — the metric measures "supported by *these* sources", not truth.
> - The raw-chunk fallback pulls from all stored chunks (including the report's own), though its sub-threshold synthetic score keeps it mostly inert.

## Credibility

Implemented in `CredibilityPipeline` (`backend/author_ai/pipelines/credibility.py`), scored per **source** at upload time from metadata gathered by `MetadataService.collect_metadata` (`backend/author_ai/services/metadata_enrichment.py`): embedded PDF metadata, first-two-pages header regexes (DOI / `Author:` / `Published:`), and — when a DOI is found — a Crossref lookup that fills title, publisher, and publication date. Metadata confidence is **HIGH** if Crossref resolved, **MEDIUM** if header regexes found anything, else **LOW**.

`score = min(100, metadata + authority + recency + confidence + user_adjustment)`

| Component | Range | Rule |
|---|---|---|
| `metadata` | 0–30 | 6 points per present field among `title`, `authors`, `publication_date`, `publisher`, `doi` |
| `authority` | 7.5–30 | Lowercased publisher contains a tier-1 keyword (`AUTHORITY_PUBLISHERS_TIER1`, default `fao,un,world bank,imf,who,unicef,oecd`) → 30; a tier-2 keyword (`AUTHORITY_PUBLISHERS_TIER2`, default `reuters,associated press,bbc,nature,science,lancet`) → 22.5; any non-empty publisher → 15; no publisher → 7.5 |
| `recency` | 3–20 | Age from publication year: ≤ 2 yrs → 20; ≤ 5 → 12; ≤ 10 → 6; older → 3; unknown date → 10 |
| `confidence` | 3–10 | Metadata confidence: HIGH → 10, MEDIUM → 7, LOW → 3 |
| `user_adjustment` | unbounded | Taken from metadata key `user_adjustment` if present; defaults to 0 (nothing in the pipeline sets it automatically) |

**Report-level aggregation** (`aggregate_report`): each source's weight is `usage_count × confidence_multiplier`, where `usage_count` is the number of evidence rows with `verdict_label = 'SUPPORTED'` that the source contributed to the report's claims (`Repository.source_usage`), and the multiplier is HIGH → 1.0, MEDIUM → 0.75, LOW → 0.5. Report credibility is the weight-normalized average of source scores. Returns `None` when no source supported any claim.

### Credibility limitations

> - Publisher matching is coarse substring containment: the tier-1 keyword `un` matches any publisher containing those two letters (e.g. "Unknown Press"), and `science` matches any name containing the word.
> - Publisher is only ever populated via Crossref, so a source without a discoverable DOI is capped at the 7.5-point "no publisher" authority tier.
> - Aggregation counts only `SUPPORTED` evidence, so a source used heavily but never confirming a claim contributes nothing to report credibility.

## Validity

Implemented in `ValidityPipeline` (`backend/author_ai/pipelines/validity.py`). Five components, each 0–100, combined by weights parsed from `VALIDITY_WEIGHTS` (default `coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1`; parse failure falls back to those defaults).

| Component | Formula | Effective floor |
|---|---|---|
| `coverage` | `60 + (covered / total) × 40` over substring hits of `VALIDITY_TOPICS` (default `climate,supply,logistics,nutrition,conflict`) in the lowercased body text | 60 |
| `consistency` | Regex `([A-Za-z\s]+)(\d+[\d,]*)(\s*\S*)` pulls entity/number/unit triples; re-seeing an entity with a different number+unit counts a contradiction. Score `= max(40, 100 − min(60, contradictions × 10))`; fewer than 2 triples → 90 | 40 |
| `methodology` | `50 + 12.5 × k`, where `k` = how many of the keywords `data`, `sample`, `method`, `limitation` appear in any section | 50 |
| `context` | `60 + min(40, hits × 8)` over geography terms `global, africa, asia, europe, america` | 60 |
| `recency` | Mean age of source publication dates in the ingestion payload: ≤ 2 yrs → 100; ≤ 5 → 80; ≤ 10 → 60; older → 40; no dates → 70 | 40 |

### Validity limitations

> - The floors mathematically inflate the score: with the default weights the minimum possible overall is ~51, so even a report failing every check looks "middling".
> - Coverage topics and geography terms are food-security-domain keyword lists — irrelevant for other domains unless `VALIDITY_TOPICS` is overridden.
> - The consistency regex keys contradictions on loose entity text and flags legitimate restatements (different years, subtotals) as conflicts.
> - The ingestion payload produced by `IngestionPipeline` currently has no `sources` key, so `_recency_score` always takes the "no dates" branch and returns the neutral 70.

## Recommendations

`RecommendationService` (`backend/author_ai/services/recommendations.py`) suggests external literature via the OpenAlex API.

- **Query construction**: report title + summary + claim texts + existing source summaries are tokenized (`[A-Za-z]{4,}`, stopword-filtered, minimal suffix stemming) into the top-16 frequency terms; the top 3 become the search string, with all 16 OR-ed onto it in parentheses (still inside the `search` parameter — the OpenAlex `filter` parameter carries only the DOI/date constraints below). The request hits `OPENALEX_BASE_URL/works` sorted by relevance, filtered to `has_doi:true` and publications from `RECOMMENDATION_PUBLICATION_CUTOFF_YEAR` (default 2018) onward; `OPENALEX_MAILTO` is attached when set.
- **On-topic gate**: a result is kept only if some topic term appears in its title or decoded abstract; results matching an existing source title are dropped.
- **Embedding relevance**: the report context (title + summary + claims, concatenated) and each candidate (title + abstract + summary) are embedded with OpenAI `text-embedding-3-large`; candidates below cosine similarity `RECOMMENDATION_SIMILARITY_THRESHOLD` (default **0.18**) are discarded, and survivors sort by (relevance, recency boost, credibility).
- **Synthetic scores** — computed from OpenAlex fields, *not* by the credibility/validity pipelines: credibility = `20 + recency (≤35, −7/yr) + min(30, log10(citations+1)×12) + 10 if DOI + min(10, 2×authors)`, clamped to [5, 100]; validity = `45` without an abstract, else `min(70, 40 + 0.05 × abstract words)`, plus `2 × max(0, 10 − age)`, clamped to [10, 100].

> Because the query already filters to `has_doi:true`, every recommendation gets the +10 DOI bonus, and the summary fallback text is food-security-specific.

## Cross-references

- Endpoint payloads exposing these scores: [api.md](api.md)
- Pipeline orchestration and the last-run-wins data model: [architecture.md](architecture.md)
- All environment variables and defaults: [configuration.md](configuration.md)
- How chat answers cite these metrics: [chat.md](chat.md)
