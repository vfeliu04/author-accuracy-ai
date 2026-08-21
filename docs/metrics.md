# Metrics: What the Numbers Mean

Every run gets three scores — **accuracy**, **credibility**, **validity** — plus a **coverage** rate. Accuracy is pure arithmetic in `backend/authorai/scoring.py`; credibility is code + live Crossref in `credibility.py`; validity is one structured LLM rubric call, code-weighted. There is deliberately **no composite "overall" score** — the three numbers measure different things and are reported independently.

## Accuracy (stance-aware report-position agreement)

Inputs: one verdict per claim (`SUPPORTED` / `CONTRADICTED` / `UNVERIFIABLE`, produced by verification — see below) and each claim's **stance**, the report's *own* position on it:

- `asserted` — the report states or relays the claim.
- `disavowed` — the report itself attaches an **explicit falsity marker** to the claim ("an event that never occurred", "a fabricated figure", a table column labeling the value fabricated). Neutral reported speech ("some analyses claim X") stays `asserted` — relaying without rebuttal lends the claim a platform.

A claim is **correct** when the verdict agrees with the report's position:

| stance | agreeing verdict |
| --- | --- |
| `asserted` | `SUPPORTED` |
| `disavowed` | `CONTRADICTED` |

The inverses are incorrect: an asserted claim the sources contradict counts against the author, and a disavowed claim the sources *support* also counts against them (no credit for debunking something true). The author is not penalized for debunking a falsehood.

```
decided  = supported + contradicted          # UNVERIFIABLE is never decided
accuracy = correct / decided                 # None (not 0.0) when decided == 0
coverage = decided / total
```

**UNVERIFIABLE claims count only in coverage, never against accuracy** — "the sources don't cover it" is not "the author lied".

Consequence worth internalizing: a run can honestly read **"9 contradicted, accuracy 0.95"** — when the contradicted claims are ones the report itself labels as fabrications, each contradiction is the author being *right*. That is exactly the recorded dev reference (below).

## Verdicts, the downgrade rule, and the year flag

The verdict judge (`claude-opus-5`) sees only retrieved SOURCE excerpts and must return a verbatim quote for any SUPPORTED/CONTRADICTED answer. Code then verifies the quote:

- **Downgrade rule**: a SUPPORTED or CONTRADICTED verdict whose quote cannot be verified against the evidence shown (normalized substring; quotes under 10 chars rejected) is downgraded to **UNVERIFIABLE**. The model's original answer is preserved in `raw_verdict`, so the downgrade rate stays measurable; a verdict is "downgraded" exactly when `raw_verdict` differs from `verdict`.
- **Year flag** (informational, never changes a verdict): for a decided verdict on a claim that names a year, `year_flag=1` when that year does not appear in the quoted chunk.

## Credibility (per source, 0–100, aggregated per run)

For each SOURCE document, an LLM (`claude-haiku-4-5`) extracts bibliographic metadata from the opening chunks **plus any imprint/citation passages found anywhere in the document** (a weighted marker scan — "recommended citation", "citation:", ISBN, "published by", copyright/©, DOI, and similar — locates them, requiring more than a lone weak marker so bibliography entries about *other* works don't qualify; institutional reports often print their publisher and ISBN on a colophon page deep in the PDF). Code then verifies the metadata against **live Crossref** and assigns a tier:

| Tier | Meaning | Points |
| --- | --- | --- |
| `VERIFIED_DOI` | The extracted DOI resolves at Crossref | 20 |
| `VERIFIED_TITLE` | No DOI, but a Crossref record matches the title exactly (normalized) **and** a second field corroborates (year ±1, author family name, or publisher) — a title-only match on a generic title is rejected | 15 |
| `VERIFIED_ISBN` | No DOI or title match, but the extracted ISBN (checksum-validated in code first) resolves at Open Library or Google Books **and** the record's title or publisher corroborates — the path for institutional books Crossref doesn't index | 12 |
| `METADATA_ONLY` | Metadata extracted but not externally verified | 5 |
| `NONE` | Nothing extractable | 0 |

Component sum (no floors — **unknown earns nothing**):

| Component | Max | Rule |
| --- | --- | --- |
| Metadata completeness | 30 | 6 pts per present field (title, authors, publisher, date, DOI) |
| Publisher authority | 30 | Word-boundary phrase match against configured tier lists: tier-1 → 30, tier-2 → 22.5, any other named publisher → 15, no publisher → 0. "UN"/"U.N." matches; "University" does not |
| Recency | 20 | Age from publication year: ≤2 y → 20, ≤5 → 12, ≤10 → 6, older → 3; no year → 0 |
| Verification | 20 | Tier points above |

Crossref semantics: 404/4xx is an *answer* (not found); 429/5xx and transport errors retry with backoff and then **raise** — treating a throttled Crossref as "not found" would silently downgrade tiers and make scores non-reproducible.

**Run-level aggregation** is a **usage-weighted mean**: each source's weight is the number of quote-verified verdicts citing it — SUPPORTED **and** CONTRADICTED both count (a source that contradicts claims is doing exactly its job). When no verdict cites any source, the result is an unweighted mean explicitly labeled `unweighted_mean_no_usage`; with no sources at all the score is `null` with method `no_sources` — never a silent zero either way.

## Validity (rubric, 0–100)

One structured call (`claude-opus-5`) scores four components of the report text itself, each 0–100 with a justification and an illustrative verbatim quote that code checks against the report (an unverifiable quote is flagged and logged, but — unlike verdicts — does not zero the component: the quote is illustrative, not verdict-grounding):

- `coverage` — does the report treat the scope its framing promises?
- `consistency` — are its statements internally coherent?
- `methodology` — does it say where figures and findings come from?
- `context` — are numbers situated (time, place, comparison), or floating free?

A fifth component, `recency`, is **pure code**: mean age of the sources' real publication years (≤2 y → 100, ≤5 → 80, ≤10 → 60, older → 40; `None` when no source has a known date — the component is then excluded and the remaining weights renormalize).

Weights come from configuration (default `coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1`) and are parsed **loudly** — unknown names, malformed entries, NaN, or weights that don't sum to 1 are errors, never silently defaulted. There are no floors: a total failure scores near 0, not ~51.

**Measured noise**: rubric component scores vary ±2–3 points across identical runs (`score_reference.json` records 34.6/39.2/35.1 over 3 same-code runs for one configuration). Treat small validity deltas as noise.

## The eval sets

Two independently audited label sets, both over deliberately fake reports checked against real sources:

- **Dev golden** — `backend/evals/golden.jsonl`, **37 records**, each with verbatim claim text, value/unit/year, `expected_verdict`, `stance`, and a grounding note. Tuned against freely.
- **Holdout** — `backend/evals/holdout/holdout.jsonl`, **27 records** (same fields, second fake report). **Scored only at phase boundaries; never tuned against.** It exists to catch overfitting to the dev report.

Every label was adversarially audited against the sources before being trusted (the first draft of the dev set had 8/32 wrong verdicts). A label added or changed is re-audited before scores based on it count.

Scorer mechanics (`evals.py`): claims match on value+year agreement (relative value tolerance 0.001) or ≥0.6 content-word overlap measured against the shorter claim; pairing is one-to-one, and for verdict/stance scoring ties go to the best-fitting row (first-match pairing once cross-paired two value=12 claims and corrupted a recorded holdout reference — 0.89 recorded, 0.96 true).

## Recorded baselines

Quoted from the JSON files in `backend/evals/`. Re-record only deliberately, with the reason in the file's `note`.

### Dev extraction — `evals/baseline.json` (recorded 2026-08-13, `claude-opus-5`)

- recall **1.0** (37/37 golden found), precision **0.974** (37/38 extracted matched)
- stance agreement **36/37** (accept threshold ≥ 35/37); the one miss is the report's weakest falsity marker, and it errs conservative (stays `asserted`)
- noise floor: "recall 1.00 precision 0.97 on ALL of 3 repeat runs (38 claims each) — fully stable"

### Dev verdicts — `evals/verdict_baseline.json` (recorded 2026-08-12, `claude-opus-5`)

- accuracy **0.838** (31/37 matched), coverage **1.0**, downgraded **0**
- per class: SUPPORTED 11/14, CONTRADICTED 8/11, UNVERIFIABLE 12/12
- noise floor: "accuracy 0.84–0.86 over 2 runs; treat |delta| ≤ 0.03 as noise"
- remaining misses are all under-commitment (S/C judged UNVERIFIABLE), never fabricated support

### Dev run scores — `evals/score_reference.json` (recorded 2026-08-13; reference snapshot, not a golden eval — no labeled truth exists for these numbers)

- accuracy **0.9524** (correct 20, incorrect 1), coverage **0.5526**; supported 12, contradicted 9, unverifiable 17, disavowed 14 — the stance-aware semantics working as designed: this report labels its fabrications, so debunking counts *for* the author. 6 disavowed claims sit UNVERIFIABLE in the coverage bucket (their fabrications reference things the sources never discuss).
- credibility **50.2** (`usage_weighted_mean`): GHI synopsis 37.0 `METADATA_ONLY` (DOI-less brochure — honest), Heliyon review 92.5 `VERIFIED_DOI` (live Crossref resolution)
- validity **42.8** (±2–3 pt rubric noise; a low score is plausibly *correct* for a deliberately self-contradicting report with no methodology section)

### Holdout extraction — `evals/holdout/holdout_reference.json` (recorded 2026-08-13)

- recall **1.0** (27/27), precision **0.871** (27/31) — the 4 precision misses are duplicate coverage (same fact in prose and table), not spurious claims; byte-identical across every prompt change since Phase 3, so the rules generalized rather than fitting the dev report
- stance agreement **27/27** — the extractor disavowed exactly the 4 table rows under the "Circulating counter-claim (fabricated)" column and kept every neutral relay and straight-stated fabrication `asserted`; the narrow falsity-marker rule did not key on the dev report's flag words
- score snapshot: accuracy **0.8077** — against the dev report's 0.95, the designed discrimination: a report that *labels* its fabrications is rewarded, one that asserts them straight is still punished

### Holdout verdicts — `evals/holdout/verdict_reference.json` (recorded 2026-08-13)

- accuracy **0.889** (24/27), coverage **1.0**, downgraded **2** (prior recording 0.963 with 0 downgrades; the judge prompt is unchanged)
- per class: SUPPORTED 15/15, CONTRADICTED 6/8, UNVERIFIABLE 3/4 — both CONTRADICTED misses are quote-check **downgrades whose raw verdicts were correct** (the downgrade policy refusing unproven judgments; batch quote-fidelity variance), and the UNVERIFIABLE miss is the same deliberately borderline Protein Accord claim as every prior recording
- dev verdict accuracy measured 0.86 the same day (within noise of its 0.838 baseline) — no dev-climbs-while-holdout-falls overfitting signature

## Related docs

- [architecture.md](architecture.md) — where each metric is computed in the pipeline
- [development.md](development.md) — running the evals and the tune-vs-holdout discipline
