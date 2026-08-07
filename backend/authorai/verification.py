"""Claim verification: retrieve source evidence, judge, verify the quote in code.

The LLM makes exactly one judgment per claim — verdict + verbatim quote — via
structured output. Everything else is plain code: retrieval is run-scoped
hybrid search restricted to SOURCE documents (a report must never be its own
evidence), the quote is verified by normalized substring against the evidence
actually shown to the model, and a SUPPORTED/CONTRADICTED verdict whose quote
cannot be verified is DOWNGRADED to UNVERIFIABLE — an unproven judgment must
never count in the headline number. The model's original verdict is preserved
in `raw_verdict` so the downgrade rate stays measurable.
"""

import re
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from authorai import db as dbmod
from authorai.db import VERDICTS
from authorai.embeddings import Embedder
from authorai.llm import BATCH_MAX_TOKENS, LLM, ParseItem
from authorai.log import setup_logger
from authorai.search import Hit, hybrid_search

logger = setup_logger(__name__)

EVIDENCE_K = 8
MAX_IMAGES = 2
# A normalized quote shorter than this ("2024", "hunger") would "verify"
# against nearly any chunk — too weak to ground a verdict.
MIN_QUOTE_CHARS = 10

VERDICT_SYSTEM = """\
You verify one claim from a report against numbered evidence excerpts taken
from SOURCE documents. Decide the verdict RELATIVE TO THIS EVIDENCE ONLY —
never from your own knowledge.

- SUPPORTED: the evidence states the claim's substance or directly entails it.
  Be strict — "the evidence vaguely gestures at this" is not entailment.
- CONTRADICTED: the evidence states a fact incompatible with the claim.
- UNVERIFIABLE: the evidence is silent or insufficient. A false-sounding claim
  about something the evidence never discusses is UNVERIFIABLE, not
  CONTRADICTED — contradiction requires an incompatible stated fact.
- If the claim attributes its content to some named report, judge the
  SUBSTANCE against the evidence; the attribution itself is not the claim.

For SUPPORTED and CONTRADICTED you MUST provide:
- `quote`: one complete clause or sentence copied VERBATIM from a single
  evidence excerpt — the exact characters, not a paraphrase, and never just a
  bare number or name. This quote is checked mechanically against the
  evidence; a quote that does not appear verbatim invalidates the verdict.
- `evidence_index`: the number of the excerpt the quote comes from.

For UNVERIFIABLE, quote and evidence_index may be null.
`rationale` is one or two sentences explaining the decision.
"""


class Verdict(BaseModel):
    """The judge's structured answer.

    Deliberately permissive — no cross-field validators. A malformed answer
    for one claim must fail that claim's code-side check (recordable, loud)
    rather than throw during batch parsing and kill the whole run.
    """

    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    quote: str | None = Field(
        default=None, description="Verbatim quote from ONE evidence excerpt; null for UNVERIFIABLE"
    )
    evidence_index: int | None = Field(
        default=None, description="1-based number of the excerpt the quote is from"
    )
    rationale: str = Field(description="One or two sentences explaining the decision")


def build_evidence_query(claim: dict) -> str:
    """Retrieval query for a claim: its verbatim text, plus the year if absent.

    The claim text already carries its numbers verbatim (extraction rule), so
    appending `value` would only add tokens like "735000000.0" that match
    nothing. The year is worth appending when missing: golden years are often
    stated near the evidence but not inside the claim sentence.
    """
    query = claim["text"]
    year = claim.get("year")
    if year is not None and str(year) not in query:
        query = f"{query} {year}"
    return query


def build_verdict_prompt(claim: dict, hits: list[Hit]) -> str:
    parts = [f"CLAIM: {claim['text']}"]
    details = ", ".join(
        f"{name}={claim[name]}" for name in ("value", "unit", "year") if claim.get(name) is not None
    )
    if details:
        parts.append(f"CLAIM DETAILS: {details}")
    parts.append("\nEVIDENCE:")
    for index, hit in enumerate(hits, start=1):
        page = f" p.{hit.page}" if hit.page is not None else ""
        parts.append(f"[{index}] ({hit.kind}{page})\n{hit.text}")
    return "\n\n".join(parts)


_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def _normalize_quote(text: str) -> str:
    """Case-fold and neutralize the artifacts PDF text carries (curly quotes,
    unicode dashes, non-breaking and doubled spaces, line breaks) so a
    faithfully-copied quote is not rejected over typography."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_TRANSLATION).lower()).strip()


def check_verdict(claim: dict, verdict: Verdict, hits: list[Hit]) -> dict:
    """Code-side checks; returns the verdicts-table row (minus model).

    Quote check: the quote must appear (normalized) in the referenced excerpt,
    or failing that in ANY excerpt shown to the model — judges sometimes cite
    the right quote with the wrong index, and provably-real evidence must not
    be discarded over an indexing slip. Failure downgrades S/C to UNVERIFIABLE.
    Year check (informational only): claim year absent from the quoted chunk.
    """
    raw = verdict.verdict
    final = raw
    quote_verified: int | None = None
    quoted_chunk_id: int | None = None

    quote = (verdict.quote or "").strip()
    normalized = _normalize_quote(quote) if quote else ""
    if quote and len(normalized) >= MIN_QUOTE_CHARS:
        candidates: list[Hit] = []
        if verdict.evidence_index is not None and 1 <= verdict.evidence_index <= len(hits):
            candidates.append(hits[verdict.evidence_index - 1])
        candidates.extend(h for h in hits if h not in candidates)
        for hit in candidates:
            if normalized in _normalize_quote(hit.text):
                quote_verified = 1
                quoted_chunk_id = hit.chunk_id
                break
        else:
            quote_verified = 0
    elif raw in ("SUPPORTED", "CONTRADICTED"):
        # Missing or trivially short quote where one is mandatory.
        quote_verified = 0

    if raw in ("SUPPORTED", "CONTRADICTED") and quote_verified != 1:
        final = "UNVERIFIABLE"
        logger.info("verdict downgraded to UNVERIFIABLE (quote check failed) claim=%s", claim["id"])

    year_flag: int | None = None
    if claim.get("year") is not None and final in ("SUPPORTED", "CONTRADICTED"):
        quoted_hit = next(h for h in hits if h.chunk_id == quoted_chunk_id)
        year_flag = 0 if re.search(rf"\b{claim['year']}\b", quoted_hit.text) else 1

    return {
        "claim_id": claim["id"],
        "verdict": final,
        "raw_verdict": raw,
        "quote": verdict.quote,
        "quote_verified": quote_verified,
        "quoted_chunk_id": quoted_chunk_id,
        "evidence_chunk_ids": [hit.chunk_id for hit in hits],
        "year_flag": year_flag,
        "rationale": verdict.rationale,
    }


def retrieve_evidence(
    conn: sqlite3.Connection,
    embedder: Embedder,
    run_id: str,
    claims: list[dict],
    k: int = EVIDENCE_K,
) -> dict[str, list[Hit]]:
    """SOURCE-only evidence per claim; one batched embedding call for all."""
    queries = [build_evidence_query(claim) for claim in claims]
    embeddings = embedder.embed(queries)
    return {
        claim["id"]: hybrid_search(conn, run_id, query, embedding, k=k, doc_kind="SOURCE")
        for claim, query, embedding in zip(claims, queries, embeddings, strict=True)
    }


def images_for(conn: sqlite3.Connection, hits: list[Hit]) -> list[Path]:
    """PNGs for the first MAX_IMAGES figure hits, so the judge sees the chart.

    A missing image file raises: the chunk text still holds the caption and
    description, so degrading to text-only would be silent and undetectable.
    """
    paths: list[Path] = []
    for hit in hits:
        if hit.kind != "figure" or hit.figure_id is None:
            continue
        row = conn.execute(
            "SELECT image_path FROM figures WHERE id = ?", (hit.figure_id,)
        ).fetchone()
        path = Path(row["image_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Figure {hit.figure_id} image missing at {path} — refusing to judge "
                "a chart claim without the chart"
            )
        paths.append(path)
        if len(paths) == MAX_IMAGES:
            break
    return paths


def verify_run(
    conn: sqlite3.Connection,
    embedder: Embedder,
    llm: LLM,
    run_id: str,
    *,
    model: str,
    k: int = EVIDENCE_K,
    batch: bool = True,
) -> dict:
    """Verify every claim in a run; stores verdicts (replacing) and returns a summary."""
    claims = dbmod.list_claims(conn, run_id)
    if not claims:
        raise ValueError(f"Run {run_id!r} has no claims — run `extract` first")
    sources = conn.execute(
        "SELECT count(*) FROM documents WHERE run_id = ? AND kind = 'SOURCE'", (run_id,)
    ).fetchone()[0]
    if not sources:
        raise ValueError(
            f"Run {run_id!r} has no SOURCE documents — every verdict would be "
            "vacuously UNVERIFIABLE; ingest sources first"
        )

    evidence = retrieve_evidence(conn, embedder, run_id, claims, k=k)

    rows: list[dict] = []
    items: list[ParseItem] = []
    judged_claims: dict[str, dict] = {}
    for claim in claims:
        hits = evidence[claim["id"]]
        if not hits:
            # Nothing retrieved: there is no judgment to make — bookkeeping,
            # not a silent fallback, and logged as such.
            logger.warning("no evidence retrieved for claim %s — UNVERIFIABLE", claim["id"])
            rows.append(
                {
                    "claim_id": claim["id"],
                    "verdict": "UNVERIFIABLE",
                    "raw_verdict": "UNVERIFIABLE",
                    "quote": None,
                    "quote_verified": None,
                    "quoted_chunk_id": None,
                    "evidence_chunk_ids": [],
                    "year_flag": None,
                    "rationale": "No evidence retrieved from the source documents.",
                }
            )
            continue
        judged_claims[claim["id"]] = claim
        items.append(
            ParseItem(
                custom_id=claim["id"],
                system=VERDICT_SYSTEM,
                prompt=build_verdict_prompt(claim, hits),
                output_type=Verdict,
                images=images_for(conn, hits) or None,
            )
        )

    if items:
        if batch:
            results = llm.parse_batch(model=model, items=items, max_tokens=BATCH_MAX_TOKENS)
        else:
            results = {
                item.custom_id: llm.parse(
                    model=model,
                    system=item.system,
                    prompt=item.prompt,
                    output_type=item.output_type,
                    images=item.images,
                )
                for item in items
            }
        for claim_id, verdict in results.items():
            rows.append(check_verdict(judged_claims[claim_id], verdict, evidence[claim_id]))

    for row in rows:
        row["model"] = model
    dbmod.add_verdicts(conn, run_id, rows, replace=True)

    return {
        "counts": {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS},
        "downgraded": sum(1 for r in rows if r["quote_verified"] == 0),
        "year_flagged": sum(1 for r in rows if r["year_flag"] == 1),
        "no_evidence": sum(1 for r in rows if not r["evidence_chunk_ids"]),
        "total": len(rows),
    }
