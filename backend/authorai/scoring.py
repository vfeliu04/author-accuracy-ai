"""Run scoring: three-way accuracy (code), validity rubric (LLM judged, code weighted).

Accuracy is pure arithmetic over stored verdicts, computed in exactly one
place. The headline number uses the DECIDED-claims denominator —
supported/(supported+contradicted) — with unverifiable claims reported as
their own coverage rate, never counted as inaccurate (v1 counted them against
the score, so missing source coverage read as lying).

Validity is the one language judgment here: a structured rubric call whose
component scores each carry a justification and an illustrative quote that
code checks against the report text. Weights are configured, parsed loudly,
and there are NO floors (v1's floors made a total failure score ~51/100).
"""

import math
import re
import sqlite3
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from authorai import db as dbmod
from authorai.config import Settings
from authorai.credibility import (
    CrossrefClient,
    IsbnClient,
    aggregate_credibility,
    evidence_usage,
    extract_metadata,
    merge_record,
    resolve_tier,
    score_source,
)
from authorai.llm import LLM
from authorai.log import setup_logger
from authorai.verification import normalize_quote, verdict_stamp

logger = setup_logger(__name__)

RUBRIC_COMPONENTS = ("coverage", "consistency", "methodology", "context")

VALIDITY_SYSTEM = """\
You assess the argumentative quality of a report using a fixed rubric. Judge
ONLY from the report text given — not from whether its claims are true (that
is verified separately) and not from your own knowledge of the topic.

Score each component 0-100, where 0 is a total absence and 100 is exemplary:

- `coverage`: does the report actually treat the scope its title and framing
  promise, or does it skip declared topics?
- `consistency`: are its statements internally coherent, or does it assert
  things that conflict with each other?
- `methodology`: does it say where its figures and findings come from —
  sources, data, methods, limitations?
- `context`: does it situate findings (time, place, comparison points), or
  present numbers floating free?

For each component give `justification` (one or two sentences) and `quote` —
a short verbatim passage from the report that ILLUSTRATES your judgment (the
strongest example either way). Quote exactly; the quote is checked against
the text.
"""


class ComponentAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    justification: str
    quote: str | None = None


class ValidityAssessment(BaseModel):
    coverage: ComponentAssessment
    consistency: ComponentAssessment
    methodology: ComponentAssessment
    context: ComponentAssessment


# The verdict that AGREES with each stance the report can take. Accuracy is
# report-position agreement: a disavowed claim (the report itself marks it
# false) is CORRECT when the sources contradict it — the author is not
# penalized for debunking a falsehood, and gets no credit if it turns out true.
_AGREEING_VERDICT = {"asserted": "SUPPORTED", "disavowed": "CONTRADICTED"}


def accuracy_scores(verdict_rows: list[dict]) -> dict:
    """Report-position agreement over a run's verdicts. Pure arithmetic, one home.

    `accuracy` is None (never a fake 0.0) when no claim was decided either way.
    UNVERIFIABLE claims are neither correct nor incorrect — they live in the
    coverage number, same as before stances existed.
    """
    counts = {
        "supported": sum(1 for r in verdict_rows if r["verdict"] == "SUPPORTED"),
        "contradicted": sum(1 for r in verdict_rows if r["verdict"] == "CONTRADICTED"),
        "unverifiable": sum(1 for r in verdict_rows if r["verdict"] == "UNVERIFIABLE"),
        "total": len(verdict_rows),
    }
    correct = sum(
        1 for r in verdict_rows if r["verdict"] == _AGREEING_VERDICT[r.get("stance") or "asserted"]
    )
    decided = counts["supported"] + counts["contradicted"]
    return {
        **counts,
        "correct": correct,
        "incorrect": decided - correct,
        "disavowed": sum(1 for r in verdict_rows if r.get("stance") == "disavowed"),
        "accuracy": round(correct / decided, 4) if decided else None,
        "coverage": round(decided / counts["total"], 4) if counts["total"] else None,
    }


def parse_weights(spec: str) -> dict[str, float]:
    """Parse 'coverage:0.25,consistency:0.25,...' — loudly.

    Unknown component names, malformed entries, or weights that do not sum to
    1 are configuration errors, not things to paper over with defaults (v1
    silently fell back to hardcoded weights on any parse error).
    """
    known = {*RUBRIC_COMPONENTS, "recency"}
    weights: dict[str, float] = {}
    for entry in spec.split(","):
        name, _, value = entry.partition(":")
        name = name.strip()
        if name not in known:
            raise ValueError(f"Unknown validity component {name!r} in weights {spec!r}")
        if name in weights:
            raise ValueError(f"Duplicate validity component {name!r} in weights {spec!r}")
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"Malformed weight {entry!r} in {spec!r}") from exc
        # NaN slips through a `abs(sum - 1) > 0.001` check (every NaN
        # comparison is False) and would surface as a NaN score in the API.
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Weight {entry!r} must be a finite non-negative number ({spec!r})")
        weights[name] = weight
    if abs(sum(weights.values()) - 1.0) > 0.001:
        raise ValueError(f"Validity weights must sum to 1, got {sum(weights.values())} ({spec!r})")
    return weights


def recency_score(source_years: list[int], current_year: int) -> float | None:
    """Code-side recency from the sources' real publication years — None (not a
    floor value) when no source has a known date. v1 read a payload key that
    was never emitted and returned a constant."""
    if not source_years:
        return None
    mean_age = sum(max(0, current_year - year) for year in source_years) / len(source_years)
    if mean_age <= 2:
        return 100.0
    if mean_age <= 5:
        return 80.0
    if mean_age <= 10:
        return 60.0
    return 40.0


def assess_validity(
    llm: LLM,
    model: str,
    sections: list[dict],
    *,
    weights: dict[str, float],
    source_years: list[int],
    current_year: int,
) -> dict:
    if not sections:
        raise ValueError("No report sections to assess — was the document ingested?")
    body = "\n\n".join(f"## {s.get('title') or '(untitled)'}\n{s['text']}" for s in sections)
    assessment = llm.parse(
        model=model,
        system=VALIDITY_SYSTEM,
        prompt=f"REPORT:\n\n{body}",
        output_type=ValidityAssessment,
    )

    normalized_body = normalize_quote(body)
    components: dict[str, dict] = {}
    for name in RUBRIC_COMPONENTS:
        part: ComponentAssessment = getattr(assessment, name)
        if part.quote:
            quote_verified = 1 if normalize_quote(part.quote) in normalized_body else 0
            if not quote_verified:
                # Illustrative, not verdict-grounding: flag and warn, don't zero
                # the score — a deliberate, documented deviation from the
                # verdict downgrade rule.
                logger.warning("validity %s quote not found in report text", name)
        else:
            # The rubric mandates a quote; omitting one must not look MORE
            # trustworthy than providing a wrong one.
            quote_verified = 0
            logger.warning("validity %s returned no quote — unverifiable against the text", name)
        components[name] = {
            "score": float(part.score),
            "justification": part.justification,
            "quote": part.quote,
            "quote_verified": quote_verified,
        }

    recency = recency_score(source_years, current_year)
    components["recency"] = {
        "score": recency,
        "justification": (
            f"Mean source age from {len(source_years)} dated source(s)."
            if recency is not None
            else "No source publication dates known — component excluded, weights renormalized."
        ),
        "quote": None,
        "quote_verified": None,
    }

    present = {name: w for name, w in weights.items() if components[name]["score"] is not None}
    weight_sum = sum(present.values())
    if weight_sum <= 0:
        raise ValueError(
            "No weighted validity component has a score — with these weights nothing "
            f"can be assessed ({weights!r})"
        )
    total = sum(components[name]["score"] * w for name, w in present.items()) / weight_sum
    return {
        "score": round(total, 1),
        "components": components,
        "weights_used": {name: round(w / weight_sum, 4) for name, w in present.items()},
    }


# Lexical markers of imprint/citation blocks, weighted by specificity: the
# true imprint hits several at once ("Recommended citation … ISBN … ©"), so
# it outranks bibliography entries that merely contain a DOI. Regexes, not
# substrings — a bare "doi" substring matched "doing" and let prose outrank
# real bibliographic text (review 2026-08-21).
_BIBLIO_MARKERS: tuple[tuple[re.Pattern, int], ...] = (
    (re.compile(r"recommended citation"), 3),
    (re.compile(r"suggested citation"), 3),
    # Bare "citation:" catches imprints that skip the adjective (observed
    # live: "Citation: WMO & WCRP…, 2025") and is still rare in body text.
    (re.compile(r"citation:"), 2),
    (re.compile(r"\bisbn\b"), 2),
    (re.compile(r"published by"), 2),
    (re.compile(r"all rights reserved"), 1),
    (re.compile(r"copyright"), 1),
    (re.compile(r"©"), 1),
    (re.compile(r"\bdoi\b|\b10\.\d{4,9}/"), 1),
)
_IMPRINT_CHUNK_LIMIT = 4
_IMPRINT_CHUNK_CHARS = 1200
# A chunk needs this much summed marker weight to qualify: weight-1 markers
# alone (a bibliography entry's DOI, a stray "copyright") describe OTHER
# works or prove nothing — feeding them to the extractor as "likely imprint"
# invited it to adopt a cited work's DOI as the document's own.
_IMPRINT_MIN_WEIGHT = 2


def _metadata_text(
    conn: sqlite3.Connection, run_id: str, doc_id: str, max_chars: int = 6000
) -> str:
    """The text shown to the metadata extractor: opening pages PLUS the
    likeliest imprint/citation passages from anywhere in the document.

    Chunks are the guaranteed text store (ingestion refuses empty documents);
    the sections in documents.metadata are a convenience that documents
    ingested before Phase 3 don't carry — reading those silently fed the
    metadata extractor an empty string and every source scored tier NONE.

    Opening pages carry the title and date, but institutional reports print
    their publisher/authors/ISBN on an imprint or recommended-citation page
    that can sit ANYWHERE (observed live: page 61 of 62 — the GHI scored
    credibility 37 instead of ~79 because the extractor never saw it). A
    plain marker scan over the chunk store finds those passages; no model
    involved in the finding. `max_chars` bounds only the opening segment; the
    total is bounded at max_chars + limit×chunk chars (~10.9k, fine for a
    Haiku call).

    Only chunks whose text FULLY made it into the opening segment are
    excluded from the marker scan — a page-2 imprint sliced off by the cap
    must stay eligible, or the cap recreates the missed-imprint bug at the
    front of the document (review 2026-08-21).
    """
    rows = conn.execute(
        """
        SELECT id, section, text FROM chunks
        WHERE run_id = ? AND doc_id = ? AND kind = 'text' AND (page IS NULL OR page <= 2)
        ORDER BY id LIMIT 8
        """,
        (run_id, doc_id),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT id, section, text FROM chunks WHERE run_id = ? AND doc_id = ?"
            " AND kind = 'text' ORDER BY id LIMIT 3",
            (run_id, doc_id),
        ).fetchall()
    # Section headings must be included: for academic PDFs the title and the
    # DOI-bearing header line land in Docling section names, not chunk text —
    # dropping them hid exactly the fields metadata extraction exists to find.
    parts: list[str] = []
    seen_sections: set[str] = set()
    used_ids: set[int] = set()
    length = 0
    for row in rows:
        addition: list[str] = []
        section = row["section"]
        if section and section not in seen_sections:
            addition.append(f"## {section}")
        addition.append(row["text"])
        added = sum(len(piece) + 2 for piece in addition)
        if parts and length + added > max_chars:
            break  # this chunk would be cut — leave it whole for the marker scan
        if section and section not in seen_sections:
            seen_sections.add(section)
        parts.extend(addition)
        used_ids.add(row["id"])
        length += added
    text = "\n\n".join(parts)[:max_chars]
    if not text.strip():
        raise ValueError(
            f"Document {doc_id!r} has no text chunks to extract metadata from — "
            "refusing to score a source that was never ingested properly"
        )

    scored: list[tuple[int, int, str]] = []
    for row in conn.execute(
        "SELECT id, text FROM chunks WHERE run_id = ? AND doc_id = ? AND kind = 'text' ORDER BY id",
        (run_id, doc_id),
    ):
        if row["id"] in used_ids:
            continue
        lowered = row["text"].lower()
        score = sum(weight for marker, weight in _BIBLIO_MARKERS if marker.search(lowered))
        if score >= _IMPRINT_MIN_WEIGHT:
            scored.append((score, row["id"], row["text"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    imprint = [chunk[:_IMPRINT_CHUNK_CHARS] for _, _, chunk in scored[:_IMPRINT_CHUNK_LIMIT]]
    if imprint:
        text += "\n\nLIKELY IMPRINT / CITATION PASSAGES:\n\n" + "\n\n".join(imprint)
    return text


def _report_sections(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    import json

    reports = conn.execute(
        "SELECT * FROM documents WHERE run_id = ? AND kind = 'REPORT'", (run_id,)
    ).fetchall()
    if not reports:
        raise ValueError(f"Run {run_id!r} has no REPORT document")
    if len(reports) > 1:
        raise ValueError(f"Run {run_id!r} has multiple REPORT documents — cannot score")
    return json.loads(reports[0]["metadata"]).get("sections", [])


def score_run(
    conn: sqlite3.Connection,
    llm: LLM,
    run_id: str,
    settings: Settings,
    crossref: CrossrefClient | None = None,
    isbn_lookup: IsbnClient | None = None,
    allow_stale: bool = False,
) -> dict:
    """Compute and persist all three scores for a run. Verdicts must exist.

    Nothing is persisted until every score (including the validity LLM call)
    is computed — a mid-way failure leaves the run's PRIOR score set intact
    and internally consistent, rather than a fresh source_credibility next to
    a stale run_scores.
    """
    if dbmod.get_run(conn, run_id) is None:
        raise ValueError(f"Unknown run {run_id!r}")
    verdict_rows = dbmod.list_verdicts(conn, run_id)
    if not verdict_rows:
        raise ValueError(f"Run {run_id!r} has no verdicts — run `verify` first")
    # The score is PERSISTED and served by the API, so scoring stale verdicts
    # is worse than the eval case the guard was built for — refuse by default.
    stale = [r for r in verdict_rows if r.get("prompt_hash") != verdict_stamp()]
    if stale and not allow_stale:
        raise ValueError(
            f"{len(stale)}/{len(verdict_rows)} verdicts were produced by a different "
            "judge prompt — re-run `verify` (or score with allow_stale=True)"
        )
    accuracy = accuracy_scores(verdict_rows)

    sources = conn.execute(
        "SELECT * FROM documents WHERE run_id = ? AND kind = 'SOURCE'", (run_id,)
    ).fetchall()
    tier1 = [p.strip() for p in settings.authority_tier1.split(",") if p.strip()]
    tier2 = [p.strip() for p in settings.authority_tier2.split(",") if p.strip()]
    current_year = datetime.now(UTC).year

    # Built only when sources exist, closed only if built here — a caller's
    # injected client stays theirs to manage.
    owns_crossref = crossref is None and bool(sources)
    if owns_crossref:
        crossref = CrossrefClient(settings.crossref_mailto)
    owns_isbn = isbn_lookup is None and bool(sources)
    if owns_isbn:
        isbn_lookup = IsbnClient(settings.crossref_mailto)
    per_source: list[dict] = []
    source_years: list[int] = []
    try:
        for document in sources:
            metadata = extract_metadata(
                llm, settings.metadata_model, _metadata_text(conn, run_id, document["id"])
            )
            tier, record = resolve_tier(metadata, crossref, isbn_lookup)
            merged = merge_record(metadata, record)
            scored = score_source(
                merged,
                tier,
                tier1_publishers=tier1,
                tier2_publishers=tier2,
                current_year=current_year,
            )
            year_match = re.search(r"\b(19|20)\d{2}\b", merged.publication_date or "")
            if year_match:
                source_years.append(int(year_match.group()))
            per_source.append(
                {
                    "doc_id": document["id"],
                    "metadata": merged.model_dump(),
                    "components": scored["components"],
                    "total": scored["total"],
                    "tier": tier,
                }
            )
    finally:
        if owns_crossref:
            crossref.close()
        if owns_isbn:
            isbn_lookup.close()
    credibility = aggregate_credibility(per_source, evidence_usage(conn, run_id))

    # The last failure-prone step (a network LLM call), computed BEFORE any
    # write so a failure here persists nothing and the prior scores stand.
    validity = assess_validity(
        llm,
        settings.validity_model,
        _report_sections(conn, run_id),
        weights=parse_weights(settings.validity_weights),
        source_years=source_years,
        current_year=current_year,
    )

    # Both writes at the end, adjacent, with no fallible work between them.
    dbmod.save_source_credibility(conn, run_id, per_source)
    dbmod.save_run_scores(
        conn, run_id, accuracy=accuracy, credibility=credibility, validity=validity
    )
    return {"accuracy": accuracy, "credibility": credibility, "validity": validity}
