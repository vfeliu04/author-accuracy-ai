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
    aggregate_credibility,
    evidence_usage,
    extract_metadata,
    merge_record,
    resolve_tier,
    score_source,
)
from authorai.llm import LLM
from authorai.log import setup_logger
from authorai.verification import normalize_quote

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


def accuracy_scores(verdict_rows: list[dict]) -> dict:
    """Three-way accuracy over a run's verdicts. Pure arithmetic, one home.

    `accuracy` is None (never a fake 0.0) when no claim was decided either way.
    """
    counts = {
        "supported": sum(1 for r in verdict_rows if r["verdict"] == "SUPPORTED"),
        "contradicted": sum(1 for r in verdict_rows if r["verdict"] == "CONTRADICTED"),
        "unverifiable": sum(1 for r in verdict_rows if r["verdict"] == "UNVERIFIABLE"),
        "total": len(verdict_rows),
    }
    decided = counts["supported"] + counts["contradicted"]
    return {
        **counts,
        "accuracy": round(counts["supported"] / decided, 4) if decided else None,
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


def _opening_text(conn: sqlite3.Connection, doc_id: str, max_chars: int = 6000) -> str:
    """The document's first pages' text, from its CHUNKS.

    Chunks are the guaranteed text store (ingestion refuses empty documents);
    the sections in documents.metadata are a convenience that documents
    ingested before Phase 3 don't carry — reading those silently fed the
    metadata extractor an empty string and every source scored tier NONE.
    """
    rows = conn.execute(
        """
        SELECT section, text FROM chunks
        WHERE doc_id = ? AND kind = 'text' AND (page IS NULL OR page <= 2)
        ORDER BY id LIMIT 8
        """,
        (doc_id,),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT section, text FROM chunks WHERE doc_id = ? AND kind = 'text'"
            " ORDER BY id LIMIT 3",
            (doc_id,),
        ).fetchall()
    # Section headings must be included: for academic PDFs the title and the
    # DOI-bearing header line land in Docling section names, not chunk text —
    # dropping them hid exactly the fields metadata extraction exists to find.
    parts = []
    seen_sections: set[str] = set()
    for row in rows:
        section = row["section"]
        if section and section not in seen_sections:
            seen_sections.add(section)
            parts.append(f"## {section}")
        parts.append(row["text"])
    text = "\n\n".join(parts)[:max_chars]
    if not text.strip():
        raise ValueError(
            f"Document {doc_id!r} has no text chunks to extract metadata from — "
            "refusing to score a source that was never ingested properly"
        )
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
) -> dict:
    """Compute and persist all three scores for a run. Verdicts must exist."""
    verdict_rows = dbmod.list_verdicts(conn, run_id)
    if not verdict_rows:
        raise ValueError(f"Run {run_id!r} has no verdicts — run `verify` first")
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
    per_source: list[dict] = []
    source_years: list[int] = []
    try:
        for document in sources:
            metadata = extract_metadata(
                llm, settings.metadata_model, _opening_text(conn, document["id"])
            )
            tier, record = resolve_tier(metadata, crossref)
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
    dbmod.save_source_credibility(conn, run_id, per_source)
    credibility = aggregate_credibility(per_source, evidence_usage(conn, run_id))

    validity = assess_validity(
        llm,
        settings.validity_model,
        _report_sections(conn, run_id),
        weights=parse_weights(settings.validity_weights),
        source_years=source_years,
        current_year=current_year,
    )

    dbmod.save_run_scores(
        conn, run_id, accuracy=accuracy, credibility=credibility, validity=validity
    )
    return {"accuracy": accuracy, "credibility": credibility, "validity": validity}
