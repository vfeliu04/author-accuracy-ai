"""Source credibility: LLM-extracted metadata, Crossref verification tiers, scoring.

The LLM reads a source's opening pages and extracts bibliographic metadata
(the language judgment); code verifies it against Crossref and does all the
arithmetic. Verification tiers, strongest first:

  VERIFIED_DOI    the extracted DOI resolves at Crossref
  VERIFIED_TITLE  no DOI, but a Crossref title search returns a record whose
                  title matches ours exactly (normalized) AND a second field
                  corroborates (year ±1, or first-author family name) — a
                  title-only match on a generic title is a rejection
  METADATA_ONLY   metadata extracted but not externally verified
  NONE            nothing extractable

v1 defects deliberately killed here: publisher was reachable ONLY via
Crossref-by-DOI (DOI-less NGO reports floored at ~32/100); publisher tier
matching used naive substrings ("un" matched "University"); the title-match
tier was designed but never built; unknown values got floor points.
"""

import re
import sqlite3
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from authorai.llm import LLM
from authorai.log import setup_logger

logger = setup_logger(__name__)

CROSSREF_BASE = "https://api.crossref.org"
CROSSREF_TIMEOUT = 10.0
CROSSREF_RETRIES = 2

Tier = Literal["VERIFIED_DOI", "VERIFIED_TITLE", "METADATA_ONLY", "NONE"]

METADATA_SYSTEM = """\
You extract bibliographic metadata from the opening pages of a document.
Report ONLY what the text actually states — never guess, never infer from
style, never fill a field from world knowledge. A field the text does not
state is null.

- `title`: the document's own title (not a chapter or figure title).
- `authors`: personal names only, as printed. Empty list if none are printed.
- `publisher`: the publishing organization as printed (imprint, institute,
  journal, agency). Null if not stated.
- `publication_date`: as printed, ideally ISO (YYYY or YYYY-MM or YYYY-MM-DD).
- `doi`: the DOI string only (10.xxxx/...), without a URL prefix.
"""


class SourceMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    publisher: str | None = None
    publication_date: str | None = None
    doi: str | None = None


def extract_metadata(llm: LLM, model: str, opening_text: str) -> SourceMetadata:
    return llm.parse(
        model=model,
        system=METADATA_SYSTEM,
        prompt=f"DOCUMENT OPENING PAGES:\n\n{opening_text}",
        output_type=SourceMetadata,
    )


class CrossrefClient:
    """Thin Crossref client: DOI resolution and bibliographic title search.

    Retries only transport failures (timeout/connect); a non-200 is an answer
    ("not found" / rejected), not an outage, and is returned as None with a
    log line. A missing mailto is allowed but logged loudly — Crossref's
    anonymous pool is slower and they ask for contact info.
    """

    def __init__(self, mailto: str | None, timeout: float = CROSSREF_TIMEOUT):
        if not mailto:
            logger.warning(
                "CROSSREF_MAILTO is not set — using Crossref's anonymous pool "
                "(slower, and impolite for sustained use)"
            )
        agent = f"AuthorAI/2.0 (mailto:{mailto})" if mailto else "AuthorAI/2.0"
        self._client = httpx.Client(
            base_url=CROSSREF_BASE, timeout=timeout, headers={"User-Agent": agent}
        )

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        for attempt in range(CROSSREF_RETRIES + 1):
            try:
                response = self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == CROSSREF_RETRIES:
                    logger.warning("Crossref unreachable after retries (%s): %s", path, exc)
                    return None
                time.sleep(1.0)
                continue
            if response.status_code != 200:
                logger.info("Crossref %s -> %s", path, response.status_code)
                return None
            return response.json().get("message")
        return None

    def by_doi(self, doi: str) -> dict | None:
        return self._get(f"/works/{doi}")

    def by_title(self, title: str, rows: int = 5) -> list[dict]:
        message = self._get("/works", params={"query.bibliographic": title, "rows": rows})
        return message.get("items", []) if message else []


_TAGS = re.compile(r"<[^>]+>")


def _normalize_title(title: str) -> str:
    """Crossref titles carry markup (<i>…</i>) and typographic noise; claims of
    'exact match' are made on this normalized form (the Phase 3 lesson: collapse
    non-alphanumerics to spaces, never delete them)."""
    return re.sub(r"[^a-z0-9]+", " ", _TAGS.sub(" ", title.lower())).strip()


def _year_of(date_string: str | None) -> int | None:
    if not date_string:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", date_string)
    return int(match.group()) if match else None


def _record_year(record: dict) -> int | None:
    for field in ("published", "published-print", "published-online", "issued"):
        parts = (record.get(field) or {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _record_titles(record: dict) -> list[str]:
    titles = [t for t in record.get("title") or [] if t]
    subtitles = [s for s in record.get("subtitle") or [] if s]
    combined = list(titles)
    if titles and subtitles:
        combined.append(f"{titles[0]} {subtitles[0]}")
    return combined


def _family_names(metadata: SourceMetadata) -> set[str]:
    names = set()
    for author in metadata.authors:
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", author)
        if tokens:
            names.add(tokens[-1].lower())  # last printed token ≈ family name
    return names


def _title_match(metadata: SourceMetadata, record: dict) -> bool:
    """Normalized-exact title match PLUS one corroborating field.

    A title match alone is not identity — 'Annual Report 2023' matches
    thousands of works. Corroboration: publication year within ±1, or the
    record's first-author family name appearing among our extracted authors.
    """
    ours = _normalize_title(metadata.title or "")
    if not ours or ours not in {_normalize_title(t) for t in _record_titles(record)}:
        return False
    our_year, record_year = _year_of(metadata.publication_date), _record_year(record)
    if our_year is not None and record_year is not None and abs(our_year - record_year) <= 1:
        return True
    record_families = {
        (a.get("family") or "").lower() for a in record.get("author") or [] if a.get("family")
    }
    return bool(record_families & _family_names(metadata))


def resolve_tier(metadata: SourceMetadata, crossref: CrossrefClient) -> tuple[Tier, dict | None]:
    if metadata.doi:
        record = crossref.by_doi(metadata.doi)
        if record:
            return "VERIFIED_DOI", record
    if metadata.title:
        for record in crossref.by_title(metadata.title):
            if _title_match(metadata, record):
                return "VERIFIED_TITLE", record
    if metadata.model_dump(exclude_defaults=True):
        return "METADATA_ONLY", None
    return "NONE", None


def merge_record(metadata: SourceMetadata, record: dict | None) -> SourceMetadata:
    """Crossref fills gaps the document's own pages left; it never overrides
    what the document states about itself."""
    if not record:
        return metadata
    updates: dict = {}
    if not metadata.publisher and record.get("publisher"):
        updates["publisher"] = record["publisher"]
    if not metadata.publication_date and _record_year(record):
        updates["publication_date"] = str(_record_year(record))
    if not metadata.title and _record_titles(record):
        updates["title"] = _record_titles(record)[0]
    return metadata.model_copy(update=updates) if updates else metadata


_TIER_POINTS = {"VERIFIED_DOI": 20.0, "VERIFIED_TITLE": 15.0, "METADATA_ONLY": 5.0, "NONE": 0.0}


def _publisher_authority(publisher: str | None, tier1: list[str], tier2: list[str]) -> float:
    """Word-boundary matching — 'un' must match 'UN' but never 'University'."""
    if not publisher:
        return 0.0
    words = set(re.findall(r"[a-z0-9]+", publisher.lower()))

    def hits(needles: list[str]) -> bool:
        return any(set(needle.lower().split()) <= words for needle in needles if needle)

    if hits(tier1):
        return 30.0
    if hits(tier2):
        return 22.5
    return 15.0


def score_source(
    metadata: SourceMetadata,
    tier: Tier,
    *,
    tier1_publishers: list[str],
    tier2_publishers: list[str],
    current_year: int,
) -> dict:
    """Component scores, 0–100 total, NO floors — unknown earns nothing.

    (v1 gave floor points for absent publishers and unknown dates, so a source
    about which nothing was known scored ~32/100 instead of reading as the
    unknown it is.)
    """
    fields = [
        metadata.title,
        metadata.authors or None,
        metadata.publisher,
        metadata.publication_date,
        metadata.doi,
    ]
    completeness = 30.0 * sum(1 for f in fields if f) / len(fields)
    authority = _publisher_authority(metadata.publisher, tier1_publishers, tier2_publishers)

    year = _year_of(metadata.publication_date)
    if year is None:
        recency = 0.0
    else:
        age = max(0, current_year - year)
        recency = 20.0 if age <= 2 else 12.0 if age <= 5 else 6.0 if age <= 10 else 3.0

    components = {
        "metadata_completeness": completeness,
        "authority": authority,
        "recency": recency,
        "verification": _TIER_POINTS[tier],
    }
    return {"components": components, "total": round(sum(components.values()), 1)}


def evidence_usage(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    """How many quote-verified verdicts cite each source document.

    SUPPORTED and CONTRADICTED both count — a source that contradicts claims
    is doing exactly its job (v1 counted only SUPPORTED, so contradicting
    sources contributed zero weight)."""
    rows = conn.execute(
        """
        SELECT c.doc_id, count(*) AS n
        FROM verdicts v JOIN chunks c ON c.id = v.quoted_chunk_id
        WHERE v.run_id = ? GROUP BY c.doc_id
        """,
        (run_id,),
    ).fetchall()
    return {row["doc_id"]: row["n"] for row in rows}


def aggregate_credibility(per_source: list[dict], usage: dict[str, int]) -> dict:
    """Usage-weighted mean of source scores; explicitly labeled unweighted mean
    when no verdict cites any source (never None → 0.0)."""
    if not per_source:
        return {"score": None, "method": "no_sources", "sources": []}
    total_usage = sum(usage.get(row["doc_id"], 0) for row in per_source)
    if total_usage:
        method = "usage_weighted_mean"
        score = sum(row["total"] * usage.get(row["doc_id"], 0) for row in per_source) / total_usage
    else:
        method = "unweighted_mean_no_usage"
        score = sum(row["total"] for row in per_source) / len(per_source)
    return {
        "score": round(score, 1),
        "method": method,
        "sources": [
            {
                "doc_id": row["doc_id"],
                "total": row["total"],
                "tier": row["tier"],
                "usage": usage.get(row["doc_id"], 0),
            }
            for row in per_source
        ],
    }
