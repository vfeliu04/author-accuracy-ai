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

import html
import re
import sqlite3
import time
from typing import Literal
from urllib.parse import quote

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


_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_DOI_SHAPE = re.compile(r"^10\.\d{4,9}/\S+$")


def clean_doi(doi: str) -> str | None:
    """Strip the URL/`doi:` prefixes the extractor is told not to emit but
    sometimes does, and reject anything that is not DOI-shaped — an
    unvalidated string in the request path would silently look up a
    DIFFERENT DOI (fragments are dropped, `?` starts a query string)."""
    candidate = _DOI_PREFIX.sub("", doi.strip())
    if not _DOI_SHAPE.match(candidate):
        logger.warning("Malformed DOI %r — skipping DOI lookup", doi)
        return None
    return candidate


class CrossrefClient:
    """Thin Crossref client: DOI resolution and bibliographic title search.

    404/4xx is an ANSWER ("not found" / rejected) and returns None. 429/5xx
    and transport failures are retried with backoff and then RAISE — treating
    a throttled or down Crossref as "not found" would silently downgrade
    verification tiers and make credibility scores non-reproducible between
    runs of identical inputs. A missing mailto is allowed but logged loudly —
    Crossref's anonymous pool is slower and they ask for contact info.
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

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        failure = ""
        for attempt in range(CROSSREF_RETRIES + 1):
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.json().get("message")
                if response.status_code != 429 and response.status_code < 500:
                    logger.info(
                        "Crossref %s -> %s (an answer: not found)", path, response.status_code
                    )
                    return None
                failure = f"HTTP {response.status_code}"
            if attempt < CROSSREF_RETRIES:
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(
            f"Crossref gave no answer after {CROSSREF_RETRIES + 1} attempts ({path}): {failure}"
        )

    def by_doi(self, doi: str) -> dict | None:
        cleaned = clean_doi(doi)
        if cleaned is None:
            return None
        return self._get(f"/works/{quote(cleaned, safe='/')}")

    def by_title(self, title: str, rows: int = 5) -> list[dict]:
        message = self._get("/works", params={"query.bibliographic": title, "rows": rows})
        return message.get("items", []) if message else []


_TAGS = re.compile(r"<[^>]+>")


def _normalize_title(title: str) -> str:
    """Crossref titles carry markup (<i>…</i>) and XML entities (&amp;);
    claims of 'exact match' are made on this normalized form (the Phase 3
    lesson: collapse non-alphanumerics to spaces, never delete them).
    `\\w` keeps the comparison Unicode-aware — an ASCII-only class would
    normalize every non-Latin-script title to '' and make VERIFIED_TITLE
    unreachable for such sources."""
    return re.sub(r"[\W_]+", " ", _TAGS.sub(" ", html.unescape(title).lower())).strip()


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


# Trailing tokens that are never family names ("John Smith Jr.", "Jane Doe, PhD").
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"}


def _family_names(metadata: SourceMetadata) -> set[str]:
    """Best-effort family names from authors "as printed".

    Academic front matter prints both "Jane Doe" and "Doe, Jane" — for the
    inverted form the family name is BEFORE the comma, so taking the last
    token of the whole string would corroborate against given names.
    Conveniently, "Jane Doe, PhD" resolves correctly under the same rule
    (last non-suffix token of the pre-comma segment)."""
    names = set()
    for author in metadata.authors:
        segment = author.split(",")[0] if "," in author else author
        tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'-]+", segment)]
        while tokens and tokens[-1] in _NAME_SUFFIXES:
            tokens.pop()
        if tokens:
            names.add(tokens[-1])
    return names


def _publishers_agree(ours: str | None, theirs: str | None) -> bool:
    """Phrase containment either way — 'Welthungerhilfe' ≡ 'Welthungerhilfe e.V.'."""
    our_tokens, their_tokens = _phrase_tokens(ours or ""), _phrase_tokens(theirs or "")
    if not our_tokens or not their_tokens:
        return False
    a, b = f" {' '.join(our_tokens)} ", f" {' '.join(their_tokens)} "
    return a in b or b in a


def _title_match(metadata: SourceMetadata, record: dict) -> bool:
    """Normalized-exact title match PLUS one corroborating field.

    A title match alone is not identity — 'Annual Report 2023' matches
    thousands of works. Corroboration: publication year within ±1, author
    family names intersecting, or publisher agreement. When the title itself
    contains a year, every same-titled work trivially agrees on it, so year
    corroboration is vacuous for exactly the generic-title class it exists to
    reject — those titles must corroborate via authors or publisher (NGO
    reports with year-bearing titles usually print no personal authors).
    """
    ours = _normalize_title(metadata.title or "")
    if not ours or ours not in {_normalize_title(t) for t in _record_titles(record)}:
        return False
    year_in_title = re.search(r"\b(19|20)\d{2}\b", ours)
    our_year, record_year = _year_of(metadata.publication_date), _record_year(record)
    if (
        not year_in_title
        and our_year is not None
        and record_year is not None
        and abs(our_year - record_year) <= 1
    ):
        return True
    record_families = {
        (a.get("family") or "").lower() for a in record.get("author") or [] if a.get("family")
    }
    if record_families & _family_names(metadata):
        return True
    return _publishers_agree(metadata.publisher, record.get("publisher"))


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


def _phrase_tokens(text: str) -> list[str]:
    """Word tokens with runs of single letters fused, so 'U.N.' ≡ 'UN'."""
    fused: list[str] = []
    run = ""
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) == 1:
            run += token
            continue
        if run:
            fused.append(run)
            run = ""
        fused.append(token)
    if run:
        fused.append(run)
    return fused


def _publisher_authority(publisher: str | None, tier1: list[str], tier2: list[str]) -> float:
    """Consecutive word-boundary phrase matching.

    'un' must match 'UN' or 'U.N.' but never 'University', and a multi-word
    needle must appear as an adjacent in-order phrase — a subset-of-words
    check would give 'World Bank' authority to any publisher containing both
    words anywhere. Caveat that stays with the CONFIG: a single generic word
    as a needle ('Science', 'Nature') matches every publisher containing that
    word; keep configured needles as specific as the real names allow.
    """
    if not publisher:
        return 0.0
    haystack = " " + " ".join(_phrase_tokens(publisher)) + " "

    def hits(needles: list[str]) -> bool:
        return any(
            f" {' '.join(_phrase_tokens(needle))} " in haystack for needle in needles if needle
        )

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
