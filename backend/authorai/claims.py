"""Claim extraction: one structured LLM call over the report's sections.

The LLM does the language judgment (what is a checkable claim, what are its
normalized value/unit/year); code does the bookkeeping. No regex cascades.
"""

from typing import Literal

from pydantic import BaseModel, Field

from authorai.llm import LLM, prompt_fingerprint

EXTRACTION_SYSTEM = """\
You extract checkable factual claims from a report so they can be verified
against source documents.

Rules:
- A claim is a statement of fact that could in principle be checked against
  external evidence: quantities, dates, trends, named events, causal assertions.
- Do NOT extract: section headings, opinions, recommendations, questions,
  descriptions of the report itself ("this report covers..."), or vague
  statements with no checkable content.
- `text` must be the claim VERBATIM from the report — never paraphrase.
- Reports sometimes present a claim as something OTHERS say: "some analyses
  claim X", "a fabricated report asserts X", "contrary coverage has claimed X".
  The checkable claim is X ITSELF — the embedded assertion. Take `text` from
  the embedded clause, verbatim from inside the frame, dropping the reporting
  frame ("some analyses claim that...") and any editorial verdict the report
  attaches ("...an event that never occurred"). The frame's existence is not a
  checkable claim; X is.
- `stance` records the REPORT'S OWN position on the claim. It is "disavowed"
  ONLY when the report attaches an explicit falsity marker to the claim itself:
  wording like "an event that never occurred", "a fabricated figure", "this is
  false", or a table column/header that labels the value fabricated. Everything
  else — including neutral reported speech — is "asserted": a report that
  relays "some analyses claim X" WITHOUT marking X false is lending X its
  platform, so that claim's stance is "asserted", not "disavowed".
- `subject` is a short noun phrase naming what the claim is about.
- `value` is the main number in the claim as a plain float (735 million -> 735000000,
  23.5% -> 23.5). Null when the claim has no central number.
- `unit` names what the value counts ("people", "percent", "tonnes", "USD"). Null if no value.
- `year` is the year the claim is about (not the publication year). Null if none stated.
- `page` is the page number from the [page N] marker of the section the claim
  appears in. Null if unknown.
- Split compound sentences into separate claims when each part is independently
  checkable.
- Be exhaustive: every checkable assertion in the text should appear in your
  output. A sentence does not need a named source, a citation, or a number to
  be a claim — general statements of fact about drivers, trends, or causes
  ("X remains the leading cause of Y", "Z destabilizes W worldwide") are
  claims too. Do not skip a topic sentence merely because a neighbouring
  sentence carries the citation.

Tables appear after the sections. Reports routinely put their most checkable
figures in a table rather than in prose, so treat them as first-class:
- Every row that states an indicator is its own claim. A row holding both a real
  and a fabricated figure for the same subject is TWO claims, not one.
- A row is not a sentence, so for table claims `text` may be composed from the
  row's cells into a readable statement — but every number, year and quoted
  phrase must be copied exactly as written. Do not round, convert or reword them.
"""


class ExtractedClaim(BaseModel):
    text: str = Field(description="The claim, verbatim from the report")
    subject: str = Field(description="Short noun phrase naming what the claim is about")
    value: float | None = Field(default=None, description="Central number, as a plain float")
    unit: str | None = Field(default=None, description="What the value counts")
    year: int | None = Field(default=None, description="Year the claim is about")
    page: int | None = Field(default=None, description="Page the claim appears on")
    stance: Literal["asserted", "disavowed"] = Field(
        default="asserted",
        description=(
            "The report's own position: 'disavowed' only when the report"
            " explicitly marks the claim false; otherwise 'asserted'"
        ),
    )


class ClaimExtraction(BaseModel):
    claims: list[ExtractedClaim]


def _page_marker(item: dict) -> str:
    page = item.get("page")
    return f"[page {page}]" if page is not None else "[page unknown]"


def build_extraction_prompt(sections: list[dict], tables: list[dict] | None = None) -> str:
    parts = ["Extract the checkable factual claims from this report.\n"]
    for section in sections:
        title = section.get("title") or "(untitled section)"
        parts.append(f"{_page_marker(section)} ## {title}\n{section['text']}")
    for table in tables or []:
        parts.append(f"{_page_marker(table)} ## TABLE\n{table['text']}")
    return "\n\n".join(parts)


# Stamped on every stored claim; eval-extract refuses rows whose hash is not
# the CURRENT prompt's — recall/precision measured on claims from an older
# extractor say nothing about the prompt being tuned (same rationale as
# verification's VERDICT_PROMPT_HASH).
EXTRACTION_PROMPT_HASH = prompt_fingerprint(
    EXTRACTION_SYSTEM,
    build_extraction_prompt(
        [{"title": "Frozen Section", "text": "Frozen section text.", "page": 1}],
        [{"text": "frozen | table | row", "page": 2}],
    ),
    # The type extract_claims actually parses — the wrapper's schema is part
    # of what the model sees, so it must be part of the hash.
    output_type=ClaimExtraction,
)


def claims_as_rows(claims: list[ExtractedClaim]) -> list[dict]:
    """Storage rows for add_claims — the ONE place the hash stamp is applied,
    so the CLI and the jobs worker cannot drift apart on it."""
    return [
        {**claim.model_dump(), "extraction_prompt_hash": EXTRACTION_PROMPT_HASH} for claim in claims
    ]


def extract_claims(
    llm: LLM,
    sections: list[dict],
    model: str,
    tables: list[dict] | None = None,
) -> list[ExtractedClaim]:
    """Extract claims from a report's prose sections and its tables.

    Tables are passed separately because ingestion stores them as their own
    chunks, not as section text — without them the report's tabulated figures
    (often the most checkable claims it makes) are invisible to extraction.
    """
    if not sections and not tables:
        raise ValueError("No sections to extract claims from — was the document ingested?")
    result = llm.parse(
        model=model,
        system=EXTRACTION_SYSTEM,
        prompt=build_extraction_prompt(sections, tables),
        output_type=ClaimExtraction,
    )
    return result.claims
