"""Bibliography scan: the works a report cites, read from its own closing pages.

A pre-upload aid. When the user picks the report PDF, the frontend asks for the
report's reference list so it can show which cited works are missing from the
user's sources and offer the free copies as links. Nothing here persists: no
run, no upload row, no file — the spooled upload is read in place.

Two judgments, split the way the rest of the pipeline splits them: the model
turns printed reference entries into fields (the language judgment), and code
decides where the bibliography is, caps what the model sees and returns, and
checks every address before it is offered. The prompt is the mirror image of
credibility.METADATA_SYSTEM, which tells the model to IGNORE reference lists:
here every entry describes a work the report CITES, never the report.
"""

import re
from typing import BinaryIO, Literal

from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

from authorai.llm import LLM
from authorai.log import setup_logger

logger = setup_logger(__name__)

# Bounds are code, not schema: structured outputs do not honour max_length
# and a client-side validation failure would 500 the scan (see
# extract_references). 30,000 characters is 4–7x the measured real
# bibliographies (~7.5k Haiku tokens); 80 entries is more than any report in
# the example sets prints; 600 pages from the end covers any whole report
# the pipeline accepts while bounding a hostile file's page walk.
REFERENCE_MAX_CHARS = 30_000
MAX_REFERENCES = 80
REFERENCE_MAX_PAGES = 600

TextSource = Literal["heading", "tail", "none"]

# A line that IS a reference-list heading — optionally numbered ("7.
# References"), optionally colon-terminated, nothing else on the line. A
# mid-line mention ("see References") never matches. Extend the alternation
# only with evidence: a miss falls back to the document tail, which still works.
_HEADING = re.compile(
    r"^\s*(?:\d+[.\s]*)?(?:references|bibliography|works cited|reference list|literature cited)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def read_pages(handle: BinaryIO, *, max_pages: int = REFERENCE_MAX_PAGES) -> list[str]:
    """The text of the LAST `max_pages` pages, one string per page (empty for
    a page without text, such as a scan).

    pypdf, not Docling: a layout pass is seconds per page, far too slow for a
    dialog that must answer while the user is still picking sources, and the
    reference list needs no layout. `strict=False` tolerates the malformed
    xref tables real PDFs carry; a file encrypted with an empty user password
    (print/copy restrictions) is opened as readers do. Anything pypdf cannot
    read — its own exception family, or the KeyError/RecursionError a hostile
    file provokes — is one ValueError, the caller's 400.
    """
    try:
        reader = PdfReader(handle, strict=False)
        if reader.is_encrypted:
            reader.decrypt("")
        pages = reader.pages
        first = max(0, len(pages) - max_pages)
        return [pages[index].extract_text() or "" for index in range(first, len(pages))]
    except Exception as exc:  # noqa: BLE001 - every reader failure is "unreadable"
        raise ValueError(f"could not read the PDF ({type(exc).__name__}: {exc})") from exc


def reference_text(
    pages: list[str], *, max_chars: int = REFERENCE_MAX_CHARS
) -> tuple[str, TextSource]:
    """The closing text the model should read, and where it came from.

    From the LAST reference-list heading forward when the report prints one
    (the 51-page Drought report drops from 72,699 to 4,443 characters), else
    the last `max_chars` of the document (two of four test reports print no
    heading). No text at all is ("", "none"): a scanned PDF must not yield an
    invented bibliography, so the caller makes no model call on that value.
    """
    text = "\n".join(pages)
    if not text.strip():
        return "", "none"
    last = None
    for match in _HEADING.finditer(text):
        last = match
    if last is not None:
        return text[last.start() : last.start() + max_chars].strip(), "heading"
    return text[-max_chars:].strip(), "tail"


class Reference(BaseModel):
    """One printed reference entry, as fields. `entry` is the anchor: the
    text as printed, shown to the user when the title is null and the proof
    that the row came from the page rather than from memory."""

    title: str | None = Field(default=None, description="The cited work's title, as printed")
    authors: list[str] = Field(
        default_factory=list, description="Author names as printed; empty when none are printed"
    )
    year: int | None = Field(default=None, description="Publication year, as printed")
    doi: str | None = Field(
        default=None, description="The DOI printed in the entry (10.xxxx/...), without a URL prefix"
    )
    url: str | None = Field(default=None, description="A web address printed in the entry")
    entry: str = Field(description="The reference entry verbatim, as printed")

    @field_validator("title", "doi", "url", "entry", mode="after")
    @classmethod
    def _blank_is_absent(cls, value: str | None) -> str | None:
        # The model sometimes prints '' where it was told to leave null; the
        # DOI lookup and the printed-URL rule key on absence.
        if value is None:
            return None
        return value.strip() or None

    @field_validator("authors", mode="after")
    @classmethod
    def _drop_blank_authors(cls, value: list[str]) -> list[str]:
        return [name.strip() for name in value if name.strip()]


class ReferenceList(BaseModel):
    references: list[Reference] = Field(default_factory=list)


REFERENCES_SYSTEM = """\
You read the closing pages of a report and list the works its reference list
cites. Every entry you return describes a work the report CITES — never the
report itself. Report ONLY what the text actually prints — never guess, never
complete an entry from world knowledge. A field the entry does not print is
null.

- `entry`: the reference as printed, verbatim (line breaks joined), so a
  reader can find it on the page.
- `title`: the cited work's title as printed. Null when the entry prints none.
- `authors`: the names as printed, personal or organizational. Empty list if
  none are printed.
- `year`: the publication year as printed.
- `doi`: ONLY a DOI printed in the entry (10.xxxx/...), without a URL prefix.
  Never supply a DOI from memory, even for a famous paper.
- `url`: ONLY a web address printed in the entry.

In-text citations such as "(Smith, 2020)", footnote markers, figure and table
sources, and the report's own title and imprint are not entries. A document
without a reference list yields an empty list — that is the correct answer,
not a failure.
"""


def extract_references(llm: LLM, model: str, closing_text: str) -> ReferenceList:
    """The model's reading of the closing text, capped in code at
    MAX_REFERENCES — the cap is deliberately NOT in the schema, where
    structured outputs may not honour it and a validation failure would fail
    the whole scan instead of trimming it."""
    result = llm.parse(
        model=model,
        system=REFERENCES_SYSTEM,
        prompt=f"CLOSING PAGES:\n\n{closing_text}",
        output_type=ReferenceList,
    )
    if len(result.references) > MAX_REFERENCES:
        logger.warning(
            "the model returned %d references — keeping the first %d",
            len(result.references),
            MAX_REFERENCES,
        )
        result = ReferenceList(references=result.references[:MAX_REFERENCES])
    return result
