"""Ingestion: a ParsedDocument → chunking → embeddings → indexes.

A document arrives either from Docling (`parse_pdf`, the only function that
touches Docling) or from a stored snapshot (`load_snapshot`: the sections a
web page or transcript yielded when it was fetched). Everything downstream
works on plain dataclasses through one path, `ingest_parsed`, so tests
exercise the full ingestion path without Docling's models. Figures are stored
as PNG files, their chunk text carrying the caption plus an LLM description.
"""

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from authorai import db as dbmod
from authorai.chunking import chunk_text
from authorai.embeddings import Embedder

if TYPE_CHECKING:
    from PIL.Image import Image

TABLE_TEXT_CAP = 4000

FIGURE_DESCRIPTION_PROMPT = (
    "Describe this figure from a document in 2-3 sentences: what it shows, its "
    "axes or categories, and the main trend or takeaway. Be specific about any "
    "values you can read; do not speculate beyond what is visible."
)


@dataclass
class ParsedSection:
    title: str
    page: int | None
    text: str
    # Transcript time locator; None for every PDF and web section.
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass
class ParsedTable:
    page: int | None
    markdown: str
    caption: str = ""


@dataclass
class ParsedFigure:
    page: int | None
    image: "Image"
    caption: str = ""


@dataclass
class ParsedDocument:
    title: str | None
    sections: list[ParsedSection]
    tables: list[ParsedTable]
    figures: list[ParsedFigure]


@lru_cache(maxsize=1)
def _get_converter():
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "Docling is not installed — PDF ingestion requires it. "
            'Install the backend with: pip install -e ".[dev]"'
        ) from exc

    options = PdfPipelineOptions()
    # Digital PDFs only for now. Scanned-PDF OCR is a deliberate later
    # addition, not a silent gap — this flag is where it will be enabled.
    options.do_ocr = False
    options.images_scale = 2.0
    options.generate_picture_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def parse_pdf(path: Path | str) -> ParsedDocument:
    """Parse a PDF into sections, tables, and figures via Docling."""
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

    document = _get_converter().convert(str(path)).document

    sections: list[ParsedSection] = []
    tables: list[ParsedTable] = []
    figures: list[ParsedFigure] = []
    current_title = ""
    current_page: int | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        if current_parts:
            sections.append(
                ParsedSection(
                    title=current_title, page=current_page, text="\n\n".join(current_parts)
                )
            )
        current_parts = []

    for item, _level in document.iterate_items():
        # SectionHeaderItem subclasses TextItem — check it first.
        if isinstance(item, SectionHeaderItem):
            flush()
            current_title = item.text.strip()
            current_page = _page_of(item)
        elif isinstance(item, TableItem):
            tables.append(
                ParsedTable(
                    page=_page_of(item),
                    markdown=item.export_to_markdown(document),
                    caption=item.caption_text(document) or "",
                )
            )
        elif isinstance(item, PictureItem):
            image = item.get_image(document)
            if image is not None:
                figures.append(
                    ParsedFigure(
                        page=_page_of(item),
                        image=image,
                        caption=item.caption_text(document) or "",
                    )
                )
        elif isinstance(item, TextItem):
            text = item.text.strip()
            if text:
                if current_page is None:
                    current_page = _page_of(item)
                current_parts.append(text)
    flush()

    title = sections[0].title if sections and sections[0].title else None
    return ParsedDocument(title=title, sections=sections, tables=tables, figures=figures)


_TIME_LOCATOR = ("start_seconds", "end_seconds")

# Version of the snapshot file format. The frontend parses snapshots too, so a
# file written by an older extractor must be refused loudly, never misread.
SNAPSHOT_SCHEMA = 1


def _section_record(section: ParsedSection) -> dict:
    """A section as stored JSON: the time locator only when it is set, so a
    PDF's stored sections keep exactly the {title, page, text} shape they have
    always had (dedup copies of older documents describe the same shape)."""
    record = asdict(section)
    for key in _TIME_LOCATOR:
        if record[key] is None:
            del record[key]
    return record


def write_snapshot(path: Path | str, parsed: ParsedDocument, provenance: dict) -> None:
    """Store what a fetched source yielded, atomically: a `.part` file renamed
    into place, so a crash can never leave a half-written snapshot that a retry
    would trust. Bytes depend only on content (sorted keys), never on dict
    order. Snapshots carry sections only — web pages keep their tables inline
    as text, and transcripts have neither tables nor figures."""
    if parsed.tables or parsed.figures:
        raise ValueError("snapshots carry sections only; tables and figures are not stored")
    path = Path(path)
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "document": {
            "title": parsed.title,
            "sections": [_section_record(section) for section in parsed.sections],
        },
        "provenance": provenance,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    part = path.with_name(path.name + ".part")
    try:
        part.write_text(text, encoding="utf-8")
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def load_snapshot(path: Path | str) -> tuple[ParsedDocument, dict]:
    """Read a stored snapshot back. Anything but a well-formed current-schema
    snapshot raises — re-fetching the source is the only safe recovery."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable snapshot {path}: {exc}") from exc
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema != SNAPSHOT_SCHEMA:
        raise ValueError(
            f"unsupported snapshot schema {schema!r} in {path} (expected {SNAPSHOT_SCHEMA}) "
            "— re-fetch the source"
        )
    document, provenance = payload.get("document"), payload.get("provenance")
    if not isinstance(document, dict) or not isinstance(provenance, dict):
        raise ValueError(f"malformed snapshot {path}: missing document or provenance")
    try:
        sections = [ParsedSection(**record) for record in document["sections"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed snapshot {path}: {exc}") from exc
    parsed = ParsedDocument(title=document.get("title"), sections=sections, tables=[], figures=[])
    return parsed, provenance


def ingest_pdf(
    conn,
    embedder: Embedder,
    run_id: str,
    path: Path | str,
    kind: str,
    figures_dir: Path | str,
    upload_id: str | None = None,
    describe: "Callable[[Image], str] | None" = None,
    fallback_title: str | None = None,
) -> str:
    """Ingest one PDF into a run (see ingest_parsed for the write contract)."""
    path = Path(path)
    return ingest_parsed(
        conn,
        embedder,
        run_id,
        parse_pdf(path),
        path=path,
        kind=kind,
        figures_dir=figures_dir,
        upload_id=upload_id,
        describe=describe,
        fallback_title=fallback_title,
    )


def ingest_snapshot(
    conn,
    embedder: Embedder,
    run_id: str,
    path: Path | str,
    *,
    kind: str,
    figures_dir: Path | str,
    upload_id: str,
    fallback_title: str | None = None,
) -> str:
    """Ingest a stored web/transcript snapshot. Its provenance (URL, fetch time,
    the page's declared metadata) is kept in documents.metadata, where
    credibility scoring reads it and a dedup copy carries it along."""
    path = Path(path)
    parsed, provenance = load_snapshot(path)
    return ingest_parsed(
        conn,
        embedder,
        run_id,
        parsed,
        path=path,
        kind=kind,
        figures_dir=figures_dir,
        upload_id=upload_id,
        fallback_title=fallback_title,
        extra_metadata={"provenance": provenance},
    )


def ingest_parsed(
    conn,
    embedder: Embedder,
    run_id: str,
    parsed: ParsedDocument,
    *,
    path: Path | None,
    kind: str,
    figures_dir: Path | str,
    upload_id: str | None = None,
    describe: "Callable[[Image], str] | None" = None,
    fallback_title: str | None = None,
    extra_metadata: dict | None = None,
) -> str:
    """Ingest one parsed document into a run: upload + document rows, chunks, figures.

    All failure-prone external work (parsing, figure descriptions, the
    embedding network call) happens BEFORE anything is written, so an
    ingestion failure cannot leave a half-ingested document behind; what
    remains after that point is a short sequence of local SQLite writes.
    When `upload_id` is not supplied (CLI path), an uploads row is recorded
    from the file itself so every document stays traceable to its source PDF.
    When `describe` is supplied, each figure's chunk text gains an LLM-written
    description — this must happen here, before embedding, because chunk text
    is immutable once written.
    """
    doc_id = dbmod.new_id()

    chunks: list[dict] = []
    for section in parsed.sections:
        for piece in chunk_text(section.text):
            chunks.append(
                {
                    "text": piece,
                    "page": section.page,
                    "section": section.title or None,
                    "kind": "text",
                    "start_seconds": section.start_seconds,
                    "end_seconds": section.end_seconds,
                }
            )

    for table in parsed.tables:
        text = f"{table.caption}\n\n{table.markdown}".strip()
        if len(text) > TABLE_TEXT_CAP:
            text = text[:TABLE_TEXT_CAP] + "\n[table truncated]"
        if text:
            chunks.append({"text": text, "page": table.page, "kind": "table"})

    figure_dir = (Path(figures_dir) / run_id / doc_id).resolve()
    planned_figures: list[tuple[str, ParsedFigure, Path, str | None]] = []
    for index, figure in enumerate(parsed.figures, start=1):
        figure_id = dbmod.new_id()
        description = describe(figure.image) if describe else None
        planned_figures.append((figure_id, figure, figure_dir / f"fig-{index}.png", description))
        caption = figure.caption.strip()
        if caption:
            text = caption
        elif figure.page is not None:
            text = f"Figure on page {figure.page}"
        else:
            text = "Figure"
        if description:
            text = f"{text}\n\n{description}"
        chunks.append({"text": text, "page": figure.page, "kind": "figure", "figure_id": figure_id})

    if not chunks:
        raise ValueError(f"No extractable content in {path} — refusing to index an empty document")

    # Network call — deliberately before any database or filesystem write.
    embeddings = embedder.embed([chunk["text"] for chunk in chunks])

    if planned_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
    for _figure_id, figure, image_path, _description in planned_figures:
        figure.image.save(image_path, format="PNG")

    if upload_id is None:
        if path is None:
            raise ValueError("ingest_parsed needs an upload_id or the source path to record one")
        upload_id = dbmod.add_upload(conn, kind, path.name, str(path.resolve()))
    dbmod.add_document(
        conn,
        run_id,
        kind,
        upload_id=upload_id,
        # API uploads live on disk under generated names — their real name is
        # the fallback, so untitled documents don't display as hex ids.
        title=parsed.title or fallback_title or (path.stem if path is not None else None),
        metadata=json.dumps(
            {
                "sections": [_section_record(section) for section in parsed.sections],
                **(extra_metadata or {}),
            }
        ),
        doc_id=doc_id,
        # Which model made this document's vectors — the dedup donor filter:
        # a future ingest under a different model must not reuse them.
        embedding_model=embedder.model,
    )
    for figure_id, figure, image_path, description in planned_figures:
        dbmod.add_figure(
            conn,
            run_id,
            doc_id,
            image_path=str(image_path),
            page=figure.page,
            caption=figure.caption or None,
            description=description,
            figure_id=figure_id,
        )
    dbmod.add_chunks(conn, run_id, doc_id, chunks, embeddings)
    return doc_id
