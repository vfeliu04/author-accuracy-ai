"""PDF ingestion: Docling parsing → chunking → embeddings → indexes.

`parse_pdf` is the only function that touches Docling; everything downstream
works on plain dataclasses, so tests exercise the full ingestion path without
Docling's models. Figures are stored as PNG files plus their caption text —
LLM-written figure descriptions arrive in Phase 3 with the shared client.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from authorai import db as dbmod
from authorai.chunking import chunk_text
from authorai.embeddings import Embedder

if TYPE_CHECKING:
    from PIL.Image import Image

TABLE_TEXT_CAP = 4000


@dataclass
class ParsedSection:
    title: str
    page: int | None
    text: str


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


def ingest_pdf(
    conn,
    embedder: Embedder,
    run_id: str,
    path: Path | str,
    kind: str,
    figures_dir: Path | str,
) -> str:
    """Ingest one PDF into a run: document row, chunks, figures, embeddings."""
    path = Path(path)
    parsed = parse_pdf(path)
    doc_id = dbmod.add_document(conn, run_id, kind, title=parsed.title or path.stem)

    chunks: list[dict] = []
    for section in parsed.sections:
        for piece in chunk_text(section.text):
            chunks.append(
                {
                    "text": piece,
                    "page": section.page,
                    "section": section.title or None,
                    "kind": "text",
                }
            )

    for table in parsed.tables:
        text = f"{table.caption}\n\n{table.markdown}".strip()
        if len(text) > TABLE_TEXT_CAP:
            text = text[:TABLE_TEXT_CAP] + "\n[table truncated]"
        if text:
            chunks.append({"text": text, "page": table.page, "kind": "table"})

    figure_dir = Path(figures_dir) / run_id / doc_id
    if parsed.figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
    for index, figure in enumerate(parsed.figures, start=1):
        image_path = figure_dir / f"fig-{index}.png"
        figure.image.save(image_path, format="PNG")
        figure_id = dbmod.add_figure(
            conn,
            run_id,
            doc_id,
            image_path=str(image_path),
            page=figure.page,
            caption=figure.caption or None,
        )
        caption = figure.caption.strip()
        if caption:
            text = caption
        elif figure.page is not None:
            text = f"Figure on page {figure.page}"
        else:
            text = "Figure"
        chunks.append({"text": text, "page": figure.page, "kind": "figure", "figure_id": figure_id})

    if not chunks:
        raise ValueError(f"No extractable content in {path} — refusing to index an empty document")

    embeddings = embedder.embed([chunk["text"] for chunk in chunks])
    dbmod.add_chunks(conn, run_id, doc_id, chunks, embeddings)
    return doc_id
