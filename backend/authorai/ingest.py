"""Ingestion: a ParsedDocument → chunking → embeddings → indexes.

A document arrives either from Docling (`parse_pdf`, the only function that
touches Docling) or from a stored snapshot (`load_snapshot`: the sections a
web page or transcript yielded when it was fetched). Everything downstream
works on plain dataclasses through one path, `ingest_parsed`, so tests
exercise the full ingestion path without Docling's models. An uploaded image
(`parse_image`) is a document of one figure. Figures are stored as PNG files
(an uploaded photo's bounded copy as JPEG), their chunk text carrying the
caption plus an LLM description.
"""

import io
import json
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, NamedTuple

import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms

from authorai import db as dbmod
from authorai.chunking import chunk_text
from authorai.embeddings import Embedder
from authorai.log import setup_logger

if TYPE_CHECKING:
    from PIL.Image import Image

logger = setup_logger(__name__)

TABLE_TEXT_CAP = 4000

FIGURE_DESCRIPTION_PROMPT = (
    "Describe this figure from a document in 2-3 sentences: what it shows, its "
    "axes or categories, and the main trend or takeaway. Be specific about any "
    "values you can read; do not speculate beyond what is visible."
)

# An uploaded image is a source in its own right: the judge may only quote the
# evidence TEXT, so the reading has to carry the image's words and numbers, not
# just its gist (a PDF figure keeps the prompt above; its page has the text).
#
# Measured (PR_C_MEASURE in the project folder: 3 real charts x 5 calls). An
# earlier wording listed a chart's rotated values apart from their labels in
# every run, so no line held a quotable pair, and named an untitled chart
# "water stress" in every run; this one paired every value it read, in every
# run, and invented no subject.
IMAGE_SOURCE_PROMPT = (
    "This image was uploaded as a source document for a fact-check; its words and "
    "numbers will be quoted as evidence, so they must be exactly what it prints.\n"
    "First write the line 'Text in the image:' and transcribe every legible piece of "
    "text exactly as printed: title, subtitle, axis titles, legend entries, data "
    "labels, table cells, annotations, notes and source lines. Write each value on "
    "one line together with everything that identifies it: its category, and its "
    "series, year or column when there is one (for example 'Region A, 2008: 12.5'). "
    "Never list the labels and the values separately. Write a table one row per "
    "line, its cells in order. Axis tick values need not be listed. Copy numbers, "
    "units, names and spellings exactly; do not round, convert, correct or update "
    "them. Leave out anything you cannot read with confidence rather than guessing.\n"
    "Then write the line 'Description:' and describe in 2-3 sentences what the image "
    "shows. Use only what the image itself prints or shows: do not name a subject, "
    "unit, place or source it does not print, and if it does not say what it "
    "measures, say so."
)
# A screenshot of a dense table transcribes to ~2k tokens; a reading cut off at
# this budget is refused, never kept (llm.describe_image).
IMAGE_DESCRIPTION_MAX_TOKENS = 4096

# What an image upload may be, keyed by the format Pillow finds in its BYTES,
# with the suffix it is stored under: the file endpoint serves that suffix's
# media type, so the content, never the client's file name, decides it. A phone
# photo carrying a depth map or preview opens as a multi-picture JPEG (MPO)
# with several frames; it is one photo, not an animation.
IMAGE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "MPO": ".jpg", "WEBP": ".webp"}
_DECODERS = ("PNG", "JPEG", "WEBP")  # Pillow plugins allowed to open an upload at all
_STILL_MULTI_FRAME = ("MPO",)
# Decoding costs ~4 bytes a pixel on the worker, and a byte cap cannot bound
# pixels: a 72-megapixel PNG of one colour is ~70 KB.
IMAGE_MAX_PIXELS = 50_000_000
# Read as one piece, a longer image is reduced until its text is unreadable (a
# 1080x15000 scroll screenshot becomes 112x1556) and every claim on it would end
# unverifiable in a run that looks fine: refused at upload instead.
MAX_IMAGE_ASPECT = 4.0
# Each progressive scan is a decoding pass over the whole image, and neither
# libjpeg nor Pillow bounds how many a file declares; encoders write about ten.
# Counted from the bytes: entropy-coded data stuffs every 0xFF, so the start-of-
# scan pair appears only as a marker.
MAX_JPEG_SCANS = 100
_SCAN_MARKER = b"\xff\xda"
_SCAN_BLOCK_BYTES = 1 << 20
# Claude's standard vision tier, where the caption model reads (the API would
# downscale anything larger).
MODEL_IMAGE_MAX_EDGE = 1568
MODEL_IMAGE_MAX_PIXELS = 1_150_000
# A smaller image is enlarged toward the tier, at most this much: at its own
# 1024x390 the caption model misread two small rotated values of a real chart in
# every run (12.5 as 12.3, 8.9 as 8.3), and enlarged it read all 28 in every
# run (PR_C_MEASURE). Enlarging adds no information, so it stops here.
MODEL_IMAGE_MAX_ENLARGEMENT = 2.0
# Bytes of one stored copy. Every claim that retrieves an image carries its copy
# into the verify batch, which the Batch API caps at 256 MB: at two images a
# claim and base64's third on top, ~160 claims still fit. A real photo fits at
# quality 90; only noise-like pixels walk down the quality ladder.
MODEL_IMAGE_MAX_BYTES = 600_000
# Lossless for charts and text; JPEG (4:4:4, so small text stays sharp) only
# where it saves at least a third: a photo's PNG is ~3x its JPEG, an enlarged
# chart's only ~1.1x, and a few KB are not worth lossy text.
_JPEG_SAVING = 1.5
_JPEG_QUALITIES = (90, 80, 70, 60, 50)
FIGURE_SUFFIXES = {"PNG": ".png", "JPEG": ".jpg"}
_SIXTEEN_BIT_GREY = ("I", "I;16", "I;16B", "I;16L", "I;16N")
# The EXIF orientation, read from its tag alone: Pillow's exif_transpose also
# rewrites the EXIF block, and on damaged EXIF (found by fuzzing) that raised
# struct.error and TypeError. Same turns as exif_transpose makes.
_ORIENTATION_TAG = 0x0112
_UPRIGHT = {
    2: PILImage.Transpose.FLIP_LEFT_RIGHT,
    3: PILImage.Transpose.ROTATE_180,
    4: PILImage.Transpose.FLIP_TOP_BOTTOM,
    5: PILImage.Transpose.TRANSPOSE,
    6: PILImage.Transpose.ROTATE_270,
    7: PILImage.Transpose.TRANSVERSE,
    8: PILImage.Transpose.ROTATE_90,
}
_PROFILED_MODES = ("RGB", "RGBA", "L", "CMYK")  # modes a colour profile is applied in


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
    # How the stored copy is encoded (FIGURE_SUFFIXES): PNG for every PDF
    # figure; an uploaded photo's copy may be JPEG (model_copy).
    fmt: Literal["PNG", "JPEG"] = "PNG"
    # The stored bytes when they are already made (an uploaded image's copy,
    # whose `image` is exactly these pixels); else `image` is encoded as `fmt`.
    encoded: bytes | None = None


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


class ImageRefused(ValueError):
    """An image source the gate will not read. Its message reads after the
    file's name: "'<name>' is animated; ..."."""


def _jpeg_scans(handle: BinaryIO) -> int:
    """How many start-of-scan markers a JPEG holds, read block by block (a
    marker split across two blocks counts once)."""
    handle.seek(0)
    count, tail = 0, b""
    while block := handle.read(_SCAN_BLOCK_BYTES):
        data = tail + block
        count += data.count(_SCAN_MARKER)
        tail = data[-1:]
    return count


def open_image_source(handle: BinaryIO) -> "Image":
    """Open an uploaded image WITHOUT decoding its pixels: the one gate both the
    upload and the later ingest pass (scan count, format, frames, pixel count,
    shape — from the bytes and the header). Raises ImageRefused. The caller's
    handle stays open whatever happens."""
    handle.seek(0)
    if handle.read(3) == b"\xff\xd8\xff":
        scans = _jpeg_scans(handle)
        if scans > MAX_JPEG_SCANS:
            raise ImageRefused(
                f"is a JPEG of {scans} progressive scans; at most {MAX_JPEG_SCANS} are accepted"
            )
    handle.seek(0)
    try:
        image = PILImage.open(handle, formats=_DECODERS)
    except PILImage.DecompressionBombError as exc:
        # Pillow's own ceiling, far above ours: raised inside open() itself.
        raise ImageRefused(
            f"is too large to open safely; images up to {IMAGE_MAX_PIXELS:,} pixels are accepted"
        ) from exc
    except (OSError, SyntaxError, ValueError) as exc:  # UnidentifiedImageError is an OSError
        raise ImageRefused("is not a PNG, JPEG or WebP image") from exc
    width, height = image.size
    tall = height > width
    if image.format not in IMAGE_FORMATS:
        reason = "is not a PNG, JPEG or WebP image"
    elif getattr(image, "is_animated", False) and image.format not in _STILL_MULTI_FRAME:
        reason = "is animated; only still images can be checked"
    elif width * height > IMAGE_MAX_PIXELS:
        reason = (
            f"is {width}×{height} pixels; images up to {IMAGE_MAX_PIXELS:,} pixels are accepted"
        )
    elif max(width, height) > MAX_IMAGE_ASPECT * min(width, height):
        reason = (
            f"is {width}×{height} pixels, more than {MAX_IMAGE_ASPECT:g} times as "
            f"{'tall' if tall else 'wide'} as it is {'wide' if tall else 'tall'}; crop it "
            "into parts that can each be read at once"
        )
    else:
        return image
    # Not image.close(): an explicit close also closes a handle Pillow was only
    # lent (its `with` exit does not). The image object is simply dropped.
    raise ImageRefused(reason)


def image_suffix(handle: BinaryIO) -> str:
    """The suffix an image upload is stored under, once the gate passes it."""
    with open_image_source(handle) as image:
        return IMAGE_FORMATS[image.format]


def encode_image(image: "Image", fmt: str = "PNG", quality: int = _JPEG_QUALITIES[0]) -> bytes:
    """The one encoder for stored figures. A PDF figure's PNG is byte for byte
    what `image.save(path, "PNG")` always wrote."""
    buffer = io.BytesIO()
    options = {"quality": quality, "subsampling": 0} if fmt == "JPEG" else {}
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


class ModelCopy(NamedTuple):
    """The copy of an uploaded image the models are shown: `data` is what is
    stored and attached for the judge, and `image` is exactly its pixels,
    which the caption model reads — the two never see different pixels."""

    image: "Image"
    fmt: Literal["PNG", "JPEG"]
    data: bytes


def _eight_bit(image: "Image") -> "Image":
    """Palette and 1-bit images are resampled by nearest neighbour whatever
    filter is asked for (a thin line kept or dropped at random), and 16-bit
    grey cannot be resampled at all: each becomes 8-bit colour or grey first,
    at full size. Any other mode is returned untouched (still undecoded, so a
    JPEG can be decoded at reduced scale)."""
    if image.mode == "P":
        return image.convert("RGBA" if image.has_transparency_data else "RGB")
    if image.mode == "1":
        return image.convert("L")
    if image.mode not in _SIXTEEN_BIT_GREY:
        return image
    values = np.asarray(image.convert("I"))
    grey = PILImage.fromarray((np.clip(values, 0, 65535) >> 8).astype(np.uint8))
    key = image.info.get("transparency")
    if isinstance(key, int):
        # Matched on the 16-bit value: scaled to 8 bits it can no longer be.
        grey.putalpha(PILImage.fromarray(np.where(values == key, 0, 255).astype(np.uint8)))
    grey.info.update({k: image.info[k] for k in ("icc_profile", "exif") if k in image.info})
    return grey


def _upright(image: "Image") -> "Image":
    """Turned by its EXIF orientation (a phone stores its photo sideways)."""
    orientation = image.getexif().get(_ORIENTATION_TAG)
    method = _UPRIGHT.get(orientation) if isinstance(orientation, int) else None
    return image if method is None else image.transpose(method)


def _in_srgb(image: "Image") -> "Image":
    """Colours converted from the image's own profile to sRGB: the copy
    carries no metadata, so a Display P3 photo or screenshot would otherwise
    be read as if its values were sRGB. A profile that cannot be applied is
    what a browser ignores too: logged, pixels kept as stored."""
    icc = image.info.get("icc_profile")
    if not icc or image.mode not in _PROFILED_MODES:
        return image
    try:
        return ImageCms.profileToProfile(
            image,
            ImageCms.ImageCmsProfile(io.BytesIO(icc)),
            ImageCms.createProfile("sRGB"),
            outputMode="RGBA" if image.mode == "RGBA" else "RGB",
        )
    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
        logger.warning("image colour profile not applied (%s); pixels read as stored", exc)
        return image


def _stored(image: "Image") -> ModelCopy:
    """PNG when it fits the byte budget and JPEG would not save a third; else
    the best JPEG quality that fits, and past the lowest, fewer pixels."""
    png = encode_image(image, "PNG")
    data = encode_image(image, "JPEG")
    if len(png) <= MODEL_IMAGE_MAX_BYTES and not len(data) * _JPEG_SAVING < len(png):
        return ModelCopy(image, "PNG", png)
    for quality in _JPEG_QUALITIES[1:]:
        if len(data) <= MODEL_IMAGE_MAX_BYTES:
            break
        data = encode_image(image, "JPEG", quality)
    while len(data) > MODEL_IMAGE_MAX_BYTES:
        size = (max(1, math.floor(image.width * 0.85)), max(1, math.floor(image.height * 0.85)))
        image = image.resize(size, PILImage.Resampling.LANCZOS)
        data = encode_image(image, "JPEG", _JPEG_QUALITIES[-1])
    stored = PILImage.open(io.BytesIO(data))
    stored.load()
    return ModelCopy(stored, "JPEG", data)


def model_copy(image: "Image") -> ModelCopy:
    """The copy of an uploaded image the models are shown.

    Brought to 8 bits, sized to the standard vision tier (reduced, or enlarged
    at most MODEL_IMAGE_MAX_ENLARGEMENT times, keeping its shape), turned
    upright, converted to sRGB, alpha laid on white, stripped of all metadata
    (nothing from the camera, GPS included, leaves with it), and stored within
    MODEL_IMAGE_MAX_BYTES (_stored).
    """
    image = _eight_bit(image)
    width, height = image.size
    scale = min(
        MODEL_IMAGE_MAX_ENLARGEMENT,
        MODEL_IMAGE_MAX_EDGE / max(width, height),
        math.sqrt(MODEL_IMAGE_MAX_PIXELS / (width * height)),
    )
    size = (max(1, math.floor(width * scale)), max(1, math.floor(height * scale)))
    if scale < 1:
        # thumbnail() decodes a JPEG at a reduced scale (draft), so a big photo
        # is never held at full size.
        image.thumbnail(size, PILImage.Resampling.LANCZOS)
    image = _in_srgb(_upright(image))
    if image.has_transparency_data:
        rgba = image.convert("RGBA")
        image = PILImage.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")
    if scale > 1:
        # Enlarged last, once the image is 8-bit RGB (upright, so width and
        # height swap with the orientation).
        image = image.resize(
            size if image.size == (width, height) else size[::-1], PILImage.Resampling.LANCZOS
        )
    image.info.clear()
    return _stored(image)


def parse_image(path: Path | str, *, name: str) -> ParsedDocument:
    """One uploaded image as a document of one figure, with no page (the image
    is the whole source) and no caption: the file's name is the user's, not the
    image's, so it stays out of the evidence text, where it could supply a
    quote or a year the image never prints ('Stunting fell to 23.2% in
    2024.png'); it titles the document instead (ingest_image). The upload's
    gate runs again first: one rule, two doors. Every error quotes `name`,
    the upload's real file name."""
    try:
        with Path(path).open("rb") as handle, open_image_source(handle) as image:
            copy = model_copy(image)
    except ImageRefused as exc:
        raise ImageRefused(f"{name!r} {exc}") from exc
    except Exception as exc:
        # Untrusted pixels: any failure while decoding them (a truncated file,
        # corrupt data, a decoder's own error type) means the image cannot be read.
        raise ValueError(
            f"{name!r} could not be read as an image: {type(exc).__name__}: {exc}"
        ) from exc
    figure = ParsedFigure(page=None, image=copy.image, fmt=copy.fmt, encoded=copy.data)
    return ParsedDocument(title=None, sections=[], tables=[], figures=[figure])


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
    with replaced_atomically(path) as part:
        part.write_text(text, encoding="utf-8")


@contextmanager
def replaced_atomically(path: Path) -> Iterator[Path]:
    """Write a file whole or not at all: the caller writes the yielded
    `<name>.part`, which is renamed over `path` once the block finishes and
    removed if anything in it (or the rename) fails. db.link_artifact_paths
    names the .part files a link upload can leave by this same convention."""
    part = path.with_name(path.name + ".part")
    try:
        yield part
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


def ingest_image(
    conn,
    embedder: Embedder,
    run_id: str,
    path: Path | str,
    *,
    name: str,
    kind: str,
    figures_dir: Path | str,
    upload_id: str,
    describe: "Callable[[Image], str] | None" = None,
) -> str:
    """Ingest one uploaded image: a single figure chunk whose text is the
    caption model's reading of it (IMAGE_SOURCE_PROMPT), titled with the file's
    stem. The original file is never rewritten; the judge is shown the bounded
    copy stored beside the run's other figures. A failed reading names the
    file: with several images in a run, "cut off at max_tokens" alone does not
    say which one to crop or replace."""
    path = Path(path)
    read = None
    if describe is not None:

        def read(image):
            try:
                return describe(image)
            except Exception as exc:
                raise RuntimeError(
                    f"{name!r} could not be read by the caption model: {type(exc).__name__}: {exc}"
                ) from exc

    return ingest_parsed(
        conn,
        embedder,
        run_id,
        parse_image(path, name=name),
        path=path,
        kind=kind,
        figures_dir=figures_dir,
        upload_id=upload_id,
        describe=read,
        fallback_title=Path(name).stem,
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
        image_path = figure_dir / f"fig-{index}{FIGURE_SUFFIXES[figure.fmt]}"
        planned_figures.append((figure_id, figure, image_path, description))
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
        encoded = figure.encoded
        image_path.write_bytes(
            encoded if encoded is not None else encode_image(figure.image, figure.fmt)
        )

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
