from __future__ import annotations

from pathlib import Path

import pytest

from pdf_pipeline.ingest import ingest_pdf_v2

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_fixture(name: str, builder) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / name
    if not path.exists():
        builder(path)
    return path


def _build_single_column(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    y = 720
    page.insert_text((72, y), "Global Food Security Report", fontsize=18)
    y -= 40
    page.insert_text((72, y), "Executive Summary", fontsize=14)
    y -= 24
    page.insert_text((72, y), "Hunger levels remain elevated in several regions.", fontsize=11)
    y -= 40
    page.insert_text((72, y), "1 Introduction", fontsize=13)
    y -= 24
    paragraph = "This section outlines the drivers of hunger and the methodology used for assessment."
    page.insert_text((72, y), paragraph, fontsize=11, wrap=440)
    y -= 80
    table_lines = [
        "Country  2020  2024",
        "Kenya    24    29",
        "Peru     18    19",
    ]
    for line in table_lines:
        page.insert_text((72, y), line, fontsize=11)
        y -= 14
    doc.save(str(path))
    doc.close()


def _build_table_appendix(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    y = 720
    page.insert_text((72, y), "Appendix A: Country Rankings", fontsize=16)
    y -= 40
    lines = [
        "Rank  Country     2000  2010  2020",
        "1     Ghana       24    16    12",
        "2     Nepal       32    28    20",
        "3     Laos        34    30    25",
    ]
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y -= 14
    doc.save(str(path))
    doc.close()


def _build_scanned(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    pil = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    img = pil.new("L", (600, 400), color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 560, 360], outline=0, width=3)
    draw.text((80, 80), "Finance Summary", fill=0)
    draw.text((80, 150), "Revenue 2024: 1,200", fill=0)
    tmp_image = path.with_suffix(".png")
    img.save(tmp_image)
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(str(tmp_image))
    page.insert_image(fitz.Rect(50, 50, 562, 360), pixmap=pix)
    tmp_image.unlink(missing_ok=True)
    doc.save(str(path))
    doc.close()


def test_single_column_ingestion_produces_sections():
    pytest.importorskip("pdfplumber")
    path = _ensure_fixture("report_single_column.pdf", _build_single_column)
    document, body_text, table_map, metrics = ingest_pdf_v2(str(path))
    assert document.title
    assert document.sections
    assert any(getattr(section, "type", None) == "executive_summary" for section in document.sections)
    assert table_map
    assert "[[TABLE-1]]" in body_text or table_map


def test_table_appendix_detects_tables():
    pytest.importorskip("pdfplumber")
    path = _ensure_fixture("table_only_appendix.pdf", _build_table_appendix)
    document, _, table_map, _ = ingest_pdf_v2(str(path))
    assert len(document.tables) >= 1 or table_map


def test_scanned_document_routes_to_ocr_flag():
    pytest.importorskip("pdfplumber")
    path = _ensure_fixture("scanned_financial.pdf", _build_scanned)
    document, _, _, metrics = ingest_pdf_v2(str(path))
    router = metrics.get("router", {})
    assert router.get("is_scanned") or document.metadata.get("is_scanned")
