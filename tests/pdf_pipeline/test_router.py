import fitz
from pdf_pipeline.router import route


def _make_pdf(tmp_path, lines):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 14
    doc.save(path)
    doc.close()
    return str(path)


def test_router_detects_report(tmp_path):
    pdf_path = _make_pdf(tmp_path, ["Executive Summary", "This is a test report."])
    decision = route(pdf_path)
    assert decision.label == "report"
    assert not decision.features["is_scanned"]
    assert decision.features["text_pages"] == 1


def test_router_detects_scanned(tmp_path):
    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    doc.new_page()  # no text inserted
    doc.save(path)
    doc.close()
    decision = route(str(path))
    assert decision.label == "scanned"
    assert decision.features["is_scanned"]


def test_router_detects_table_heavy(tmp_path):
    pdf_path = _make_pdf(tmp_path, ["1 2 3 4", "5 6 7 8", "9 10 11 12", "13 14 15 16"])
    decision = route(pdf_path)
    assert decision.features["table_like_pages"] >= 1
