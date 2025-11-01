from pdf_pipeline.schema import Section, Table, new_document

from src.hallcheck.ingest_pdf import _document_to_pdfcontent


def test_document_to_pdfcontent_creates_placeholders():
    document = new_document("fixture.pdf")
    document.title = "Sample"
    document.authors = ["Author"]
    document.date = "2024"
    document.sections = [
        Section(
            title="Section",
            level=1,
            type="generic",
            text="Body text",
            page_range=(1, 1),
            confidence=0.6,
        )
    ]
    document.tables = [
        Table(
            id="TABLE-1",
            page=1,
            caption="Table caption",
            html="<table></table>",
            text="row",
            bbox=None,
        )
    ]
    body = "Body text\n\n[[TABLE-1]]"
    table_map = {"TABLE-1": {"html": "<table></table>", "text": "row", "page_number": 1, "caption": "Table caption"}}

    content = _document_to_pdfcontent(document, body, table_map)
    assert content.title == "Sample"
    assert content.tables
    assert "[[TABLE-1]]" in content.tables
