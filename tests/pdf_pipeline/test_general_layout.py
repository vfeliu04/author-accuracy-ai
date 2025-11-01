from pdf_pipeline.config import pipeline_config
from pdf_pipeline.extractors.general_layout import _assemble_sections, Line
from pdf_pipeline.ingest import build_body_text_with_tables
from pdf_pipeline.schema import Section, Table, new_document


def test_assemble_sections_builds_hierarchy():
    base_font = 10.0
    pages = [
        [
            Line("Executive Summary", font_size=14.0, page_index=1, bbox=(0, 0, 100, 10), confidence=0.0),
            Line("This is the summary text.", font_size=10.0, page_index=1, bbox=(0, 20, 100, 30), confidence=0.0),
        ],
        [
            Line("1 Introduction", font_size=12.0, page_index=2, bbox=(0, 0, 100, 10), confidence=0.0),
            Line("Body content for introduction.", font_size=10.0, page_index=2, bbox=(0, 20, 100, 30), confidence=0.0),
        ],
    ]

    sections = _assemble_sections(pages, base_font=base_font, cfg=pipeline_config)
    assert len(sections) == 2
    assert sections[0].title == "Executive Summary"
    assert sections[0].type == "executive_summary"
    assert "summary text" in sections[0].text.lower()
    assert sections[1].level >= 1
    assert "introduction" in sections[1].title.lower()


def test_build_body_text_with_tables_inserts_placeholders():
    document = new_document("sample.pdf")
    document.sections = [
        Section(
            title="Introduction",
            level=1,
            type="generic",
            text="This section discusses data.",
            page_range=(1, 1),
            confidence=0.7,
        )
    ]
    document.tables = [
        Table(
            id="TABLE-1",
            page=1,
            caption="Summary table",
            html=None,
            text="A\tB",
            bbox=None,
        )
    ]

    body_text, table_map = build_body_text_with_tables(document)
    assert "[[TABLE-1]]" in body_text
    assert "TABLE-1" in table_map
