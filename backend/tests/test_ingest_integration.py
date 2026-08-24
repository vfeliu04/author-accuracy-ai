"""Real-Docling integration test — excluded in CI (downloads layout models).

Run locally with: python -m pytest -m integration -q
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "example_sources"
    / "example source one"
    / "2025_world_hunger.pdf"
)


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example PDF not present")
def test_real_docling_parses_example_pdf():
    from authorai.ingest import parse_pdf

    parsed = parse_pdf(EXAMPLE)
    assert parsed.sections, "expected at least one section"
    total_text = sum(len(section.text) for section in parsed.sections)
    assert total_text > 1000, f"suspiciously little text extracted: {total_text} chars"
