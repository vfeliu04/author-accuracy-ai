from pdf_pipeline.postprocess.header_footer import (
    identify_repeated_lines,
    strip_repeated_lines,
)


def test_identify_and_strip_repeated_lines():
    pages = [
        ["Report Title", "Body line 1", "Page 1"],
        ["Report Title", "Body line 2", "Page 2"],
        ["Report Title", "Body line 3", "Page 3"],
    ]
    headers, footers = identify_repeated_lines(pages, threshold=0.6)
    assert "report title" in headers
    assert "page 1" not in headers

    cleaned = strip_repeated_lines(pages[0], headers, footers)
    assert cleaned[0] == "Body line 1"
    assert cleaned[-1] != "Page 1"
