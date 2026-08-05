import pytest

from authorai.chunking import chunk_text


def test_short_text_single_chunk():
    text = "Global hunger fell slightly in 2023."
    assert chunk_text(text, max_chars=500, overlap=50) == [text]


def test_chunks_are_bounded_and_lossless():
    paragraphs = [f"Paragraph {i:02d} discusses topic number {i} in detail." for i in range(40)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    joined = " ".join(chunks)
    for i in range(40):
        assert f"Paragraph {i:02d}" in joined


def test_overlap_is_carried_between_chunks():
    # ~60-char paragraphs with max 120 force one paragraph per chunk,
    # each new chunk opening with the 40-char tail of its predecessor.
    paragraphs = [
        f"Sentence number {i} about world hunger statistics padded out." for i in range(4)
    ]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=120, overlap=40)
    assert len(chunks) >= 2
    assert chunks[1].startswith(chunks[0][-40:])


def test_oversized_paragraph_is_split_on_sentences_in_order():
    paragraph = " ".join(f"Fact {i:02d} is stated here in sentence form." for i in range(30))
    chunks = chunk_text(paragraph, max_chars=200, overlap=0)
    assert all(len(chunk) <= 200 for chunk in chunks)
    joined = " ".join(chunks)
    # Every sentence present AND in document order — chunk text is quoted as
    # evidence downstream, so reordering is a correctness bug, not a nit.
    positions = [joined.index(f"Fact {i:02d}") for i in range(30)]
    assert positions == sorted(positions)


def test_long_sentence_after_short_ones_keeps_document_order():
    # Regression: the hard-wrap path used to emit the wrapped segments BEFORE
    # the accumulated preceding sentences, splicing non-adjacent text.
    paragraph = "Short intro sentence here. " + "X" * 500
    chunks = chunk_text(paragraph, max_chars=200, overlap=0)
    joined = "".join(chunks)
    assert chunks[0].startswith("Short intro sentence here.")
    assert joined.index("Short intro") < joined.index("X" * 50)
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_single_sentence_longer_than_cap_is_hard_wrapped():
    text = "x" * 950
    chunks = chunk_text(text, max_chars=300, overlap=0)
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == 950


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello", max_chars=100, overlap=100)
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("hello", max_chars=0, overlap=0)
    with pytest.raises(ValueError, match="non-negative"):
        chunk_text("hello", max_chars=100, overlap=-1)
