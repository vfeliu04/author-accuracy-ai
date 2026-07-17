from author_ai.pipelines.ingestion import chunk_text


def test_short_text_returns_single_chunk():
    text = "Global hunger fell slightly in 2023."
    assert chunk_text(text, max_chars=500, overlap=0) == [text]


def test_long_text_splits_into_bounded_chunks():
    sentence = "Cereal production reached 2800 million tonnes in 2023. "
    text = (sentence * 40).strip()
    chunks = chunk_text(text, max_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_no_paragraph_is_lost():
    text = "\n\n".join(
        f"Paragraph {i:02d} reports coverage figures for the region." for i in range(30)
    )
    chunks = chunk_text(text, max_chars=300, overlap=50)
    joined = " ".join(chunks)
    for i in range(30):
        assert f"Paragraph {i:02d}" in joined
