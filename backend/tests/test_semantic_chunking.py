import numpy as np

from author_ai.pipelines.ingestion import chunk_text


def test_semantic_chunk_merge(monkeypatch):
    """
    Adjacent similar sentences should merge; dissimilar ones should split.
    """

    sentences = [
        "Cats purr softly.",
        "Cats meow when hungry.",
        "The global economy is slowing down.",
    ]
    text = " ".join(sentences)

    # Craft embeddings: first two are similar, third is orthogonal.
    vectors = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    monkeypatch.setattr(
        "author_ai.pipelines.ingestion.embed_texts",
        lambda inputs: vectors[: len(inputs)],
    )

    chunks = chunk_text(text, max_chars=500, overlap=0)

    assert len(chunks) == 2, f"expected 2 chunks, got {len(chunks)}"
    assert "Cats" in chunks[0]
    assert "economy" in chunks[1].lower()
