import pytest

from authorai.embeddings import FakeEmbedder, OpenAIEmbedder, normalize


def test_openai_embedder_refuses_to_run_without_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(api_key=None, model="text-embedding-3-large", dim=8)


def test_fake_embedder_is_deterministic_and_normalized():
    embedder = FakeEmbedder(dim=8)
    first = embedder.embed(["some text never seen before"])[0]
    second = embedder.embed(["some text never seen before"])[0]
    assert first == second
    assert abs(sum(x * x for x in first) - 1.0) < 1e-5


def test_fake_embedder_mapping_wins_over_pseudo():
    fixed = [1.0] + [0.0] * 7
    embedder = FakeEmbedder(dim=8, mapping={"known": fixed})
    assert embedder.embed(["known"])[0] == fixed


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero"):
        normalize([0.0] * 8)
