"""Hybrid search behavior on a small hand-built corpus across two runs."""

import pytest

from authorai import db as dbmod
from authorai.embeddings import FakeEmbedder
from authorai.search import hybrid_search, keyword_search, vector_search
from tests.conftest import DIM


def _vec(*values):
    padded = list(values) + [0.0] * (DIM - len(values))
    return padded


HUNGER_2023 = "Global hunger affected 735 million people in 2023."
FOOD_INSECURITY = "Food insecurity is rising across many regions of the world."
HARVEST_2024 = "The 2024 harvest exceeded expectations in Europe."
OTHER_RUN_TEXT = "Run B copy: hunger figure of 735 million appears here too."


# Hand-picked vectors per text, delivered through FakeEmbedder the way the
# real pipeline will deliver OpenAI vectors through OpenAIEmbedder.
EMBEDDER = FakeEmbedder(
    dim=DIM,
    mapping={
        HUNGER_2023: _vec(1.0),
        FOOD_INSECURITY: _vec(0.9, 0.4),
        HARVEST_2024: _vec(0.0, 0.0, 1.0),
        OTHER_RUN_TEXT: _vec(1.0),
    },
)


@pytest.fixture()
def corpus(conn):
    run_a = dbmod.create_run(conn)
    run_b = dbmod.create_run(conn)
    doc_a = dbmod.add_document(conn, run_a, "SOURCE", title="Doc A")
    doc_b = dbmod.add_document(conn, run_b, "SOURCE", title="Doc B")

    texts_a = [HUNGER_2023, FOOD_INSECURITY, HARVEST_2024]
    ids_a = dbmod.add_chunks(
        conn, run_a, doc_a, [{"text": t} for t in texts_a], EMBEDDER.embed(texts_a)
    )
    ids_b = dbmod.add_chunks(
        conn, run_b, doc_b, [{"text": OTHER_RUN_TEXT}], EMBEDDER.embed([OTHER_RUN_TEXT])
    )
    return {"run_a": run_a, "run_b": run_b, "a": ids_a, "b": ids_b}


def test_vector_search_ranks_by_similarity(conn, corpus):
    hits = vector_search(conn, corpus["run_a"], _vec(1.0, 0.1))
    assert hits[0] == corpus["a"][0]
    assert hits[1] == corpus["a"][1]


def test_keyword_search_finds_exact_number(conn, corpus):
    hits = keyword_search(conn, corpus["run_a"], "735 million")
    assert hits == [corpus["a"][0]]


def test_run_isolation_both_channels(conn, corpus):
    # Run B's chunk shares the same text terms and an identical vector,
    # yet must never leak into run A's results.
    keyword_hits = keyword_search(conn, corpus["run_a"], "735 million")
    vector_hits = vector_search(conn, corpus["run_a"], _vec(1.0))
    assert corpus["b"][0] not in keyword_hits
    assert corpus["b"][0] not in vector_hits

    assert keyword_search(conn, corpus["run_b"], "735 million") == [corpus["b"][0]]


def test_fts_syntax_characters_are_neutralized(conn, corpus):
    # Quotes, parens, and FTS operators in user text must not crash the query,
    # and the harmless tokens ("hunger", "735") must still match via OR.
    result = keyword_search(conn, corpus["run_a"], 'hunger "NEAR(" AND -735')
    assert corpus["a"][0] in result


def test_keyword_search_uses_or_semantics_for_sentence_queries(conn, corpus):
    # Claim-length queries contain words no chunk has; a single missing token
    # must not blank the whole keyword channel (implicit AND would).
    result = keyword_search(
        conn, corpus["run_a"], "global hunger reached 735 million people worldwide"
    )
    assert corpus["a"][0] in result


def test_hybrid_ranks_dual_channel_hit_first(conn, corpus):
    hits = hybrid_search(conn, corpus["run_a"], "hunger 735 million", _vec(1.0, 0.1))
    assert hits[0].chunk_id == corpus["a"][0]
    assert set(hits[0].channels) == {"vector", "keyword"}
    assert hits[0].text == HUNGER_2023
    single_channel = [h for h in hits if h.chunk_id != corpus["a"][0]]
    assert all(h.score < hits[0].score for h in single_channel)


def test_hybrid_empty_query_still_uses_vector_channel(conn, corpus):
    hits = hybrid_search(conn, corpus["run_a"], "", _vec(0.0, 0.0, 1.0))
    assert hits[0].chunk_id == corpus["a"][2]
    assert hits[0].channels == ("vector",)
