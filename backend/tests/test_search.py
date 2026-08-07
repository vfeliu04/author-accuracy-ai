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


REPORT_TEXT = "Report copy: global hunger affected 735 million people in 2023."


@pytest.fixture()
def mixed_kind_corpus(conn):
    """One run holding a REPORT whose chunk out-ranks every SOURCE chunk."""
    run_id = dbmod.create_run(conn)
    source_doc = dbmod.add_document(conn, run_id, "SOURCE", title="Source")
    report_doc = dbmod.add_document(conn, run_id, "REPORT", title="Report")

    embedder = FakeEmbedder(
        dim=DIM,
        mapping={
            REPORT_TEXT: _vec(1.0),  # identical to the query — always rank 1
            HUNGER_2023: _vec(0.9, 0.3),
            FOOD_INSECURITY: _vec(0.5, 0.5),
        },
    )
    source_ids = dbmod.add_chunks(
        conn,
        run_id,
        source_doc,
        [{"text": HUNGER_2023}, {"text": FOOD_INSECURITY}],
        embedder.embed([HUNGER_2023, FOOD_INSECURITY]),
    )
    report_ids = dbmod.add_chunks(
        conn, run_id, report_doc, [{"text": REPORT_TEXT}], embedder.embed([REPORT_TEXT])
    )
    return {"run": run_id, "source": source_ids, "report": report_ids}


def test_doc_kind_filter_excludes_top_ranked_report_chunk(conn, mixed_kind_corpus):
    run = mixed_kind_corpus["run"]
    # The report chunk is the closest vector AND a keyword match — with the
    # filter it must appear in neither channel, and the SOURCE chunks must
    # still fill the results (over-fetch: filtering the top k post-hoc would
    # starve the vector channel instead).
    vector_hits = vector_search(conn, run, _vec(1.0), k=2, doc_kind="SOURCE")
    assert vector_hits == mixed_kind_corpus["source"]

    keyword_hits = keyword_search(conn, run, "hunger 735 million", doc_kind="SOURCE")
    assert mixed_kind_corpus["report"][0] not in keyword_hits
    assert mixed_kind_corpus["source"][0] in keyword_hits

    hybrid_hits = hybrid_search(conn, run, "hunger 735 million", _vec(1.0), doc_kind="SOURCE")
    assert mixed_kind_corpus["report"][0] not in [h.chunk_id for h in hybrid_hits]
    assert hybrid_hits[0].chunk_id == mixed_kind_corpus["source"][0]


def test_doc_kind_none_is_unfiltered(conn, mixed_kind_corpus):
    run = mixed_kind_corpus["run"]
    hits = hybrid_search(conn, run, "hunger 735 million", _vec(1.0))
    assert hits[0].chunk_id == mixed_kind_corpus["report"][0]


def test_hit_carries_figure_id(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    figure_id = dbmod.add_figure(conn, run_id, doc_id, image_path="/tmp/fig.png", page=1)
    text = "Figure on page 1\n\nA bar chart of hunger levels."
    embedder = FakeEmbedder(dim=DIM, mapping={text: _vec(1.0)})
    dbmod.add_chunks(
        conn,
        run_id,
        doc_id,
        [{"text": text, "kind": "figure", "figure_id": figure_id}],
        embedder.embed([text]),
    )
    [hit] = hybrid_search(conn, run_id, "bar chart", _vec(1.0))
    assert hit.kind == "figure"
    assert hit.figure_id == figure_id
