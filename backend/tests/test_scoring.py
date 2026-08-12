"""Scoring tests: three-way accuracy, validity rubric mechanics, score_run persistence."""

import json

import pytest

from authorai import db as dbmod
from authorai.config import Settings
from authorai.credibility import SourceMetadata
from authorai.scoring import (
    ComponentAssessment,
    ValidityAssessment,
    accuracy_scores,
    assess_validity,
    parse_weights,
    recency_score,
    score_run,
)
from tests.conftest import DIM, FakeLLM


def _verdict(kind):
    return {"verdict": kind}


# --- accuracy ---------------------------------------------------------------


def test_accuracy_uses_decided_denominator():
    rows = (
        [_verdict("SUPPORTED")] * 20
        + [_verdict("CONTRADICTED")] * 5
        + [_verdict("UNVERIFIABLE")] * 12
    )
    scores = accuracy_scores(rows)
    assert scores["accuracy"] == 0.8  # 20/25 — unverifiable NOT in the denominator
    assert scores["coverage"] == round(25 / 37, 4)
    assert scores["supported"] == 20
    assert scores["unverifiable"] == 12


def test_accuracy_is_none_not_zero_when_nothing_decided():
    scores = accuracy_scores([_verdict("UNVERIFIABLE")] * 4)
    assert scores["accuracy"] is None  # a fake 0.0 would read as "all wrong"
    assert scores["coverage"] == 0.0


def test_accuracy_empty_run():
    scores = accuracy_scores([])
    assert scores["accuracy"] is None
    assert scores["coverage"] is None


# --- validity ----------------------------------------------------------------


def test_parse_weights_is_loud_about_garbage():
    parse_weights("coverage:0.5,consistency:0.5")  # valid subset is fine
    with pytest.raises(ValueError, match="Unknown validity component"):
        parse_weights("coverage:0.5,vibes:0.5")
    with pytest.raises(ValueError, match="Malformed weight"):
        parse_weights("coverage:lots")
    with pytest.raises(ValueError, match="must sum to 1"):
        parse_weights("coverage:0.9,consistency:0.9")
    # NaN passes an `abs(sum - 1) > 0.001` check (NaN comparisons are False)
    # and would surface as a NaN score serialized into the API.
    with pytest.raises(ValueError, match="finite"):
        parse_weights("coverage:nan,consistency:0.5,methodology:0.5")
    with pytest.raises(ValueError, match="finite"):
        parse_weights("coverage:-0.5,consistency:1.5")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_weights("coverage:0.5,coverage:0.5")


def test_recency_from_real_years_and_none_when_unknown():
    assert recency_score([2025, 2024], current_year=2026) == 100.0
    assert recency_score([2018], current_year=2026) == 60.0
    assert recency_score([], current_year=2026) is None  # v1 returned a constant 70.0


SECTIONS = [
    {"title": "Intro", "text": "Hunger rose sharply in 2023 across three regions."},
    {"title": "Methods", "text": "Data was collected from FAO national surveys."},
]


def _assessment(quote="Data was collected from FAO national surveys."):
    part = ComponentAssessment(score=80, justification="States its data source.", quote=quote)
    return ValidityAssessment(coverage=part, consistency=part, methodology=part, context=part)


def test_validity_weighted_total_and_quote_verification():
    llm = FakeLLM(parse_results={ValidityAssessment: _assessment()})
    result = assess_validity(
        llm,
        "model-m",
        SECTIONS,
        weights=parse_weights(
            "coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1"
        ),
        source_years=[2025],
        current_year=2026,
    )
    # 80 on the four rubric components, 100 recency, weights as configured.
    assert result["score"] == 82.0
    assert result["components"]["methodology"]["quote_verified"] == 1
    assert result["components"]["recency"]["score"] == 100.0


def test_validity_unfindable_quote_flags_but_does_not_zero():
    llm = FakeLLM(parse_results={ValidityAssessment: _assessment(quote="not in the report at all")})
    result = assess_validity(
        llm,
        "model-m",
        SECTIONS,
        weights=parse_weights("coverage:1.0"),
        source_years=[],
        current_year=2026,
    )
    assert result["components"]["coverage"]["quote_verified"] == 0
    assert result["components"]["coverage"]["score"] == 80.0  # flagged, not punished


def test_validity_missing_quote_is_flagged_not_trusted():
    """Omitting the mandated quote must not look MORE trustworthy than
    providing a wrong one."""
    llm = FakeLLM(parse_results={ValidityAssessment: _assessment(quote=None)})
    result = assess_validity(
        llm,
        "model-m",
        SECTIONS,
        weights=parse_weights("coverage:1.0"),
        source_years=[],
        current_year=2026,
    )
    assert result["components"]["coverage"]["quote_verified"] == 0
    assert result["components"]["coverage"]["score"] == 80.0  # flagged, not punished


def test_validity_no_scorable_component_is_loud():
    # recency carries all weight but no source has a date -> nothing to weigh.
    llm = FakeLLM(parse_results={ValidityAssessment: _assessment()})
    with pytest.raises(ValueError, match="No weighted validity component"):
        assess_validity(
            llm,
            "model-m",
            SECTIONS,
            weights=parse_weights("recency:1.0"),
            source_years=[],
            current_year=2026,
        )


def test_validity_missing_recency_renormalizes_weights():
    llm = FakeLLM(parse_results={ValidityAssessment: _assessment()})
    result = assess_validity(
        llm,
        "model-m",
        SECTIONS,
        weights=parse_weights("coverage:0.5,recency:0.5"),
        source_years=[],  # no dates -> recency excluded, coverage carries all weight
        current_year=2026,
    )
    assert result["score"] == 80.0
    assert result["weights_used"] == {"coverage": 1.0}
    assert result["components"]["recency"]["score"] is None


def test_validity_requires_sections():
    with pytest.raises(ValueError, match="No report sections"):
        assess_validity(
            FakeLLM(),
            "m",
            [],
            weights=parse_weights("coverage:1.0"),
            source_years=[],
            current_year=2026,
        )


# --- score_run orchestration --------------------------------------------------


class _NoNetworkCrossref:
    """Crossref stub: everything is unverifiable offline."""

    def by_doi(self, doi):
        return None

    def by_title(self, title, rows=5):
        return []


@pytest.fixture()
def scored_run(conn):
    from authorai.embeddings import FakeEmbedder

    run_id = dbmod.create_run(conn)
    source = dbmod.add_document(
        conn,
        run_id,
        "SOURCE",
        title="Source A",
        metadata=json.dumps(
            {"sections": [{"title": "T", "page": 1, "text": "Published 2025 by FAO."}]}
        ),
    )
    report = dbmod.add_document(
        conn,
        run_id,
        "REPORT",
        metadata=json.dumps(
            {"sections": [{"title": "Intro", "page": 1, "text": "Hunger rose in 2023."}]}
        ),
    )
    embedder = FakeEmbedder(dim=DIM)
    [chunk_id] = dbmod.add_chunks(
        conn, run_id, source, [{"text": "evidence text"}], embedder.embed(["evidence text"])
    )
    [claim_id] = dbmod.add_claims(conn, run_id, report, [{"text": "Hunger rose in 2023."}])
    dbmod.add_verdicts(
        conn,
        run_id,
        [
            {
                "claim_id": claim_id,
                "verdict": "SUPPORTED",
                "raw_verdict": "SUPPORTED",
                "quote": "evidence text",
                "quote_verified": 1,
                "quoted_chunk_id": chunk_id,
                "evidence_chunk_ids": [chunk_id],
                "rationale": "r",
                "model": "m",
            }
        ],
    )
    return {"run": run_id, "source": source}


def test_score_run_persists_all_three_scores(conn, scored_run):
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(
                title="Source A", publisher="FAO", publication_date="2025"
            ),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    result = score_run(conn, llm, scored_run["run"], settings, crossref=_NoNetworkCrossref())

    assert result["accuracy"]["accuracy"] == 1.0
    assert result["credibility"]["method"] == "usage_weighted_mean"
    [source_row] = dbmod.list_source_credibility(conn, scored_run["run"])
    assert source_row["tier"] == "METADATA_ONLY"
    assert source_row["components"]["authority"] == 30.0  # FAO, word-boundary tier1

    persisted = dbmod.get_run_scores(conn, scored_run["run"])
    assert persisted["accuracy"]["supported"] == 1
    assert persisted["validity"]["score"] == result["validity"]["score"]

    # Re-scoring replaces, never duplicates.
    rescore_llm = FakeLLM(parse_results=llm._parse_results)
    score_run(conn, rescore_llm, scored_run["run"], settings, crossref=_NoNetworkCrossref())
    assert len(dbmod.list_source_credibility(conn, scored_run["run"])) == 1


def test_score_run_requires_verdicts(conn):
    run_id = dbmod.create_run(conn)
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    with pytest.raises(ValueError, match="no verdicts"):
        score_run(conn, FakeLLM(), run_id, settings, crossref=_NoNetworkCrossref())
