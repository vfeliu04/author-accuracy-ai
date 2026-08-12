"""Credibility tests: Crossref tiers (respx-recorded), publisher matching, aggregation."""

import httpx
import pytest
import respx

from authorai.credibility import (
    CROSSREF_BASE,
    CrossrefClient,
    SourceMetadata,
    _publisher_authority,
    _title_match,
    aggregate_credibility,
    merge_record,
    resolve_tier,
    score_source,
)

META = SourceMetadata(
    title="Global Hunger Index 2025",
    authors=["Jane Q. Author"],
    publisher="Welthungerhilfe",
    publication_date="2025-10",
    doi=None,
)


def _client():
    return CrossrefClient(mailto="test@example.com")


# --- Crossref tiers --------------------------------------------------------


@respx.mock
def test_doi_resolution_wins_tier_verified_doi():
    respx.get(f"{CROSSREF_BASE}/works/10.1000/xyz").mock(
        return_value=httpx.Response(
            200, json={"message": {"title": ["Anything"], "publisher": "X"}}
        )
    )
    metadata = META.model_copy(update={"doi": "10.1000/xyz"})
    tier, record = resolve_tier(metadata, _client())
    assert tier == "VERIFIED_DOI"
    assert record["publisher"] == "X"


@respx.mock
def test_unresolvable_doi_falls_through_to_title_search():
    respx.get(f"{CROSSREF_BASE}/works/10.1000/nope").mock(return_value=httpx.Response(404))
    respx.get(f"{CROSSREF_BASE}/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "title": ["Global Hunger Index 2025"],
                            "published": {"date-parts": [[2025, 10]]},
                        }
                    ]
                }
            },
        )
    )
    metadata = META.model_copy(update={"doi": "10.1000/nope"})
    tier, _ = resolve_tier(metadata, _client())
    assert tier == "VERIFIED_TITLE"


@respx.mock
def test_title_match_requires_corroboration():
    # Exact title, but the year is far off and no author matches -> rejected.
    respx.get(f"{CROSSREF_BASE}/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "title": ["Global Hunger Index 2025"],
                            "published": {"date-parts": [[2011]]},
                            "author": [{"family": "Someone"}],
                        }
                    ]
                }
            },
        )
    )
    tier, _ = resolve_tier(META, _client())
    assert tier == "METADATA_ONLY"


def test_title_match_normalizes_markup_and_accepts_author_corroboration():
    record = {
        "title": ["<i>Global</i> Hunger  Index — 2025"],
        "author": [{"family": "Author"}],
    }
    assert _title_match(META, record)  # markup stripped, family name corroborates


@respx.mock
def test_crossref_timeout_retries_then_gives_up_loudly():
    route = respx.get(f"{CROSSREF_BASE}/works/10.1/slow")
    route.side_effect = httpx.ConnectTimeout("slow")
    metadata = SourceMetadata(doi="10.1/slow", title=None)
    tier, record = resolve_tier(metadata, _client())
    assert record is None
    assert tier == "METADATA_ONLY"  # doi present but unverifiable -> metadata only
    assert route.call_count == 3  # initial + 2 retries


def test_merge_record_fills_gaps_but_never_overrides():
    metadata = SourceMetadata(title="Our Title", publisher=None, publication_date=None)
    record = {"publisher": "Nature", "published": {"date-parts": [[2024]]}, "title": ["Theirs"]}
    merged = merge_record(metadata, record)
    assert merged.publisher == "Nature"
    assert merged.publication_date == "2024"
    assert merged.title == "Our Title"  # the document's own statement wins


# --- component scoring -----------------------------------------------------


def test_publisher_matching_is_word_boundary_not_substring():
    tier1 = ["UN", "FAO"]
    # THE v1 defect: "un" as a substring matched "University".
    assert _publisher_authority("Cambridge University Press", tier1, []) == 15.0
    assert _publisher_authority("UN World Food Programme", tier1, []) == 30.0
    assert _publisher_authority("FAO", tier1, []) == 30.0
    assert _publisher_authority(None, tier1, []) == 0.0  # no floor points


def test_score_source_no_floors_for_unknowns():
    empty = SourceMetadata()
    scored = score_source(
        empty, "NONE", tier1_publishers=[], tier2_publishers=[], current_year=2026
    )
    assert scored["total"] == 0.0  # v1 gave ~32/100 for a source it knew nothing about


def test_score_source_full_marks_shape():
    metadata = SourceMetadata(
        title="T", authors=["A"], publisher="FAO", publication_date="2025", doi="10.1/x"
    )
    scored = score_source(
        metadata, "VERIFIED_DOI", tier1_publishers=["FAO"], tier2_publishers=[], current_year=2026
    )
    assert scored["components"] == {
        "metadata_completeness": 30.0,
        "authority": 30.0,
        "recency": 20.0,
        "verification": 20.0,
    }
    assert scored["total"] == 100.0


# --- aggregation -------------------------------------------------------------


def test_aggregation_weights_by_usage_including_contradictions():
    per_source = [
        {"doc_id": "a", "total": 90.0, "tier": "VERIFIED_DOI"},
        {"doc_id": "b", "total": 30.0, "tier": "NONE"},
    ]
    # Source "a" cited by 3 verdicts (any verdict class counts), "b" by 1.
    result = aggregate_credibility(per_source, {"a": 3, "b": 1})
    assert result["method"] == "usage_weighted_mean"
    assert result["score"] == 75.0  # (90*3 + 30*1) / 4


def test_aggregation_zero_usage_is_labeled_unweighted_never_zero():
    per_source = [
        {"doc_id": "a", "total": 80.0, "tier": "METADATA_ONLY"},
        {"doc_id": "b", "total": 40.0, "tier": "NONE"},
    ]
    result = aggregate_credibility(per_source, {})
    assert result["method"] == "unweighted_mean_no_usage"
    assert result["score"] == 60.0


def test_aggregation_no_sources_is_explicit():
    result = aggregate_credibility([], {})
    assert result["score"] is None
    assert result["method"] == "no_sources"


def test_metadata_extraction_prompt_carries_opening_text():
    from authorai.credibility import extract_metadata
    from tests.conftest import FakeLLM

    llm = FakeLLM(parse_results={SourceMetadata: META})
    result = extract_metadata(llm, "model-m", "OPENING PAGES TEXT HERE")
    assert result.title == "Global Hunger Index 2025"
    assert "OPENING PAGES TEXT HERE" in llm.parse_calls[0]["prompt"]
    assert "never guess" in llm.parse_calls[0]["system"]


@pytest.mark.parametrize("mailto", [None, "dev@example.com"])
def test_client_constructs_with_and_without_mailto(mailto):
    CrossrefClient(mailto=mailto)  # missing mailto logs loudly but works
