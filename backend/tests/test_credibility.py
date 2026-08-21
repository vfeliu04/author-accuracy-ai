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
    evidence_usage,
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
                            # The title embeds its year, so the year cannot
                            # corroborate — the publisher does.
                            "publisher": "Welthungerhilfe e.V.",
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
def test_crossref_timeout_retries_then_raises_loudly(monkeypatch):
    """An unreachable Crossref is an outage, not 'not found' — silently
    downgrading tiers would make credibility scores non-reproducible."""
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    route = respx.get(f"{CROSSREF_BASE}/works/10.1000/slow")
    route.side_effect = httpx.ConnectTimeout("slow")
    metadata = SourceMetadata(doi="10.1000/slow", title=None)
    with pytest.raises(RuntimeError, match="no answer after 3 attempts"):
        resolve_tier(metadata, _client())
    assert route.call_count == 3  # initial + 2 retries


@respx.mock
def test_crossref_throttling_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    route = respx.get(f"{CROSSREF_BASE}/works/10.1000/busy")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"message": {"title": ["Anything"]}}),
    ]
    metadata = SourceMetadata(doi="10.1000/busy", title=None)
    tier, record = resolve_tier(metadata, _client())
    assert tier == "VERIFIED_DOI"
    assert route.call_count == 2


@respx.mock
def test_crossref_server_errors_raise_after_retries(monkeypatch):
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    route = respx.get(f"{CROSSREF_BASE}/works/10.1000/down")
    route.mock(return_value=httpx.Response(503))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        resolve_tier(SourceMetadata(doi="10.1000/down", title=None), _client())
    assert route.call_count == 3


@respx.mock
def test_malformed_doi_is_skipped_without_a_request():
    # No route mocked: any HTTP call would make respx raise. The URL-prefixed
    # and shapeless forms both fall through to METADATA_ONLY with a warning.
    for bad in ("https://doi.org/nope", "not-a-doi", "10.1/x"):
        tier, record = resolve_tier(SourceMetadata(doi=bad, title=None), _client())
        assert (tier, record) == ("METADATA_ONLY", None)


@respx.mock
def test_url_prefixed_doi_is_cleaned_before_lookup():
    route = respx.get(f"{CROSSREF_BASE}/works/10.1000/xyz").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Anything"]}})
    )
    metadata = SourceMetadata(doi="https://doi.org/10.1000/xyz", title=None)
    tier, _ = resolve_tier(metadata, _client())
    assert tier == "VERIFIED_DOI"
    assert route.call_count == 1


def test_year_bearing_title_cannot_corroborate_by_year_alone():
    """Finding: 'Annual Report 2023' from a DIFFERENT organization trivially
    agrees on the year — the corroboration must come from an independent field."""
    record = {
        "title": ["Global Hunger Index 2025"],
        "published": {"date-parts": [[2025]]},  # matches META's year exactly
    }
    assert not _title_match(META, record)  # no author, no publisher -> rejected
    assert _title_match(META, {**record, "publisher": "Welthungerhilfe e.V."})


def test_family_names_handle_inverted_and_suffixed_authors():
    def match(printed):
        metadata = META.model_copy(update={"authors": [printed], "publication_date": None})
        record = {"title": ["Global Hunger Index 2025"], "author": [{"family": "Smith"}]}
        return _title_match(metadata, record)

    assert match("John Smith")
    assert match("Smith, John")  # family name printed BEFORE the comma
    assert match("John Smith Jr.")  # suffix is not a family name
    assert match("John Smith, PhD")  # degree is not a family name
    assert not match("John Watson")


def test_title_normalization_handles_xml_entities_and_non_latin():
    metadata = META.model_copy(update={"title": "Health & Safety Report"})
    record = {"title": ["Health &amp; Safety Report"], "author": [{"family": "Author"}]}
    assert _title_match(metadata, record)
    # A non-Latin title must not normalize to '' (which would match nothing).
    chinese = META.model_copy(update={"title": "中国农业发展报告"})
    record = {"title": ["中国农业发展报告"], "author": [{"family": "Author"}]}
    assert _title_match(chinese, record)


def test_merge_record_fills_gaps_but_never_overrides():
    metadata = SourceMetadata(title="Our Title", publisher=None, publication_date=None)
    record = {"publisher": "Nature", "published": {"date-parts": [[2024]]}, "title": ["Theirs"]}
    merged = merge_record(metadata, record)
    assert merged.publisher == "Nature"
    assert merged.publication_date == "2024"
    assert merged.title == "Our Title"  # the document's own statement wins


# --- component scoring -----------------------------------------------------


def test_evidence_usage_counts_supported_and_contradicted_from_the_db(conn):
    """Executes the real SQL: a contradicting source is doing its job and must
    carry weight (v1 counted only SUPPORTED). Quote-less verdicts count nothing."""
    from authorai import db as dbmod
    from authorai.embeddings import FakeEmbedder

    run_id = dbmod.create_run(conn)
    report = dbmod.add_document(conn, run_id, "REPORT")
    source_a = dbmod.add_document(conn, run_id, "SOURCE")
    source_b = dbmod.add_document(conn, run_id, "SOURCE")
    embedder = FakeEmbedder(dim=8)
    [chunk_a] = dbmod.add_chunks(conn, run_id, source_a, [{"text": "aa"}], embedder.embed(["aa"]))
    [chunk_b] = dbmod.add_chunks(conn, run_id, source_b, [{"text": "bb"}], embedder.embed(["bb"]))
    claims = dbmod.add_claims(conn, run_id, report, [{"text": f"c{i}"} for i in range(3)])
    dbmod.add_verdicts(
        conn,
        run_id,
        [
            {
                "claim_id": claims[0],
                "verdict": "SUPPORTED",
                "raw_verdict": "SUPPORTED",
                "quoted_chunk_id": chunk_a,
                "rationale": "r",
                "model": "m",
            },
            {
                "claim_id": claims[1],
                "verdict": "CONTRADICTED",
                "raw_verdict": "CONTRADICTED",
                "quoted_chunk_id": chunk_b,
                "rationale": "r",
                "model": "m",
            },
            {
                "claim_id": claims[2],
                "verdict": "UNVERIFIABLE",
                "raw_verdict": "UNVERIFIABLE",
                "quoted_chunk_id": None,
                "rationale": "r",
                "model": "m",
            },
        ],
    )
    assert evidence_usage(conn, run_id) == {source_a: 1, source_b: 1}


def test_publisher_matching_is_word_boundary_not_substring():
    tier1 = ["UN", "FAO"]
    # THE v1 defect: "un" as a substring matched "University".
    assert _publisher_authority("Cambridge University Press", tier1, []) == 15.0
    assert _publisher_authority("UN World Food Programme", tier1, []) == 30.0
    assert _publisher_authority("FAO", tier1, []) == 30.0
    assert _publisher_authority(None, tier1, []) == 0.0  # no floor points


def test_publisher_matching_requires_adjacent_in_order_phrase():
    tier1 = ["World Bank", "UN"]
    # Both words present but scattered must NOT match the phrase.
    assert _publisher_authority("International Bank for Reconstruction, World", tier1, []) == 15.0
    assert _publisher_authority("The World Bank Group", tier1, []) == 30.0
    # Dotted initialisms are the same word: 'U.N.' == 'UN'.
    assert _publisher_authority("U.N. Development Programme", tier1, []) == 30.0


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


def test_default_authority_tiers_cover_live_run_publishers():
    """The orgs added 2026-08-21 (observed as publishers in live runs) must
    match at their intended tiers via the DEFAULT config, in both acronym and
    spelled-out forms — and 'UN' must still never match 'University'."""
    from authorai.config import Settings
    from authorai.credibility import _publisher_authority

    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    tier1 = [p.strip() for p in settings.authority_tier1.split(",") if p.strip()]
    tier2 = [p.strip() for p in settings.authority_tier2.split(",") if p.strip()]

    for publisher in (
        "WMO",
        "World Meteorological Organization",
        "UNCCD",
        "United Nations Convention to Combat Desertification",
        "Welthungerhilfe (WHH), Concern Worldwide, and IFHV",
        "World Health Organization",
    ):
        assert _publisher_authority(publisher, tier1, tier2) == 30.0, publisher

    for publisher in (
        "United States National Drought Mitigation Center",
        "NDMC",
        "International Water Management Institute",
        "IWMI",
        "WCRP's Climate and the Cryosphere Project",
    ):
        assert _publisher_authority(publisher, tier1, tier2) == 22.5, publisher

    assert _publisher_authority("Unseen University Press", tier1, tier2) == 15.0


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
