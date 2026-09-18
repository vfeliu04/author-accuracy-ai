"""Credibility tests: Crossref tiers (respx-recorded), publisher matching, aggregation."""

import httpx
import pytest
import respx

from authorai.credibility import (
    CROSSREF_BASE,
    CrossrefClient,
    SourceMetadata,
    _publisher_authority,
    _same_site,
    _title_match,
    aggregate_credibility,
    authority_needles,
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
    # The record is the document's OWN: its title corroborates the DOI.
    respx.get(f"{CROSSREF_BASE}/works/10.1000/xyz").mock(
        return_value=httpx.Response(
            200, json={"message": {"title": [META.title], "publisher": "X"}}
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
    # Title matching the record's, so the DOI corroborates and the test measures
    # the retry alone.
    metadata = SourceMetadata(doi="10.1000/busy", title="Anything")
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
def test_malformed_200_body_raises_instead_of_reading_as_not_found():
    """A 200 whose body is not a JSON object (proxy/CDN interference) is a
    malfunction, not an answer — reading it as 'not found' would silently
    downgrade tiers, the exact failure the retry policy exists to refuse."""
    respx.get(f"{CROSSREF_BASE}/works/10.1000/xyz").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(RuntimeError, match="non-object body"):
        _client().by_doi("10.1000/xyz")


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
    # Title matching the record's, so the DOI corroborates and the test measures
    # the cleaning alone.
    metadata = SourceMetadata(doi="https://doi.org/10.1000/xyz", title="Anything")
    tier, _ = resolve_tier(metadata, _client())
    assert tier == "VERIFIED_DOI"
    assert route.call_count == 1


PAGE_META = SourceMetadata(
    title="Ten Facts About Drinking Water",
    authors=["Anon Blogger"],
    publisher="Daily Water Blog",
    publication_date="2026-01",
    doi="10.1038/famous",
)
# The record of a well-known paper the page does not describe.
STRANGER_RECORD = {
    "title": ["Can Quantum-Mechanical Description of Reality Be Considered Complete?"],
    "publisher": "American Physical Society",
    "author": [{"family": "Einstein"}, {"family": "Podolsky"}],
    "published": {"date-parts": [[1935]]},
}


def _crossref_doi(doi: str, record: dict) -> None:
    respx.get(f"{CROSSREF_BASE}/works/{doi}").mock(
        return_value=httpx.Response(200, json={"message": record})
    )


def _crossref_no_titles() -> None:
    respx.get(f"{CROSSREF_BASE}/works").mock(
        return_value=httpx.Response(200, json={"message": {"items": []}})
    )


@respx.mock
def test_a_doi_resolving_to_another_work_is_not_verified(credibility_log):
    """A web page's DOI is read from its own citation_doi tag — invisible text
    the page owner writes. A resolved record that shares no field with the
    document is not the document's own record: it must earn no tier, and the
    record must not come back from resolve_tier, or merge_record would import
    the stranger's publisher, date and title onto the page."""
    _crossref_doi("10.1038/famous", STRANGER_RECORD)
    _crossref_no_titles()
    tier, record = resolve_tier(PAGE_META, _client())
    assert (tier, record) == ("METADATA_ONLY", None)
    merged = merge_record(PAGE_META, record)
    assert (merged.publisher, merged.publication_date) == ("Daily Water Blog", "2026-01")
    assert "10.1038/famous" in credibility_log.text
    assert "corroborat" in credibility_log.text


@respx.mock
@pytest.mark.parametrize(
    "corroborating",
    [
        {"title": ["Ten Facts About <i>Drinking</i> Water"]},
        {"publisher": "Daily Water Blog"},
        {"author": [{"family": "Blogger"}, {"family": "Someone"}]},
    ],
    ids=["title", "publisher", "author-family-name"],
)
def test_a_corroborated_doi_still_earns_the_top_tier(corroborating):
    """One agreeing field of our own is enough — a scholarly page or PDF whose
    own title, publisher or author matches the record it resolves to keeps
    VERIFIED_DOI, and keeps the record for gap-merging."""
    _crossref_doi("10.1038/famous", {**STRANGER_RECORD, **corroborating})
    tier, record = resolve_tier(PAGE_META, _client())
    assert tier == "VERIFIED_DOI"
    assert record is not None


@respx.mock
def test_a_document_stating_nothing_but_a_doi_cannot_corroborate_it(credibility_log):
    """Nothing of ours can agree with the record, so the DOI stays unverified.
    Degenerate in this app (a page's declaration reaches scoring only with a
    structured field, and its <title> is all but always present; the model
    extraction all but always returns a title), while accepting it would leave
    a page one invisible tag away from the top tier."""
    _crossref_doi("10.1000/bare", {"title": ["Anything"], "publisher": "Elsevier"})
    tier, record = resolve_tier(SourceMetadata(doi="10.1000/bare", title=None), _client())
    assert (tier, record) == ("METADATA_ONLY", None)
    assert "10.1000/bare" in credibility_log.text


@respx.mock
def test_an_uncorroborated_doi_falls_through_to_the_title_path():
    """Falling through, not returning: a document whose own title is registered
    still reaches VERIFIED_TITLE after its DOI is rejected."""
    _crossref_doi("10.1038/famous", STRANGER_RECORD)
    respx.get(f"{CROSSREF_BASE}/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "title": ["Ten Facts About Drinking Water"],
                            "published": {"date-parts": [[2026]]},
                        }
                    ]
                }
            },
        )
    )
    tier, record = resolve_tier(PAGE_META, _client())
    assert tier == "VERIFIED_TITLE"
    assert record["title"] == ["Ten Facts About Drinking Water"]


# --- the DOI a PAGE declares: the record must point back at the page ---------

DOI = "10.1016/j.heliyon.2024.e34730"
# A page that declares a paper's DOI AND, in the same invisible tags, that
# paper's title, authors and publisher: corroboration alone cannot tell it
# from the paper's own landing page.
PAPER_META = SourceMetadata(
    title="Rainfall variability in the Ebro basin",
    authors=["Lucía Fernández Ibáñez"],
    publisher="Heliyon",
    publication_date="2024",
    doi=DOI,
)
PAPER_RECORD = {
    "title": ["Rainfall variability in the Ebro basin"],
    "publisher": "Elsevier BV",
    "author": [{"family": "Fernández Ibáñez"}],
    "published": {"date-parts": [[2024]]},
    "resource": {"primary": {"URL": "https://www.sciencedirect.com/science/article/pii/S3407"}},
}
PAPER_PAGE = "https://www.sciencedirect.com/science/article/pii/S3407"


@respx.mock
@pytest.mark.parametrize(
    "page_url",
    [
        PAPER_PAGE,
        "https://sciencedirect.com/science/article/pii/S3407",  # no www.
        "https://secure.sciencedirect.com/science/article/pii/S3407",  # another subdomain
    ],
    ids=["same-host", "without-www", "subdomain"],
)
def test_a_scholarly_page_the_record_points_at_keeps_the_top_tier(page_url):
    """The paper's own landing page: the record's resource.primary.URL shares
    its registrable domain, so the declared DOI still earns VERIFIED_DOI."""
    _crossref_doi(DOI, PAPER_RECORD)
    tier, record = resolve_tier(PAPER_META, _client(), page_url=page_url)
    assert tier == "VERIFIED_DOI"
    assert record is not None


@respx.mock
def test_the_same_page_on_an_unrelated_domain_is_not_verified(credibility_log):
    """The hole corroboration alone left: a page's title, authors and publisher
    come from its own invisible tags, exactly like its DOI, so declaring the
    paper's title beside the paper's DOI passes corroboration. The record must
    also point at the page — and it must not come back, or merge_record would
    import the paper's publisher and date onto the impostor."""
    _crossref_doi(DOI, PAPER_RECORD)
    _crossref_no_titles()
    tier, record = resolve_tier(
        PAPER_META, _client(), page_url="https://water-truths.example/ten-facts"
    )
    assert (tier, record) == ("METADATA_ONLY", None)
    assert merge_record(PAPER_META, record).publisher == "Heliyon"
    warnings = [r for r in credibility_log.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert DOI in warnings[0].getMessage()
    assert "water-truths.example" in warnings[0].getMessage()


@respx.mock
@pytest.mark.parametrize(
    "page_url",
    [
        f"https://doi.org/{DOI}",
        f"https://onlinelibrary.example/doi/full/{DOI}",
        f"https://journals.example/article?id={DOI.replace('/', '%2F')}",
    ],
    ids=["doi-resolver", "doi-in-path", "doi-percent-encoded"],
)
def test_a_page_whose_address_carries_the_doi_keeps_the_top_tier(page_url):
    """The other half of the gate: many publishers put the DOI in the article's
    address, and the record's landing page is often on a sibling corporate
    domain (linkinghub.elsevier.com for a sciencedirect.com article)."""
    _crossref_doi(DOI, {**PAPER_RECORD, "resource": {"primary": {"URL": "https://elsewhere.test"}}})
    tier, _ = resolve_tier(PAPER_META, _client(), page_url=page_url)
    assert tier == "VERIFIED_DOI"


@respx.mock
def test_a_record_without_a_primary_resource_falls_back_to_its_url():
    """Crossref's older records carry only the top-level URL."""
    record = {k: v for k, v in PAPER_RECORD.items() if k != "resource"}
    _crossref_doi(DOI, {**record, "URL": f"https://doi.org/{DOI}"})
    _crossref_no_titles()
    assert resolve_tier(PAPER_META, _client(), page_url=PAPER_PAGE)[0] == "METADATA_ONLY"
    assert (
        resolve_tier(PAPER_META, _client(), page_url="https://doi.org/whatever")[0]
        == "VERIFIED_DOI"
    )


@pytest.mark.parametrize(
    ("landing", "page", "shared"),
    [
        ("https://www.nature.com/articles/x", "https://nature.com/articles/x", True),
        ("https://nature.com/articles/x", "https://blogs.nature.com/post", True),
        ("https://NATURE.com:443/articles/x", "https://www.nature.com/p", True),
        # The two shapes a look-alike host takes.
        ("https://www.nature.com/articles/x", "https://nature.com.attacker.example/p", False),
        ("https://www.nature.com/articles/x", "https://attacker-nature.com/p", False),
        ("https://www.nature.com/articles/x", "https://sciencedirect.com/p", False),
        ("not a url", "https://nature.com/p", False),
    ],
)
def test_same_site_compares_registrable_domains(landing, page, shared):
    """The security predicate itself: a suffix must fall on a label boundary,
    or every look-alike host would inherit the real site's records."""
    assert _same_site(landing, page) is shared


@respx.mock
def test_a_pdf_with_the_same_metadata_is_unaffected(credibility_log):
    """A PDF has no address to compare, and its metadata is read from its own
    pages, not from tags only a machine sees: the gate must not touch it."""
    _crossref_doi(DOI, PAPER_RECORD)
    tier, record = resolve_tier(PAPER_META, _client())
    assert tier == "VERIFIED_DOI"
    assert record is not None
    assert not [r for r in credibility_log.records if r.levelname == "WARNING"]


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
    assert result["excluded"] == []


def test_aggregation_carries_excluded_sources_with_their_usage():
    per_source = [{"doc_id": "a", "total": 80.0, "tier": "METADATA_ONLY"}]
    result = aggregate_credibility(
        per_source, {"a": 2, "img": 3}, excluded=[{"doc_id": "img", "reason": "image"}]
    )
    assert result["score"] == 80.0
    assert result["method"] == "usage_weighted_mean"
    # Usage of an excluded source is reported, never folded into the mean.
    assert result["excluded"] == [{"doc_id": "img", "reason": "image", "usage": 3}]


def test_aggregation_with_only_excluded_sources_is_labeled_not_zeroed():
    result = aggregate_credibility([], {"img": 1}, excluded=[{"doc_id": "img", "reason": "image"}])
    assert result["score"] is None
    assert result["method"] == "no_scorable_sources"


def test_metadata_from_provenance_maps_declared_fields():
    from authorai.credibility import metadata_from_provenance

    meta = metadata_from_provenance(
        {
            "url": "https://example.org/a",
            "final_url": "https://example.org/a",
            "title": "T",
            "authors": ["Ana Pérez"],
            "publisher": "World Health Organization",
            "publication_date": "2026-03-01",
            "doi": "10.1000/xyz",
            "scholarly": True,
        }
    )
    assert meta == SourceMetadata(
        title="T",
        authors=["Ana Pérez"],
        publisher="World Health Organization",
        publication_date="2026-03-01",
        doi="10.1000/xyz",
    )


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {"url": "https://example.org/a", "title": "Only a title"},
        {"title": "T", "authors": [], "publisher": None, "publication_date": "", "doi": None},
    ],
)
def test_metadata_from_provenance_is_none_without_structured_fields(provenance):
    from authorai.credibility import metadata_from_provenance

    # A bare <title> is not bibliographic metadata: the caller falls back to
    # the model extraction instead of scoring a title-only source.
    assert metadata_from_provenance(provenance) is None


class _MatchingTitleCrossref:
    def __init__(self):
        self.title_calls = 0

    def by_doi(self, doi):
        return None

    def by_title(self, title, rows=5):
        self.title_calls += 1
        return [{"title": ["Global Hunger Index 2025"], "author": [{"family": "Author"}]}]


def test_resolve_tier_title_search_defaults_on_and_can_be_disabled():
    crossref = _MatchingTitleCrossref()
    assert resolve_tier(META, crossref)[0] == "VERIFIED_TITLE"
    assert resolve_tier(META, crossref, title_search=False)[0] == "METADATA_ONLY"
    assert crossref.title_calls == 1  # the disabled call never queried


# --- ISBN verification (Wave 2) --------------------------------------------

OPENLIBRARY_URL = "https://openlibrary.org/api/books"
GOOGLEBOOKS_URL = "https://www.googleapis.com/books/v1/volumes"


def test_clean_isbn_validates_checksums():
    from authorai.credibility import clean_isbn

    # Real ISBNs from live-run sources: the 2025 GHI (13) and IWMI RR19 (10).
    assert clean_isbn("978-1-9191958-0-3") == "9781919195803"
    assert clean_isbn("92-9090-354-6") == "9290903546"
    assert clean_isbn("ISBN 0-9752298-0-X") == "097522980X"  # X check digit
    # The printed prefix family — imprints write all of these "as printed".
    assert clean_isbn("ISBN: 92-9090-354-6") == "9290903546"
    assert clean_isbn("ISBN-13: 978-0-306-40615-7") == "9780306406157"
    assert clean_isbn("ISBN-10: 0-306-40615-2") == "0306406152"
    assert clean_isbn("978–0–306–40615–7") == "9780306406157"  # en dashes
    assert clean_isbn("978-1-9191958-0-4") is None  # checksum off by one
    assert clean_isbn("12345") is None
    assert clean_isbn("not-an-isbn") is None


def _isbn_client():
    from authorai.credibility import IsbnClient

    return IsbnClient(mailto="test@example.com")


@respx.mock
def test_isbn_open_library_hit_returns_crossref_shaped_record():
    respx.get(OPENLIBRARY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "ISBN:9781919195803": {
                    "title": "2025 Global Hunger Index",
                    "publishers": [{"name": "Welthungerhilfe"}],
                    "publish_date": "October 2025",
                }
            },
        )
    )
    record = _isbn_client().by_isbn("978-1-9191958-0-3")
    assert record["title"] == ["2025 Global Hunger Index"]
    assert record["publisher"] == "Welthungerhilfe"
    assert record["published"] == {"date-parts": [[2025]]}


@respx.mock
def _gb_item(title, publisher, date, isbn13):
    return {
        "volumeInfo": {
            "title": title,
            "publisher": publisher,
            "publishedDate": date,
            "industryIdentifiers": [{"type": "ISBN_13", "identifier": isbn13}],
        }
    }


@respx.mock
def test_isbn_falls_back_to_google_books_on_miss_and_outage(monkeypatch):
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    # Open Library is DOWN (retried, then treated as one provider's outage)...
    respx.get(OPENLIBRARY_URL).mock(return_value=httpx.Response(503))
    # ...Google Books answers WITH a matching identifier — the hit wins and
    # the OL outage never needs to be raised.
    respx.get(GOOGLEBOOKS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"items": [_gb_item("2025 Global Hunger Index", "WHH", "2025", "9781919195803")]},
        )
    )
    record = _isbn_client().by_isbn("9781919195803")
    assert record["title"] == ["2025 Global Hunger Index"]


@respx.mock
def test_isbn_google_books_fuzzy_results_without_the_identifier_are_rejected():
    """q=isbn: is a SEARCH — an unrelated book with a colliding generic title
    must not be adopted just for being items[0]."""
    respx.get(OPENLIBRARY_URL).mock(return_value=httpx.Response(200, json={}))
    respx.get(GOOGLEBOOKS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"items": [_gb_item("Annual Report 2023", "Someone", "2023", "9780306406157")]},
        )
    )
    assert _isbn_client().by_isbn("9781919195803") is None


@respx.mock
def test_isbn_ten_thirteen_equivalence_matches_the_same_book():
    respx.get(OPENLIBRARY_URL).mock(return_value=httpx.Response(200, json={}))
    # We ask with the ISBN-10; the registry lists the 978- ISBN-13 form.
    respx.get(GOOGLEBOOKS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"items": [_gb_item("Numerical Recipes", "CUP", "1986", "9780306406157")]},
        )
    )
    record = _isbn_client().by_isbn("0-306-40615-2")
    assert record["title"] == ["Numerical Recipes"]


@respx.mock
def test_isbn_outage_with_no_hit_raises_instead_of_downgrading(monkeypatch):
    """The reproducibility invariant: 'not found' is only an answer when every
    provider actually answered. One registry down + no hit elsewhere must be
    loud, or the same source scores differently across an outage."""
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    respx.get(OPENLIBRARY_URL).mock(return_value=httpx.Response(503))
    respx.get(GOOGLEBOOKS_URL).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
    with pytest.raises(RuntimeError, match="unreachable"):
        _isbn_client().by_isbn("9781919195803")


@respx.mock
def test_isbn_both_providers_down_raises_loudly(monkeypatch):
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    respx.get(OPENLIBRARY_URL).mock(return_value=httpx.Response(503))
    respx.get(GOOGLEBOOKS_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(RuntimeError, match="unreachable"):
        _isbn_client().by_isbn("9781919195803")


@respx.mock
def test_isbn_not_found_anywhere_is_an_answer():
    respx.get(OPENLIBRARY_URL).mock(return_value=httpx.Response(200, json={}))
    respx.get(GOOGLEBOOKS_URL).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
    assert _isbn_client().by_isbn("9781919195803") is None


@respx.mock
def test_resolve_tier_isbn_path_requires_corroboration():
    from authorai.credibility import resolve_tier

    # No DOI; Crossref title search finds nothing.
    respx.get(f"{CROSSREF_BASE}/works").mock(
        return_value=httpx.Response(200, json={"message": {"items": []}})
    )
    respx.get(OPENLIBRARY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "ISBN:9781919195803": {
                    "title": "A Completely Different Book",
                    "publishers": [{"name": "Someone Else Press"}],
                }
            },
        )
    )
    metadata = META.model_copy(update={"isbn": "978-1-9191958-0-3"})
    # Resolves, but neither title nor publisher corroborates -> unverified.
    tier, _ = resolve_tier(metadata, _client(), _isbn_client())
    assert tier == "METADATA_ONLY"

    # Same lookup with an agreeing publisher -> VERIFIED_ISBN, record merged.
    respx.get(OPENLIBRARY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "ISBN:9781919195803": {
                    "title": "A Completely Different Book",
                    "publishers": [{"name": "Welthungerhilfe e.V."}],
                    "publish_date": "2025",
                }
            },
        )
    )
    tier, record = resolve_tier(metadata, _client(), _isbn_client())
    assert tier == "VERIFIED_ISBN"
    assert record["publisher"] == "Welthungerhilfe e.V."


def test_verified_isbn_tier_points():
    from authorai.credibility import score_source

    scored = score_source(
        META.model_copy(update={"isbn": "978-1-9191958-0-3"}),
        "VERIFIED_ISBN",
        tier1_publishers=["Welthungerhilfe"],
        tier2_publishers=[],
        current_year=2026,
    )
    assert scored["components"]["verification"] == 12.0


def test_default_authority_tiers_cover_live_run_publishers():
    """The orgs added 2026-08-21 (observed as publishers in live runs) must
    match at their intended tiers via the DEFAULT config, in both acronym and
    spelled-out forms — and 'UN' must still never match 'University'."""
    from authorai.config import Settings
    from authorai.credibility import _publisher_authority

    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    tier1 = authority_needles(settings.authority_tier1)
    tier2 = authority_needles(settings.authority_tier2)

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


def test_acronym_needles_that_are_ordinary_words_match_only_in_capitals():
    """'WHO' and 'UN' are tier-1 acronyms but also the word 'who' and the
    Spanish/French/Italian article 'un'. A web page's publisher is the site's
    own declared name, so a case-folded match gave 'Who What Wear' tier-1
    authority. An all-caps needle is an acronym and matches only in capitals;
    spelled-out needles stay case-insensitive."""
    from authorai.config import Settings

    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    tier1 = authority_needles(settings.authority_tier1)
    tier2 = authority_needles(settings.authority_tier2)

    for publisher in (
        "Who What Wear",
        "Marquis Who's Who",
        "Doctor Who News",
        "Diario de un Hidrólogo",
        "Un Mundo Sostenible",
        "Pourquoi un tel choix ?",
    ):
        assert _publisher_authority(publisher, tier1, tier2) == 15.0, publisher

    for publisher in (
        "WHO",
        "UN News",
        "U.N. Development Programme",
        "World Health Organization: WHO",
        "WHO/UNICEF Joint Monitoring Programme",
        "world health organization",  # a spelled-out needle ignores case
    ):
        assert _publisher_authority(publisher, tier1, tier2) == 30.0, publisher

    # Another spelling of an acronym is accepted only as its own configured needle.
    assert _publisher_authority("Unicef", ["UNICEF"], []) == 15.0
    assert _publisher_authority("Unicef", ["UNICEF", "Unicef"], []) == 30.0


def test_default_authority_tiers_accept_the_mixed_case_spellings_sites_use():
    """Sites styling an acronym as a word ('Unicef', 'Bbc') lost authority once
    acronym needles matched only in capitals, so the defaults carry those
    spellings as needles of their own. Acronyms that are also ordinary words
    ('Who', 'Un') are never added, and the eval sources' publishers keep their
    scores."""
    from authorai.config import Settings

    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    tier1 = authority_needles(settings.authority_tier1)
    tier2 = authority_needles(settings.authority_tier2)

    for publisher in ("Unicef", "Fao", "Oecd", "Unicef Office of Research", "Oecd Publishing"):
        assert _publisher_authority(publisher, tier1, tier2) == 30.0, publisher
    for publisher in ("Bbc", "Bbc News"):
        assert _publisher_authority(publisher, tier1, tier2) == 22.5, publisher
    for publisher in ("who cares blog", "un mundo"):
        assert _publisher_authority(publisher, tier1, tier2) == 15.0, publisher
    # 'who' and 'un' are only ever acronym (all-caps) needles, never case-folded ones.
    assert not {"who", "un"} & {needle.lower() for needle in tier1 + tier2 if not needle.isupper()}

    # The eval and end-to-end source sets' publishers score as before.
    for publisher, score in (
        ("Welthungerhilfe", 30.0),  # the Global Hunger Index synopsis
        ("Elsevier", 22.5),  # the Heliyon review
        ("Food and Agriculture Organization of the United Nations", 30.0),
        ("UNCCD", 30.0),
        ("World Meteorological Organization", 30.0),
        ("International Water Management Institute", 22.5),
        ("NDMC", 22.5),
    ):
        assert _publisher_authority(publisher, tier1, tier2) == score, publisher


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
