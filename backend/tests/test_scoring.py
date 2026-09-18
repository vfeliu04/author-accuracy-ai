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
from authorai.verification import verdict_stamp
from tests.conftest import DIM, FakeLLM


def _verdict(kind, stance="asserted"):
    return {"verdict": kind, "stance": stance}


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


def test_accuracy_is_report_position_agreement():
    # The full stance × verdict matrix: agreement (asserted+SUPPORTED,
    # disavowed+CONTRADICTED) is correct; the inverses are incorrect;
    # UNVERIFIABLE is neither, whatever the stance.
    rows = (
        [_verdict("SUPPORTED")] * 3  # asserted, supported → correct
        + [_verdict("CONTRADICTED", "disavowed")] * 2  # report called it false → correct
        + [_verdict("CONTRADICTED")] * 1  # asserted falsehood → incorrect
        + [_verdict("SUPPORTED", "disavowed")] * 1  # report wrongly disavowed → incorrect
        + [_verdict("UNVERIFIABLE")] * 1
        + [_verdict("UNVERIFIABLE", "disavowed")] * 1
    )
    scores = accuracy_scores(rows)
    assert scores["correct"] == 5
    assert scores["incorrect"] == 2
    assert scores["disavowed"] == 4
    assert scores["accuracy"] == round(5 / 7, 4)
    # Verdict counts stay raw — the stance never rewrites what the sources said.
    assert scores["supported"] == 4
    assert scores["contradicted"] == 3
    assert scores["coverage"] == round(7 / 9, 4)


def test_accuracy_defaults_missing_stance_to_asserted():
    # Rows from pre-stance code paths (or eval fixtures) carry no stance key;
    # they must score exactly as before the field existed.
    scores = accuracy_scores([{"verdict": "SUPPORTED"}, {"verdict": "CONTRADICTED"}])
    assert scores["correct"] == 1
    assert scores["incorrect"] == 1
    assert scores["disavowed"] == 0
    assert scores["accuracy"] == 0.5


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


def test_opening_text_includes_section_headings_and_respects_the_page_window(conn):
    """Real chunks carry sections and pages (the tested fixture previously had
    neither, leaving the heading block and page filter dead under test).
    Docling puts titles and DOI-bearing header lines in SECTION names — the
    exact fields metadata extraction exists to find."""
    from authorai import db as dbmod
    from authorai.embeddings import FakeEmbedder
    from authorai.scoring import _metadata_text

    run_id = dbmod.create_run(conn)
    doc = dbmod.add_document(conn, run_id, "SOURCE")
    embedder = FakeEmbedder(dim=8)
    chunks = [
        {"text": "First page body\nwith a line break.", "page": 1, "section": "The Real Title"},
        {"text": "Second page body.", "page": 2, "section": "The Real Title"},
        {"text": "Deep-page body that must NOT appear.", "page": 9, "section": "Conclusion"},
    ]
    dbmod.add_chunks(conn, run_id, doc, chunks, embedder.embed([c["text"] for c in chunks]))

    text = _metadata_text(conn, run_id, doc)
    assert "## The Real Title" in text  # heading included, once
    assert text.count("## The Real Title") == 1
    assert "First page body" in text and "Second page body." in text
    assert "must NOT appear" not in text  # page window is page <= 2


def test_metadata_text_includes_deep_imprint_passages(conn):
    """The observed live failure: the GHI's publisher/ISBN sit on page 61 —
    outside any opening-pages window. The marker scan must surface them, and
    multi-marker imprint chunks must outrank doi-only bibliography entries
    when the slots run out."""
    from authorai import db as dbmod
    from authorai.embeddings import FakeEmbedder
    from authorai.scoring import _IMPRINT_CHUNK_LIMIT, _metadata_text

    run_id = dbmod.create_run(conn)
    doc = dbmod.add_document(conn, run_id, "SOURCE")
    embedder = FakeEmbedder(dim=8)
    chunks = [
        {"text": "Opening page text.", "page": 1, "section": "Title"},
        # Enough doi-only bibliography chunks to fill every imprint slot...
        *[
            {"text": f"Reference {i}: Some Author (2020). doi.org/10.1234/x{i}", "page": 40 + i}
            for i in range(_IMPRINT_CHUNK_LIMIT + 2)
        ],
        # ...but the true imprint (many markers at once) must still win a slot.
        {
            "text": "Recommended citation: Welthungerhilfe (WHH). ISBN 978-1-9191958-0-3. "
            "© 2025. All rights reserved.",
            "page": 61,
        },
    ]
    dbmod.add_chunks(conn, run_id, doc, chunks, embedder.embed([c["text"] for c in chunks]))

    text = _metadata_text(conn, run_id, doc)
    assert "Opening page text." in text
    assert "LIKELY IMPRINT / CITATION PASSAGES" in text
    assert "Recommended citation: Welthungerhilfe" in text
    assert "ISBN 978-1-9191958-0-3" in text


def test_opening_text_is_loud_for_a_document_with_no_text(conn):
    from authorai import db as dbmod
    from authorai.scoring import _metadata_text

    run_id = dbmod.create_run(conn)
    doc = dbmod.add_document(conn, run_id, "SOURCE")
    with pytest.raises(ValueError, match="no text chunks"):
        _metadata_text(conn, run_id, doc)


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
                "prompt_hash": verdict_stamp(),
            }
        ],
    )
    return {"run": run_id, "source": source, "claim": claim_id}


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


def test_score_run_unknown_run_is_named_correctly(conn):
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    with pytest.raises(ValueError, match="Unknown run"):
        score_run(conn, FakeLLM(), "no-such-run", settings, crossref=_NoNetworkCrossref())


def test_score_run_refuses_stale_verdicts(conn, scored_run):
    """The persisted+served accuracy must never be computed from verdicts of a
    different judge prompt — unless explicitly allowed."""
    conn.execute(
        "UPDATE verdicts SET prompt_hash = 'OLD' WHERE claim_id = ?", (scored_run["claim"],)
    )
    conn.commit()
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    with pytest.raises(ValueError, match="different judge prompt"):
        score_run(conn, FakeLLM(), scored_run["run"], settings, crossref=_NoNetworkCrossref())
    # A failed guard persists nothing.
    assert dbmod.get_run_scores(conn, scored_run["run"]) is None

    # allow_stale scores them anyway.
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(title="Source A"),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    score_run(
        conn, llm, scored_run["run"], settings, crossref=_NoNetworkCrossref(), allow_stale=True
    )
    assert dbmod.get_run_scores(conn, scored_run["run"]) is not None


# --- source types: web provenance and image exclusion ----------------------


def _add_source(conn, run_id, *, source_type, title, metadata, text, chunk_kind="text", url=None):
    from authorai.embeddings import FakeEmbedder

    upload_id = dbmod.add_upload(
        conn, "SOURCE", title, f"/tmp/{title}", source_type=source_type, url=url
    )
    doc_id = dbmod.add_document(
        conn, run_id, "SOURCE", upload_id=upload_id, title=title, metadata=json.dumps(metadata)
    )
    [chunk_id] = dbmod.add_chunks(
        conn,
        run_id,
        doc_id,
        [{"text": text, "kind": chunk_kind}],
        FakeEmbedder(dim=DIM).embed([text]),
    )
    return doc_id, chunk_id


def _cite(conn, run_id, chunk_id, claim_text):
    report = dbmod.get_report_doc_id(conn, run_id)
    [claim_id] = dbmod.add_claims(conn, run_id, report, [{"text": claim_text}])
    dbmod.add_verdicts(
        conn,
        run_id,
        [
            {
                "claim_id": claim_id,
                "verdict": "SUPPORTED",
                "raw_verdict": "SUPPORTED",
                "quote": "cited text here",
                "quote_verified": 1,
                "quoted_chunk_id": chunk_id,
                "evidence_chunk_ids": [chunk_id],
                "rationale": "r",
                "model": "m",
                "prompt_hash": verdict_stamp(),
            }
        ],
    )


WHO_PROVENANCE = {
    "url": "https://www.who.int/news-room/fact-sheets/detail/drinking-water",
    "final_url": "https://www.who.int/news-room/fact-sheets/detail/drinking-water",
    "title": "Drinking-water",
    "authors": [],
    "publisher": "World Health Organization",
    "publication_date": "2026-03-01",
    "doi": None,
    "scholarly": False,
}


class _RecordingCrossref(_NoNetworkCrossref):
    def __init__(self, records: dict[str, list[dict]] | None = None):
        self.titles: list[str] = []
        self.records = records or {}

    def by_title(self, title, rows=5):
        self.titles.append(title)
        return self.records.get(title, [])


def _metadata_calls(llm):
    return [c for c in llm.parse_calls if c["output_type"] is SourceMetadata]


def test_web_source_is_scored_from_page_provenance_without_a_metadata_call(conn, scored_run):
    run_id = scored_run["run"]
    web_doc, web_chunk = _add_source(
        conn,
        run_id,
        source_type="web",
        title="Drinking-water",
        metadata={"sections": [], "provenance": WHO_PROVENANCE},
        text="Safely managed drinking water reached 73 percent.",
        url=WHO_PROVENANCE["url"],
    )
    _cite(conn, run_id, web_chunk, "Safely managed water reached 73 percent.")
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(
                title="Source A", publisher="FAO", publication_date="2025"
            ),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    source_a = {"title": ["Source A"], "published": {"date-parts": [[2025]]}}
    crossref = _RecordingCrossref(records={"Source A": [source_a]})
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    score_run(conn, llm, run_id, settings, crossref=crossref)

    # Only the PDF source costs a metadata extraction; the page declared its own.
    assert len(_metadata_calls(llm)) == 1
    rows = {r["doc_id"]: r for r in dbmod.list_source_credibility(conn, run_id)}
    web = rows[web_doc]
    assert web["metadata"]["publisher"] == "World Health Organization"
    assert web["components"]["authority"] == 30.0  # tier-1 publisher list, same as a PDF
    assert web["tier"] == "METADATA_ONLY"
    # A non-scholarly page never enters Crossref title search (headline false positives).
    assert "Drinking-water" not in crossref.titles
    # The PDF source (no upload row, so a PDF) still gets the title search, and a
    # corroborated match still verifies it.
    assert "Source A" in crossref.titles
    assert rows[scored_run["source"]]["tier"] == "VERIFIED_TITLE"


def test_scholarly_web_page_keeps_crossref_title_search(conn, scored_run):
    run_id = scored_run["run"]
    provenance = {**WHO_PROVENANCE, "title": "A Scholarly Landing Page", "scholarly": True}
    _add_source(
        conn,
        run_id,
        source_type="web",
        title="Landing",
        metadata={"sections": [], "provenance": provenance},
        text="Landing page abstract text.",
    )
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(title="Source A"),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    crossref = _RecordingCrossref()
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    score_run(conn, llm, run_id, settings, crossref=crossref)
    assert "A Scholarly Landing Page" in crossref.titles


PAPER_DOI = "10.1016/j.heliyon.2024.e34730"
PAPER_RECORD = {
    "title": ["Rainfall variability in the Ebro basin"],
    "publisher": "Elsevier BV",
    "resource": {"primary": {"URL": "https://www.sciencedirect.com/science/article/pii/S3407"}},
}


class _DoiCrossref(_NoNetworkCrossref):
    """Crossref stub resolving one DOI to the paper's record."""

    def by_doi(self, doi):
        return PAPER_RECORD if doi == PAPER_DOI else None


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("https://www.sciencedirect.com/science/article/pii/S3407", "VERIFIED_DOI"),
        ("https://water-truths.example/ten-facts", "METADATA_ONLY"),
    ],
    ids=["the-papers-own-page", "an-impostor-page"],
)
def test_a_pages_declared_doi_is_verified_only_where_the_record_points(
    conn, scored_run, page, expected
):
    """The page's address reaches resolve_tier from its provenance: two pages
    declaring the SAME DOI, title and publisher score differently, because only
    one of them is the page the Crossref record names."""
    run_id = scored_run["run"]
    provenance = {
        **WHO_PROVENANCE,
        "url": page,
        "final_url": page,
        "title": "Rainfall variability in the Ebro basin",
        "publisher": "Heliyon",
        "doi": PAPER_DOI,
    }
    web_doc, _ = _add_source(
        conn,
        run_id,
        source_type="web",
        title="Rainfall",
        metadata={"sections": [], "provenance": provenance},
        text="Rainfall in the basin fell by a fifth.",
        url=page,
    )
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(title="Source A"),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    score_run(conn, llm, run_id, settings, crossref=_DoiCrossref())
    rows = {r["doc_id"]: r for r in dbmod.list_source_credibility(conn, run_id)}
    assert rows[web_doc]["tier"] == expected
    # The rejected record is never merged: the page keeps its own publisher.
    assert rows[web_doc]["metadata"]["publisher"] == "Heliyon"


def test_web_page_declaring_only_a_title_falls_back_to_the_metadata_call(conn, scored_run):
    run_id = scored_run["run"]
    bare = {"url": "https://example.org/a", "final_url": "https://example.org/a", "title": "A"}
    _add_source(
        conn,
        run_id,
        source_type="web",
        title="A",
        metadata={"sections": [], "provenance": bare},
        text="Some page text about water.",
    )
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(title="Extracted", publisher="FAO"),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    score_run(conn, llm, run_id, settings, crossref=_NoNetworkCrossref())
    assert len(_metadata_calls(llm)) == 2  # the PDF source AND the title-only page


def test_image_source_is_excluded_from_credibility_but_its_usage_is_kept(conn, scored_run):
    run_id = scored_run["run"]
    image_doc, image_chunk = _add_source(
        conn,
        run_id,
        source_type="image",
        title="chart",
        metadata={"sections": []},
        text="chart\n\nA bar chart of water access by region.",
        chunk_kind="figure",
    )
    _cite(conn, run_id, image_chunk, "Water access varies by region.")
    llm = FakeLLM(
        parse_results={
            SourceMetadata: SourceMetadata(title="Source A", publisher="FAO"),
            ValidityAssessment: _assessment(quote="Hunger rose in 2023."),
        }
    )
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    result = score_run(conn, llm, run_id, settings, crossref=_NoNetworkCrossref())

    assert len(_metadata_calls(llm)) == 1  # the image never reaches metadata extraction
    assert [r["doc_id"] for r in dbmod.list_source_credibility(conn, run_id)] != []
    assert image_doc not in {r["doc_id"] for r in dbmod.list_source_credibility(conn, run_id)}
    assert result["credibility"]["excluded"] == [
        {"doc_id": image_doc, "reason": "image", "usage": 1}
    ]
    persisted = dbmod.get_run_scores(conn, run_id)
    assert persisted["credibility"]["excluded"] == result["credibility"]["excluded"]


def test_run_whose_only_source_is_an_image_is_labeled_not_zeroed(conn):
    run_id = dbmod.create_run(conn)
    dbmod.add_document(
        conn,
        run_id,
        "REPORT",
        metadata=json.dumps(
            {"sections": [{"title": "Intro", "page": 1, "text": "Water access varies."}]}
        ),
    )
    image_doc, image_chunk = _add_source(
        conn,
        run_id,
        source_type="image",
        title="chart",
        metadata={"sections": []},
        text="chart",
        chunk_kind="figure",
    )
    _cite(conn, run_id, image_chunk, "Water access varies.")
    llm = FakeLLM(parse_results={ValidityAssessment: _assessment(quote="Water access varies.")})
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    result = score_run(conn, llm, run_id, settings, crossref=_NoNetworkCrossref())
    assert result["credibility"]["score"] is None
    assert result["credibility"]["method"] == "no_scorable_sources"
    assert result["credibility"]["excluded"][0]["doc_id"] == image_doc
