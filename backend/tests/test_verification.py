"""Verdict logic tests: prompt building, quote checks, downgrade, orchestration."""

import pytest

from authorai import db as dbmod
from authorai.embeddings import FakeEmbedder
from authorai.search import Hit
from authorai.verification import (
    Verdict,
    build_evidence_query,
    build_verdict_prompt,
    check_verdict,
    verdict_stamp,
    verify_run,
)
from tests.conftest import DIM, FakeLLM

CLAIM = {"id": "claim-1", "text": "Hunger affected 735 million people in 2023.", "year": 2023}


def _hit(chunk_id, text, kind="text", figure_id=None):
    return Hit(
        chunk_id=chunk_id,
        doc_id="doc",
        page=1,
        section=None,
        kind=kind,
        text=text,
        score=1.0,
        channels=("vector",),
        figure_id=figure_id,
    )


HITS = [
    _hit(11, "In 2023, hunger affected 735 million people worldwide."),
    _hit(12, "Food insecurity is rising across many regions."),
]


# --- build_evidence_query -------------------------------------------------


def test_query_is_claim_text_plus_missing_year():
    assert build_evidence_query({"text": "Wheat exports fell.", "year": 2019}) == (
        "Wheat exports fell. 2019"
    )
    # Year already present -> not duplicated; value never appended.
    assert build_evidence_query(CLAIM) == CLAIM["text"]
    assert build_evidence_query({"text": "No year here.", "year": None}) == "No year here."


def test_prompt_numbers_evidence_and_carries_claim_details():
    prompt = build_verdict_prompt({**CLAIM, "value": 735e6, "unit": "people"}, HITS)
    assert "CLAIM: Hunger affected 735 million people in 2023." in prompt
    assert "value=735000000.0" in prompt
    assert "[1] (text p.1)" in prompt
    assert "[2] (text p.1)" in prompt
    assert HITS[0].text in prompt


# --- check_verdict --------------------------------------------------------


def _verdict(**overrides):
    fields = {
        "verdict": "SUPPORTED",
        "quote": "hunger affected 735 million people",
        "evidence_index": 1,
        "rationale": "Stated in the source.",
    }
    fields.update(overrides)
    return Verdict(**fields)


def test_verified_quote_passes_and_resolves_chunk():
    row = check_verdict(CLAIM, _verdict(), HITS)
    assert row["verdict"] == "SUPPORTED"
    assert row["quote_verified"] == 1
    assert row["quoted_chunk_id"] == 11
    assert row["evidence_chunk_ids"] == [11, 12]
    assert row["year_flag"] == 0  # 2023 appears in the quoted chunk


def test_quote_survives_pdf_typography():
    hits = [_hit(21, "Hunger  ‘remains’ severe —\naffecting  millions worldwide.")]
    verdict = _verdict(quote="Hunger 'remains' severe - affecting millions worldwide.")
    row = check_verdict({"id": "c", "year": None}, verdict, hits)
    assert row["quote_verified"] == 1


def test_unfindable_quote_downgrades_but_preserves_raw():
    verdict = _verdict(quote="a sentence that appears in no evidence chunk at all")
    row = check_verdict(CLAIM, verdict, HITS)
    assert row["verdict"] == "UNVERIFIABLE"
    assert row["raw_verdict"] == "SUPPORTED"
    assert row["quote_verified"] == 0
    assert row["quoted_chunk_id"] is None
    assert row["year_flag"] is None  # no verified quote -> year check n/a


def test_wrong_index_right_quote_verifies_with_corrected_chunk():
    verdict = _verdict(quote="food insecurity is rising", evidence_index=1)  # quote is in [2]
    row = check_verdict({"id": "c", "year": None}, verdict, HITS)
    assert row["quote_verified"] == 1
    assert row["quoted_chunk_id"] == 12


def test_contradicted_without_quote_downgrades():
    verdict = _verdict(verdict="CONTRADICTED", quote=None, evidence_index=None)
    row = check_verdict(CLAIM, verdict, HITS)
    assert row["verdict"] == "UNVERIFIABLE"
    assert row["raw_verdict"] == "CONTRADICTED"
    assert row["quote_verified"] == 0


def test_trivially_short_quote_is_rejected():
    verdict = _verdict(quote="2023")  # would substring-match nearly anything
    row = check_verdict(CLAIM, verdict, HITS)
    assert row["verdict"] == "UNVERIFIABLE"
    assert row["quote_verified"] == 0


def test_unverifiable_without_quote_is_not_a_failure():
    verdict = _verdict(verdict="UNVERIFIABLE", quote=None, evidence_index=None)
    row = check_verdict(CLAIM, verdict, HITS)
    assert row["verdict"] == "UNVERIFIABLE"
    assert row["quote_verified"] is None  # n/a, distinct from failed
    assert row["year_flag"] is None


def test_year_missing_from_cited_chunk_sets_flag():
    hits = [_hit(31, "Hunger affected 735 million people worldwide.")]  # no year anywhere
    row = check_verdict(CLAIM, _verdict(quote="hunger affected 735 million people"), hits)
    assert row["verdict"] == "SUPPORTED"
    assert row["year_flag"] == 1


# --- batch request shape --------------------------------------------------


def test_batch_request_uses_output_config_with_transformed_schema(tmp_path):
    from authorai.llm import ParseItem, build_batch_request

    image = tmp_path / "fig.png"
    image.write_bytes(b"\x89PNG fake")
    item = ParseItem(
        custom_id="claim-1", system="sys", prompt="judge this", output_type=Verdict, images=[image]
    )
    request = build_batch_request(item, model="claude-opus-5", max_tokens=123)

    assert request["custom_id"] == "claim-1"
    params = request["params"]
    assert params["model"] == "claude-opus-5"
    assert params["max_tokens"] == 123
    # The batch param is output_config — output_format is parse()-only sugar
    # and would be silently dropped here.
    fmt = params["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "verdict" in fmt["schema"]["properties"]
    # Images precede the text block, same as the sync path.
    content = params["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[-1] == {"type": "text", "text": "judge this"}


def _stub_batch_client(results):
    """A fake anthropic client whose batch lifecycle returns canned results."""
    from types import SimpleNamespace

    batches = SimpleNamespace(
        create=lambda requests: SimpleNamespace(id="batch-1"),
        retrieve=lambda batch_id: SimpleNamespace(id=batch_id, processing_status="ended"),
        results=lambda batch_id: iter(results),
    )
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _succeeded(custom_id, text):
    from types import SimpleNamespace

    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    return SimpleNamespace(
        custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message)
    )


def _errored(custom_id):
    from types import SimpleNamespace

    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


def _items():
    from authorai.llm import ParseItem

    return [
        ParseItem(custom_id="ok", system="s", prompt="p1", output_type=Verdict),
        ParseItem(custom_id="bad", system="s", prompt="p2", output_type=Verdict),
    ]


def _stub_batch_client_with_create(results, *, existing=None, request_count=2):
    """Like _stub_batch_client, plus a create() counter and a retrievable
    pre-existing batch for the resume path."""
    from types import SimpleNamespace

    counts = SimpleNamespace(
        processing=0, succeeded=request_count, errored=0, canceled=0, expired=0
    )
    created: list = []

    def create(requests):
        created.append(len(requests))
        return SimpleNamespace(id="batch-new")

    def retrieve(batch_id):
        if existing is not None and batch_id == existing:
            return SimpleNamespace(id=batch_id, processing_status="ended", request_counts=counts)
        if batch_id == "batch-new":
            return SimpleNamespace(id=batch_id, processing_status="ended", request_counts=counts)
        raise ValueError(f"no such batch {batch_id}")

    batches = SimpleNamespace(
        create=create, retrieve=retrieve, results=lambda batch_id: iter(results)
    )
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    return client, created


def test_batch_resume_polls_stored_batch_without_creating(monkeypatch):
    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    stub, created = _stub_batch_client_with_create(
        [
            _succeeded("ok", _verdict().model_dump_json()),
            _succeeded("bad", _verdict().model_dump_json()),
        ],
        existing="batch-old",
    )
    client._client = stub
    results = client.parse_batch(model="m", items=_items(), resume_batch_id="batch-old")
    assert set(results) == {"ok", "bad"}
    assert created == []  # the stored batch was reused — nothing was paid twice


def test_batch_resume_rejects_expired_batches(monkeypatch):
    """An ended batch whose requests expired must NOT resume: every item
    would fail and funnel through full-price sync retries (~2x the cost of a
    fresh half-price batch). Falling back to a new submission is cheaper."""
    from types import SimpleNamespace

    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    counts = SimpleNamespace(processing=0, succeeded=0, errored=0, canceled=0, expired=2)
    created: list[int] = []
    batches = SimpleNamespace(
        create=lambda requests: (created.append(len(requests)), SimpleNamespace(id="batch-new"))[1],
        retrieve=lambda batch_id: SimpleNamespace(
            id=batch_id,
            processing_status="ended",
            request_counts=counts
            if batch_id == "batch-dead"
            else SimpleNamespace(processing=0, succeeded=2, errored=0, canceled=0, expired=0),
        ),
        results=lambda batch_id: iter(
            [
                _succeeded("ok", _verdict().model_dump_json()),
                _succeeded("bad", _verdict().model_dump_json()),
            ]
        ),
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    results = client.parse_batch(model="m", items=_items(), resume_batch_id="batch-dead")
    assert set(results) == {"ok", "bad"}
    assert created == [2]  # the dead batch was NOT resumed — fresh submission


def test_batch_resume_falls_back_when_stored_batch_unusable(monkeypatch):
    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    stub, created = _stub_batch_client_with_create(
        [
            _succeeded("ok", _verdict().model_dump_json()),
            _succeeded("bad", _verdict().model_dump_json()),
        ],
        existing=None,  # the stored id is not retrievable
    )
    client._client = stub
    remembered: list[str] = []
    results = client.parse_batch(
        model="m",
        items=_items(),
        resume_batch_id="batch-gone",
        on_batch_created=remembered.append,
    )
    assert set(results) == {"ok", "bad"}
    assert created == [2]  # fell back to a fresh submission
    assert remembered == ["batch-new"]  # and persisted the new id


def test_batch_resume_refuses_foreign_custom_ids(monkeypatch):
    import pytest

    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    stub, _ = _stub_batch_client_with_create(
        [
            _succeeded("someone-elses-claim", _verdict().model_dump_json()),
            _succeeded("ok", _verdict().model_dump_json()),
        ],
        existing="batch-old",
    )
    client._client = stub
    with pytest.raises(RuntimeError, match="unknown custom_id"):
        client.parse_batch(model="m", items=_items(), resume_batch_id="batch-old")


def test_batch_failed_item_is_retried_sync_once(monkeypatch):
    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client._client = _stub_batch_client(
        [_succeeded("ok", _verdict().model_dump_json()), _errored("bad")]
    )
    retried = []

    def fake_parse(*, model, system, prompt, output_type, max_tokens, images=None, timeout=None):
        retried.append({"prompt": prompt, "max_tokens": max_tokens, "timeout": timeout})
        return _verdict(verdict="UNVERIFIABLE", quote=None, evidence_index=None)

    monkeypatch.setattr(client, "parse", fake_parse)
    results = client.parse_batch(model="m", items=_items(), max_tokens=32000)

    assert set(results) == {"ok", "bad"}
    assert results["bad"].verdict == "UNVERIFIABLE"
    [retry] = retried
    assert retry["prompt"] == "p2"
    # Full batch headroom — a capped retry would re-truncate the exact
    # thinking-heavy items the retry exists for. The explicit timeout lifts
    # the SDK's non-streaming guard.
    assert retry["max_tokens"] == 32000
    assert retry["timeout"] == 600.0


def test_batch_item_failing_twice_raises_listing_it(monkeypatch):
    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client._client = _stub_batch_client(
        [_succeeded("ok", _verdict().model_dump_json()), _errored("bad")]
    )

    def failing_parse(**kwargs):
        raise RuntimeError("still broken")

    monkeypatch.setattr(client, "parse", failing_parse)
    with pytest.raises(RuntimeError, match="bad: errored.*still broken"):
        client.parse_batch(model="m", items=_items())


# --- verify_run orchestration --------------------------------------------


@pytest.fixture()
def verified_run(conn):
    """A run with one SOURCE doc (two chunks) and one REPORT doc (two claims)."""
    run_id = dbmod.create_run(conn)
    source = dbmod.add_document(conn, run_id, "SOURCE")
    report = dbmod.add_document(conn, run_id, "REPORT")
    texts = [
        "In 2023, hunger affected 735 million people worldwide.",
        "Report chunk that must never be evidence: hunger affected 735 million.",
    ]
    embedder = FakeEmbedder(dim=DIM)
    dbmod.add_chunks(conn, run_id, source, [{"text": texts[0]}], embedder.embed([texts[0]]))
    dbmod.add_chunks(conn, run_id, report, [{"text": texts[1]}], embedder.embed([texts[1]]))
    claim_ids = dbmod.add_claims(
        conn,
        run_id,
        report,
        [
            # Explicit pages: list_claims orders by (page, id), and with NULL
            # pages the uuid ids would make claim order — and therefore the
            # pairing with the canned verdict list — nondeterministic.
            {"text": "Hunger affected 735 million people in 2023.", "year": 2023, "page": 1},
            {"text": "Food aid doubled last year.", "page": 2},
        ],
    )
    return {"run": run_id, "claims": claim_ids, "embedder": embedder}


def _canned_verdicts():
    return [
        _verdict(quote="hunger affected 735 million people worldwide"),
        _verdict(verdict="UNVERIFIABLE", quote=None, evidence_index=None),
    ]


def test_verify_run_stores_verdicts_and_summarizes(conn, verified_run):
    llm = FakeLLM(parse_results={Verdict: _canned_verdicts()})
    summary = verify_run(
        conn, verified_run["embedder"], llm, verified_run["run"], model="m", batch=False
    )
    assert summary["total"] == 2
    assert summary["counts"]["SUPPORTED"] == 1
    assert summary["counts"]["UNVERIFIABLE"] == 1
    assert summary["downgraded"] == 0

    verdicts = dbmod.list_verdicts(conn, verified_run["run"])
    assert len(verdicts) == 2
    supported = next(v for v in verdicts if v["verdict"] == "SUPPORTED")
    assert supported["quote_verified"] == 1
    assert supported["model"] == "m"
    assert supported["prompt_hash"] == verdict_stamp()


def test_verify_run_stamps_the_actual_k(conn, verified_run):
    # A -k override is a different judge configuration; the stamp must carry
    # the real k so the freshness guards flag those rows against the default.
    llm = FakeLLM(parse_results={Verdict: _canned_verdicts()})
    verify_run(
        conn, verified_run["embedder"], llm, verified_run["run"], model="m", batch=False, k=3
    )
    for row in dbmod.list_verdicts(conn, verified_run["run"]):
        assert row["prompt_hash"] == verdict_stamp(k=3)
        assert row["prompt_hash"] != verdict_stamp()


def test_verify_run_batch_matches_sync(conn, verified_run):
    for batch in (False, True):
        llm = FakeLLM(parse_results={Verdict: _canned_verdicts()})
        verify_run(conn, verified_run["embedder"], llm, verified_run["run"], model="m", batch=batch)
        verdicts = {v["text"]: v["verdict"] for v in dbmod.list_verdicts(conn, verified_run["run"])}
        assert verdicts == {
            "Hunger affected 735 million people in 2023.": "SUPPORTED",
            "Food aid doubled last year.": "UNVERIFIABLE",
        }


def test_verify_run_never_shows_report_chunks_to_the_judge(conn, verified_run):
    llm = FakeLLM(parse_results={Verdict: _canned_verdicts()})
    verify_run(conn, verified_run["embedder"], llm, verified_run["run"], model="m", batch=False)
    for call in llm.parse_calls:
        assert "must never be evidence" not in call["prompt"]


def test_verify_run_guards(conn):
    embedder = FakeEmbedder(dim=DIM)
    with pytest.raises(ValueError, match="Unknown run"):
        verify_run(conn, embedder, FakeLLM(), "no-such-run", model="m")

    run_id = dbmod.create_run(conn)
    with pytest.raises(ValueError, match="no claims"):
        verify_run(conn, embedder, FakeLLM(), run_id, model="m")

    report = dbmod.add_document(conn, run_id, "REPORT")
    dbmod.add_claims(conn, run_id, report, [{"text": "A claim."}])
    with pytest.raises(ValueError, match="no SOURCE"):
        verify_run(conn, embedder, FakeLLM(), run_id, model="m")

    # A SOURCE document with no chunks is still no evidence — the guard must
    # count source CONTENT, not source documents.
    dbmod.add_document(conn, run_id, "SOURCE")
    with pytest.raises(ValueError, match="no SOURCE content"):
        verify_run(conn, embedder, FakeLLM(), run_id, model="m")

    with pytest.raises(ValueError, match="k must be >= 1"):
        verify_run(conn, embedder, FakeLLM(), run_id, model="m", k=0)


def test_figure_evidence_attaches_image(conn, tmp_path):
    run_id = dbmod.create_run(conn)
    source = dbmod.add_document(conn, run_id, "SOURCE")
    report = dbmod.add_document(conn, run_id, "REPORT")
    image = tmp_path / "fig.png"
    image.write_bytes(b"\x89PNG fake")
    figure_id = dbmod.add_figure(conn, run_id, source, image_path=str(image), page=2)
    text = "Figure on page 2\n\nBar chart: hunger by region."
    embedder = FakeEmbedder(dim=DIM)
    dbmod.add_chunks(
        conn,
        run_id,
        source,
        [{"text": text, "kind": "figure", "figure_id": figure_id}],
        embedder.embed([text]),
    )
    dbmod.add_claims(conn, run_id, report, [{"text": "The chart shows hunger by region."}])

    llm = FakeLLM(parse_results={Verdict: _verdict(quote="bar chart: hunger by region")})
    verify_run(conn, embedder, llm, run_id, model="m", batch=False)
    assert llm.parse_calls[0]["images"] == [image]

    # A dangling image path must raise, not silently degrade to text-only.
    image.unlink()
    with pytest.raises(FileNotFoundError, match="refusing to judge"):
        verify_run(conn, embedder, llm, run_id, model="m", batch=False)
