import sqlite3

import pytest

from authorai import db as dbmod
from authorai.claims import ClaimExtraction, ExtractedClaim, extract_claims
from authorai.llm import AnthropicClient
from tests.conftest import FakeLLM

SECTIONS = [
    {"title": "Overview", "page": 1, "text": "Global hunger affected 735 million people in 2023."},
    {"title": "Methods", "page": 2, "text": "Data was collected from national surveys."},
]

CANNED = ClaimExtraction(
    claims=[
        ExtractedClaim(
            text="Global hunger affected 735 million people in 2023.",
            subject="global hunger",
            value=735_000_000,
            unit="people",
            year=2023,
            page=1,
        )
    ]
)


def test_extract_claims_builds_page_tagged_prompt_and_returns_claims():
    llm = FakeLLM(parse_results={ClaimExtraction: CANNED})
    claims = extract_claims(llm, SECTIONS, model="claude-haiku-4-5")

    assert len(claims) == 1
    assert claims[0].value == 735_000_000

    call = llm.parse_calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["output_type"] is ClaimExtraction
    assert "[page 1] ## Overview" in call["prompt"]
    assert "[page 2] ## Methods" in call["prompt"]
    assert "verbatim" in call["system"].lower()


def test_extract_claims_requires_sections():
    with pytest.raises(ValueError, match="No sections"):
        extract_claims(FakeLLM(), [], model="claude-haiku-4-5")


def test_tables_reach_the_extraction_prompt():
    # Reports put their most checkable figures in tables, and tables are stored
    # as chunks rather than metadata sections — if they do not reach the prompt,
    # every tabulated claim is silently unextractable.
    llm = FakeLLM(parse_results={ClaimExtraction: CANNED})
    tables = [{"page": 2, "text": "| Country | Fabricated |\n| Somalia | 98% decrease |"}]
    extract_claims(llm, SECTIONS, model="claude-haiku-4-5", tables=tables)

    prompt = llm.parse_calls[0]["prompt"]
    assert "[page 2] ## TABLE" in prompt
    assert "98% decrease" in prompt
    # Sections still come through alongside them.
    assert "[page 1] ## Overview" in prompt


def test_claims_persistence_roundtrip(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "REPORT")
    ids = dbmod.add_claims(conn, run_id, doc_id, [claim.model_dump() for claim in CANNED.claims])
    assert len(ids) == 1
    stored = dbmod.list_claims(conn, run_id)
    assert stored[0]["text"] == CANNED.claims[0].text
    assert stored[0]["value"] == 735_000_000
    assert stored[0]["year"] == 2023


def test_replace_swaps_claims_atomically(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "REPORT")
    dbmod.add_claims(conn, run_id, doc_id, [{"text": "stale claim"}])

    # A failed re-extraction must roll the DELETE back too, so the document is
    # never left with neither its old claims nor the new ones.
    with pytest.raises(sqlite3.IntegrityError):
        dbmod.add_claims(conn, run_id, doc_id, [{"text": None}], replace=True)
    assert [c["text"] for c in dbmod.list_claims(conn, run_id)] == ["stale claim"]

    dbmod.add_claims(conn, run_id, doc_id, [{"text": "fresh claim"}], replace=True)
    assert [c["text"] for c in dbmod.list_claims(conn, run_id)] == ["fresh claim"]


def test_anthropic_client_refuses_missing_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(api_key=None)


def test_the_client_sends_temperature_only_when_a_caller_sets_one():
    """Every existing caller is unchanged: with no temperature the SDK call
    carries no `temperature` key at all (Opus 4.7+ and Sonnet 5 refuse the
    parameter with a 400, its default value included), and a caller that
    sets one — the reference scan, on Haiku 4.5 — has it passed through as
    given, with nothing else about the call changed."""
    from types import SimpleNamespace

    sent: list[dict] = []

    def parse(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(
            parsed_output=CANNED,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    client = AnthropicClient(api_key="test-key")
    client._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    common = dict(model="m", system="s", prompt="p", output_type=ClaimExtraction)
    assert client.parse(**common) is CANNED
    assert "temperature" not in sent[0]
    assert client.parse(**common, temperature=0.0) is CANNED
    assert sent[1]["temperature"] == 0.0
    assert {key: value for key, value in sent[1].items() if key != "temperature"} == sent[0]
