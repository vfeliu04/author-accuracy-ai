"""Chat tests: context assembly, prompt-cache breakpoint, mode/guard behavior."""

import pytest

from authorai import chat as chatmod
from authorai import db as dbmod
from authorai.config import Settings
from authorai.embeddings import FakeEmbedder
from tests.conftest import DIM, FakeLLM

SETTINGS = Settings(anthropic_api_key="x", openai_api_key="x")


def _scored_run(conn) -> str:
    """A DONE run with one supported + one contradicted verdict and scores."""
    run_id = dbmod.create_run(conn)
    report = dbmod.add_document(conn, run_id, "REPORT")
    source = dbmod.add_document(conn, run_id, "SOURCE", title="World Hunger 2025")
    embedder = FakeEmbedder(dim=DIM)
    [chunk_id] = dbmod.add_chunks(
        conn, run_id, source, [{"text": "hunger fell", "page": 3}], embedder.embed(["hunger fell"])
    )
    [claim_a, claim_b] = dbmod.add_claims(
        conn,
        run_id,
        report,
        [{"text": "hunger fell in 2025", "page": 1}, {"text": "hunger doubled", "page": 2}],
    )
    dbmod.add_verdicts(
        conn,
        run_id,
        [
            {
                "claim_id": claim_a,
                "verdict": "SUPPORTED",
                "raw_verdict": "SUPPORTED",
                "quote": "hunger fell",
                "quote_verified": 1,
                "quoted_chunk_id": chunk_id,
                "rationale": "stated verbatim",
                "model": "m",
            },
            {
                "claim_id": claim_b,
                "verdict": "CONTRADICTED",
                "raw_verdict": "CONTRADICTED",
                "quote": None,
                "quote_verified": None,
                "quoted_chunk_id": None,
                "rationale": "the source shows a decrease",
                "model": "m",
            },
        ],
    )
    dbmod.save_run_scores(
        conn,
        run_id,
        accuracy={
            "supported": 1,
            "contradicted": 1,
            "unverifiable": 0,
            "total": 2,
            "accuracy": 0.5,
            "coverage": 1.0,
        },
        credibility={"score": 80.0, "method": "usage_weighted_mean"},
        validity={"score": 60.0, "components": {}},
    )
    dbmod.save_source_credibility(
        conn,
        run_id,
        [
            {
                "doc_id": source,
                "metadata": {"title": "World Hunger 2025"},
                "components": {},
                "total": 80.0,
                "tier": "VERIFIED_DOI",
            }
        ],
    )
    dbmod.set_run_status(conn, run_id, "DONE")
    return run_id


def test_context_contains_claims_verdicts_scores_and_sources(conn):
    run_id = _scored_run(conn)
    context = chatmod.build_context(conn, run_id)
    assert "hunger fell in 2025" in context
    assert "[SUPPORTED]" in context and "[CONTRADICTED]" in context
    assert "the source shows a decrease" in context  # rationale
    assert "World Hunger 2025" in context  # source title
    assert "VERIFIED_DOI" in context
    assert "credibility 80.0" in context  # aggregate score rendered


def test_answer_caches_the_static_block_and_keeps_mode_after_it(conn):
    run_id = _scored_run(conn)
    llm = FakeLLM(chat_answer="Two claims: one supported, one contradicted.")
    reply = chatmod.answer(
        conn,
        llm,
        run_id,
        "Which claims are contradicted?",
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        mode="evidence",
        settings=SETTINGS,
    )
    assert reply == "Two claims: one supported, one contradicted."
    [call] = llm.chat_calls
    system = call["system_blocks"]
    # Block 0 is the large static context and carries the cache breakpoint;
    # the mode instruction sits AFTER it (uncached, varies by mode).
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "ANALYSIS" in system[0]["text"]
    assert "cache_control" not in system[1]
    assert "MODE: evidence" in system[1]["text"]
    # History precedes the new question; the question is the last message.
    assert call["messages"][-1] == {"role": "user", "content": "Which claims are contradicted?"}
    assert call["model"] == SETTINGS.chat_model


def test_mode_switches_the_second_block_only(conn):
    run_id = _scored_run(conn)
    llm = FakeLLM()
    chatmod.answer(conn, llm, run_id, "q", [], "guidance", SETTINGS)
    [call] = llm.chat_calls
    assert "MODE: guidance" in call["system_blocks"][1]["text"]


def test_history_is_trimmed_to_the_configured_turns(conn):
    run_id = _scored_run(conn)
    settings = Settings(anthropic_api_key="x", openai_api_key="x", chat_history_turns=2)
    llm = FakeLLM()
    history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    chatmod.answer(conn, llm, run_id, "q", history, "evidence", settings)
    [call] = llm.chat_calls
    # 2 trimmed history turns + the new question.
    assert len(call["messages"]) == 3
    assert call["messages"][0]["content"] == "m8"


def test_history_turns_zero_sends_no_history_not_all(conn):
    """The -0 slice gotcha: history[-0:] would keep EVERYTHING; 0 means none."""
    run_id = _scored_run(conn)
    settings = Settings(anthropic_api_key="x", openai_api_key="x", chat_history_turns=0)
    llm = FakeLLM()
    history = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    chatmod.answer(conn, llm, run_id, "q", history, "evidence", settings)
    [call] = llm.chat_calls
    assert call["messages"] == [{"role": "user", "content": "q"}]


def test_leading_assistant_turns_are_dropped(conn):
    """The conversation must start with a user turn — an assistant-first
    history (after trimming) would make the API reject the request."""
    run_id = _scored_run(conn)
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    llm = FakeLLM()
    history = [
        {"role": "assistant", "content": "leading"},
        {"role": "user", "content": "real"},
    ]
    chatmod.answer(conn, llm, run_id, "q", history, "evidence", settings)
    [call] = llm.chat_calls
    assert call["messages"][0] == {"role": "user", "content": "real"}


def test_unknown_mode_is_rejected(conn):
    run_id = _scored_run(conn)
    with pytest.raises(ValueError, match="Unknown chat mode"):
        chatmod.answer(conn, FakeLLM(), run_id, "q", [], "nonsense", SETTINGS)


def test_context_reports_unscored_run(conn):
    run_id = dbmod.create_run(conn)
    dbmod.add_document(conn, run_id, "REPORT")
    context = chatmod.build_context(conn, run_id)
    assert "not been scored" in context
