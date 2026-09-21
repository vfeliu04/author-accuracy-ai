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


def test_context_lists_image_and_unscored_sources_honestly(conn):
    run_id = _scored_run(conn)
    image = dbmod.add_upload(conn, "SOURCE", "chart.png", "/tmp/chart.png", source_type="image")
    dbmod.add_document(conn, run_id, "SOURCE", upload_id=image, title="Water chart")
    dbmod.add_document(conn, run_id, "SOURCE", title="Unscored Source")
    context = chatmod.build_context(conn, run_id)
    assert "- 'World Hunger 2025': tier VERIFIED_DOI, credibility 80.0/100" in context
    assert "- 'Water chart': not scorable (image)" in context
    assert "- 'Unscored Source': not scored" in context


def test_context_spells_out_the_tier_a_name_alone_would_overstate(conn):
    """The chat answers a user in prose, so a bare MATCHED_RECORD in its
    context is a tier name the model can only guess at — and the likeliest
    guess, 'a record matched, so it is verified', is the very claim this tier
    exists to deny. VERIFIED_DOI says what it means on its own; this one is
    given the clause that makes it true."""
    run_id = _scored_run(conn)
    dbmod.save_source_credibility(
        conn,
        run_id,
        [
            {
                "doc_id": conn.execute(
                    "SELECT id FROM documents WHERE run_id = ? AND kind = 'SOURCE'", (run_id,)
                ).fetchone()["id"],
                "metadata": {"title": "World Hunger 2025"},
                "components": {},
                "total": 75.0,
                "tier": "MATCHED_RECORD",
            }
        ],
    )
    context = chatmod.build_context(conn, run_id)
    assert (
        "- 'World Hunger 2025': tier MATCHED_RECORD (a registry record matches this source's "
        "details, but does not name the address it was fetched from, so it is NOT verified), "
        "credibility 75.0/100"
    ) in context


def test_context_says_which_pages_were_read_only_in_part(conn):
    """Chat reads the same source list the report's panel does, and that panel
    marks a capped page 'Read in part'. Told nothing, the model answers from the
    head of a page as though it had the whole page, next to a panel saying it
    did not — so the two numbers travel into the context too."""
    import json

    run_id = _scored_run(conn)
    web = dbmod.add_upload(
        conn, "SOURCE", "who.int", "/tmp/page.json", source_type="web", url="https://who.int/facts"
    )
    dbmod.add_document(
        conn,
        run_id,
        "SOURCE",
        upload_id=web,
        title="Drinking-water",
        metadata=json.dumps(
            {
                "sections": [],
                "provenance": {"truncated": {"kept_chars": 200_000, "dropped_chars": 51_234}},
            }
        ),
    )
    whole = dbmod.add_upload(
        conn, "SOURCE", "un.org", "/tmp/whole.json", source_type="web", url="https://un.org/facts"
    )
    dbmod.add_document(
        conn,
        run_id,
        "SOURCE",
        upload_id=whole,
        title="Water report",
        metadata=json.dumps({"sections": [], "provenance": {"url": "https://un.org/facts"}}),
    )
    context = chatmod.build_context(conn, run_id)
    assert (
        "- 'Drinking-water': not scored — READ IN PART: only the first 200,000 of 251,234 "
        "characters of this page were analysed, so nothing later in it is covered"
    ) in context
    # A page read whole says nothing new, and neither does a PDF.
    assert context.splitlines()[-1] == "- 'Water report': not scored"
    assert "READ IN PART" not in context.split("- 'Drinking-water'")[0]
    assert "READ IN PART" in chatmod.CHAT_SYSTEM  # the model is told what it means


@pytest.mark.parametrize(
    "truncated",
    [
        {"kept_chars": 200_000},
        {"kept_chars": "many", "dropped_chars": "some"},
        {},
        [200_000, 51_234],
    ],
    ids=["missing-key", "not-numbers", "empty-object", "not-an-object"],
)
def test_a_malformed_page_cap_record_degrades_instead_of_failing_the_chat(
    conn, chat_log, truncated
):
    """The two numbers come out of a stored provenance, and the note quoting
    them is a footnote on one line of the context. A shape the writer never
    produces — a hand-edited row, a future schema — must not take the whole chat
    endpoint down with a KeyError: the line degrades to the standing it always
    had, and the log says which source could not be described."""
    import json

    run_id = _scored_run(conn)
    upload = dbmod.add_upload(
        conn, "SOURCE", "who.int", "/tmp/page.json", source_type="web", url="https://who.int/facts"
    )
    dbmod.add_document(
        conn,
        run_id,
        "SOURCE",
        upload_id=upload,
        title="Drinking-water",
        metadata=json.dumps({"sections": [], "provenance": {"truncated": truncated}}),
    )
    context = chatmod.build_context(conn, run_id)
    assert context.splitlines()[-1] == "- 'Drinking-water': not scored"
    assert "READ IN PART" not in context
    assert "Drinking-water" in chat_log.text


def test_context_phrases_each_evidence_locator_by_source_type(conn):
    run_id = _scored_run(conn)
    report = dbmod.get_report_doc_id(conn, run_id)
    embedder = FakeEmbedder(dim=DIM)

    def cited(source_type, title, chunk):
        upload = dbmod.add_upload(conn, "SOURCE", title, f"/tmp/{title}", source_type=source_type)
        doc = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload, title=title)
        [chunk_id] = dbmod.add_chunks(conn, run_id, doc, [chunk], embedder.embed([chunk["text"]]))
        [claim] = dbmod.add_claims(conn, run_id, report, [{"text": f"claim about {title}"}])
        dbmod.add_verdicts(
            conn,
            run_id,
            [
                {
                    "claim_id": claim,
                    "verdict": "SUPPORTED",
                    "raw_verdict": "SUPPORTED",
                    "quote": chunk["text"],
                    "quote_verified": 1,
                    "quoted_chunk_id": chunk_id,
                    "rationale": "r",
                    "model": "m",
                }
            ],
        )

    cited("web", "Drinking-water", {"text": "73 percent", "section": "Access to services"})
    cited("youtube", "Water talk", {"text": "two billion", "start_seconds": 754.0})
    cited("youtube", "Long lecture", {"text": "an hour in", "start_seconds": 3723.0})
    cited("image", "Chart", {"text": "a bar chart", "kind": "figure"})
    # Docling leaves page unset for a PDF item without provenance; its heading
    # stays the section, and that must not make it read like a web page.
    cited("pdf", "No provenance", {"text": "unlocated finding", "section": "Methods"})
    context = chatmod.build_context(conn, run_id)
    assert "(source 'World Hunger 2025' p.3)" in context  # PDF phrasing unchanged
    assert "(source 'Drinking-water' § Access to services)" in context
    assert "(source 'Water talk' at 12:34)" in context
    assert "(source 'Long lecture' at 1:02:03)" in context
    assert "(source 'Chart', image)" in context
    assert "(source 'No provenance')" in context
