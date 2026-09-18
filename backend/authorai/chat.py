"""Grounded chat over a scored run.

The per-run context (every claim with its verdict, rationale, and evidence,
plus the source credibility and the three scores) is STATIC once a run is
DONE, so it is assembled once and sent as a cache_control'd system block: the
large prefix is cached and only the user's question varies per turn, and a
mode switch still reads the cached prefix (the breakpoint sits on the static
block, ahead of the small mode instruction). The model answers ONLY from this
context — the pipeline already did the judgment; chat surfaces it.
"""

import sqlite3

from authorai import db as dbmod
from authorai.config import Settings
from authorai.llm import LLM

CHAT_SYSTEM = """\
You help a user understand a completed fact-checking analysis of a report. A
pipeline extracted the report's claims, verified each against ingested source
documents, and scored the report. Everything you know about this run is in the
ANALYSIS block below. Answer ONLY from it — do not invent claims, verdicts, or
sources, and when the analysis does not cover something, say so plainly.
A source marked READ IN PART is a web page the pipeline read only the
beginning of — treat the rest of that page as never seen, and say so rather
than implying the whole page was checked.
Verdicts mean: SUPPORTED / CONTRADICTED / UNVERIFIABLE *relative to the
ingested sources only*. A claim marked "disavowed by the report" is one the
report ITSELF calls false — a CONTRADICTED verdict there means the report was
right to reject it, and accuracy counts stance-verdict agreement, so do not
present disavowed-CONTRADICTED claims as errors by the report."""

MODE_INSTRUCTIONS = {
    "evidence": (
        "MODE: evidence. Answer precisely from the verdicts and quoted evidence. "
        "Cite the specific claims and sources you rely on."
    ),
    "guidance": (
        "MODE: guidance. Help the author improve the report. Ground every "
        "suggestion in the contradicted or unverifiable claims and the weakest sources."
    ),
    "creative": (
        "MODE: creative. Brainstorm freely, but stay anchored to this report's "
        "topic and findings — do not contradict the analysis."
    ),
}

CHAT_MODES = tuple(MODE_INSTRUCTIONS)


def _fmt_score(scores: dict | None) -> str:
    if scores is None:
        return "The report has not been scored yet."
    acc = scores["accuracy"]
    cred = scores["credibility"]["score"]
    val = scores["validity"]["score"]
    # Pre-stance runs stored the old supported/decided number with no
    # correct/incorrect breakdown — label it as what it IS, or the model
    # would confidently misdescribe the metric on old runs.
    if acc.get("correct") is not None:
        label = (
            f"report-position agreement: {acc['correct']} agree with the report's stance, "
            f"{acc['incorrect']} do not"
        )
    else:
        label = "supported/decided; scored before stance-aware accuracy"
    return (
        f"accuracy {acc.get('accuracy')} ({label}), "
        f"coverage {acc.get('coverage')}, credibility {cred}/100, validity {val}/100. "
        f"{acc.get('supported')} supported, {acc.get('contradicted')} contradicted, "
        f"{acc.get('unverifiable')} unverifiable of {acc.get('total')} claims."
    )


def format_timestamp(seconds: float) -> str:
    """12:34 under an hour, 1:02:03 beyond: how video players label time."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _evidence_locator(row: dict) -> str:
    """Where in its source the quoted evidence sits, phrased by SOURCE TYPE and
    never inferred from which locator happens to be null: Docling leaves page
    unset for some PDF items, and that must not read as a web section."""
    source_type = row.get("evidence_source_type") or "pdf"
    if source_type == "image":
        return ", image"
    if source_type == "youtube":
        start = row.get("evidence_start_seconds")
        return f" at {format_timestamp(start)}" if start is not None else ""
    if source_type == "web":
        section = row.get("evidence_section")
        return f" § {section}" if section else ""
    page = row.get("evidence_page")
    return f" p.{page}" if page is not None else ""


def build_context(conn: sqlite3.Connection, run_id: str) -> str:
    """The static per-run analysis, rendered for the model. Reuses the same db
    reads the /report endpoint uses so the two cannot describe different runs."""
    verdicts = dbmod.list_verdicts_with_evidence(conn, run_id)
    scores = dbmod.get_run_scores(conn, run_id)
    sources = dbmod.list_run_sources(conn, run_id)

    lines = ["=== ANALYSIS ===", "", "SCORES: " + _fmt_score(scores), "", "CLAIMS:"]
    for row in verdicts:
        evidence = ""
        if row["quote"] and row["evidence_doc_title"]:
            locator = _evidence_locator(row)
            evidence = (
                f' — evidence: "{row["quote"]}" (source {row["evidence_doc_title"]!r}{locator})'
            )
        disavowed = " (disavowed by the report)" if row.get("stance") == "disavowed" else ""
        lines.append(
            f'- [{row["verdict"]}]{disavowed} "{row["text"]}" — {row["rationale"]}{evidence}'
        )
    lines += ["", "SOURCES:"]
    for source in sources:
        if source["source_type"] == "image":
            standing = "not scorable (image)"
        elif source["total"] is None:
            standing = "not scored"
        else:
            standing = f"tier {source['tier']}, credibility {source['total']}/100"
        lines.append(f"- {source['doc_title']!r}: {standing}{_partial_note(source)}")
    return "\n".join(lines)


def _partial_note(source: dict) -> str:
    """What the page cap left out of a web source, read from the same row the
    source list beside the chat marks 'Read in part'.

    Without it the model answers from the head of a page as though it had the
    page, while the panel next to it says the page was read in part. The
    analysis never saw the rest, so a question about the rest has no answer
    here, and saying so is the only honest one."""
    truncated = source.get("truncated")
    if not truncated:
        return ""
    kept, dropped = truncated["kept_chars"], truncated["dropped_chars"]
    return (
        f" — READ IN PART: only the first {kept:,} of {kept + dropped:,} characters of this "
        "page were analysed, so nothing later in it is covered"
    )


def answer(
    conn: sqlite3.Connection,
    llm: LLM,
    run_id: str,
    question: str,
    history: list[dict],
    mode: str,
    settings: Settings,
) -> str:
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"Unknown chat mode {mode!r}; expected one of {CHAT_MODES}")
    context = build_context(conn, run_id)
    system_blocks = [
        {
            "type": "text",
            "text": f"{CHAT_SYSTEM}\n\n{context}",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": MODE_INSTRUCTIONS[mode]},
    ]
    # Trim history to the most recent turns (guarding the -0 slice, which would
    # keep ALL history). The conversation must start with a user turn, so drop
    # any leading assistant turns left after trimming — the API 400s otherwise.
    n = settings.chat_history_turns
    turns = history[-n:] if n > 0 else []
    while turns and turns[0].get("role") != "user":
        turns = turns[1:]
    messages = [*turns, {"role": "user", "content": question}]
    return llm.chat(
        model=settings.chat_model,
        system_blocks=system_blocks,
        messages=messages,
        max_tokens=settings.chat_max_tokens,
    )
