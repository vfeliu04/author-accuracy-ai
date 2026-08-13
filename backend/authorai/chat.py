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
Verdicts mean: SUPPORTED / CONTRADICTED / UNVERIFIABLE *relative to the
ingested sources only*."""

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
    return (
        f"accuracy {acc.get('accuracy')} (supported/decided), "
        f"coverage {acc.get('coverage')}, credibility {cred}/100, validity {val}/100. "
        f"{acc.get('supported')} supported, {acc.get('contradicted')} contradicted, "
        f"{acc.get('unverifiable')} unverifiable of {acc.get('total')} claims."
    )


def build_context(conn: sqlite3.Connection, run_id: str) -> str:
    """The static per-run analysis, rendered for the model. Reuses the same db
    reads the /report endpoint uses so the two cannot describe different runs."""
    verdicts = dbmod.list_verdicts_with_evidence(conn, run_id)
    scores = dbmod.get_run_scores(conn, run_id)
    sources = dbmod.list_source_credibility(conn, run_id)

    lines = ["=== ANALYSIS ===", "", "SCORES: " + _fmt_score(scores), "", "CLAIMS:"]
    for row in verdicts:
        evidence = ""
        if row["quote"] and row["evidence_doc_title"]:
            page = f" p.{row['evidence_page']}" if row["evidence_page"] is not None else ""
            evidence = f' — evidence: "{row["quote"]}" (source {row["evidence_doc_title"]!r}{page})'
        lines.append(f'- [{row["verdict"]}] "{row["text"]}" — {row["rationale"]}{evidence}')
    lines += ["", "SOURCES:"]
    for source in sources:
        lines.append(
            f"- {source['doc_title']!r}: tier {source['tier']}, credibility {source['total']}/100"
        )
    return "\n".join(lines)


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
    # Trim history to whole turns; the current question is the only new input.
    turns = history[-settings.chat_history_turns :]
    messages = [*turns, {"role": "user", "content": question}]
    return llm.chat(
        model=settings.chat_model,
        system_blocks=system_blocks,
        messages=messages,
        max_tokens=settings.chat_max_tokens,
    )
