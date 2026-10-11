"""Grounded chat over a scored run.

The per-run analysis (every claim with its verdict, rationale, and evidence,
plus the source credibility and the three scores) is STATIC once a run is
DONE. It holds text the app did not write — claim texts, evidence quotes, the
judge's rationales, source titles, section headings — so it never sits in the
system prompt: it travels as ONE plain-text document block, first in the first
user turn, and the system prompt is the app's own words only. That block
carries the cache breakpoint: the system prompt and the analysis are the
cached prefix, and only the turns after it vary. (The mode instruction is part
of the system prompt, ahead of the document, so each mode has its own cache
entry.) The model answers ONLY from the analysis — the pipeline already did
the judgment; chat surfaces it.
"""

import json
import sqlite3

from authorai import db as dbmod
from authorai.config import Settings
from authorai.llm import LLM, TextDocument, render_user_content
from authorai.log import setup_logger
from authorai.video import channel_verified, clock

logger = setup_logger(__name__)

CHAT_SYSTEM = """\
You help a user understand a completed fact-checking analysis of a report. A
pipeline extracted the report's claims, verified each against ingested source
documents, and scored the report. Everything you know about this run is in the
document titled "Analysis" at the start of the conversation. Answer ONLY from
it — do not invent claims, verdicts, or sources, and when the analysis does not
cover something, say so plainly.
A source marked READ IN PART is a web page or a video the pipeline read only
the beginning of — treat the rest of it as never seen, and say so rather than
implying the whole source was checked.
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

ANALYSIS_TITLE = "Analysis"
# What the document is, in the app's words (the block's `context` field).
ANALYSIS_CONTEXT = (
    "The stored analysis of one fact-checking run: its scores, every claim with its verdict, "
    "rationale and quoted evidence, and its sources. Each piece of text in double quotes is a "
    "JSON string holding words taken from the report or from a source, or the judge's "
    "explanation of a verdict; everything outside double quotes was written by the pipeline."
)


def _quoted(value: str) -> str:
    """One piece of outside text as a JSON string on one line.

    The analysis is read line by line and its labels are the pipeline's
    record, so no outside text may be able to end its line, close its own
    quotes or write a label: a web page's heading could otherwise add a second
    verdict to a claim's line. JSON escapes the quote, the backslash and the
    control characters; every other character that does not print (the Unicode
    line and paragraph separators, zero-width and direction marks, a lone
    surrogate) is escaped the same way. Ordinary text comes out as itself
    between double quotes."""
    return "".join(
        char if char.isprintable() else json.dumps(char)[1:-1]
        for char in json.dumps(value, ensure_ascii=False)
    )


def _title(value: "str | None") -> str:
    """A source's title, quoted — or the app's own word for one it lacks."""
    return _quoted(value) if value else "(untitled)"


# A tier name the model would otherwise have to guess at. VERIFIED_* and
# METADATA_ONLY say what they mean; MATCHED_RECORD does not, and the guess it
# invites ("a record matched, so this is verified") is exactly what it denies.
_TIER_GLOSS = {
    "MATCHED_RECORD": (
        " (a registry record matches this source's details, but does not name the address it"
        " was fetched from, so it is NOT verified)"
    )
}


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


def _evidence_locator(row: dict) -> str:
    """Where in its source the quoted evidence sits, phrased by SOURCE TYPE and
    never inferred from which locator happens to be null: Docling leaves page
    unset for some PDF items, and that must not read as a web section."""
    source_type = row.get("evidence_source_type") or "pdf"
    if source_type == "image":
        return ", image"
    if source_type == "youtube":
        start = row.get("evidence_start_seconds")
        return f" at {clock(start)}" if start is not None else ""
    if source_type == "web":
        section = row.get("evidence_section")
        return f" § {_quoted(section)}" if section else ""
    page = row.get("evidence_page")
    return f" p.{page}" if page is not None else ""


def build_context(conn: sqlite3.Connection, run_id: str) -> str:
    """The static per-run analysis, rendered for the model. Reuses the same db
    reads the /report endpoint uses so the two cannot describe different runs.

    Every string the app did not write goes through _quoted (a title through
    _title), so each line's structure is the app's alone."""
    verdicts = dbmod.list_verdicts_with_evidence(conn, run_id)
    scores = dbmod.get_run_scores(conn, run_id)
    sources = dbmod.list_run_sources(conn, run_id)

    lines = ["=== ANALYSIS ===", "", "SCORES: " + _fmt_score(scores), "", "CLAIMS:"]
    for row in verdicts:
        evidence = ""
        # A quote is shown only with the source chunk it was verified in.
        if row["quote"] and row["evidence_doc_id"]:
            source = f"{_title(row['evidence_doc_title'])}{_evidence_locator(row)}"
            evidence = f" — evidence: {_quoted(row['quote'])} (source {source})"
        disavowed = " (disavowed by the report)" if row.get("stance") == "disavowed" else ""
        lines.append(
            f"- [{row['verdict']}]{disavowed} {_quoted(row['text'])}"
            f" — {_quoted(row['rationale'])}{evidence}"
        )
    lines += ["", "SOURCES:"]
    for source in sources:
        if source["source_type"] == "image":
            standing = "not scorable (image)"
        elif source["total"] is None:
            standing = "not scored"
        else:
            standing = (
                f"tier {source['tier']}{_TIER_GLOSS.get(source['tier'], '')}, "
                f"credibility {source['total']}/100"
            )
        title = _title(source["doc_title"])
        lines.append(f"- {title}: {standing}{_video_note(source)}{_partial_note(source)}")
    return "\n".join(lines)


def _video_note(source: dict) -> str:
    """What a video source was read from: its captions — and when they are
    YouTube's speech recognition, that they can mishear a word a quote then
    carries — and whether YouTube verified its channel, which is why its name
    may or may not have earned a publisher's authority. Empty for every other
    source, and for a video whose record is not the shape the reader writes."""
    video = source.get("video")
    if source.get("source_type") != "youtube" or not isinstance(video, dict):
        return ""
    captions = video.get("captions") if isinstance(video.get("captions"), dict) else {}
    read_from = (
        "YouTube's automatic captions (speech recognition, which can mishear words)"
        if captions.get("kind") == "automatic"
        else "captions written for it"
    )
    verified = "verified" if channel_verified(video) else "not verified"
    return f" — a YouTube video, read from {read_from}; its channel is {verified} by YouTube"


def _partial_note(source: dict) -> str:
    """What the page cap left out of a web source, read from the same row the
    source list beside the chat marks 'Read in part'.

    Without it the model answers from the head of a page as though it had the
    page, while the panel next to it says the page was read in part. The
    analysis never saw the rest, so a question about the rest has no answer
    here, and saying so is the only honest one.

    Degrades rather than raises: this is a footnote on one line of the context,
    and the two numbers come from a stored provenance. A shape cap_sections
    never writes (a hand-edited row, a future schema) must not take the whole
    chat endpoint down — the line keeps the standing it always had, and the log
    names the source nobody could describe."""
    truncated = source.get("truncated")
    if truncated is None:  # the normal case: the page was read whole
        return ""
    kept = truncated.get("kept_chars") if isinstance(truncated, dict) else None
    dropped = truncated.get("dropped_chars") if isinstance(truncated, dict) else None
    if not isinstance(kept, int) or not isinstance(dropped, int):
        logger.warning(
            "source %r has a malformed page-cap record (%s) — the chat context cannot say how "
            "much of the page was read",
            source.get("doc_title"),
            type(truncated).__name__,
        )
        return ""
    video = source.get("source_type") == "youtube"
    until = truncated.get("kept_until_seconds")
    if video and isinstance(until, int | float):
        read = f"{clock(until)} of this video's captions"
    else:
        read = f"{kept:,} of {kept + dropped:,} characters of "
        read += "this video's captions" if video else "this page"
    return (
        f" — READ IN PART: only the first {read} were analysed, so nothing later in it is covered"
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
    analysis = TextDocument(
        title=ANALYSIS_TITLE, context=ANALYSIS_CONTEXT, data=build_context(conn, run_id)
    )
    system_blocks = [
        {"type": "text", "text": CHAT_SYSTEM},
        {"type": "text", "text": MODE_INSTRUCTIONS[mode]},
    ]
    # Trim history to the most recent turns (guarding the -0 slice, which would
    # keep ALL history). The conversation must start with a user turn, so drop
    # any leading assistant turns left after trimming — the API 400s otherwise.
    n = settings.chat_history_turns
    turns = history[-n:] if n > 0 else []
    while turns and turns[0].get("role") != "user":
        turns = turns[1:]
    first, *later = [*turns, {"role": "user", "content": question}]
    # The analysis rides on whichever user turn comes first, ahead of that
    # turn's own text, so the prefix (system prompt + analysis) is the same
    # for every turn of a conversation however the history window slides. The
    # breakpoint is set here, on this one block: llm's document block is also
    # what the verdict and metadata fingerprints are computed over.
    document, *text = render_user_content(first["content"], documents=[analysis])
    messages = [
        {
            "role": "user",
            "content": [{**document, "cache_control": {"type": "ephemeral"}}, *text],
        },
        *later,
    ]
    return llm.chat(
        model=settings.chat_model,
        system_blocks=system_blocks,
        messages=messages,
        max_tokens=settings.chat_max_tokens,
    )
