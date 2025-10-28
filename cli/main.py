"""Command-line entry points for extraction (legacy) and verification (Stage A–E).

The CLI exposes two commands:
    * author-extract — original regex-based extractor (JSON optional)
    * author-verify  — full pipeline producing JSONL + HTML artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import click

from author_ai.claims import extract as legacy_extract
from author_ai.claims.schema import Claim as LegacyClaim
from author_ai.config import load_default_config
from author_ai.models import Claim, EvidenceSpan, Score, VerificationResult
from author_ai.pipeline import VerificationPipeline


# --------------------------------------------------------------------------- helpers


def _read_input(infile: Optional[Path]) -> str:
    """Read from a file or stdin, keeping the legacy behaviour intact."""

    if infile:
        return infile.read_text(encoding="utf-8")
    return click.get_text_stream("stdin").read()


def _emit_spans(claims: Iterable[LegacyClaim]) -> None:
    """Log claim spans to stderr, matching the original CLI feature."""

    for claim in claims:
        click.echo(
            f"[{claim.span.start}:{claim.span.end}] {claim.type} -> {claim.text!r}",
            err=True,
        )


def _serialise_verification(
    claim: Claim,
    verification: VerificationResult | None,
    score: Score | None,
    evidence: Iterable[EvidenceSpan],
) -> dict:
    """Combine all stage outputs into a JSON serialisable payload."""

    return {
        "claim": claim.model_dump(),
        "verification": verification.model_dump() if verification else None,
        "score": score.model_dump() if score else None,
        "evidence": [span.model_dump() for span in evidence],
    }


# --------------------------------------------------------------------------- author-extract


@click.command("author-extract")
@click.option(
    "--infile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read input text from the given file (defaults to stdin).",
)
@click.option(
    "--json/--no-json",
    "json_output",
    default=True,
    help="Emit machine-readable JSON output (default).",
)
@click.option(
    "--pretty/--no-pretty",
    default=False,
    help="Pretty-print JSON output.",
)
@click.option(
    "--show-spans",
    is_flag=True,
    help="Echo detected spans to stderr for debugging.",
)
def extract_cli(
    infile: Optional[Path],
    json_output: bool,
    pretty: bool,
    show_spans: bool,
) -> None:
    """Extract structured quantitative claims from text."""

    raw_text = _read_input(infile)
    claims = legacy_extract.extract_claims(raw_text)

    if show_spans:
        _emit_spans(claims)

    if json_output:
        payload = [claim.model_dump() for claim in claims]
        indent = 2 if pretty else None
        click.echo(json.dumps(payload, indent=indent, ensure_ascii=False))
    else:
        for claim in claims:
            click.echo(f"{claim.type}: {claim.text}")


# --------------------------------------------------------------------------- author-verify


@click.command("author-verify")
@click.option(
    "--infile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read input text from the given file (defaults to stdin).",
)
@click.option(
    "--sources",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing source documents (txt/md/json).",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional directory to write report artifacts into.",
)
@click.option(
    "--html-report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit path for the HTML report (overrides --outdir default).",
)
@click.option(
    "--jsonl-report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit path for the JSONL report (overrides --outdir default).",
)
def verify_cli(
    infile: Optional[Path],
    sources: Path,
    outdir: Optional[Path],
    html_report: Optional[Path],
    jsonl_report: Optional[Path],
) -> None:
    """Run the full Stage A–E pipeline and emit JSONL + optional reports."""

    text = _read_input(infile)
    pipeline = VerificationPipeline(load_default_config())
    output = pipeline.run(text, sources)

    # Prepare lookups for JSONL emission.
    verification_by_id = {item.claim_id: item for item in output.verifications}
    score_by_id = {item.claim_id: item for item in output.scores}

    for claim in output.claims:
        record = _serialise_verification(
            claim=claim,
            verification=verification_by_id.get(claim.id),
            score=score_by_id.get(claim.id),
            evidence=output.evidence.get(claim.id, []),
        )
        click.echo(json.dumps(record, ensure_ascii=False))

    # Write artefacts if Stage E returned a report and the user requested output files.
    if output.report:
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            default_html = outdir / "author_ai_report.html"
            default_jsonl = outdir / "author_ai_report.jsonl"
        else:
            default_html = default_jsonl = None

        html_path = html_report or default_html
        jsonl_path = jsonl_report or default_jsonl

        if html_path:
            html_path.write_text(output.report.html, encoding="utf-8")
            click.echo(f"[author-verify] Wrote HTML report to {html_path}", err=True)
        if jsonl_path:
            jsonl_path.write_text(output.report.jsonl, encoding="utf-8")
            click.echo(f"[author-verify] Wrote JSONL report to {jsonl_path}", err=True)


__all__ = ["extract_cli", "verify_cli"]


def main() -> None:  # pragma: no cover - entrypoint glue
    """Backward-compatible entrypoint that defaults to extraction."""

    extract_cli.main(prog_name="author-extract")  # type: ignore[call-arg]


if __name__ == "__main__":  # pragma: no cover
    main()
