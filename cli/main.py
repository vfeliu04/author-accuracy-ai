"""Command-line interface for extracting claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import click

from author_ai.claims import extract
from author_ai.claims.schema import Claim


def _read_input(infile: Optional[Path]) -> str:
    if infile:
        return infile.read_text(encoding="utf-8")
    return click.get_text_stream("stdin").read()


def _emit_spans(claims: Iterable[Claim]) -> None:
    for claim in claims:
        click.echo(
            f"[{claim.span.start}:{claim.span.end}] {claim.type} -> {claim.text!r}",
            err=True,
        )


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
def main(
    infile: Optional[Path],
    json_output: bool,
    pretty: bool,
    show_spans: bool,
) -> None:
    """Extract structured quantitative claims from text."""
    raw_text = _read_input(infile)
    claims = extract.extract_claims(raw_text)

    if show_spans:
        _emit_spans(claims)

    if json_output:
        payload = [claim.model_dump() for claim in claims]
        indent = 2 if pretty else None
        click.echo(json.dumps(payload, indent=indent, ensure_ascii=False))
    else:
        for claim in claims:
            click.echo(f"{claim.type}: {claim.text}")


if __name__ == "__main__":
    main()
