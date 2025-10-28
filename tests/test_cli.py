"""CLI smoke tests for both commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from cli.main import extract_cli, verify_cli


def test_author_extract_cli_outputs_json() -> None:
    runner = CliRunner()
    result = runner.invoke(extract_cli, ["--json"], input="Inflation was 6% in 2023.")
    assert result.exit_code == 0
    assert '"text": "Inflation was 6% in 2023."' in result.output


def test_author_verify_cli_runs_pipeline(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "doc1.txt").write_text("Inflation reached 6.0% in 2023 according to the ONS.", encoding="utf-8")

    infile = tmp_path / "input.txt"
    infile.write_text("Inflation was 6% in 2023.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        verify_cli,
        ["--infile", str(infile), "--sources", str(sources), "--outdir", str(tmp_path)],
    )

    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if line.strip() and not line.startswith("[")]
    assert lines, "author-verify should emit JSON lines per claim"
    html_report = tmp_path / "author_ai_report.html"
    assert html_report.exists()
