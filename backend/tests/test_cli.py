from pathlib import Path

import authorai.cli as cli
from authorai import db as dbmod
from authorai.embeddings import FakeEmbedder
from tests.conftest import DIM


def test_run_ingest_prints_run_first_and_survives_a_bad_pdf(conn, tmp_path, monkeypatch, capsys):
    def fake_ingest_pdf(conn_, embedder, run_id, path, kind, figures_dir, describe=None):
        if path.name == "bad.pdf":
            raise ValueError("No extractable content")
        return dbmod.add_document(conn_, run_id, kind, title=path.stem)

    monkeypatch.setattr(cli, "ingest_pdf", fake_ingest_pdf)

    run_id, failures = cli.run_ingest(
        conn,
        FakeEmbedder(dim=DIM),
        figures_dir=tmp_path,
        pdfs=[Path("bad.pdf"), Path("good.pdf")],
        run=None,
        kind="SOURCE",
    )

    assert failures == 1
    lines = capsys.readouterr().out.strip().splitlines()
    # Run id comes FIRST, so a partial batch never loses it.
    assert lines[0] == f"run: {run_id}"
    assert lines[1].startswith("FAILED bad.pdf")
    assert lines[2].startswith("ingested good.pdf")
    # The good PDF landed in the run despite the earlier failure.
    docs = conn.execute("SELECT title FROM documents").fetchall()
    assert [d["title"] for d in docs] == ["good"]


def test_stale_verdicts_are_refused(capsys):
    import pytest

    from authorai.verification import VERDICT_PROMPT_HASH

    fresh = {"model": "m", "prompt_hash": VERDICT_PROMPT_HASH}
    stale = {"model": "old-model", "prompt_hash": "not-the-current-hash"}
    unstamped = {"model": "older-model", "prompt_hash": None}  # pre-migration row

    cli._assert_verdicts_fresh([fresh], allow_stale=False)  # fresh rows pass silently

    with pytest.raises(SystemExit, match="DIFFERENT judge prompt"):
        cli._assert_verdicts_fresh([fresh, stale], allow_stale=False)
    with pytest.raises(SystemExit, match="DIFFERENT judge prompt"):
        cli._assert_verdicts_fresh([unstamped], allow_stale=False)

    # --allow-stale converts the refusal into a loud warning.
    cli._assert_verdicts_fresh([stale], allow_stale=True)
    assert "WARNING (--allow-stale)" in capsys.readouterr().out


def test_run_ingest_rejects_unknown_run(conn, tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="Unknown run"):
        cli.run_ingest(
            conn,
            FakeEmbedder(dim=DIM),
            figures_dir=tmp_path,
            pdfs=[],
            run="nope",
            kind="SOURCE",
        )
