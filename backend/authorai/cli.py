"""Command-line interface: ingest PDFs into a run and search it.

Usage (inside the conda env, from backend/):
    python -m authorai.cli ingest ../example_sources/*.pdf
    python -m authorai.cli search <run_id> hunger 735 million
"""

import argparse
import json
import sqlite3
from pathlib import Path

from authorai import db as dbmod
from authorai.claims import extract_claims
from authorai.config import Settings
from authorai.embeddings import OpenAIEmbedder
from authorai.evals import load_golden, score_extraction
from authorai.ingest import FIGURE_DESCRIPTION_PROMPT, ingest_pdf
from authorai.llm import AnthropicClient
from authorai.search import hybrid_search

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"


def _setup() -> tuple[Settings, sqlite3.Connection]:
    settings = Settings()
    return settings, dbmod.connect(settings.db_path, settings.embedding_dim)


def _embedder(settings: Settings) -> OpenAIEmbedder:
    """Built only by commands that actually embed.

    `extract` and `eval-extract` touch no vectors; constructing the embedder
    for them would demand an OPENAI_API_KEY they never use and turn a working
    offline scoring run into a hard failure.
    """
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )


def run_ingest(
    conn,
    embedder,
    figures_dir: Path,
    pdfs: list[Path],
    run: str | None,
    kind: str,
    describe=None,
) -> tuple[str, int]:
    """Ingest PDFs into a run; returns (run_id, failure_count).

    The run id is printed FIRST so a failure partway through a batch never
    loses the id of the partially-populated run, and one bad PDF does not
    abort the rest of the batch.
    """
    if run:
        if dbmod.get_run(conn, run) is None:
            raise SystemExit(f"Unknown run {run!r}")
        run_id = run
    else:
        run_id = dbmod.create_run(conn)
    print(f"run: {run_id}")

    failures = 0
    for pdf in pdfs:
        try:
            doc_id = ingest_pdf(
                conn, embedder, run_id, pdf, kind=kind, figures_dir=figures_dir, describe=describe
            )
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failures += 1
            print(f"FAILED {pdf.name}: {exc}")
            continue
        counts = dict(
            conn.execute(
                "SELECT kind, count(*) FROM chunks WHERE doc_id = ? GROUP BY kind",
                (doc_id,),
            ).fetchall()
        )
        print(f"ingested {pdf.name}: {counts}")
    return run_id, failures


def cmd_ingest(args: argparse.Namespace) -> None:
    settings, conn = _setup()
    embedder = _embedder(settings)
    describe = None
    if not args.no_describe_figures:
        llm = AnthropicClient(settings.anthropic_api_key)
        model = settings.caption_model

        def describe(image):
            return llm.describe_image(model=model, image=image, prompt=FIGURE_DESCRIPTION_PROMPT)

    _, failures = run_ingest(
        conn, embedder, settings.figures_dir, args.pdfs, args.run, args.kind, describe=describe
    )
    if failures:
        raise SystemExit(f"{failures} PDF(s) failed to ingest")


def cmd_extract(args: argparse.Namespace) -> None:
    settings, conn = _setup()
    if args.doc:
        document = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND run_id = ?", (args.doc, args.run_id)
        ).fetchone()
        if document is None:
            raise SystemExit(f"No document {args.doc!r} in run {args.run_id!r}")
    else:
        reports = conn.execute(
            "SELECT * FROM documents WHERE run_id = ? AND kind = 'REPORT'", (args.run_id,)
        ).fetchall()
        if not reports:
            raise SystemExit(f"Run {args.run_id!r} has no REPORT document — ingest one first")
        if len(reports) > 1:
            ids = ", ".join(row["id"] for row in reports)
            raise SystemExit(f"Run has multiple REPORT documents ({ids}) — pass --doc")
        document = reports[0]

    sections = json.loads(document["metadata"]).get("sections", [])
    if not sections:
        raise SystemExit(
            "Document has no stored sections (ingested before Phase 3?) — re-ingest it"
        )

    llm = AnthropicClient(settings.anthropic_api_key)
    claims = extract_claims(llm, sections, settings.extraction_model)

    # Re-extraction replaces the document's previous claims — deterministic reruns.
    dbmod.add_claims(
        conn,
        args.run_id,
        document["id"],
        [claim.model_dump() for claim in claims],
        replace=True,
    )

    for claim in claims:
        details = ", ".join(
            f"{name}={value}"
            for name, value in (
                ("value", claim.value),
                ("unit", claim.unit),
                ("year", claim.year),
                ("page", claim.page),
            )
            if value is not None
        )
        print(f"- {claim.text}  [{details}]" if details else f"- {claim.text}")
    print(f"{len(claims)} claims extracted and stored")


def cmd_eval_extract(args: argparse.Namespace) -> None:
    _, conn = _setup()
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"Golden set not found at {GOLDEN_PATH}")
    golden = load_golden(GOLDEN_PATH)
    extracted = dbmod.list_claims(conn, args.run_id)
    if not extracted:
        raise SystemExit(f"Run {args.run_id!r} has no claims — run `extract` first")
    score = score_extraction(extracted, golden)
    print(score.summary())
    for text in score.missed:
        print(f"MISSED: {text}")
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        print(
            f"baseline: recall {baseline['recall']:.2f}, precision {baseline['precision']:.2f} "
            f"(delta: recall {score.recall - baseline['recall']:+.2f}, "
            f"precision {score.precision - baseline['precision']:+.2f})"
        )
    else:
        print(f"no baseline yet — to accept this as baseline, write {BASELINE_PATH}")


def cmd_search(args: argparse.Namespace) -> None:
    settings, conn = _setup()
    embedder = _embedder(settings)
    query = " ".join(args.query)
    [query_embedding] = embedder.embed([query])
    hits = hybrid_search(conn, args.run_id, query, query_embedding, k=args.k)
    if not hits:
        print("no results")
        return
    for hit in hits:
        channels = "+".join(hit.channels)
        page = f" p.{hit.page}" if hit.page is not None else ""
        print(f"[{hit.score:.4f} {channels}{page} {hit.kind}] {hit.text[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="authorai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest PDFs into a run")
    ingest.add_argument("pdfs", nargs="+", type=Path)
    ingest.add_argument("--run", help="Existing run id (default: create a new run)")
    ingest.add_argument("--kind", choices=["SOURCE", "REPORT"], default="SOURCE")
    ingest.add_argument(
        "--no-describe-figures",
        action="store_true",
        help="Skip LLM figure descriptions (default: on, requires ANTHROPIC_API_KEY)",
    )
    ingest.set_defaults(func=cmd_ingest)

    extract = subparsers.add_parser("extract", help="Extract claims from a run's report")
    extract.add_argument("run_id")
    extract.add_argument("--doc", help="Document id (default: the run's single REPORT)")
    extract.set_defaults(func=cmd_extract)

    eval_extract = subparsers.add_parser(
        "eval-extract", help="Score extracted claims against the golden set"
    )
    eval_extract.add_argument("run_id")
    eval_extract.set_defaults(func=cmd_eval_extract)

    search = subparsers.add_parser("search", help="Hybrid-search a run")
    search.add_argument("run_id")
    search.add_argument("query", nargs="+")
    search.add_argument("-k", type=int, default=10)
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
