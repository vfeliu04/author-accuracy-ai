"""Command-line interface: ingest PDFs into a run and search it.

Usage (inside the conda env, from backend/):
    python -m authorai.cli ingest ../example_sources/*.pdf
    python -m authorai.cli search <run_id> hunger 735 million
"""

import argparse
from pathlib import Path

from authorai import db as dbmod
from authorai.config import Settings
from authorai.embeddings import OpenAIEmbedder
from authorai.ingest import ingest_pdf
from authorai.search import hybrid_search


def _setup() -> tuple[Settings, object, OpenAIEmbedder]:
    settings = Settings()
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    return settings, conn, embedder


def run_ingest(
    conn,
    embedder,
    figures_dir: Path,
    pdfs: list[Path],
    run: str | None,
    kind: str,
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
            doc_id = ingest_pdf(conn, embedder, run_id, pdf, kind=kind, figures_dir=figures_dir)
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
    settings, conn, embedder = _setup()
    _, failures = run_ingest(conn, embedder, settings.figures_dir, args.pdfs, args.run, args.kind)
    if failures:
        raise SystemExit(f"{failures} PDF(s) failed to ingest")


def cmd_search(args: argparse.Namespace) -> None:
    _, conn, embedder = _setup()
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
    ingest.set_defaults(func=cmd_ingest)

    search = subparsers.add_parser("search", help="Hybrid-search a run")
    search.add_argument("run_id")
    search.add_argument("query", nargs="+")
    search.add_argument("-k", type=int, default=10)
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
