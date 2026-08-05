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


def cmd_ingest(args: argparse.Namespace) -> None:
    settings, conn, embedder = _setup()
    if args.run:
        if dbmod.get_run(conn, args.run) is None:
            raise SystemExit(f"Unknown run {args.run!r}")
        run_id = args.run
    else:
        run_id = dbmod.create_run(conn)
    for pdf in args.pdfs:
        doc_id = ingest_pdf(
            conn, embedder, run_id, pdf, kind=args.kind, figures_dir=settings.figures_dir
        )
        counts = dict(
            conn.execute(
                "SELECT kind, count(*) FROM chunks WHERE doc_id = ? GROUP BY kind",
                (doc_id,),
            ).fetchall()
        )
        print(f"ingested {pdf.name}: {counts}")
    print(f"run: {run_id}")


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
