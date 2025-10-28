"""Command line interface for the hallucination checker MVP."""

from __future__ import annotations

import argparse
import sys

from .verify import index_sources, verify_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallcheck",
        description="Minimal hallucination checker for comparing report claims against sources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a list of source PDFs.")
    index_parser.add_argument("--sources", nargs="+", required=True, help="List of PDF paths to index.")
    index_parser.add_argument("--index-name", default="sources", help="Name for the FAISS index (default: sources).")

    verify_parser = subparsers.add_parser("verify", help="Verify a report PDF against the indexed sources.")
    verify_parser.add_argument("--report", required=True, help="Path to the report PDF.")
    verify_parser.add_argument("--index-name", default="sources", help="Name of the index to query (default: sources).")
    verify_parser.add_argument("--topk", type=int, default=5, help="Number of evidence chunks to retrieve (default: 5).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "index":
            index_sources(args.sources, index_name=args.index_name)
            print("Indexing complete.")
        elif args.command == "verify":
            verify_report(args.report, index_name=args.index_name, topk=args.topk)
            print("Verification complete.")
        else:
            parser.print_help()
            return 1
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

