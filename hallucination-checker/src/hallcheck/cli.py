"""Command line interface for the hallucination checker MVP."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from pdf_pipeline.config import pipeline_config
from pdf_pipeline.ingest import ingest_pdf_v2
from pdf_pipeline.schema import asdict as document_asdict

from .ingest_pdf import legacy_extract_pdf_content
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

    ingest_parser = subparsers.add_parser("ingest-pdf-v2", help="Run the mixed PDF extraction pipeline on a single PDF.")
    ingest_parser.add_argument("path", help="Path to the PDF file.")
    ingest_parser.add_argument("--show-json", action="store_true", help="Print the structured document JSON.")
    ingest_parser.add_argument("--no-ocr", action="store_true", help="Disable OCR for this run.")
    ingest_parser.add_argument("--legacy", action="store_true", help="Use the legacy extraction pipeline.")

    debug_parser = subparsers.add_parser("debug-pdf", help="Inspect routing and extraction diagnostics for a PDF.")
    debug_parser.add_argument("path", help="Path to the PDF file.")
    debug_parser.add_argument("--no-ocr", action="store_true", help="Disable OCR for this run.")

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
        elif args.command == "ingest-pdf-v2":
            if args.legacy:
                content = legacy_extract_pdf_content(args.path)
                print("Legacy pipeline result:")
                print(f"Title: {content.title}")
                print(f"Authors: {', '.join(content.authors) if content.authors else 'n/a'}")
                print(f"Year: {content.year or 'n/a'}")
                print(f"Tables: {len(content.tables)}")
                if args.show_json:
                    payload = {
                        "title": content.title,
                        "authors": list(content.authors),
                        "year": content.year,
                        "tables": content.tables,
                        "text_length": len(content.text),
                    }
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                cfg = pipeline_config
                if args.no_ocr:
                    cfg = replace(cfg, ocr_enabled=False)
                document, body_text, table_map, metrics = ingest_pdf_v2(args.path, config=cfg)
                print(f"Router label: {document.metadata.get('router_label')}")
                print(f"OCR used: {bool(document.metadata.get('ocr_used'))}")
                print(f"Sections: {len(document.sections)}")
                print(f"Tables: {len(document.tables)}")
                print(f"Body length: {len(body_text)} characters")
                if args.show_json:
                    print(json.dumps(document_asdict(document), ensure_ascii=False, indent=2))
                else:
                    print(json.dumps(metrics, ensure_ascii=False, indent=2))
            print("Ingestion complete.")
        elif args.command == "debug-pdf":
            cfg = pipeline_config
            if args.no_ocr:
                cfg = replace(cfg, ocr_enabled=False)
            document, _, _, metrics = ingest_pdf_v2(args.path, config=cfg)
            router = metrics.get("router", {})
            layout_metrics = metrics.get("extractors", {}).get("layout", {})
            print("Router decision:")
            print(json.dumps(router, ensure_ascii=False, indent=2))
            print("OCR used:", bool(metrics.get("ocr_used")))
            headers = layout_metrics.get("headers_removed", [])
            footers = layout_metrics.get("footers_removed", [])
            print("Repeated headers:", headers if headers else "<none>")
            print("Repeated footers:", footers if footers else "<none>")
            print("Sections:")
            for section in document.sections:
                title = section.title or "(untitled)"
                print(f"  - L{section.level} {title} [p{section.page_range[0]}-{section.page_range[1]}] conf={section.confidence:.2f}")
            print("Tables:")
            for table in document.tables:
                label = table.caption or "(no caption)"
                print(f"  - {table.id} page {table.page}: {label}")
        else:
            parser.print_help()
            return 1
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
