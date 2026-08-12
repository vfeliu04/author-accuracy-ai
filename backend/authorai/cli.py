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
from authorai.evals import load_golden, score_extraction, score_verdicts
from authorai.ingest import FIGURE_DESCRIPTION_PROMPT, ingest_pdf
from authorai.llm import AnthropicClient
from authorai.scoring import score_run
from authorai.search import hybrid_search
from authorai.verification import EVIDENCE_K, VERDICT_PROMPT_HASH, verify_run

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"
VERDICT_BASELINE_PATH = Path(__file__).resolve().parent.parent / "evals" / "verdict_baseline.json"


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

    # Tables live in their own chunks, never in metadata sections — pass them in
    # explicitly or every tabulated figure in the report goes unchecked.
    tables = dbmod.list_chunks_by_kind(conn, document["id"], "table")

    llm = AnthropicClient(settings.anthropic_api_key)
    claims = extract_claims(llm, sections, settings.extraction_model, tables=tables)

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


def _resolve_baseline(args: argparse.Namespace, default_path: Path) -> Path | None:
    """A baseline only means something for the golden set it was recorded
    against — comparing a holdout score to the dev baseline would mislead.
    Resolve before comparing so the dev golden passed as a relative path
    still counts as the dev golden."""
    if args.baseline:
        return args.baseline
    return default_path if args.golden.resolve() == GOLDEN_PATH.resolve() else None


def _load_golden_or_exit(golden_path: Path) -> list[dict]:
    if not golden_path.exists():
        raise SystemExit(f"Golden set not found at {golden_path}")
    return load_golden(golden_path)


def _print_baseline_delta(baseline_path: Path | None, delta_line) -> None:
    if baseline_path is None:
        return
    if baseline_path.exists():
        print(delta_line(json.loads(baseline_path.read_text())))
    else:
        print(f"no baseline yet — to accept this as baseline, write {baseline_path}")


def cmd_eval_extract(args: argparse.Namespace) -> None:
    _, conn = _setup()
    golden = _load_golden_or_exit(args.golden)
    extracted = dbmod.list_claims(conn, args.run_id)
    if not extracted:
        raise SystemExit(f"Run {args.run_id!r} has no claims — run `extract` first")
    score = score_extraction(extracted, golden)
    print(score.summary())
    for text in score.missed:
        print(f"MISSED: {text}")
    _print_baseline_delta(
        _resolve_baseline(args, BASELINE_PATH),
        lambda b: (
            f"baseline: recall {b['recall']:.2f}, precision {b['precision']:.2f} "
            f"(delta: recall {score.recall - b['recall']:+.2f}, "
            f"precision {score.precision - b['precision']:+.2f})"
        ),
    )


def cmd_verify(args: argparse.Namespace) -> None:
    settings, conn = _setup()
    embedder = _embedder(settings)
    llm = AnthropicClient(settings.anthropic_api_key)
    summary = verify_run(
        conn,
        embedder,
        llm,
        args.run_id,
        model=settings.verdict_model,
        k=args.k,
        batch=not args.sync,
    )
    for verdict, count in summary["counts"].items():
        print(f"{verdict}: {count}")
    print(
        f"downgraded (quote check failed): {summary['downgraded']}, "
        f"year-flagged: {summary['year_flagged']}, "
        f"no evidence: {summary['no_evidence']}, total: {summary['total']}"
    )


def _assert_verdicts_fresh(verdict_rows: list[dict], allow_stale: bool) -> None:
    """Refuse to score verdicts produced by a different judge prompt.

    Scoring stale verdicts as if they were fresh already produced one wrong
    conclusion (MISTAKES.md 2026-08-08) — this makes that class of error
    impossible instead of relying on operator memory.
    """
    stale = [r for r in verdict_rows if r.get("prompt_hash") != VERDICT_PROMPT_HASH]
    if not stale:
        return
    models = sorted({r["model"] for r in stale})
    message = (
        f"{len(stale)}/{len(verdict_rows)} verdicts were produced by a DIFFERENT judge "
        f"prompt than the current one (models: {', '.join(models)}). The score would say "
        "nothing about the current prompt — rerun `verify` first."
    )
    if allow_stale:
        print(f"WARNING (--allow-stale): {message}")
        return
    raise SystemExit(message + " (or pass --allow-stale to score them anyway)")


def cmd_eval_verdict(args: argparse.Namespace) -> None:
    _, conn = _setup()
    golden = _load_golden_or_exit(args.golden)
    verdict_rows = dbmod.list_verdicts(conn, args.run_id)
    if not verdict_rows:
        raise SystemExit(f"Run {args.run_id!r} has no verdicts — run `verify` first")
    _assert_verdicts_fresh(verdict_rows, args.allow_stale)
    models = sorted({r["model"] for r in verdict_rows})
    print(f"judge model(s): {', '.join(models)}")
    score = score_verdicts(verdict_rows, golden)
    print(score.summary())
    for expected, predicted_counts in score.confusion.items():
        wrong = {p: n for p, n in predicted_counts.items() if n and p != expected}
        if wrong:
            print(f"confused {expected} -> {wrong}")
    _print_baseline_delta(
        _resolve_baseline(args, VERDICT_BASELINE_PATH),
        lambda b: (
            f"baseline: accuracy {b['accuracy']:.2f} (delta: {score.accuracy - b['accuracy']:+.2f})"
        ),
    )


def cmd_score(args: argparse.Namespace) -> None:
    settings, conn = _setup()
    # Validate the run BEFORE building the client — otherwise a typo'd id (or a
    # missing key) fails with an unrelated message instead of "unknown run".
    if dbmod.get_run(conn, args.run_id) is None:
        raise SystemExit(f"Unknown run {args.run_id!r}")
    llm = AnthropicClient(settings.anthropic_api_key)
    result = score_run(conn, llm, args.run_id, settings, allow_stale=args.allow_stale)
    accuracy = result["accuracy"]
    print(
        f"accuracy: {accuracy['accuracy']} "
        f"(supported {accuracy['supported']}, contradicted {accuracy['contradicted']}, "
        f"unverifiable {accuracy['unverifiable']}; coverage {accuracy['coverage']})"
    )
    credibility = result["credibility"]
    print(f"credibility: {credibility['score']} ({credibility['method']})")
    for source in credibility["sources"]:
        print(
            f"  - {source['doc_id'][:8]}: {source['total']} "
            f"[{source['tier']}] usage {source['usage']}"
        )
    print(f"validity: {result['validity']['score']}")


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


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id")
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help="Golden JSONL to score against (default: the dev set)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON for the delta line (default: the dev baseline, dev golden only)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="authorai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest PDFs into a run")
    ingest.add_argument("pdfs", nargs="+", type=Path)
    ingest.add_argument("--run", help="Existing run id (default: create a new run)")
    ingest.add_argument("--kind", choices=dbmod.DOC_KINDS, default="SOURCE")
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
    _add_eval_args(eval_extract)
    eval_extract.set_defaults(func=cmd_eval_extract)

    verify = subparsers.add_parser("verify", help="Verify a run's claims against its sources")
    verify.add_argument("run_id")
    verify.add_argument(
        "--sync",
        action="store_true",
        help="Per-claim sync calls instead of the Batch API (full price, immediate)",
    )
    verify.add_argument("-k", type=int, default=EVIDENCE_K, help="Evidence chunks per claim")
    verify.set_defaults(func=cmd_verify)

    eval_verdict = subparsers.add_parser(
        "eval-verdict", help="Score stored verdicts against the golden set"
    )
    _add_eval_args(eval_verdict)
    eval_verdict.add_argument(
        "--allow-stale",
        action="store_true",
        help="Score verdicts even if they were produced by an older judge prompt",
    )
    eval_verdict.set_defaults(func=cmd_eval_verdict)

    score = subparsers.add_parser("score", help="Compute accuracy/credibility/validity for a run")
    score.add_argument("run_id")
    score.add_argument(
        "--allow-stale",
        action="store_true",
        help="Score even if the verdicts came from an older judge prompt",
    )
    score.set_defaults(func=cmd_score)

    search = subparsers.add_parser("search", help="Hybrid-search a run")
    search.add_argument("run_id")
    search.add_argument("query", nargs="+")
    search.add_argument("-k", type=int, default=10)
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
