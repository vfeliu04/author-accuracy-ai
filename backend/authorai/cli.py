"""Command-line interface: ingest PDFs into a run and search it.

Usage (inside the conda env, from backend/):
    python -m authorai.cli ingest "../example_sources/example source one/"*.pdf
    python -m authorai.cli search <run_id> hunger 735 million
"""

import argparse
import json
import sqlite3
from pathlib import Path

from authorai import db as dbmod
from authorai.claims import EXTRACTION_PROMPT_HASH, claims_as_rows, extract_claims
from authorai.config import Settings
from authorai.embeddings import OpenAIEmbedder
from authorai.evals import (
    load_golden,
    score_extraction,
    score_injection,
    score_stance,
    score_verdicts,
    verdict_pairs,
)
from authorai.ingest import (
    FIGURE_DESCRIPTION_PROMPT,
    IMAGE_DESCRIPTION_MAX_TOKENS,
    IMAGE_SOURCE_PROMPT,
    ingest_image,
    ingest_pdf,
)
from authorai.llm import AnthropicClient
from authorai.scoring import score_run
from authorai.search import hybrid_search
from authorai.verification import EVIDENCE_K, verdict_stamp, verify_run

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"
VERDICT_BASELINE_PATH = Path(__file__).resolve().parent.parent / "evals" / "verdict_baseline.json"
INJECTION_GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "evals" / "injection" / "injection.jsonl"
)


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
    image: bool = False,
) -> tuple[str, int]:
    """Ingest PDFs (or, with `image`, PNG/JPEG/WebP image sources) into a run;
    returns (run_id, failure_count).

    The run id is printed FIRST so a failure partway through a batch never
    loses the id of the partially-populated run, and one bad PDF does not
    abort the rest of the batch. An image is ingested the way the jobs worker
    ingests an uploaded one: an `image` upload row, then its reading by
    `describe` as the document's only text.
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
            if image:
                # Like an API upload, the row exists before the reading: a
                # failed reading leaves it, with no document, as one would.
                upload_id = dbmod.add_upload(
                    conn, kind, pdf.name, str(pdf.resolve()), source_type="image"
                )
                doc_id = ingest_image(
                    conn,
                    embedder,
                    run_id,
                    pdf,
                    name=pdf.name,
                    kind=kind,
                    figures_dir=figures_dir,
                    upload_id=upload_id,
                    describe=describe,
                )
            else:
                doc_id = ingest_pdf(
                    conn,
                    embedder,
                    run_id,
                    pdf,
                    kind=kind,
                    figures_dir=figures_dir,
                    describe=describe,
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
    if args.image and args.no_describe_figures:
        raise SystemExit(
            "An image source's reading is its only text; --no-describe-figures cannot apply"
        )
    settings, conn = _setup()
    embedder = _embedder(settings)
    describe = None
    if args.image:
        llm = AnthropicClient(settings.anthropic_api_key)
        model = settings.caption_model

        def describe(image):
            # The same reading the jobs worker asks of an uploaded image.
            return llm.describe_image(
                model=model,
                image=image,
                prompt=IMAGE_SOURCE_PROMPT,
                max_tokens=IMAGE_DESCRIPTION_MAX_TOKENS,
            )
    elif not args.no_describe_figures:
        llm = AnthropicClient(settings.anthropic_api_key)
        model = settings.caption_model

        def describe(image):
            return llm.describe_image(model=model, image=image, prompt=FIGURE_DESCRIPTION_PROMPT)

    _, failures = run_ingest(
        conn,
        embedder,
        settings.figures_dir,
        args.pdfs,
        args.run,
        args.kind,
        describe=describe,
        image=args.image,
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
    # Tables live in their own chunks, never in metadata sections — pass them in
    # explicitly or every tabulated figure in the report goes unchecked.
    tables = dbmod.list_chunks_by_kind(conn, document["id"], "table")
    if not sections and not tables:
        raise SystemExit(
            "Document has no stored sections or table chunks (ingested before Phase 3?)"
            " — re-ingest it"
        )

    llm = AnthropicClient(settings.anthropic_api_key)
    claims = extract_claims(llm, sections, settings.extraction_model, tables=tables)

    # Re-extraction replaces the document's previous claims — deterministic reruns.
    dbmod.add_claims(conn, args.run_id, document["id"], claims_as_rows(claims), replace=True)

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


def _refuse_stale(message: str, allow_stale: bool) -> None:
    """The one refusal policy both freshness guards share: refuse loudly, or
    convert to a loud warning under --allow-stale. Scoring stale rows as if
    they were fresh already produced one wrong conclusion (MISTAKES.md
    2026-08-08) — the policy living here once keeps the two guards identical.
    """
    if allow_stale:
        print(f"WARNING (--allow-stale): {message}")
        return
    raise SystemExit(message + " (or pass --allow-stale to score them anyway)")


def _assert_claims_fresh(claim_rows: list[dict], allow_stale: bool) -> None:
    """Refuse to score claims produced by a different extraction prompt.

    The extraction twin of _assert_verdicts_fresh: recall/precision measured
    on stale claims say nothing about the prompt being tuned. NULL
    (pre-guard rows) counts as stale.
    """
    stale = [r for r in claim_rows if r.get("extraction_prompt_hash") != EXTRACTION_PROMPT_HASH]
    if not stale:
        return
    _refuse_stale(
        f"{len(stale)}/{len(claim_rows)} claims were produced by a DIFFERENT extraction "
        "prompt than the current one. The score would say nothing about the current "
        "prompt — rerun `extract` first.",
        allow_stale,
    )


def cmd_eval_extract(args: argparse.Namespace) -> None:
    _, conn = _setup()
    golden = _load_golden_or_exit(args.golden)
    extracted = dbmod.list_claims(conn, args.run_id)
    if not extracted:
        raise SystemExit(f"Run {args.run_id!r} has no claims — run `extract` first")
    _assert_claims_fresh(extracted, args.allow_stale)
    score = score_extraction(extracted, golden)
    print(score.summary())
    for text in score.missed:
        print(f"MISSED: {text}")
    stance = score_stance(extracted, golden)
    if stance.labeled or stance.unpaired:
        print(stance.summary())
        for line in stance.disagreements:
            print(f"STANCE MISMATCH: {line}")
    _print_baseline_delta(
        _resolve_baseline(args, BASELINE_PATH),
        lambda b: (
            f"baseline: recall {b['recall']:.2f}, precision {b['precision']:.2f} "
            f"(delta: recall {score.recall - b['recall']:+.2f}, "
            f"precision {score.precision - b['precision']:+.2f})"
            + (
                f"; stance {stance.agreed}/{stance.labeled} vs baseline "
                f"{b['stance_agreed']}/{b['stance_labeled']}"
                if stance.labeled and b.get("stance_agreed") is not None
                else ""
            )
        ),
    )


def cmd_verify(args: argparse.Namespace) -> None:
    if args.sync and args.resume_batch:
        raise SystemExit("--resume-batch resumes a Batch API run; it cannot apply with --sync")
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
        resume_batch_id=args.resume_batch,
        # Printed the moment it exists: a queue that outlasts the client's wait
        # is resumed with --resume-batch instead of paid for twice.
        on_batch_created=lambda batch_id: print(f"batch: {batch_id}", flush=True),
    )
    for verdict, count in summary["counts"].items():
        print(f"{verdict}: {count}")
    print(
        f"downgraded (quote check failed): {summary['downgraded']}, "
        f"year-flagged: {summary['year_flagged']}, "
        f"no evidence: {summary['no_evidence']}, total: {summary['total']}"
    )


def _assert_verdicts_fresh(verdict_rows: list[dict], allow_stale: bool) -> None:
    """Refuse to score verdicts produced by a different judge configuration
    (prompt or evidence-k). NULL (pre-guard rows) counts as stale."""
    stale = [r for r in verdict_rows if r.get("prompt_hash") != verdict_stamp()]
    if not stale:
        return
    models = sorted({r["model"] for r in stale})
    _refuse_stale(
        f"{len(stale)}/{len(verdict_rows)} verdicts were produced by a DIFFERENT judge "
        f"prompt than the current one (models: {', '.join(models)}). The score would say "
        "nothing about the current prompt — rerun `verify` first.",
        allow_stale,
    )


def cmd_eval_verdict(args: argparse.Namespace) -> None:
    _, conn = _setup()
    golden = _load_golden_or_exit(args.golden)
    verdict_rows = dbmod.list_verdicts(conn, args.run_id)
    if not verdict_rows:
        raise SystemExit(f"Run {args.run_id!r} has no verdicts — run `verify` first")
    # Both guards: the verdict stamp proves the JUDGE configuration, but says
    # nothing about the claim population — verdicts freshly judged over claims
    # from an older extraction prompt would measure the judge on outdated
    # inputs (list_verdicts joins extraction_prompt_hash for exactly this).
    _assert_claims_fresh(verdict_rows, args.allow_stale)
    _assert_verdicts_fresh(verdict_rows, args.allow_stale)
    models = sorted({r["model"] for r in verdict_rows})
    print(f"judge model(s): {', '.join(models)}")
    score = score_verdicts(verdict_rows, golden)
    print(score.summary())
    for expected, predicted_counts in score.confusion.items():
        wrong = {p: n for p, n in predicted_counts.items() if n and p != expected}
        if wrong:
            print(f"confused {expected} -> {wrong}")
    if args.save:
        _save_json(
            args.save,
            {
                "run_id": args.run_id,
                "models": models,
                "prompt_hash": verdict_stamp(),
                "accuracy": score.accuracy,
                "correct": score.correct,
                "matched": score.matched,
                "coverage": score.coverage,
                "downgraded": score.downgraded,
                "claims": verdict_pairs(verdict_rows, golden),
            },
        )
    _print_baseline_delta(
        _resolve_baseline(args, VERDICT_BASELINE_PATH),
        lambda b: (
            f"baseline: accuracy {b['accuracy']:.2f} (delta: {score.accuracy - b['accuracy']:+.2f})"
        ),
    )


def _save_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved {path}")


def cmd_eval_injection(args: argparse.Namespace) -> None:
    _, conn = _setup()
    golden = _load_golden_or_exit(args.golden)
    verdict_rows = dbmod.list_verdicts(conn, args.run_id)
    if not verdict_rows:
        raise SystemExit(f"Run {args.run_id!r} has no verdicts — run `verify` first")
    _assert_claims_fresh(verdict_rows, args.allow_stale)
    _assert_verdicts_fresh(verdict_rows, args.allow_stale)
    chunk_text_by_id = dict(
        conn.execute("SELECT id, text FROM chunks WHERE run_id = ?", (args.run_id,)).fetchall()
    )
    score = score_injection(verdict_rows, golden, chunk_text_by_id)
    models = sorted({r["model"] for r in verdict_rows})
    print(f"judge model(s): {', '.join(models)}")
    for outcome in [*score.attacks, *score.controls]:
        raw = f" (raw {outcome['raw']})" if outcome["raw"] != outcome["final"] else ""
        print(
            f"{outcome['id']:>4} {outcome['status']:<10} expected {outcome['expected']}, "
            f"final {outcome['final']}{raw} — {outcome['claim'][:70]}"
        )
    print(score.summary())
    if args.save:
        _save_json(
            args.save,
            {
                "run_id": args.run_id,
                "models": models,
                "prompt_hash": verdict_stamp(),
                **score.as_dict(),
            },
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
        f"(agreement: {accuracy['correct']} correct, {accuracy['incorrect']} incorrect; "
        f"supported {accuracy['supported']}, contradicted {accuracy['contradicted']}, "
        f"unverifiable {accuracy['unverifiable']}, disavowed {accuracy['disavowed']}; "
        f"coverage {accuracy['coverage']})"
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
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Score rows even if they were produced by an older prompt",
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
    ingest.add_argument(
        "--image",
        action="store_true",
        help="The files are PNG/JPEG/WebP image sources, read by the caption model",
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
    verify.add_argument(
        "--resume-batch",
        metavar="BATCH_ID",
        help="Poll this already-submitted batch (the id `verify` printed) instead of a new one",
    )
    verify.set_defaults(func=cmd_verify)

    eval_verdict = subparsers.add_parser(
        "eval-verdict", help="Score stored verdicts against the golden set"
    )
    _add_eval_args(eval_verdict)
    eval_verdict.add_argument(
        "--save", type=Path, help="Also write the score and the per-claim pairing to this JSON file"
    )
    eval_verdict.set_defaults(func=cmd_eval_verdict)

    eval_injection = subparsers.add_parser(
        "eval-injection", help="Score stored verdicts against the prompt-injection set"
    )
    eval_injection.add_argument("run_id")
    eval_injection.add_argument(
        "--golden",
        type=Path,
        default=INJECTION_GOLDEN_PATH,
        help="Injection-set JSONL (default: evals/injection/injection.jsonl)",
    )
    eval_injection.add_argument(
        "--save", type=Path, help="Also write the per-attack outcomes to this JSON file"
    )
    eval_injection.add_argument(
        "--allow-stale",
        action="store_true",
        help="Score rows even if they were produced by an older prompt",
    )
    eval_injection.set_defaults(func=cmd_eval_injection)

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
