"""Command-line interface: ingest PDFs into a run and search it.

Usage (inside the conda env, from backend/):
    python -m authorai.cli ingest "../example_sources/example source one/"*.pdf
    python -m authorai.cli search <run_id> hunger 735 million
"""

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
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
from authorai.ingest import figure_describer, image_source_reader, ingest_image, ingest_pdf
from authorai.llm import AnthropicClient
from authorai.scoring import score_run
from authorai.search import hybrid_search
from authorai.verification import (
    EVIDENCE_K,
    judge_identity,
    stale_verdicts,
    verify_run,
)

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
    paths: list[Path],
    run: str | None,
    kind: str,
    describe=None,
    image: bool = False,
) -> tuple[str, int]:
    """Ingest PDFs (or, with `image`, PNG/JPEG/WebP image sources) into a run;
    returns (run_id, failure_count).

    The run id is printed FIRST so a failure partway through a batch never
    loses the id of the partially-populated run, and one bad file does not
    abort the rest of the batch. An image's reading by `describe` is its only
    text, and its upload row is written after the reading, so a failed one
    leaves nothing behind.
    """
    if run:
        if dbmod.get_run(conn, run) is None:
            raise SystemExit(f"Unknown run {run!r}")
        run_id = run
    else:
        run_id = dbmod.create_run(conn)
    print(f"run: {run_id}")

    failures = 0
    for path in paths:
        extra = {"name": path.name} if image else {}
        try:
            doc_id = (ingest_image if image else ingest_pdf)(
                conn,
                embedder,
                run_id,
                path,
                kind=kind,
                figures_dir=figures_dir,
                describe=describe,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failures += 1
            print(f"FAILED {path.name}: {exc}")
            continue
        counts = dict(
            conn.execute(
                "SELECT kind, count(*) FROM chunks WHERE doc_id = ? GROUP BY kind",
                (doc_id,),
            ).fetchall()
        )
        print(f"ingested {path.name}: {counts}")
    return run_id, failures


def cmd_ingest(args: argparse.Namespace) -> None:
    if args.image and args.kind == "REPORT":
        raise SystemExit("A report must be a PDF; --image ingests image sources only")
    if args.image and args.no_describe_figures:
        raise SystemExit(
            "An image source's reading is its only text; --no-describe-figures cannot apply"
        )
    settings, conn = _setup()
    embedder = _embedder(settings)
    describe = None
    if not args.no_describe_figures:  # always so with --image (refused above)
        llm = AnthropicClient(settings.anthropic_api_key)
        reader = image_source_reader if args.image else figure_describer
        describe = reader(llm, settings.caption_model)
    _, failures = run_ingest(
        conn,
        embedder,
        settings.figures_dir,
        args.files,
        args.run,
        args.kind,
        describe=describe,
        image=args.image,
    )
    if failures:
        raise SystemExit(f"{failures} file(s) failed to ingest")


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


def _batch_record(settings: Settings, batch_id: str) -> Path:
    """Where `verify` records what judged a batch it submitted: beside the
    database, so a resume is checked against the database it writes to."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", batch_id):
        raise SystemExit(f"{batch_id!r} is not a batch id")
    return Path(settings.db_path).parent / "batches" / f"{batch_id}.json"


def _read_record(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_record(path: Path, record: dict) -> None:
    """Whole or not at all: a torn record would hide an unstored batch from the
    guard below, or stop its resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_text(json.dumps(record))
    part.replace(path)


def _check_batch_record(settings: Settings, batch_id: str, judge: dict) -> dict:
    """Refuse to resume a batch this judge configuration did not submit, or
    whose record holds no evidence; returns the record.

    A resumed batch's answers are stored under the CURRENT code's stamp, and
    the freshness guards then trust them: a batch submitted before a prompt
    change, with another -k, for another run or on another model would be
    scored as if the current judge had given it.
    """
    path = _batch_record(settings, batch_id)
    if not path.exists():
        raise SystemExit(
            f"There is no record of batch {batch_id} under {path.parent}: `verify` did not "
            "submit it against this database, so nothing proves which judge answered it."
        )
    record = _read_record(path)
    if record is None:
        raise SystemExit(f"The record of batch {batch_id} ({path}) cannot be read.")
    for field, now in judge.items():
        if record.get(field) != now:
            raise SystemExit(
                f"Batch {batch_id} cannot be resumed here: its {field} differs (submitted "
                f"with {record.get(field)!r}, now {now!r})."
            )
    if not isinstance(record.get("evidence"), dict):
        raise SystemExit(
            f"Batch {batch_id} cannot be resumed here: its record holds no evidence, so its "
            "answers could not be checked against what the judge was shown."
        )
    return record


def _batch_records(settings: Settings, judge: dict) -> list[tuple[str, dict]]:
    """Every readable batch record for this run and judge, oldest first."""
    folder = Path(settings.db_path).parent / "batches"
    found = []
    for path in folder.glob("*.json") if folder.is_dir() else []:
        record = _read_record(path)
        if record is not None and all(record.get(f) == v for f, v in judge.items()):
            found.append((path.stem, record))
    return sorted(found, key=lambda item: str(item[1].get("submitted", "")))


def _unstored(record: dict) -> bool:
    """Submitted and paid for, its answers never stored (a queue that outlasted
    the wait, an item that failed twice), and no later run stored this run's."""
    return record.get("stored") is False and not record.get("superseded_by")


def cmd_verify(args: argparse.Namespace) -> None:
    resuming = args.resume_batch is not None
    if args.sync and resuming:
        raise SystemExit("--resume-batch resumes a Batch API run; it cannot apply with --sync")
    if args.pay_again and resuming:
        raise SystemExit("--pay-again judges anew; it cannot apply with --resume-batch")
    settings, conn = _setup()
    judge = {"run_id": args.run_id, **judge_identity(settings.verdict_model, args.k)}
    resumed = _check_batch_record(settings, args.resume_batch, judge) if resuming else None
    if not resuming and not args.pay_again:
        # New answers are paid for, by batch or --sync: while a batch for this
        # run and judge sits unstored, judging again would pay twice.
        unstored = [
            batch_id for batch_id, record in _batch_records(settings, judge) if _unstored(record)
        ]
        if unstored:
            raise SystemExit(
                f"Batch {unstored[-1]} was submitted for this run and judge and its answers "
                f"were never stored: resume it with `verify {args.run_id} --resume-batch "
                f"{unstored[-1]}`, or pass --pay-again to pay for new answers"
                + (f" ({len(unstored)} such batches)." if len(unstored) > 1 else ".")
            )
    sent: dict = {}
    batch = {"id": args.resume_batch}

    def check_requests(requests: str, evidence: dict) -> None:
        # Called before any batch work. A resume is rebuilt from the evidence
        # its record holds, so the requests hash the same unless something
        # else changed (the claims, a figure, the code): then its answers are
        # not this judgment.
        if resumed is not None and resumed.get("requests") != requests:
            raise SystemExit(
                f"Batch {args.resume_batch} cannot be resumed here: the requests rebuilt from "
                "its recorded evidence differ from the ones it was sent (a figure or the code "
                "changed since)."
            )
        sent.update(requests=requests, evidence=evidence)

    def announce(batch_id: str) -> None:
        # Printed the moment it exists, so a queue that outlasts the client's
        # wait is resumed with --resume-batch instead of paid for twice; then
        # recorded with the judge, the requests and the evidence, so that
        # resume proves it is the same judgment. The batch already exists, so
        # a failed record write must not fail this run.
        print(f"batch: {batch_id}", flush=True)
        batch["id"] = batch_id
        try:
            _write_record(
                _batch_record(settings, batch_id),
                {**judge, **sent, "submitted": datetime.now(UTC).isoformat(), "stored": False},
            )
        except OSError as exc:
            print(f"WARNING: could not record batch {batch_id} ({exc}); it cannot be resumed")

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
        require_resume=resuming,
        on_batch_created=announce,
        on_requests=check_requests,
        evidence_ids=resumed["evidence"] if resumed is not None else None,
    )
    # This run's answers are stored now: its batch is, and no batch of the
    # same run and judge left unstored should stop the next judgment.
    for batch_id, record in _batch_records(settings, judge):
        if batch_id == batch["id"]:
            record["stored"] = True
        elif _unstored(record):
            record["superseded_by"] = batch["id"] or "sync"
        else:
            continue
        try:
            _write_record(_batch_record(settings, batch_id), record)
        except OSError:
            print(f"WARNING: could not update the record of batch {batch_id}")
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
    stale = stale_verdicts(verdict_rows)
    if not stale:
        return
    models = sorted({r["model"] for r in stale})
    _refuse_stale(
        f"{len(stale)}/{len(verdict_rows)} verdicts were produced by a DIFFERENT judge "
        f"prompt than the current one (models: {', '.join(models)}). The score would say "
        "nothing about the current prompt — rerun `verify` first.",
        allow_stale,
    )


def _fresh_verdicts(args: argparse.Namespace) -> tuple[sqlite3.Connection, list[dict], list[dict]]:
    """The run's verdicts and the golden set, for the eval commands that score
    verdicts; refuses stale rows (unless --allow-stale) and names the judge."""
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
    print(f"judge model(s): {', '.join(sorted({r['model'] for r in verdict_rows}))}")
    return conn, golden, verdict_rows


def _save_scored(args: argparse.Namespace, verdict_rows: list[dict], fields: dict) -> None:
    """With --save, write the score beside what produced it: the run, the
    models and the stamps on the rows scored (under --allow-stale, not the
    current code's)."""
    if not args.save:
        return
    record = {
        "run_id": args.run_id,
        "models": sorted({row["model"] for row in verdict_rows}),
        "prompt_hashes": sorted({row["prompt_hash"] for row in verdict_rows}, key=str),
        **fields,
    }
    args.save.write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved {args.save}")


def cmd_eval_verdict(args: argparse.Namespace) -> None:
    _, golden, verdict_rows = _fresh_verdicts(args)
    score = score_verdicts(verdict_rows, golden)
    print(score.summary())
    for expected, predicted_counts in score.confusion.items():
        wrong = {p: n for p, n in predicted_counts.items() if n and p != expected}
        if wrong:
            print(f"confused {expected} -> {wrong}")
    _save_scored(
        args,
        verdict_rows,
        {
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


def cmd_eval_injection(args: argparse.Namespace) -> None:
    conn, golden, verdict_rows = _fresh_verdicts(args)
    chunks_by_id = {
        row["id"]: {"text": row["text"], "kind": row["kind"], "figure_id": row["figure_id"]}
        for row in conn.execute(
            "SELECT id, text, kind, figure_id FROM chunks WHERE run_id = ?", (args.run_id,)
        )
    }
    score = score_injection(verdict_rows, golden, chunks_by_id)
    for outcome in [*score.attacks, *score.controls]:
        raw = f" (raw {outcome['raw']})" if outcome["raw"] != outcome["final"] else ""
        print(
            f"{outcome['id']:>4} {outcome['status']:<12} expected {outcome['expected']}, "
            f"final {outcome['final']}{raw} — {outcome['claim'][:70]}"
        )
        if outcome["status"] == "CROSS-PAIRED":
            print(f"       paired with: {outcome['row_claim'][:70]}")
    print(score.summary())
    _save_scored(args, verdict_rows, score.as_dict())


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


def _add_eval_args(
    parser: argparse.ArgumentParser,
    golden: Path = GOLDEN_PATH,
    golden_help: str = "the dev set",
    baseline: bool = True,
) -> None:
    parser.add_argument("run_id")
    parser.add_argument(
        "--golden",
        type=Path,
        default=golden,
        help=f"Golden JSONL to score against (default: {golden_help})",
    )
    if baseline:
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

    ingest = subparsers.add_parser(
        "ingest", help="Ingest PDFs (or, with --image, images) into a run"
    )
    ingest.add_argument("files", nargs="+", type=Path)
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
        help=(
            "Poll this already-submitted batch (the id `verify` printed) instead of a new one, "
            "on the evidence it was sent; its answers are read again at no cost, but items "
            "that failed are retried once each at full price"
        ),
    )
    verify.add_argument(
        "--pay-again",
        action="store_true",
        help=(
            "Judge anew (batch or --sync) even though a batch for this run and judge was "
            "never stored; its answers are paid for again"
        ),
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
    _add_eval_args(
        eval_injection,
        golden=INJECTION_GOLDEN_PATH,
        golden_help="evals/injection/injection.jsonl",
        baseline=False,
    )
    eval_injection.add_argument(
        "--save", type=Path, help="Also write the per-attack outcomes to this JSON file"
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
