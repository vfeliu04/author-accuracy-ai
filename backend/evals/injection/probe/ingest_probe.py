"""Ingest one build of the probe as a run, through the app's own ingest step.

The PDF and the fares board are uploads. The web page and the captions are
link sources, and the app reads those only from the network. So this script
runs the real jobs.step_ingest with the two network readers replaced:

    jobs.fetch_url           returns the build's HTML file as the page served
                             at make_probe_set.WEB_URL;
    jobs.read_video_bounded  hands the build's caption file to the real
                             video.read_video, as yt-dlp would have.

Everything after the read is the app's code: the page reader in its bounded
child process, the cap (make_probe_set.WEB_MAX_CHARS, which makes both link
sources "read in part"), the stored snapshot, chunking, embedding, the image
readers. Nothing here is a product entry point: a way to ingest a local file
under a claimed address would be a way around the fetch guards.

usage, from backend/, with the code to measure first on PYTHONPATH:
    python evals/injection/probe/ingest_probe.py BUILD [--dry DIR]

BUILD      clean | hostile | claimline (make_probe_set.BUILDS)
--dry DIR  a scratch database in DIR with a stand-in embedder and stand-in
           image readers: no key, no network, no cost. Shows what a real run
           would hold.

Prints `run: <id>` first, like `authorai.cli ingest`, then what the run holds,
and exits non-zero when that is not what the set was sized for.
"""

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_NAME = "Pellmere_on_Two_Wheels.pdf"
SOURCE_NAME = "Pellmere_Operations_Statement_2025.pdf"
BOARD_NAME = "pellmere_fares_board.png"


def generator():
    spec = importlib.util.spec_from_file_location("make_probe_set", HERE / "make_probe_set.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalYDL:
    """Stands in for yt_dlp.YoutubeDL inside video.read_video: answers with
    the build's video record and serves its caption file."""

    def __init__(self, info: dict, captions: bytes):
        self.info, self.captions = info, captions

    def __call__(self, _options):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True, process=True):
        return json.loads(json.dumps(self.info))

    def urlopen(self, url):
        return io.BytesIO(self.captions)


class _StandInReader:
    """--dry only: answers every image call with a fixed line."""

    def describe_image(self, **_kwargs) -> str:
        return "A stand-in reading (dry run)."


def files_of(build: str, probe) -> dict[str, Path]:
    """The files a run of `build` ingests. The claimline run reads the clean
    PDF beside its own board and has no link source."""
    folder = HERE / build
    if build == "claimline":
        return {
            "report": folder / probe.REPORT_PDF,
            "pdf": HERE / "clean" / probe.SOURCE_PDF,
            "board": folder / probe.BOARD_PNG,
        }
    return {
        "report": folder / probe.REPORT_PDF,
        "pdf": folder / probe.SOURCE_PDF,
        "board": folder / probe.BOARD_PNG,
        "page": folder / probe.PAGE_HTML,
        "captions": folder / probe.CAPTIONS,
    }


def ingest(context, build: str, probe) -> str:
    """Create the run and its uploads, then run the app's ingest step with the
    two network readers replaced. Returns the run id."""
    from authorai import db as dbmod
    from authorai import jobs as jobsmod
    from authorai import video as videomod
    from authorai.fetch import FetchedResponse

    conn, settings = context.conn, context.settings
    files = files_of(build, probe)
    for path in files.values():
        if not path.is_file():
            raise SystemExit(f"{path} is missing: build the set first (make_probe_set.py)")
    run_id = dbmod.create_run(conn)
    print(f"run: {run_id}", flush=True)
    report = dbmod.add_upload(conn, "REPORT", REPORT_NAME, str(files["report"]))
    sources = [
        dbmod.add_upload(conn, "SOURCE", SOURCE_NAME, str(files["pdf"])),
        dbmod.add_upload(conn, "SOURCE", BOARD_NAME, str(files["board"]), source_type="image"),
    ]
    if "page" in files:
        for url, source_type in ((probe.WEB_URL, "web"), (probe.VIDEO_URL, "youtube")):
            planned = Path(settings.uploads_dir) / f"{dbmod.new_id()}.json"
            sources.append(
                dbmod.add_upload(
                    conn, "SOURCE", url, str(planned), source_type=source_type, url=url
                )
            )
        page = files["page"].read_bytes()
        captions = files["captions"].read_bytes()

        def fetch_url(url, _settings, **_kwargs):
            if url != probe.WEB_URL:
                raise SystemExit(f"the probe has no page at {url}")
            return FetchedResponse(
                url=url,
                final_url=url,
                content_type="text/html",
                charset="utf-8",
                body=page,
                is_pdf=False,
            )

        def read_video_bounded(video_id, *, timeout, max_chars):
            if video_id != probe.VIDEO_ID:
                raise SystemExit(f"the probe has no video {video_id}")
            return videomod.read_video(
                video_id, max_chars=max_chars, ydl_class=LocalYDL(probe.VIDEO_INFO, captions)
            )

        # The two names jobs.py reads the network through. A rename there must
        # fail here, never fall through to a real fetch.
        for name in ("fetch_url", "read_video_bounded"):
            if not hasattr(jobsmod, name):
                raise SystemExit(
                    f"authorai.jobs no longer has {name}: the stand-in would not apply"
                )
        jobsmod.fetch_url = fetch_url
        jobsmod.read_video_bounded = read_video_bounded
    label = jobsmod.step_ingest(
        context, run_id, {"report_upload_id": report, "source_upload_ids": sources}
    )
    print(label)
    return run_id


def holdings(conn, run_id: str) -> dict:
    """What the run holds, as the judge's retrieval will find it."""
    documents = conn.execute(
        "SELECT d.id, d.kind, d.title, d.metadata, u.source_type FROM documents d"
        " LEFT JOIN uploads u ON u.id = d.upload_id WHERE d.run_id = ? ORDER BY d.rowid",
        (run_id,),
    ).fetchall()
    rows = []
    for document in documents:
        chunks = conn.execute(
            "SELECT kind, section, length(text) AS chars, start_seconds FROM chunks"
            " WHERE doc_id = ? ORDER BY id",
            (document["id"],),
        ).fetchall()
        provenance = (json.loads(document["metadata"] or "{}")).get("provenance") or {}
        rows.append(
            {
                "kind": document["kind"],
                "source_type": document["source_type"] or "pdf",
                "title": document["title"],
                "chunks": [
                    (chunk["kind"], chunk["section"], chunk["chars"], chunk["start_seconds"])
                    for chunk in chunks
                ],
                "truncated": provenance.get("truncated"),
                "captions": (provenance.get("video") or {}).get("captions"),
            }
        )
    figures = conn.execute(
        "SELECT f.image_path FROM figures f JOIN documents d ON d.id = f.doc_id"
        " WHERE d.run_id = ? ORDER BY f.rowid",
        (run_id,),
    ).fetchall()
    return {"documents": rows, "figures": [row["image_path"] for row in figures]}


def check(held: dict, build: str, probe) -> list[str]:
    """Why this run is not what the set was sized for (empty when it is)."""
    problems = []
    sources = [document for document in held["documents"] if document["kind"] == "SOURCE"]
    chunks = sum(len(document["chunks"]) for document in sources)
    links = build != "claimline"
    expected = probe.EXPECTED_SOURCE_CHUNKS if links else 4
    if chunks != expected:
        problems.append(f"{chunks} SOURCE chunks, expected {expected}")
    if len(held["figures"]) != probe.EXPECTED_FIGURES:
        problems.append(f"{len(held['figures'])} pictures, expected {probe.EXPECTED_FIGURES}")
    kinds = sorted(document["source_type"] for document in sources)
    wanted = ["image", "pdf", "web", "youtube"] if links else ["image", "pdf"]
    if kinds != wanted:
        problems.append(f"source kinds {kinds}, expected {wanted}")
    for document in sources:
        if document["source_type"] in ("web", "youtube"):
            if len(document["chunks"]) != 2:
                problems.append(f"{document['source_type']}: {len(document['chunks'])} chunks kept")
            if not document["truncated"]:
                problems.append(f"{document['source_type']}: not read in part")
        if document["source_type"] == "youtube" and document["captions"] != {
            "kind": "automatic",
            "language": "en",
        }:
            problems.append(f"captions read as {document['captions']}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ingest one build of the probe as a run.")
    parser.add_argument("build", choices=("clean", "hostile", "claimline"))
    parser.add_argument("--dry", type=Path, help="scratch folder: no key, no network, no cost")
    args = parser.parse_args(argv)
    probe = generator()
    if args.build not in probe.builds():
        raise SystemExit(f"nothing to ingest for {args.build!r}: payloads.py holds no text for it")

    from authorai import db as dbmod
    from authorai.config import Settings
    from authorai.jobs import PipelineContext

    if args.dry:
        from authorai.embeddings import FakeEmbedder

        args.dry.mkdir(parents=True, exist_ok=True)
        settings = Settings(
            anthropic_api_key="dry",
            openai_api_key="dry",
            db_path=args.dry / "probe.db",
            figures_dir=args.dry / "figures",
            uploads_dir=args.dry / "uploads",
            embedding_dim=8,
            web_max_chars=probe.WEB_MAX_CHARS,
        )
        context = PipelineContext(dbmod.connect(settings.db_path, settings.embedding_dim), settings)
        context._embedder = FakeEmbedder(dim=8)  # type: ignore[assignment]
        context._llm = _StandInReader()  # type: ignore[assignment]
    else:
        settings = Settings(web_max_chars=probe.WEB_MAX_CHARS)
        context = PipelineContext(dbmod.connect(settings.db_path, settings.embedding_dim), settings)
    run_id = ingest(context, args.build, probe)
    held = holdings(context.conn, run_id)
    for document in held["documents"]:
        print(f"{document['kind']:6} {document['source_type']:8} {document['title']!r}")
        for kind, section, chars, start in document["chunks"]:
            where = f" at {start:.0f}s" if start is not None else ""
            print(f"         {kind:6} {chars:5} chars  {section!r}{where}")
        if document["truncated"]:
            print(f"         read in part: {document['truncated']}")
    for picture in held["figures"]:
        print(f"picture: {picture}")
    problems = check(held, args.build, probe)
    for problem in problems:
        print(f"NOT AS SIZED: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
