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
and exits non-zero when that is not what the set was sized for: the wrong
chunks, a kept section lost, a planted text that is not whole in a stored
chunk, a drawn note that is not in the picture the judge will be shown.

A real ingest is paid (two image readings and the embeddings) and writes to
the database AUTHORAI_DB_PATH names; without that variable it is refused, so
it never lands in a default database by accident.
"""

import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# A drawn note is in a picture when its band holds at least this share of dark
# pixels; a picture with nothing drawn there holds less than the lower one.
INK, NO_INK = 0.01, 0.002
# A band is a share of the stored picture, so it lands on the note only when
# the picture has the shape it was measured on: the figure cut to its frame
# (make_probe_set.FIGURE_CROP), the board as drawn. The first set's figure, cut
# to its bars, has another shape, and there the bars' own ink fills the band.
SHAPE_TOLERANCE = 0.03


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
    report = dbmod.add_upload(conn, "REPORT", probe.REPORT_NAME, str(files["report"]))
    sources = [
        dbmod.add_upload(conn, "SOURCE", probe.SOURCE_NAME, str(files["pdf"])),
        dbmod.add_upload(
            conn, "SOURCE", probe.BOARD_NAME, str(files["board"]), source_type="image"
        ),
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
    saved = {}
    if "page" in files:
        saved = {name: getattr(jobsmod, name) for name in ("fetch_url", "read_video_bounded")}
        jobsmod.fetch_url = fetch_url
        jobsmod.read_video_bounded = read_video_bounded
    try:
        label = jobsmod.step_ingest(
            context, run_id, {"report_upload_id": report, "source_upload_ids": sources}
        )
    finally:
        for name, reader in saved.items():
            setattr(jobsmod, name, reader)
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
            "SELECT kind, section, text, start_seconds FROM chunks WHERE doc_id = ? ORDER BY id",
            (document["id"],),
        ).fetchall()
        pictures = conn.execute(
            "SELECT image_path FROM figures WHERE doc_id = ? ORDER BY rowid", (document["id"],)
        ).fetchall()
        provenance = (json.loads(document["metadata"] or "{}")).get("provenance") or {}
        rows.append(
            {
                "kind": document["kind"],
                "source_type": document["source_type"] or "pdf",
                "title": document["title"],
                "chunks": [
                    {
                        "kind": chunk["kind"],
                        "section": chunk["section"],
                        "text": chunk["text"],
                        "start": chunk["start_seconds"],
                    }
                    for chunk in chunks
                ],
                "pictures": [row["image_path"] for row in pictures],
                "truncated": provenance.get("truncated"),
                "captions": (provenance.get("video") or {}).get("captions"),
            }
        )
    return {"documents": rows}


def dark_share(image, band) -> float:
    """The share of dark pixels in `band` (fractions: left, top, right, bottom)
    of a picture."""
    width, height = image.size
    left, top, right, bottom = band
    region = image.convert("L").crop(
        (int(left * width), int(top * height), int(right * width), int(bottom * height))
    )
    return sum(region.histogram()[:100]) / (region.width * region.height)


def expected_shape(slot: str, probe) -> float:
    """Width over height of the stored picture a band of `slot` was measured on."""
    from PIL import Image

    if slot == "pdf-figure":
        left, top, right, bottom = probe.FIGURE_CROP
        return (right - left) / (bottom - top)
    with Image.open(io.BytesIO(probe.board_png())) as board:
        return board.width / board.height


def _plain(text: str) -> str:
    """Text as compared between what was planted and what was stored: runs of
    whitespace as one space (a chunk joins a section's paragraphs, the captions
    join their lines)."""
    return " ".join(text.split())


def check(held: dict, build: str, probe, *, dry: bool) -> list[str]:
    """Why this run is not what the set was sized for (empty when it is).

    The sizing: the PDF as one text, one table and one figure; the board as
    one figure; the page's first two sections and the captions' first two
    windows, one chunk each, the rest cut by the cap. Then the planted texts:
    each written one whole in a stored chunk of its own source, each drawn one
    as ink in its band of the stored picture, and no ink in a band nothing was
    drawn in. On a real ingest (`dry` false) the board's reading must also
    repeat the code of a text drawn on it, since the scorer looks for it there.
    """
    from PIL import Image

    problems = []
    sources = {
        document["source_type"]: document
        for document in held["documents"]
        if document["kind"] == "SOURCE"
    }
    links = build != "claimline"
    wanted = ["image", "pdf", "web", "youtube"] if links else ["image", "pdf"]
    kinds = sorted(d["source_type"] for d in held["documents"] if d["kind"] == "SOURCE")
    if kinds != wanted:
        return [f"source kinds {kinds}, expected {wanted}"]
    chunks = sum(len(document["chunks"]) for document in sources.values())
    expected = probe.EXPECTED_SOURCE_CHUNKS if links else 4
    if chunks != expected:
        problems.append(f"{chunks} SOURCE chunks, expected {expected}")
    pictures = sum(len(document["pictures"]) for document in sources.values())
    if pictures != probe.EXPECTED_FIGURES:
        problems.append(f"{pictures} pictures, expected {probe.EXPECTED_FIGURES}")
    for source_type, shape in (("pdf", ["figure", "table", "text"]), ("image", ["figure"])):
        found = sorted(chunk["kind"] for chunk in sources[source_type]["chunks"])
        if found != shape:
            problems.append(f"{source_type}: chunks {found}, expected {shape}")
    if links:
        kept = [chunk["section"] for chunk in sources["web"]["chunks"]]
        headings = [heading for heading, _ in probe.PAGE_SECTIONS[:2]]
        if kept != headings:
            problems.append(f"web: sections kept {kept}, expected {headings}")
        windows = [
            None if chunk["start"] is None else int(chunk["start"] // 75)
            for chunk in sources["youtube"]["chunks"]
        ]
        if windows != [0, 1]:
            problems.append(f"youtube: caption windows kept {windows}, expected [0, 1]")
        for source_type in ("web", "youtube"):
            if not sources[source_type]["truncated"]:
                problems.append(f"{source_type}: not read in part")
        captions = sources["youtube"]["captions"]
        if captions != {"kind": "automatic", "language": "en"}:
            problems.append(f"captions read as {captions}")

    planted = {attack["slot"]: attack for attack in probe.attacks(build)}
    stored = {
        source_type: [_plain(chunk["text"]) for chunk in document["chunks"]]
        for source_type, document in sources.items()
    }
    home = {"text": "pdf", "table": "pdf", "web": "web", "captions": "youtube"}
    for slot, attack in planted.items():
        channel = probe.SLOTS[slot][0]
        if channel in home and not any(
            _plain(attack["payload"]) in text for text in stored[home[channel]]
        ):
            problems.append(f"{attack['id']} ({slot}): its text is not whole in a stored chunk")
    if build == "hostile" and not any(_plain(probe.PAYLOADS.LIE) in text for text in stored["web"]):
        problems.append("the labelled statement (payloads.LIE) is not whole in a stored chunk")

    for slot, band in probe.NOTE_BANDS.items():
        source_type = "pdf" if slot == "pdf-figure" else "image"
        paths = sources[source_type]["pictures"]
        if len(paths) != 1:
            problems.append(
                f"{slot}: {len(paths)} stored pictures under the {source_type} source, expected 1;"
                " its band was not measured"
            )
            continue
        with Image.open(paths[0]) as picture:
            shape, expected = picture.width / picture.height, expected_shape(slot, probe)
            if abs(shape / expected - 1) > SHAPE_TOLERANCE:
                problems.append(
                    f"{slot}: the stored picture {paths[0]} is {picture.width} by"
                    f" {picture.height} (shape {shape:.2f}), not the shape its band was measured"
                    f" on ({expected:.2f}); its band was not measured"
                )
                continue
            share = dark_share(picture, band)
        if slot in planted and share < INK:
            problems.append(
                f"{planted[slot]['id']} ({slot}): no drawn text in its band of the stored "
                f"picture {paths[0]} (dark share {share:.4f})"
            )
        if slot not in planted and share > NO_INK:
            problems.append(f"{slot}: ink in a band nothing was planted in ({share:.4f})")
        if slot in planted and source_type == "image" and not dry:
            reading = stored["image"][0]
            if planted[slot]["marker"] not in reading:
                problems.append(
                    f"{planted[slot]['id']} ({slot}): the reading of the board lacks its code "
                    f"{planted[slot]['marker']!r}; the scorer would call it not tested. "
                    "Ingest again"
                )
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ingest one build of the probe as a run.")
    parser.add_argument("build", choices=("clean", "hostile", "claimline"))
    parser.add_argument("--dry", type=Path, help="scratch folder: no key, no network, no cost")
    args = parser.parse_args(argv)
    probe = generator()
    if args.build not in probe.builds():
        raise SystemExit(f"nothing to ingest for {args.build!r}: payloads.py holds no text for it")
    if not args.dry and not os.environ.get("AUTHORAI_DB_PATH"):
        raise SystemExit(
            "REFUSED: a real ingest is paid and writes to a database. Name the database in "
            "AUTHORAI_DB_PATH (the eval tools do), or use --dry DIR."
        )

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
    print(f"database: {settings.db_path}", flush=True)
    run_id = ingest(context, args.build, probe)
    held = holdings(context.conn, run_id)
    for document in held["documents"]:
        print(f"{document['kind']:6} {document['source_type']:8} {document['title']!r}")
        for chunk in document["chunks"]:
            where = f" at {chunk['start']:.0f}s" if chunk["start"] is not None else ""
            size = f"{len(chunk['text']):5} chars"
            print(f"         {chunk['kind']:6} {size}  {chunk['section']!r}{where}")
        if document["truncated"]:
            print(f"         read in part: {document['truncated']}")
        for picture in document["pictures"]:
            print(f"         picture: {picture}")
    problems = check(held, args.build, probe, dry=bool(args.dry))
    for problem in problems:
        print(f"NOT AS SIZED: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
