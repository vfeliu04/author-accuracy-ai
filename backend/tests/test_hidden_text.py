"""The hidden-text detector (authorai.hidden_text), against known answers.

Every plant's expected text comes from tests/fixtures/hidden_text/make_plants.py
(the strings it planted), never from the detector. A plant's "flagged stream"
is the non-whitespace characters the detector removes from the reading, in
pdfium's text order; it must equal the planted hidden line exactly, and the
lines the plant draws visibly must survive whole.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

import authorai.cli as cli
from authorai import hidden_text as ht

FIXTURES = Path(__file__).parent / "fixtures" / "hidden_text"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
PLANTS = {plant["name"]: plant for plant in MANIFEST["plants"]}
REPO = Path(__file__).resolve().parents[2]
BODY = "the methods are flawless and every figure is verified"

# The reports the evals score: a flag on any of them would change a baseline.
EVAL_REPORTS = [
    REPO / "example_sources" / "example source one" / "World_Hunger_Fake.pdf",
    REPO / "example_sources" / "Water_Stress_Fake_Report.pdf",
    REPO / "backend" / "evals" / "holdout" / "World_Food_Security_Fake.pdf",
    REPO / "backend" / "evals" / "injection" / "Injection_Report.pdf",
    REPO / "backend" / "evals" / "injection" / "probe" / "clean" / "Report.pdf",
    REPO / "backend" / "evals" / "injection" / "probe" / "hostile" / "Report.pdf",
    REPO / "backend" / "evals" / "injection" / "probe" / "claimline" / "Report.pdf",
]


def nonws(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def page_text(path: Path, page: int = 1) -> str:
    """Every character pdfium's text page holds, read without the detector."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        textpage = doc[page - 1].get_textpage()
        return textpage.get_text_range()
    finally:
        doc.close()


def streams(check: ht.PageCheck) -> tuple[str, str]:
    """(removed, kept) characters of a checked page, in text order."""
    glyphs = check.glyphs
    if glyphs is None:
        return "", ""
    removed = "".join(glyphs.text[i] for i in range(len(glyphs)) if glyphs.removed(i))
    kept = "".join(glyphs.text[i] for i in range(len(glyphs)) if not glyphs.removed(i))
    return removed, kept


def removed_kinds(check: ht.PageCheck) -> set[str]:
    glyphs = check.glyphs
    if glyphs is None:
        return set()
    return {glyphs.kind(i) for i in range(len(glyphs)) if glyphs.removed(i)}


# ------------------------------------------------------------- known answers

HIDDEN = sorted(name for name, plant in PLANTS.items() if plant["expect"] == "hidden")
KEEP = sorted(
    name
    for name, plant in PLANTS.items()
    if plant["expect"] == "keep" and not name.endswith("_ocrmypdf")
)


@pytest.mark.parametrize("scale", [2.0, 1.5])
@pytest.mark.parametrize("name", HIDDEN)
def test_a_hidden_plant_loses_exactly_its_planted_line(name, scale):
    plant = PLANTS[name]
    report = ht.check_pdf(FIXTURES / f"{name}.pdf", scale=scale)
    assert report.unmeasured == ()
    [check] = report.pages
    removed, kept = streams(check)
    assert removed == nonws(plant["hidden"]), f"{name}: removed {removed!r}"
    if isinstance(plant["kind"], list):  # each removed character is one of these kinds
        assert removed_kinds(check) <= set(plant["kind"])
    else:
        assert removed_kinds(check) == {plant["kind"]}
    assert check.hidden_chars == len(nonws(plant["hidden"]))
    assert check.ocr_layer_kept == plant.get("ocr_kept", 0)
    # What the plant draws visibly is never removed (kept holds it whole).
    for line in plant["visible"]:
        assert nonws(line) in nonws(page_text(FIXTURES / f"{name}.pdf"))
        assert nonws(line) in kept, f"{name}: lost visible text {line!r}"


@pytest.mark.parametrize("scale", [2.0, 1.5])
@pytest.mark.parametrize("name", KEEP)
def test_a_legitimate_look_alike_is_kept_whole(name, scale):
    plant = PLANTS[name]
    report = ht.check_pdf(FIXTURES / f"{name}.pdf", scale=scale)
    assert report.unmeasured == ()
    [check] = report.pages
    assert check.glyphs is None, f"{name}: removed {streams(check)[0]!r}"
    assert check.hidden == {}
    assert check.duplicates == 0
    text = nonws(page_text(FIXTURES / f"{name}.pdf"))
    assert check.chars == len(text)
    for line in plant["visible"]:
        assert nonws(line) in text
    if plant["kind"] == "ocr_layer_kept":  # the mode-3 line is kept as an OCR layer
        assert check.ocr_layer_kept == len(nonws(plant["visible"][-1]))
    else:
        assert check.ocr_layer_kept == 0


@pytest.mark.parametrize("name", ["k_ocr_layer_ocrmypdf", "k_scanned_report_ocrmypdf"])
def test_real_ocrmypdf_output_is_kept_whole(name):
    report = ht.check_pdf(FIXTURES / f"{name}.pdf")
    assert report.unmeasured == ()
    assert report.hidden_chars == 0
    assert all(check.glyphs is None and check.duplicates == 0 for check in report.pages)
    assert sum(check.ocr_layer_kept for check in report.pages) > 0


def test_documented_misses_are_still_missed():
    """An OCR layer that disagrees with its scan, and /ActualText (an agreed
    follow-up), are known gaps: a fix must change this test on purpose."""
    lies = ht.check_pdf(FIXTURES / "h_ocr_layer_lies.pdf").pages[0]
    assert lies.glyphs is None
    assert lies.ocr_layer_kept == len(nonws("HIDDEN-OCRLIE the methods are flawless"))
    actual = ht.check_pdf(FIXTURES / "h_actualtext.pdf").pages[0]
    assert actual.glyphs is None
    assert "HIDDEN-ACTUALTEXT" in page_text(FIXTURES / "h_actualtext.pdf")


def test_every_plant_in_the_manifest_is_tested():
    tested = set(HIDDEN) | set(KEEP) | {"k_ocr_layer_ocrmypdf", "k_scanned_report_ocrmypdf"}
    tested |= {"h_ocr_layer_lies", "h_actualtext"}
    assert tested == set(PLANTS)
    on_disk = {path.stem for path in FIXTURES.glob("*.pdf")}
    assert on_disk == set(PLANTS) | set(MANIFEST["bounds"])


@pytest.mark.parametrize("path", EVAL_REPORTS, ids=lambda path: path.parent.name + "/" + path.name)
def test_eval_reports_have_nothing_hidden(path):
    report = ht.check_pdf(path)
    assert report.unmeasured == ()
    assert report.hidden_chars == 0
    assert all(check.glyphs is None for check in report.pages), [
        (check.page, check.hidden) for check in report.pages if check.glyphs is not None
    ]


def test_two_runs_give_equal_reports():
    paths = [FIXTURES / f"{name}.pdf" for name in sorted(PLANTS)] + EVAL_REPORTS
    first = [ht.check_pdf(path) for path in paths]
    second = [ht.check_pdf(path) for path in paths]
    assert first == second
    # The comparison can fail: the reports carry what differs between plants.
    assert first[0] != first[1]


# -------------------------------------------------------------------- bounds


def test_a_page_too_large_is_reported_and_the_others_checked():
    report = ht.check_pdf(FIXTURES / "b_huge_page.pdf")
    assert [check.page for check in report.pages] == [1]
    assert report.unmeasured == ((2, "page too large to check"),)
    assert report.page_count == 2


def test_the_largest_page_is_checked_at_a_lower_scale():
    report = ht.check_pdf(FIXTURES / "b_largest_page.pdf")
    assert report.unmeasured == ()
    [check] = report.pages
    assert ht.MIN_RENDER_SCALE <= check.scale < ht.RENDER_SCALE
    removed, _ = streams(check)
    assert removed == nonws("HIDDEN-BIG " + BODY)


def test_too_many_overlapping_objects_leave_the_page_unchecked():
    report = ht.check_pdf(FIXTURES / "b_overlap_65.pdf")
    assert report.pages == ()
    assert report.unmeasured == ((1, "too many overlapping text objects"),)


def test_a_page_that_fails_to_render_is_reported_and_the_others_checked(monkeypatch):
    real = ht._render

    def render(page, scale, *, number):
        if number == 2:
            raise RuntimeError("planted render failure")
        return real(page, scale, number=number)

    monkeypatch.setattr(ht, "_render", render)
    report = ht.check_pdf(FIXTURES / "b_three_pages.pdf")
    assert [check.page for check in report.pages] == [1, 3]
    assert report.unmeasured == ((2, "the page could not be rendered"),)


def test_a_page_with_too_much_text_is_reported(monkeypatch):
    monkeypatch.setattr(ht, "PAGE_MAX_CHARS", 10)
    report = ht.check_pdf(FIXTURES / "h_white_on_white.pdf")
    assert report.unmeasured == ((1, "too much text on the page"),)


def test_glyphs_past_the_result_budget_leave_their_page_unchecked(monkeypatch):
    pikepdf = pytest.importorskip("pikepdf")
    merged = pikepdf.new()
    for name in ("h_white_on_white", "h_mode3_plain"):
        with pikepdf.open(FIXTURES / f"{name}.pdf") as part:
            merged.pages.extend(part.pages)
    buf = io.BytesIO()
    merged.save(buf)
    both = buf.getvalue()
    whole = ht.check_pdf(both)
    first = len(whole.pages[0].glyphs)
    monkeypatch.setattr(ht, "RESULT_MAX_GLYPHS", first)
    report = ht.check_pdf(both)
    assert [check.page for check in report.pages] == [1]
    assert report.unmeasured == ((2, "result too large"),)


@pytest.mark.parametrize("name", ["b_giant_glyph", "b_huge_cropbox"])
def test_a_hostile_size_is_checked_within_the_bounds(name):
    """One glyph far larger than the page, or a CropBox far larger than the
    MediaBox, once took 15 s and 1.7 GB, or 2.8 GB, for one page: the check
    now completes in the bounded child (deadline and memory bound)."""
    report = ht.detect_hidden_text(FIXTURES / f"{name}.pdf", timeout=60)
    assert report.unmeasured == ()
    assert [check.page for check in report.pages] == [1]


def test_a_huge_cropbox_reads_the_page_pdfium_draws():
    [check] = ht.check_pdf(FIXTURES / "b_huge_cropbox.pdf").pages
    assert check.glyphs is None  # the visible sentence lies inside the drawn page


def test_text_too_dense_to_compare_leaves_its_page_unchecked():
    started = time.perf_counter()
    report = ht.check_pdf(FIXTURES / "b_dense_stack.pdf")
    assert report.unmeasured == ((1, ht.TOO_DENSE),)
    assert time.perf_counter() - started < 15  # was 55 s, quadratic


def test_many_small_text_objects_are_checked_in_reasonable_time():
    started = time.perf_counter()
    report = ht.check_pdf(FIXTURES / "b_one_object_per_char.pdf")
    assert report.unmeasured == ()
    assert time.perf_counter() - started < 15  # was 25 s, a set rebuilt per character


def test_many_page_sized_glyphs_leave_their_page_unchecked_quickly():
    """300 glyphs each about the size of the page cost minutes of per-pixel
    work and timed out the whole document; the page's pixel work is bounded."""
    started = time.perf_counter()
    report = ht.check_pdf(FIXTURES / "b_many_page_glyphs.pdf")
    assert report.unmeasured == ((1, ht.TOO_MUCH_AREA),)
    assert time.perf_counter() - started < 15


def test_a_page_sized_glyph_on_the_largest_page_stays_under_the_memory_bound():
    from authorai.references import READER_MEMORY_BYTES

    report = ht.detect_hidden_text(FIXTURES / "b_page_glyph_memory.pdf", timeout=120)
    assert report.unmeasured == ()
    assert report.peak_memory_bytes is not None
    assert report.peak_memory_bytes < READER_MEMORY_BYTES


def test_the_child_removes_the_handed_over_file_before_checking(tmp_path, monkeypatch):
    """handover_file's contract: the child deletes its copy of the user's file
    as soon as it has it open, since a server killed outright never does."""
    copy = tmp_path / "handed-over.pdf"
    copy.write_bytes((FIXTURES / "h_covered_rect.pdf").read_bytes())
    seen = []
    real = ht.check_pdf

    def checking(source, **kwargs):
        seen.append(copy.exists())
        return real(source, **kwargs)

    monkeypatch.setattr(ht, "check_pdf", checking)
    report = ht._read_and_check(str(copy), None)
    assert seen == [False]
    assert report.hidden_chars == len(nonws(PLANTS["h_covered_rect"]["hidden"]))


def test_pages_can_be_chosen():
    report = ht.check_pdf(FIXTURES / "b_three_pages.pdf", pages=[3, 1])
    assert [check.page for check in report.pages] == [1, 3]
    with pytest.raises(ValueError, match="page 4"):
        ht.check_pdf(FIXTURES / "b_three_pages.pdf", pages=[4])


def test_an_encrypted_pdf_with_an_empty_password_is_read():
    report = ht.check_pdf(FIXTURES / "b_encrypted_empty_password.pdf")
    removed, _ = streams(report.pages[0])
    assert removed == nonws("HIDDEN-LOCKED " + BODY)


def test_an_unreadable_file_is_a_value_error():
    with pytest.raises(ValueError, match="could not be opened"):
        ht.check_pdf(b"%PDF-1.7 this is not a pdf")


# ---------------------------------------------------------- the bounded child


def without_timing(report: ht.HiddenTextReport) -> ht.HiddenTextReport:
    return replace(report, peak_memory_bytes=None)


@pytest.mark.parametrize("name", ["h_covered_rect", "k_fake_bold", "t_ocr_page_55"])
def test_the_child_reports_what_the_core_does(name):
    path = FIXTURES / f"{name}.pdf"
    with path.open("rb") as handle:
        bounded = ht.detect_hidden_text(handle, timeout=60)
    assert bounded == ht.check_pdf(path)  # timings and memory are not compared
    assert bounded.peak_memory_bytes is not None


def test_the_child_reads_a_path_and_bytes_too():
    path = FIXTURES / "h_covered_rect.pdf"
    by_path = ht.detect_hidden_text(path, timeout=60)
    by_bytes = ht.detect_hidden_text(path.read_bytes(), timeout=60)
    assert by_path == by_bytes == ht.check_pdf(path)


def test_the_deadline_is_a_check_error():
    with pytest.raises(ht.HiddenTextCheckError, match="longer than 0.5 seconds"):
        ht.detect_hidden_text(FIXTURES / "k_scanned_report_ocrmypdf.pdf", timeout=0.5)


def test_a_garbage_file_is_a_check_error():
    with pytest.raises(ht.HiddenTextCheckError, match="could not be opened"):
        ht.detect_hidden_text(b"%PDF-1.7 this is not a pdf", timeout=60)


def test_the_largest_page_stays_under_the_memory_bound(capsys):
    from authorai.references import READER_MEMORY_BYTES

    report = ht.detect_hidden_text(FIXTURES / "b_largest_page.pdf", timeout=120)
    assert report.unmeasured == ()
    peak = report.peak_memory_bytes
    assert peak is not None
    with capsys.disabled():
        print(f"\nlargest page: peak resident {peak / 2**20:.0f} MiB")
    assert peak < READER_MEMORY_BYTES


# -------------------------------------------------------------- version, CLI


def test_the_rules_are_pinned_to_the_version():
    """A change to any rule changes what is read: it bumps DETECTOR_VERSION
    (which R2's dedup stamp names) and this pin together."""
    digest = hashlib.sha256(json.dumps(ht.RULES, sort_keys=True).encode()).hexdigest()[:12]
    assert (ht.DETECTOR_VERSION, digest) == ("hidden-text v2", RULES_DIGEST)


RULES_DIGEST = "24bd3de2989e"


def test_the_cli_prints_each_page(capsys, monkeypatch):
    path = FIXTURES / "h_white_on_white.pdf"
    monkeypatch.setattr(sys, "argv", ["authorai", "hidden-text", str(path)])
    cli.main()
    out = capsys.readouterr().out
    assert "page 1:" in out
    planted = nonws(PLANTS["h_white_on_white"]["hidden"])
    assert f"same_colour_as_background {len(planted)} " in out
    assert repr(planted) in out


def test_the_cli_refuses_a_garbage_file(tmp_path, monkeypatch):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 this is not a pdf")
    monkeypatch.setattr(sys, "argv", ["authorai", "hidden-text", str(bad)])
    with pytest.raises(SystemExit, match="could not be opened"):
        cli.main()


# ------------------------------------------------------------------ fixtures


def test_the_committed_fixtures_are_the_generators(tmp_path):
    """Rebuilt into a temporary folder, every file the generator makes holds
    the same objects as the committed one, compared decoded (file bytes can
    differ by zlib build)."""
    pikepdf = pytest.importorskip("pikepdf")
    result = subprocess.run(
        [sys.executable, str(FIXTURES / "make_plants.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    rebuilt = json.loads((tmp_path / "manifest.json").read_text())
    assert rebuilt == MANIFEST
    made = sorted(path.name for path in tmp_path.glob("*.pdf"))
    assert made == sorted(
        f"{name}.pdf" for name in set(PLANTS) | set(MANIFEST["bounds"]) if "ocrmypdf" not in name
    )
    for name in made:
        assert _decoded(pikepdf, tmp_path / name) == _decoded(pikepdf, FIXTURES / name), name


def _decoded(pikepdf, path: Path):
    with pikepdf.open(path) as pdf:
        return _walk(pikepdf, pdf.Root, set())


def _walk(pikepdf, obj, seen):
    if isinstance(obj, pikepdf.Stream):
        # /Length and the filter describe the encoding, not the content.
        keys = {k: v for k, v in obj.items() if k not in ("/Length", "/Filter", "/DecodeParms")}
        return ("stream", _walk(pikepdf, keys, seen), obj.read_bytes())
    if isinstance(obj, pikepdf.Dictionary):
        key = obj.objgen if obj.is_indirect else None
        if key and key in seen:
            return ("ref",)
        if key:
            seen.add(key)
        return {k: _walk(pikepdf, v, seen) for k, v in sorted(obj.items()) if k != "/Parent"}
    if isinstance(obj, dict):
        return {k: _walk(pikepdf, v, seen) for k, v in sorted(obj.items()) if k != "/Parent"}
    if isinstance(obj, pikepdf.Array):
        return [_walk(pikepdf, v, seen) for v in obj]
    return repr(obj)


# ------------------------------------------------------- real PDFs (local)

EXAMPLES = REPO / "example_sources"
GHI_2024 = EXAMPLES / "example source one" / "2024 Global Wolrd Hunger Index.pdf"
HH = EXAMPLES / "example source one" / "HH_Nov24-May25_FINAL.pdf"


def _page(path: Path, number: int) -> ht.PageCheck:
    [check] = ht.check_pdf(path, pages=[number]).pages
    return check


@pytest.mark.integration
def test_real_pages_the_research_read_by_render():
    """Pages the research rendered with pdfium and Ghostscript and read by
    eye (author_ai/report-side-research/hidden-text/shots): the hidden copy
    is removed, the copy a reader sees is kept."""
    # GHI 2024 p48: last edition's header lies under this edition's.
    removed, kept = streams(_page(GHI_2024, 48))
    assert "2000,2005,2010,AND2019GLOBALHUNGERINDEXSCORES,ANDCHANGESINCE2000" in removed
    assert "2000,2008,2016,AND2024GLOBALHUNGERINDEXSCORES" in kept
    # GHI 2024 p23: a clipped copy of a figure's title beside the visible one.
    removed, kept = streams(_page(GHI_2024, 23))
    assert "FIGURE1.5GENDERINEQUALITYGOESHANDINHANDWITHHUNGER" in removed
    assert "GENDERINEQUALITYGOESHANDINHANDWITHHUNGER" in kept
    # HH p14: a stale "19.0" under the visible "14.6".
    removed, kept = streams(_page(HH, 14))
    assert removed == "19.0"
    assert "14.6" in kept


@pytest.mark.integration
def test_every_example_pdf_is_checked_whole(capsys):
    """Every page of every example PDF is checked (none left unchecked), and
    the fake reports have nothing hidden. Totals and timings are printed as
    information, not asserted."""
    lines = []
    for path in sorted(EXAMPLES.rglob("*.pdf")):
        report = ht.check_pdf(path)
        assert report.unmeasured == (), (path.name, report.unmeasured)
        if "Fake" in path.name:
            assert report.hidden_chars == 0, path.name
        seconds = sum(check.seconds for check in report.pages)
        lines.append(
            f"{path.name[:44]:44s} {report.page_count:3d} pages {seconds:6.1f} s "
            f"hidden {report.hidden_chars:5d} on {len(report.pages_with_hidden)} pages"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))


def test_the_document_is_reopened_every_few_pages(monkeypatch):
    """pdfium keeps what it loaded until the document is closed, so a long
    scan read in one opening grew past the memory bound: check_pdf opens it
    once to count the pages, then once per PAGES_PER_OPEN pages."""
    opened = []
    real = ht._open

    def counting(source):
        opened.append(source)
        return real(source)

    path = FIXTURES / "b_three_pages.pdf"
    whole = ht.check_pdf(path)
    monkeypatch.setattr(ht, "_open", counting)
    monkeypatch.setattr(ht, "PAGES_PER_OPEN", 2)
    assert ht.check_pdf(path) == whole
    assert len(opened) == 1 + 2
