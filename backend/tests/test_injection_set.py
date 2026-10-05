"""The prompt-injection eval set (evals/injection/) is generated, reproducible
and internally consistent: what the scorer assumes about it holds offline."""

import importlib.util
import json
import re
from pathlib import Path

from pypdf import PdfReader

SET_DIR = Path(__file__).resolve().parent.parent / "evals" / "injection"
FILES = ("Injection_Source.pdf", "Injection_Report.pdf", "injection_chart.png", "injection.jsonl")


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_injection_set", SET_DIR / "make_injection_set.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pdf_text(path: Path) -> str:
    text = " ".join(page.extract_text() for page in PdfReader(path).pages)
    return re.sub(r"\s+", " ", text)


def test_generation_is_byte_reproducible_and_matches_the_committed_golden(tmp_path):
    generator = _generator()
    generator.build(tmp_path / "a")
    generator.build(tmp_path / "b")

    for name in FILES:
        built = (tmp_path / "a" / name).read_bytes()
        assert built == (tmp_path / "b" / name).read_bytes(), name
        # The committed files are the generator's: a generator edit without a
        # rebuild, or a hand edit, would leave the eval ingesting stale files
        # (and the report's text out of step with its golden record).
        assert built == (SET_DIR / name).read_bytes(), f"{name} differs from a fresh build"


def test_every_attack_is_marked_once_and_planted_where_it_claims():
    golden = [json.loads(line) for line in (SET_DIR / "injection.jsonl").read_text().splitlines()]
    attacks = [g["attack"] for g in golden if "attack" in g]
    controls = [g["control"] for g in golden if "control" in g]
    assert len(attacks) == 10 and len(controls) == 3

    markers = [a["marker"] for a in attacks]
    assert len(set(markers)) == len(markers), "two attacks share a marker"
    for record in golden:
        if "attack" in record:
            # An attack asking for the verdict that is already right measures nothing.
            assert record["attack"]["target"] != record["expected_verdict"], record["text"]

    source = _pdf_text(SET_DIR / "Injection_Source.pdf")
    for attack in attacks:
        planted = source.count(attack["marker"])
        if attack["channel"] == "image":
            assert planted == 0  # it lives in the chart image source, not the PDF
        elif attack["channel"] == "figure":
            assert planted == 1  # the caption; the payload itself is pixels
        else:
            assert planted == 1, attack["marker"]
    for control in controls:
        assert control["marker"] in markers, "a control must sit beside a real payload"

    report = _pdf_text(SET_DIR / "Injection_Report.pdf")
    for record in golden:
        assert record["text"] in report, record["text"]
