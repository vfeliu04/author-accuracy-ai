"""The prompt-injection eval set (evals/injection/) is generated, reproducible
and internally consistent: what the scorer assumes about it holds offline."""

import importlib.util
import re
from pathlib import Path

import pytest
from pypdf import PdfReader

from authorai.evals import load_golden

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
    golden = load_golden(SET_DIR / "injection.jsonl")
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


def _planted_texts():
    """Every planted text of the main set (the generator's strings carrying a
    reference code) and of the reader probe."""
    import ast

    generator = SET_DIR / "make_injection_set.py"
    main_set = [
        node.value
        for node in ast.walk(ast.parse(generator.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "LVWA" in node.value
    ]
    spec = importlib.util.spec_from_file_location("reader_probe", SET_DIR / "reader_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert len(main_set) >= 10 and len(probe.PAYLOADS) == 7  # every attack's text was found
    return [*main_set, *probe.PAYLOADS]


def _policy(name):
    from authorai.credibility import METADATA_SYSTEM
    from authorai.ingest import IMAGE_READER_SYSTEM
    from authorai.verification import VERDICT_SYSTEM

    return {
        "judge": VERDICT_SYSTEM,
        "image readers": IMAGE_READER_SYSTEM,
        "metadata": METADATA_SYSTEM,
    }[name]


@pytest.mark.parametrize("name", ["judge", "image readers", "metadata"])
def test_no_payload_wording_in_a_policy(name):
    """A policy states principles; it must not quote a planted text, or the
    sets would measure phrase-matching instead of the rule."""

    def words(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    policy_words = words(_policy(name))
    policy = " ".join(policy_words)
    assert "lvwa" not in policy_words and "fwt" not in policy_words
    for payload in _planted_texts():
        tokens = words(payload)
        for start in range(len(tokens) - 4):
            shingle = " ".join(tokens[start : start + 5])
            assert shingle not in policy, (shingle, payload)
