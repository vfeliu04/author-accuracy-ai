from collections.abc import Callable

import pytest
from pydantic import BaseModel

from authorai import db as dbmod
from authorai.llm import BATCH_MAX_TOKENS, PARSE_MAX_TOKENS

DIM = 8


@pytest.fixture()
def conn(tmp_path):
    connection = dbmod.connect(tmp_path / "test.db", embedding_dim=DIM)
    yield connection
    connection.close()


def _attached_to(logger, caplog):
    """authorai loggers do not propagate to the root logger (log.setup_logger),
    so caplog's handler is attached to the module logger itself."""
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


@pytest.fixture()
def web_log(caplog):
    """caplog, capturing authorai.web's records (imported here, not at the top,
    so collecting unrelated tests does not import the page reader)."""
    import authorai.web as web_mod

    yield from _attached_to(web_mod.logger, caplog)


@pytest.fixture()
def jobs_log(caplog):
    """caplog, capturing authorai.jobs's records."""
    from authorai import jobs as jobsmod

    yield from _attached_to(jobsmod.logger, caplog)


@pytest.fixture()
def chat_log(caplog):
    """caplog, capturing authorai.chat's records."""
    from authorai import chat as chat_mod

    yield from _attached_to(chat_mod.logger, caplog)


@pytest.fixture()
def credibility_log(caplog):
    """caplog, capturing authorai.credibility's records."""
    from authorai import credibility as credibility_mod

    yield from _attached_to(credibility_mod.logger, caplog)


@pytest.fixture()
def references_log(caplog):
    """caplog, capturing authorai.references's records."""
    from authorai import references as references_mod

    yield from _attached_to(references_mod.logger, caplog)


def pdf_from_objects(objects: list[bytes]) -> bytes:
    """A real PDF file around the given object bodies, numbered from 1 with
    object 1 the catalog: header, the objects, an xref table and a trailer.
    The one place the file syntax is written, so a test that needs a page
    tree or a font pypdf must READ a particular way builds only the objects."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def pdf_with_pages(pages: list[str]) -> bytes:
    """A real, minimal PDF written from raw PDF syntax — one page per string,
    an empty string making a page with no text (a scanned page's shape).

    Hand-built rather than written by a library so the tests exercise pypdf's
    READING of an ordinary xref-table file offline, not a writer's round
    trip. Helvetica is one of the standard 14 fonts every reader must carry,
    so no font is embedded; text is Latin-1.
    """
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        content_number = 5 + 2 * i
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            + f"/Contents {content_number} 0 R >>".encode()
        )
        operators = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in text.split("\n"):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operators.append(f"({escaped}) Tj T*")
        operators.append("ET")
        stream = "\n".join(operators).encode("latin-1")
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    return pdf_from_objects(objects)


def poison_providers(monkeypatch):
    """Make any provider work during an ingest reuse a test failure — not just
    calls: CONSTRUCTING a client already means the dedup path leaked. The one
    definition both the jobs tests and the API seam test use, so the
    reuse-recomputes-nothing contract cannot silently stop being guarded in
    one of them."""
    from authorai import jobs as jobsmod

    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: pytest.fail("re-ingested"))
    monkeypatch.setattr(
        jobsmod, "ingest_snapshot", lambda *a, **k: pytest.fail("re-ingested a stored page")
    )
    monkeypatch.setattr(jobsmod, "fetch_url", lambda *a, **k: pytest.fail("fetched a link"))
    monkeypatch.setattr(
        jobsmod, "OpenAIEmbedder", lambda *a, **k: pytest.fail("constructed an embedder")
    )
    monkeypatch.setattr(
        jobsmod, "AnthropicClient", lambda *a, **k: pytest.fail("constructed an LLM client")
    )


class FakeLLM:
    """Canned LLM for tests: returns pre-set objects per output type and
    records every call so tests can assert on prompts.

    A parse_results value may be a single instance (returned every call), a
    list of instances popped in call order — verification tests need a
    different Verdict per claim — or a callable taking the prompt and
    returning the instance. The callable is for callers that parse in
    PARALLEL (the chunked reference scan): there, pop order would follow the
    pool's scheduling and a list could hand chunk 2's answer to chunk 1 on
    one run in ten, while an answer keyed on the prompt's own content is the
    same under every interleaving.
    """

    def __init__(
        self,
        parse_results: dict[type, BaseModel | list[BaseModel] | Callable[[str], BaseModel]]
        | None = None,
        image_description: str = "A fake description.",
        chat_answer: str = "A fake answer.",
    ):
        self._parse_results = parse_results or {}
        self._image_description = image_description
        self._chat_answer = chat_answer
        self.parse_calls: list[dict] = []
        self.image_calls: int = 0
        self.chat_calls: list[dict] = []

    def parse(
        self, *, model, system, prompt, output_type, max_tokens=PARSE_MAX_TOKENS, images=None
    ):
        self.parse_calls.append(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "output_type": output_type,
                "images": images,
            }
        )
        result = self._parse_results[output_type]
        if isinstance(result, list):
            return result.pop(0)
        if callable(result):
            return result(prompt)
        return result

    def parse_batch(
        self,
        *,
        model,
        items,
        max_tokens=BATCH_MAX_TOKENS,
        resume_batch_id=None,
        on_batch_created=None,
    ):
        # Mirrors AnthropicClient.parse_batch's contract: dict keyed by custom_id.
        self.resume_batch_ids = [*getattr(self, "resume_batch_ids", []), resume_batch_id]
        if on_batch_created is not None and resume_batch_id is None:
            on_batch_created("fake-batch-1")
        return {
            item.custom_id: self.parse(
                model=model,
                system=item.system,
                prompt=item.prompt,
                output_type=item.output_type,
                max_tokens=max_tokens,
                images=item.images,
            )
            for item in items
        }

    def describe_image(self, *, model, image, prompt, max_tokens=512):
        self.image_calls += 1
        return self._image_description

    def chat(self, *, model, system_blocks, messages, max_tokens=2048):
        self.chat_calls.append(
            {
                "model": model,
                "system_blocks": system_blocks,
                "messages": messages,
            }
        )
        return self._chat_answer
