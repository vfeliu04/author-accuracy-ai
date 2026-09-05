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


def poison_providers(monkeypatch):
    """Make any provider work during an ingest reuse a test failure — not just
    calls: CONSTRUCTING a client already means the dedup path leaked. The one
    definition both the jobs tests and the API seam test use, so the
    reuse-recomputes-nothing contract cannot silently stop being guarded in
    one of them."""
    from authorai import jobs as jobsmod

    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: pytest.fail("re-ingested"))
    monkeypatch.setattr(
        jobsmod, "OpenAIEmbedder", lambda *a, **k: pytest.fail("constructed an embedder")
    )
    monkeypatch.setattr(
        jobsmod, "AnthropicClient", lambda *a, **k: pytest.fail("constructed an LLM client")
    )


class FakeLLM:
    """Canned LLM for tests: returns pre-set objects per output type and
    records every call so tests can assert on prompts.

    A parse_results value may be a single instance (returned every call) or a
    list of instances popped in call order — verification tests need a
    different Verdict per claim.
    """

    def __init__(
        self,
        parse_results: dict[type, BaseModel | list[BaseModel]] | None = None,
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
