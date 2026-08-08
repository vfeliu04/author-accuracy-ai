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
    ):
        self._parse_results = parse_results or {}
        self._image_description = image_description
        self.parse_calls: list[dict] = []
        self.image_calls: int = 0

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

    def parse_batch(self, *, model, items, max_tokens=BATCH_MAX_TOKENS):
        # Mirrors AnthropicClient.parse_batch's contract: dict keyed by custom_id.
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
