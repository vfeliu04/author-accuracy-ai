import pytest
from pydantic import BaseModel

from authorai import db as dbmod

DIM = 8


@pytest.fixture()
def conn(tmp_path):
    connection = dbmod.connect(tmp_path / "test.db", embedding_dim=DIM)
    yield connection
    connection.close()


class FakeLLM:
    """Canned LLM for tests: returns pre-set objects per output type and
    records every call so tests can assert on prompts."""

    def __init__(
        self,
        parse_results: dict[type, BaseModel] | None = None,
        image_description: str = "A fake description.",
    ):
        self._parse_results = parse_results or {}
        self._image_description = image_description
        self.parse_calls: list[dict] = []
        self.image_calls: int = 0

    def parse(self, *, model, system, prompt, output_type, max_tokens=8192):
        self.parse_calls.append(
            {"model": model, "system": system, "prompt": prompt, "output_type": output_type}
        )
        return self._parse_results[output_type]

    def describe_image(self, *, model, image, prompt, max_tokens=512):
        self.image_calls += 1
        return self._image_description
