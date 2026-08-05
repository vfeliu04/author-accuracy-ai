"""The one LLM client module — ALL Anthropic traffic goes through here.

Pipeline code depends on the `LLM` protocol; tests inject a fake. The real
client refuses to construct without a key (no silent degradation), raises
loudly when a call yields no usable output, and logs token usage per call so
cost stays visible. Retries are the SDK's built-in ones — no hand-rolled loop.
"""

import base64
import io
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel

from authorai.log import setup_logger

if TYPE_CHECKING:
    from PIL.Image import Image

logger = setup_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLM(Protocol):
    def parse(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[ModelT],
        max_tokens: int = 8192,
    ) -> ModelT: ...

    def describe_image(
        self, *, model: str, image: "Image", prompt: str, max_tokens: int = 512
    ) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str | None):
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. LLM calls require a real provider — "
                "refusing to start rather than degrading silently."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def parse(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[ModelT],
        max_tokens: int = 8192,
    ) -> ModelT:
        response = self._client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_type,
        )
        self._log_usage(model, response)
        if response.parsed_output is None:
            raise RuntimeError(
                f"LLM call produced no parseable {output_type.__name__} "
                f"(stop_reason={response.stop_reason!r})"
            )
        return response.parsed_output

    def describe_image(
        self, *, model: str, image: "Image", prompt: str, max_tokens: int = 512
    ) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        self._log_usage(model, response)
        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text.strip():
            raise RuntimeError(
                f"LLM image call returned no text (stop_reason={response.stop_reason!r})"
            )
        return text.strip()

    def _log_usage(self, model: str, response) -> None:
        usage = response.usage
        logger.info(
            "llm call model=%s input_tokens=%s output_tokens=%s",
            model,
            usage.input_tokens,
            usage.output_tokens,
        )
