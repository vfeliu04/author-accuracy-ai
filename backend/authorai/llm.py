"""The one LLM client module — ALL Anthropic traffic goes through here.

Pipeline code depends on the `LLM` protocol; tests inject a fake. The real
client refuses to construct without a key (no silent degradation), raises
loudly when a call yields no usable output, and logs token usage per call so
cost stays visible. Retries are the SDK's built-in ones — no hand-rolled loop.
"""

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel

from authorai.log import setup_logger

if TYPE_CHECKING:
    from PIL.Image import Image

logger = setup_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


# Claude Opus 5 thinks by default, and thinking shares this budget with the
# response — a report yielding dozens of claims truncates at 8192 and comes back
# unparseable. 16000 is the ceiling that still stays under the SDK's
# non-streaming HTTP timeout.
PARSE_MAX_TOKENS = 16000

# Batch requests face no HTTP timeout, so they can afford more headroom —
# observed live: a hard verdict item burned the full 16k on thinking and came
# back truncated.
BATCH_MAX_TOKENS = 32000

BATCH_POLL_SECONDS = 10
BATCH_TIMEOUT_SECONDS = 3600


@dataclass
class ParseItem:
    """One structured call in a batch. Carries the same payload the sync path
    takes — including images — so the two paths stay interchangeable."""

    custom_id: str
    system: str
    prompt: str
    output_type: type[BaseModel]
    images: "list[Path] | None" = None


class LLM(Protocol):
    def parse(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[ModelT],
        max_tokens: int = PARSE_MAX_TOKENS,
        images: "list[Path] | None" = None,
    ) -> ModelT: ...

    def parse_batch(
        self,
        *,
        model: str,
        items: list[ParseItem],
        max_tokens: int = PARSE_MAX_TOKENS,
    ) -> dict[str, BaseModel]: ...

    def describe_image(
        self, *, model: str, image: "Image", prompt: str, max_tokens: int = 512
    ) -> str: ...


def _content(prompt: str, images: "list[Path] | None"):
    """User-message content: the plain string, or image blocks before the text."""
    if not images:
        return prompt
    blocks = []
    for path in images:
        data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            }
        )
    blocks.append({"type": "text", "text": prompt})
    return blocks


def build_batch_request(item: ParseItem, model: str, max_tokens: int = PARSE_MAX_TOKENS):
    """The Batch API request for one ParseItem.

    transform_schema() produces exactly the schema messages.parse() would send,
    so the batch path and the sync path make provably identical requests. The
    batch param is `output_config` — `output_format` is parse()-only sugar and
    would be silently dropped from batch params.
    """
    from anthropic import transform_schema
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    return Request(
        custom_id=item.custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=max_tokens,
            system=item.system,
            messages=[{"role": "user", "content": _content(item.prompt, item.images)}],
            output_config={
                "format": {"type": "json_schema", "schema": transform_schema(item.output_type)}
            },
        ),
    )


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
        max_tokens: int = PARSE_MAX_TOKENS,
        images: "list[Path] | None" = None,
    ) -> ModelT:
        response = self._client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": _content(prompt, images)}],
            output_format=output_type,
        )
        self._log_usage(model, response)
        if response.parsed_output is None:
            raise RuntimeError(
                f"LLM call produced no parseable {output_type.__name__} "
                f"(stop_reason={response.stop_reason!r})"
            )
        return response.parsed_output

    def parse_batch(
        self,
        *,
        model: str,
        items: list[ParseItem],
        max_tokens: int = PARSE_MAX_TOKENS,
        poll_seconds: float = BATCH_POLL_SECONDS,
        timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    ) -> dict[str, BaseModel]:
        """Run all items through the Batch API; returns results keyed by custom_id.

        A failed item gets ONE sync retry (logged — observed live failure modes
        are transient: corrupted verdict literals, max_tokens truncation), then
        the call is ALL-OR-NOTHING: items that fail twice raise after the batch
        ends, listing every failure — storing partial results would silently
        change the denominator of any score computed on them. The batch id is
        logged the moment it exists so a timeout or interrupt never orphans a
        paid batch.
        """
        requests = [build_batch_request(item, model, max_tokens) for item in items]
        batch = self._client.messages.batches.create(requests=requests)
        logger.info(
            "batch %s created (%d items) — reattach by id if interrupted", batch.id, len(items)
        )

        deadline = time.monotonic() + timeout_seconds
        while True:
            batch = self._client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {batch.id} still {batch.processing_status!r} after "
                    f"{timeout_seconds}s — it is still queued server-side; re-poll or cancel it"
                )
            time.sleep(poll_seconds)

        output_types = {item.custom_id: item.output_type for item in items}
        parsed: dict[str, BaseModel] = {}
        failures: list[str] = []
        for result in self._client.messages.batches.results(batch.id):
            custom_id = result.custom_id
            if result.result.type != "succeeded":
                failures.append(f"{custom_id}: {result.result.type}")
                continue
            message = result.result.message
            self._log_usage(model, message)
            if message.stop_reason == "max_tokens":
                failures.append(f"{custom_id}: output truncated at max_tokens")
                continue
            # Thinking blocks precede the text block — never index content[0].
            text = next((b.text for b in message.content if b.type == "text"), None)
            if text is None:
                failures.append(f"{custom_id}: no text block (stop_reason={message.stop_reason!r})")
                continue
            try:
                parsed[custom_id] = output_types[custom_id].model_validate_json(text)
            except Exception as exc:  # noqa: BLE001 - recorded per item, raised in aggregate
                failures.append(f"{custom_id}: unparseable output ({exc})")
        for custom_id in output_types:
            if custom_id not in parsed and not any(f.startswith(custom_id) for f in failures):
                failures.append(f"{custom_id}: no result returned")

        if failures:
            # One sync retry per failed item — loudly logged, never silent.
            items_by_id = {item.custom_id: item for item in items}
            still_failing: list[str] = []
            for failure in failures:
                custom_id = failure.split(":", 1)[0]
                item = items_by_id[custom_id]
                logger.warning("batch item failed (%s) — retrying once sync", failure)
                try:
                    parsed[custom_id] = self.parse(
                        model=model,
                        system=item.system,
                        prompt=item.prompt,
                        output_type=item.output_type,
                        # Sync must stay under the SDK's non-streaming ceiling
                        # even when the batch ran with more headroom.
                        max_tokens=min(max_tokens, PARSE_MAX_TOKENS),
                        images=item.images,
                    )
                except Exception as exc:  # noqa: BLE001 - aggregated below
                    still_failing.append(f"{failure}; retry: {exc}")
            if still_failing:
                raise RuntimeError(
                    f"Batch {batch.id}: {len(still_failing)}/{len(items)} items failed even "
                    "after a sync retry — storing nothing (rerun is safe, replace "
                    "semantics): " + "; ".join(still_failing)
                )
        return parsed

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
