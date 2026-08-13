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

# Chat answers are grounded Q&A over a cached run context — a bounded,
# non-reasoning task, so thinking is disabled to keep the whole budget for the
# visible answer and responses fast and direct.
CHAT_MAX_TOKENS = 2048


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
        max_tokens: int = BATCH_MAX_TOKENS,
    ) -> dict[str, BaseModel]: ...

    def describe_image(
        self, *, model: str, image: "Image", prompt: str, max_tokens: int = 512
    ) -> str: ...

    def chat(
        self,
        *,
        model: str,
        system_blocks: list[dict],
        messages: list[dict],
        max_tokens: int = CHAT_MAX_TOKENS,
    ) -> str: ...


def _image_block(png_bytes: bytes) -> dict:
    """THE Anthropic image content block — every image this module sends goes
    through here, so the wire shape exists once."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("ascii"),
        },
    }


def _content(prompt: str, images: "list[Path] | None"):
    """User-message content: the plain string, or image blocks before the text."""
    if not images:
        return prompt
    blocks = [_image_block(Path(path).read_bytes()) for path in images]
    blocks.append({"type": "text", "text": prompt})
    return blocks


def _first_text(message) -> str | None:
    """The first text block's text — thinking blocks precede it, so never
    index content[0]."""
    return next((b.text for b in message.content if b.type == "text"), None)


def build_batch_request(item: ParseItem, model: str, max_tokens: int = BATCH_MAX_TOKENS):
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
        timeout: float | None = None,
    ) -> ModelT:
        # An explicit timeout lifts the SDK's non-streaming max_tokens guard —
        # used by the batch retry so a thinking-heavy item keeps its headroom.
        client = self._client if timeout is None else self._client.with_options(timeout=timeout)
        response = client.messages.parse(
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
        max_tokens: int = BATCH_MAX_TOKENS,
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

        deadline = time.monotonic() + BATCH_TIMEOUT_SECONDS
        while True:
            batch = self._client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {batch.id} still {batch.processing_status!r} after "
                    f"{BATCH_TIMEOUT_SECONDS}s — it is still queued server-side; "
                    "re-poll or cancel it"
                )
            time.sleep(BATCH_POLL_SECONDS)

        output_types = {item.custom_id: item.output_type for item in items}
        parsed: dict[str, BaseModel] = {}
        # Keyed by custom_id so nothing ever parses ids back out of message
        # strings (and a custom_id that prefixes another can't mask a failure).
        failures: dict[str, str] = {}
        for result in self._client.messages.batches.results(batch.id):
            custom_id = result.custom_id
            if result.result.type != "succeeded":
                failures[custom_id] = result.result.type
                continue
            message = result.result.message
            self._log_usage(model, message)
            if message.stop_reason == "max_tokens":
                failures[custom_id] = "output truncated at max_tokens"
                continue
            text = _first_text(message)
            if text is None:
                failures[custom_id] = f"no text block (stop_reason={message.stop_reason!r})"
                continue
            try:
                parsed[custom_id] = output_types[custom_id].model_validate_json(text)
            except Exception as exc:  # noqa: BLE001 - recorded per item, raised in aggregate
                failures[custom_id] = f"unparseable output ({exc})"
        for custom_id in output_types:
            if custom_id not in parsed and custom_id not in failures:
                failures[custom_id] = "no result returned"

        if failures:
            # One sync retry per failed item — loudly logged, never silent.
            items_by_id = {item.custom_id: item for item in items}
            still_failing: list[str] = []
            for custom_id, reason in failures.items():
                item = items_by_id[custom_id]
                logger.warning("batch item %s failed (%s) — retrying once sync", custom_id, reason)
                try:
                    parsed[custom_id] = self.parse(
                        model=model,
                        system=item.system,
                        prompt=item.prompt,
                        output_type=item.output_type,
                        # Full batch headroom: capping the retry at the sync
                        # ceiling would re-truncate the exact thinking-heavy
                        # items the retry exists for. The explicit timeout
                        # lifts the SDK's non-streaming guard.
                        max_tokens=max_tokens,
                        images=item.images,
                        timeout=600.0,
                    )
                except Exception as exc:  # noqa: BLE001 - aggregated below
                    still_failing.append(f"{custom_id}: {reason}; retry: {exc}")
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
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        _image_block(buffer.getvalue()),
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        self._log_usage(model, response)
        text = _first_text(response) or ""
        if not text.strip():
            raise RuntimeError(
                f"LLM image call returned no text (stop_reason={response.stop_reason!r})"
            )
        return text.strip()

    def chat(
        self,
        *,
        model: str,
        system_blocks: list[dict],
        messages: list[dict],
        max_tokens: int = CHAT_MAX_TOKENS,
    ) -> str:
        """One grounded chat answer. `system_blocks` is a list so a block can
        carry cache_control — the large static run context is cached and the
        per-turn question is the only uncached input, so repeat turns for the
        same run reuse the cached prefix (see _log_usage's cache-token line)."""
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=messages,
            thinking={"type": "disabled"},
        )
        self._log_usage(model, response)
        text = _first_text(response) or ""
        if not text.strip():
            raise RuntimeError(f"Chat call returned no text (stop_reason={response.stop_reason!r})")
        return text.strip()

    def _log_usage(self, model: str, response) -> None:
        usage = response.usage
        logger.info(
            "llm call model=%s input_tokens=%s output_tokens=%s cache_write=%s cache_read=%s",
            model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
        )
