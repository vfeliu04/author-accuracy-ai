from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

import tiktoken


# The default model to use for tokenization if none is specified.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# This function gets the correct tokenizer for a given OpenAI model.
# `@lru_cache` is used to cache the result, so we don't have to load the tokenizer repeatedly.
@lru_cache(maxsize=4)
def _get_encoding(model: str = DEFAULT_EMBEDDING_MODEL) -> tiktoken.Encoding:
    """Return encoding for model with safe fallback."""
    try:
        # `tiktoken` is OpenAI's library for tokenization.
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # If the model name isn't recognized, fall back to a generic but compatible tokenizer.
        return tiktoken.get_encoding("cl100k_base")


# Converts a string of text into a list of token integers.
def encode_text(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[int]:
    """Encode text into token ids for a model."""
    encoding = _get_encoding(model)
    # The `encode` method performs the conversion.
    return encoding.encode(text, disallowed_special=())


# Converts a list of token integers back into a human-readable string.
def decode_tokens(tokens: Sequence[int], model: str = DEFAULT_EMBEDDING_MODEL) -> str:
    """Decode token ids into text for a model."""
    encoding = _get_encoding(model)
    return encoding.decode(tokens)


# A simple utility to count the number of tokens in a piece of text.
def count_tokens(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Return token count for text."""
    # We just encode the text and check the length of the resulting list.
    return len(encode_text(text, model))


# This function truncates a piece of text to a maximum number of tokens.
def trim_to_token_limit(text: str, max_tokens: int, model: str = DEFAULT_EMBEDDING_MODEL) -> str:
    """Trim text to at most max_tokens while preserving order."""
    token_ids = encode_text(text, model=model)
    if len(token_ids) <= max_tokens:
        return text
    # If the text is too long, we slice the list of token IDs and then decode it back to text.
    return decode_tokens(token_ids[:max_tokens], model=model)


# A generic helper function to split a sequence into smaller batches of a given size.
# This is useful for sending data to APIs that have batch size limits.
def batch(iterable: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield successive batches from a sequence."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
