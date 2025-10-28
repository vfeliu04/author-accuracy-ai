"""Embedding utilities backed by OpenAI."""

from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from .config import settings


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = OpenAI(api_key=settings.openai_api_key or None)
    response = client.embeddings.create(model=settings.openai_embed_model, input=texts)
    vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0)

