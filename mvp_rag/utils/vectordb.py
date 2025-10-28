from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from mvp_rag.config import Settings


# Standard logger setup.
LOGGER = logging.getLogger(__name__)

# The size of the vector embeddings. `text-embedding-3-small` produces vectors of this size.
# This must match the model's output dimension.
VECTOR_SIZE = 1536


# A client class to encapsulate all interactions with the Qdrant vector database.
@dataclass(slots=True)
class VectorStore:
    client: QdrantClient  # The official Qdrant client instance.
    collection_name: str  # The name of the collection where vectors will be stored.

    # A factory method to create a VectorStore instance from our application settings.
    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorStore":
        if not settings.qdrant_url:
            raise RuntimeError("QDRANT_URL is not configured.")
        # Initialize the Qdrant client with the URL, API key, and a timeout.
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.request_timeout_seconds,
        )
        store = cls(client=client, collection_name=settings.collection_name)
        # Make sure the collection exists in Qdrant before returning the client.
        store.ensure_collection()
        return store

    # This method checks if the required collection exists in Qdrant and creates it if it doesn't.
    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections or []
        names = {collection.name for collection in collections}
        if self.collection_name in names:
            return  # The collection already exists, so we do nothing.

        LOGGER.info("Creating Qdrant collection %s", self.collection_name)
        # Define the configuration for the new collection.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE  # Cosine similarity is a common choice for text embeddings.
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(
                vacuum_min_vector_number=1000,  # Optimization settings for the collection.
            ),
            on_disk_payload=True,  # Store the payload data on disk for efficiency.
        )

    # This is a helper method to fetch existing chunk keys for a given document.
    # It's used to prevent ingesting the exact same chunk multiple times.
    def _existing_dedup_keys(self, doc_title: str) -> Set[str]:
        dedup_keys: Set[str] = set()
        # We filter by `doc_title` to only check for duplicates within the same document.
        scroll_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_title", match=qmodels.MatchValue(value=doc_title))]
        )
        offset: str | None = None
        # The `scroll` API is used to iterate through all points in a collection that match a filter.
        while True:
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,  # Process points in batches of 256.
                with_vectors=False,  # We don't need the vector data, just the payload.
                with_payload=True,
                offset=offset,
            )
            points, offset = response
            # For each point found, construct a unique key and add it to our set.
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                page_start = payload.get("page_start")
                if chunk_id is None or page_start is None:
                    continue
                dedup_keys.add(f"{page_start}:{chunk_id}")
            # When `offset` is `None`, we've reached the end of the results.
            if offset is None:
                break
        return dedup_keys

    # This method "upserts" (updates or inserts) chunks into the vector store.
    def upsert_chunks(
        self, doc_title: str, chunks: Sequence[Dict[str, int | str]], embeddings: Sequence[Sequence[float]]
    ) -> Tuple[int, int]:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")

        existing_keys = self._existing_dedup_keys(doc_title)
        points: List[qmodels.PointStruct] = []  # A list to hold the points we'll send to Qdrant.
        skipped = 0

        # Iterate through the chunks and their corresponding embeddings.
        for chunk, embedding in zip(chunks, embeddings):
            key = f"{chunk['page_start']}:{chunk['chunk_id']}"
            # If we've already ingested this exact chunk, skip it.
            if key in existing_keys:
                skipped += 1
                continue
            # The "payload" is the metadata we store alongside the vector.
            # This is what we get back when we perform a search.
            payload = {
                "doc_title": doc_title,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "chunk_text": chunk["chunk_text"],
                "chunk_id": chunk["chunk_id"],
                "token_count": chunk["token_count"],
            }
            # Create a unique ID for this point.
            point_id = f"{doc_title}::{chunk['page_start']}::{chunk['chunk_id']}"
            # Construct the `PointStruct` object that Qdrant expects.
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )
            existing_keys.add(key)

        # If there are new points to add, send them to Qdrant in a single batch.
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,  # `wait=True` ensures the operation is complete before proceeding.
            )

        return len(points), skipped

    # This method searches the collection for the `top_k` most similar vectors to the query embedding.
    def query(self, embedding: Sequence[float], top_k: int = 4) -> List[qmodels.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,  # The number of results to return.
            with_payload=True,  # We need the payload to get the original text and metadata.
            with_vectors=False,  # We don't need the vector itself in the search results.
        )
