"""Hybrid search: vector similarity + keyword match, fused with RRF.

Vector search (sqlite-vec) is good at paraphrase and bad at exact numbers;
keyword search (FTS5/BM25) is the reverse. Reciprocal Rank Fusion merges the
two ranked lists so a chunk found by both channels outranks single-channel
hits. Every query is scoped to one run via SQL — no post-filtering.
"""

import sqlite3
from dataclasses import dataclass

from sqlite_vec import serialize_float32

from authorai.db import get_chunks
from authorai.embeddings import normalize

RRF_K = 60  # standard damping constant: score = sum(1 / (RRF_K + rank))
CHANNEL_K = 20  # candidates fetched per channel before fusion


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    page: int | None
    section: str | None
    kind: str
    text: str
    score: float
    channels: tuple[str, ...]


def vector_search(
    conn: sqlite3.Connection, run_id: str, query_embedding: list[float], k: int = CHANNEL_K
) -> list[int]:
    """Chunk ids by ascending vector distance, scoped to one run."""
    rows = conn.execute(
        """
        SELECT chunk_id FROM chunks_vec
        WHERE embedding MATCH ? AND k = ? AND run_id = ?
        ORDER BY distance
        """,
        (serialize_float32(normalize(query_embedding)), k, run_id),
    ).fetchall()
    return [row["chunk_id"] for row in rows]


def keyword_search(
    conn: sqlite3.Connection, run_id: str, query_text: str, k: int = CHANNEL_K
) -> list[int]:
    """Chunk ids by BM25 relevance, scoped to one run."""
    match = _fts_query(query_text)
    if match is None:
        return []
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ? AND c.run_id = ?
        ORDER BY f.rank
        LIMIT ?
        """,
        (match, run_id, k),
    ).fetchall()
    return [row["chunk_id"] for row in rows]


def _fts_query(query_text: str) -> str | None:
    """Quote each token (so user text can't break FTS5 syntax) and OR them.

    OR, not the default implicit AND: queries are claim-length sentences, and
    requiring every token to appear would make the keyword channel return
    nothing in practice. BM25 still ranks chunks matching more tokens higher.
    """
    tokens = [token for token in query_text.split() if token.strip('"')]
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def hybrid_search(
    conn: sqlite3.Connection,
    run_id: str,
    query_text: str,
    query_embedding: list[float],
    k: int = 10,
) -> list[Hit]:
    channel_k = max(k, CHANNEL_K)
    vector_ids = vector_search(conn, run_id, query_embedding, channel_k)
    keyword_ids = keyword_search(conn, run_id, query_text, channel_k)

    scores: dict[int, float] = {}
    channels: dict[int, list[str]] = {}
    for channel, ids in (("vector", vector_ids), ("keyword", keyword_ids)):
        for rank, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            channels.setdefault(chunk_id, []).append(channel)

    top = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:k]
    rows = get_chunks(conn, top)
    return [
        Hit(
            chunk_id=chunk_id,
            doc_id=rows[chunk_id]["doc_id"],
            page=rows[chunk_id]["page"],
            section=rows[chunk_id]["section"],
            kind=rows[chunk_id]["kind"],
            text=rows[chunk_id]["text"],
            score=scores[chunk_id],
            channels=tuple(channels[chunk_id]),
        )
        for chunk_id in top
    ]
