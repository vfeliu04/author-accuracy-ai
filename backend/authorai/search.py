"""Hybrid search: vector similarity + keyword match, fused with RRF.

Vector search (sqlite-vec) is good at paraphrase and bad at exact numbers;
keyword search (FTS5/BM25) is the reverse. Reciprocal Rank Fusion merges the
two ranked lists so a chunk found by both channels outranks single-channel
hits. Every query is scoped to one run via SQL — no post-filtering.
"""

import sqlite3
from collections import defaultdict
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
    figure_id: str | None = None


def vector_search(
    conn: sqlite3.Connection,
    run_id: str,
    query_embedding: list[float],
    k: int = CHANNEL_K,
    doc_kind: str | None = None,
) -> list[int]:
    """Chunk ids by ascending vector distance, scoped to one run.

    `doc_kind` is a PARTITION KEY on chunks_vec (like run_id), so the filter
    is index-native and exact — no over-fetching, no post-filter, and no k cap
    to starve the channel when the excluded document dominates the neighborhood.
    """
    kind_filter = "" if doc_kind is None else "AND doc_kind = ?"
    params: tuple = (serialize_float32(normalize(query_embedding)), k, run_id)
    if doc_kind is not None:
        params = (*params, doc_kind)
    rows = conn.execute(
        f"""
        SELECT chunk_id FROM chunks_vec
        WHERE embedding MATCH ? AND k = ? AND run_id = ? {kind_filter}
        ORDER BY distance
        """,
        params,
    ).fetchall()
    return [row["chunk_id"] for row in rows]


def keyword_search(
    conn: sqlite3.Connection,
    run_id: str,
    query_text: str,
    k: int = CHANNEL_K,
    doc_kind: str | None = None,
) -> list[int]:
    """Chunk ids by BM25 relevance, scoped to one run (and optionally one doc kind)."""
    match = _fts_query(query_text)
    if match is None:
        return []
    kind_filter = "" if doc_kind is None else "AND d.kind = ?"
    params: tuple = (match, run_id) if doc_kind is None else (match, run_id, doc_kind)
    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        JOIN documents d ON d.id = c.doc_id
        WHERE chunks_fts MATCH ? AND c.run_id = ? {kind_filter}
        ORDER BY f.rank
        LIMIT ?
        """,
        (*params, k),
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
    doc_kind: str | None = None,
) -> list[Hit]:
    """Fuse the vector and keyword channels with Reciprocal Rank Fusion.

    Returns at most `k` hits. Each channel contributes a candidate pool of
    max(k, CHANNEL_K) — deliberately wider than `k` so the two rankings have
    enough overlap for the fusion to be meaningful. With `doc_kind`, both
    channels only surface chunks from documents of that kind (verification
    retrieves evidence with doc_kind="SOURCE" so a report can never be its
    own evidence).
    """
    channel_k = max(k, CHANNEL_K)
    vector_ids = vector_search(conn, run_id, query_embedding, channel_k, doc_kind=doc_kind)
    keyword_ids = keyword_search(conn, run_id, query_text, channel_k, doc_kind=doc_kind)

    scores: defaultdict[int, float] = defaultdict(float)
    channels: defaultdict[int, list[str]] = defaultdict(list)
    for channel, ids in (("vector", vector_ids), ("keyword", keyword_ids)):
        for rank, chunk_id in enumerate(ids):
            scores[chunk_id] += 1.0 / (RRF_K + rank + 1)
            channels[chunk_id].append(channel)

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
            figure_id=rows[chunk_id]["figure_id"],
        )
        for chunk_id in top
    ]
