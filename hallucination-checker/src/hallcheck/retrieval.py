"""Vector retrieval helpers backed by FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from .config import settings


INDEX_DIR = (settings.project_root or Path(__file__).resolve().parents[2]) / "data" / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_faiss(embeddings: np.ndarray, dim: int, out_name: str) -> faiss.IndexFlatIP:
    """Create and persist a FAISS index."""
    if embeddings.size == 0:
        raise ValueError("Cannot build index from empty embeddings.")
    normalized = np.array(embeddings, dtype=np.float32, copy=True)
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    index_path = INDEX_DIR / f"{out_name}.faiss"
    faiss.write_index(index, str(index_path))
    return index


def load_faiss(out_name: str) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk."""
    index_path = INDEX_DIR / f"{out_name}.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    return faiss.read_index(str(index_path))


def search_faiss(idx: faiss.IndexFlatIP, query_vecs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index with L2-normalized queries."""
    if query_vecs.size == 0:
        return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)
    queries = np.array(query_vecs, dtype=np.float32, copy=True)
    faiss.normalize_L2(queries)
    distances, indices = idx.search(queries, k)
    return distances, indices


def save_metadata(out_name: str, doc_ids: Sequence[int], chunk_payloads: Sequence[dict]) -> None:
    """Persist index metadata aligned with the FAISS rows."""
    doc_ids_array = np.asarray(doc_ids, dtype=np.int64)
    doc_path = INDEX_DIR / f"{out_name}_docids.npy"
    np.save(doc_path, doc_ids_array)

    chunk_path = INDEX_DIR / f"{out_name}_chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as handle:
        for payload in chunk_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_metadata(out_name: str) -> Tuple[np.ndarray, List[dict]]:
    """Load stored metadata aligned with the FAISS rows."""
    doc_path = INDEX_DIR / f"{out_name}_docids.npy"
    chunk_path = INDEX_DIR / f"{out_name}_chunks.jsonl"
    if not doc_path.exists() or not chunk_path.exists():
        raise FileNotFoundError(f"Metadata not found for index '{out_name}'.")
    doc_ids = np.load(doc_path)
    chunk_payloads: List[dict] = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk_payloads.append(json.loads(line))
    if len(doc_ids) != len(chunk_payloads):
        raise ValueError("Metadata size mismatch between doc IDs and chunk payloads.")
    return doc_ids, chunk_payloads

