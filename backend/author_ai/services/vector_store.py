"""FAISS-backed vector store utility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterable, Any

import faiss  # type: ignore
import numpy as np

from ..config import get_settings
from ..services.logger import setup_logger


logger = setup_logger(__name__)


class VectorStore:
    def __init__(self, name: str = "sources"):
        self.settings = get_settings()
        self.name = name
        self.index_path = self.settings.faiss_index_dir / f"{name}.index"
        self.meta_path = self.settings.faiss_index_dir / f"{name}.meta.json"
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Loaded FAISS index %s with %d vectors", self.index_path, self.index.ntotal)
        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    def _persist(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, vectors: Iterable[List[float]], metadatas: Iterable[Dict[str, Any]]) -> None:
        vectors = list(vectors)
        metas = list(metadatas)
        if not vectors:
            return
        arr = np.array(vectors, dtype="float32")
        self._ensure_index(arr.shape[1])
        self._normalize(arr)
        self.index.add(arr)
        self.metadata.extend(metas)
        self._persist()
        logger.info("Indexed %d vectors into %s", len(vectors), self.name)

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or not self.metadata:
            return []
        arr = np.array([vector], dtype="float32")
        self._normalize(arr)
        distances, indices = self.index.search(arr, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({"score": float(score), **meta})
        return results


VECTOR_STORE = VectorStore()
