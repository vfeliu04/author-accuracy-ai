"""Vector store abstraction (LangChain FAISS when available, legacy FAISS fallback otherwise)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS as LangChainFAISS
except ModuleNotFoundError as exc:  # pragma: no cover
    LangChainEmbeddings = None  # type: ignore
    Document = None  # type: ignore
    LangChainFAISS = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import faiss  # type: ignore

from ..config import get_settings
from ..services.logger import setup_logger
from ..services.embedding import embed_texts


logger = setup_logger(__name__)


def _normalize_vector(vector: List[float]) -> List[float]:
    arr = np.array(vector, dtype="float32")
    norm = np.linalg.norm(arr)
    if norm == 0:
        return list(arr)
    return list(arr / norm)


if LangChainFAISS is not None and LangChainEmbeddings is not None and Document is not None:

    class _EmbeddingAdapter(LangChainEmbeddings):  # type: ignore[misc]
        def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
            vectors = embed_texts(texts)
            return [_normalize_vector(vec) for vec in vectors]

        def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
            vectors = embed_texts([text])
            return _normalize_vector(vectors[0]) if vectors else []

    class VectorStore:
        def __init__(self, name: str = "sources", base_dir: Path | None = None):
            self.settings = get_settings()
            storage_root = Path(base_dir) if base_dir else self.settings.faiss_index_dir
            self.store_dir = storage_root / name
            self.store_dir.parent.mkdir(parents=True, exist_ok=True)
            self.embedding = _EmbeddingAdapter()
            self.store: LangChainFAISS | None = None  # type: ignore[type-arg]
            self._load()

        def _load(self) -> None:
            if not self.store_dir.exists():
                return
            try:
                self.store = LangChainFAISS.load_local(
                    str(self.store_dir),
                    embeddings=self.embedding,
                    allow_dangerous_deserialization=True,
                )
                self.store.normalize_L2 = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load FAISS index %s: %s", self.store_dir, exc)
                self.store = None

        def _persist(self) -> None:
            if self.store is None:
                if self.store_dir.exists():
                    shutil.rmtree(self.store_dir)
                return
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.store_dir))

        @staticmethod
        def _to_documents(texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]) -> List[Document]:
            docs: List[Document] = []
            for text, metadata in zip(texts, metadatas):
                content = (text or "").strip()
                if not content:
                    continue
                docs.append(Document(page_content=content, metadata=metadata or {}))
            return docs

        def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
            docs = self._to_documents(texts, metadatas)
            if not docs:
                return
            if self.store is None:
                self.store = LangChainFAISS.from_documents(
                    docs,
                    embedding=self.embedding,
                    normalize_L2=True,
                )
            else:
                self.store.add_documents(docs)
            self._persist()

        def overwrite(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
            docs = self._to_documents(texts, metadatas)
            if not docs:
                self.store = None
                self._persist()
                return
            self.store = LangChainFAISS.from_documents(
                docs,
                embedding=self.embedding,
                normalize_L2=True,
            )
            self._persist()

        def similarity_search(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
            if not text or self.store is None:
                return []
            docs_with_scores = self.store.similarity_search_with_score(text, k=top_k)
            return [self._format_result(doc, score) for doc, score in docs_with_scores]

        def similarity_search_by_doc(self, text: str, doc_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
            """Return up to top_k hits filtered to a specific doc_id.

            LangChain FAISS does not support metadata filtering natively, so we
            over-fetch (top_k * 10) and filter by doc_id after the search.
            """
            if not text or self.store is None:
                return []
            fetch_k = top_k * 10
            docs_with_scores = self.store.similarity_search_with_score(text, k=fetch_k)
            results = []
            for doc, score in docs_with_scores:
                if doc.metadata.get("doc_id") == doc_id:
                    results.append(self._format_result(doc, score))
                    if len(results) >= top_k:
                        break
            return results

        def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
            if not vector or self.store is None:
                return []
            normalized = _normalize_vector(vector)
            docs_with_scores = self.store.similarity_search_with_score_by_vector(normalized, k=top_k)
            return [self._format_result(doc, score) for doc, score in docs_with_scores]

        @staticmethod
        def _cosine_from_distance(distance: float) -> float:
            # LangChain FAISS returns L2 distance when normalize_L2=True. Convert back to cosine similarity.
            cosine = 1.0 - (distance / 2.0)
            return max(-1.0, min(1.0, cosine))

        def _format_result(self, doc: Document, score: float) -> Dict[str, Any]:
            metadata = dict(doc.metadata)
            metadata.setdefault("snippet", doc.page_content)
            metadata["score"] = float(self._cosine_from_distance(score))
            return metadata


else:

    class VectorStore:
        # Simple FAISS wrapper without LangChain; used when LC deps are missing.
        def __init__(self, name: str = "sources", base_dir: Path | None = None):
            self.settings = get_settings()
            storage_dir = Path(base_dir) if base_dir else self.settings.faiss_index_dir
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = storage_dir / f"{name}.index"
            self.meta_path = storage_dir / f"{name}.meta.json"
            self.index: Optional[faiss.IndexFlatIP] = None
            self.metadata: List[Dict[str, Any]] = []
            self._load()

        def _load(self) -> None:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            if self.meta_path.exists():
                self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

        def _ensure_index(self, dim: int) -> None:
            if self.index is None:
                self.index = faiss.IndexFlatIP(dim)

        def _persist(self) -> None:
            if self.index is None:
                if self.index_path.exists():
                    self.index_path.unlink()
                if self.meta_path.exists():
                    self.meta_path.unlink()
                return
            faiss.write_index(self.index, str(self.index_path))
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

        @staticmethod
        def _normalize(arr: np.ndarray) -> np.ndarray:
            faiss.normalize_L2(arr)
            return arr

        def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
            vectors = embed_texts(texts)
            self.add(vectors, metadatas)

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

        def overwrite(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
            vectors = embed_texts(texts)
            metas = list(metadatas)
            self.index = None
            self.metadata = []
            if self.index_path.exists():
                self.index_path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()
            if not vectors:
                self._persist()
                return
            arr = np.array(vectors, dtype="float32")
            self._ensure_index(arr.shape[1])
            self._normalize(arr)
            self.index.add(arr)
            self.metadata = metas
            self._persist()

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

        def similarity_search(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
            vectors = embed_texts([text])
            if not vectors:
                return []
            return self.search(vectors[0], top_k=top_k)

        def similarity_search_by_doc(self, text: str, doc_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
            """Return up to top_k hits filtered to a specific doc_id.

            Over-fetches (top_k * 10) from the FAISS index and filters by doc_id
            post-search since FAISS has no native metadata filtering.
            """
            vectors = embed_texts([text])
            if not vectors:
                return []
            fetch_k = top_k * 10
            all_hits = self.search(vectors[0], top_k=fetch_k)
            results = []
            for hit in all_hits:
                if hit.get("doc_id") == doc_id:
                    results.append(hit)
                    if len(results) >= top_k:
                        break
            return results


VECTOR_STORE = VectorStore()
