"""FAISS-backed vector store utility with LangChain retriever support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterable, Any, Optional

import faiss  # type: ignore
import numpy as np

try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS as LangChainFAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
except ImportError:  # pragma: no cover
    LangChainEmbeddings = None  # type: ignore
    Document = None  # type: ignore
    LangChainFAISS = None  # type: ignore
    InMemoryDocstore = None  # type: ignore

from ..config import get_settings
from ..services.logger import setup_logger
from ..services.embedding import embed_texts


logger = setup_logger(__name__)


if LangChainEmbeddings is not None:

    class _EmbeddingAdapter(LangChainEmbeddings):  # type: ignore[misc]
        """Adapter to reuse Author AI's embedding service inside LangChain retrievers."""

        def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
            return embed_texts(texts)

        def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
            vectors = embed_texts([text])
            return vectors[0] if vectors else []

else:  # pragma: no cover
    _EmbeddingAdapter = None  # type: ignore[assignment]


class VectorStore:
    def __init__(self, name: str = "sources", base_dir: Path | None = None):
        self.settings = get_settings()
        self.name = name
        storage_dir = Path(base_dir) if base_dir else self.settings.faiss_index_dir
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = storage_dir / f"{name}.index"
        self.meta_path = storage_dir / f"{name}.meta.json"
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._langchain_store: Optional[LangChainFAISS] = None  # type: ignore[type-arg]
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
        self._langchain_store = None

    def _build_langchain_store(self) -> Optional[LangChainFAISS]:
        if (
            LangChainFAISS is None
            or InMemoryDocstore is None
            or Document is None
            or LangChainEmbeddings is None
            or _EmbeddingAdapter is None
        ):
            return None
        if self.index is None or not self.metadata:
            return None
        if self._langchain_store:
            return self._langchain_store

        documents = {}
        index_to_docstore_id: Dict[int, str] = {}
        for idx, meta in enumerate(self.metadata):
            doc_id = meta.get("chunk_id") or f"chunk-{idx}"
            page_content = meta.get("snippet") or meta.get("text") or ""
            documents[doc_id] = Document(page_content=page_content, metadata=meta)
            index_to_docstore_id[idx] = doc_id

        docstore = InMemoryDocstore(documents)
        self._langchain_store = LangChainFAISS(
            embedding_function=_EmbeddingAdapter(),
            index=self.index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=True,
        )
        return self._langchain_store

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

    def similarity_search(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        store = self._build_langchain_store()
        if store is None:
            vectors = embed_texts([text])
            if not vectors:
                return []
            return self.search(vectors[0], top_k=top_k)
        docs_with_scores = store.similarity_search_with_score(text, k=top_k)
        results: List[Dict[str, Any]] = []
        for doc, score in docs_with_scores:
            metadata = dict(doc.metadata)
            metadata.setdefault("snippet", doc.page_content)
            metadata["score"] = float(score)
            results.append(metadata)
        return results

    def overwrite(self, vectors: Iterable[List[float]], metadatas: Iterable[Dict[str, Any]]) -> None:
        vectors = list(vectors)
        metas = list(metadatas)
        self.index = None
        self.metadata = []
        self._langchain_store = None
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


VECTOR_STORE = VectorStore()
