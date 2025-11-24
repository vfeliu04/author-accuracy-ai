"""
LlamaIndex-powered section summarizer for richer parent context.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from llama_index.core import Document, SummaryIndex
    try:
        from llama_index.core import Settings as LlamaSettings  # type: ignore
    except ImportError:  # pragma: no cover - older LlamaIndex versions
        LlamaSettings = None  # type: ignore
    try:
        from llama_index.core import ServiceContext  # type: ignore
    except ImportError:  # pragma: no cover - newer LlamaIndex versions may not expose this
        ServiceContext = None  # type: ignore
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
except ImportError:  # pragma: no cover - optional dependency
    Document = None  # type: ignore
    SummaryIndex = None  # type: ignore
    LlamaSettings = None  # type: ignore
    ServiceContext = None  # type: ignore
    LlamaOpenAI = None  # type: ignore

from ..config import get_settings
from ..services.logger import setup_logger
from ..services.summarizer import summarize_text


logger = setup_logger(__name__)


class SectionIndexer:
    def __init__(self):
        self.settings = get_settings()
        self.enabled = bool(
            Document and SummaryIndex and LlamaOpenAI and self.settings.openai_api_key
        )
        self.llm = None
        if self.enabled:
            try:
                self.llm = LlamaOpenAI(model=self.settings.explanation_model, api_key=self.settings.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LlamaIndex initialization failed, falling back to heuristic summaries: %s", exc)
                self.enabled = False
                self.llm = None
        self.has_settings = bool(LlamaSettings)
        self.service_context_cls = ServiceContext

    def summarize_sections(self, sections: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Return a mapping of section_id -> summary text.
        Falls back to the built-in summarizer when LlamaIndex is unavailable.
        """

        if not sections:
            return {}
        if not self.enabled or not self.llm:
            return self._fallback(sections)

        docs = []
        for section in sections:
            text = (section.get("text") or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    doc_id=section["id"],
                    metadata={"title": section.get("title"), "page": section.get("page")},
                )
            )

        if not docs:
            return {}
        index = self._build_index(docs)
        if not index:
            return self._fallback(sections)

        summaries: Dict[str, str] = {}
        for doc in docs:
            try:
                summary = index.get_document_summary(doc.doc_id)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Summary generation failed for %s: %s", doc.doc_id, exc)
                summary = None
            if summary:
                summaries[doc.doc_id] = summary.strip()
        missing = [section for section in sections if section["id"] not in summaries]
        if missing:
            summaries.update(self._fallback(missing))
        return summaries

    def _build_index(self, docs: List[Document]):
        # Try Settings-based API first (newer releases).
        if self.has_settings and LlamaSettings is not None:
            previous_llm = getattr(LlamaSettings, "llm", None)
            try:
                LlamaSettings.llm = self.llm
                return SummaryIndex.from_documents(docs)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Settings-based SummaryIndex failed: %s", exc)
            finally:
                if hasattr(LlamaSettings, "llm"):
                    LlamaSettings.llm = previous_llm

        # Fallback to ServiceContext API (older releases).
        if self.service_context_cls:
            try:
                ctx = self.service_context_cls.from_defaults(llm=self.llm)  # type: ignore[arg-type]
                return SummaryIndex.from_documents(docs, service_context=ctx)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ServiceContext-based SummaryIndex failed: %s", exc)

        return None

    @staticmethod
    def _fallback(sections: List[Dict[str, str]]) -> Dict[str, str]:
        return {section["id"]: summarize_text(section.get("text") or "", word_limit=80) for section in sections if section.get("text")}


SECTION_INDEXER = SectionIndexer()
