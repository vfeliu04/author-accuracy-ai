"""
LlamaIndex-powered section summarizer for richer parent context.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from llama_index.core import Document, SummaryIndex, ServiceContext
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
except ImportError:  # pragma: no cover - optional dependency
    Document = None  # type: ignore
    SummaryIndex = None  # type: ignore
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
            Document and SummaryIndex and ServiceContext and LlamaOpenAI and self.settings.openai_api_key
        )
        if self.enabled:
            try:
                self.service_context = ServiceContext.from_defaults(
                    llm=LlamaOpenAI(model=self.settings.explanation_model, api_key=self.settings.openai_api_key),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LlamaIndex initialization failed, falling back to heuristic summaries: %s", exc)
                self.enabled = False
                self.service_context = None
        else:
            self.service_context = None

    def summarize_sections(self, sections: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Return a mapping of section_id -> summary text.
        Falls back to the built-in summarizer when LlamaIndex is unavailable.
        """

        if not sections:
            return {}
        if not self.enabled or not self.service_context:
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
        try:
            index = SummaryIndex.from_documents(docs, service_context=self.service_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LlamaIndex SummaryIndex failed (%s), using fallback summaries.", exc)
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

    @staticmethod
    def _fallback(sections: List[Dict[str, str]]) -> Dict[str, str]:
        return {section["id"]: summarize_text(section.get("text") or "", word_limit=80) for section in sections if section.get("text")}


SECTION_INDEXER = SectionIndexer()
