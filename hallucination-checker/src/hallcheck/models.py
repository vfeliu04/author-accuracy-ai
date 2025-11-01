"""Database models for hallucination checker."""

from __future__ import annotations

import enum
from typing import List, Optional

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class."""


class DocumentKind(str, enum.Enum):
    REPORT = "report"
    SOURCE = "source"


class VerdictStatus(str, enum.Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"


class Document(Base):
    """A PDF document ingested by the system."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[DocumentKind] = mapped_column(Enum(DocumentKind), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    author: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    router_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_scanned: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=False)
    extractor_chain: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    document_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    claims: Mapped[List["Claim"]] = relationship("Claim", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A chunk of text derived from a document."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tables: Mapped[Optional[List[dict]]] = mapped_column(JSON, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")


class Claim(Base):
    """A claim extracted from a report document."""

    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    sentence: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    units: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="claims")
    verdicts: Mapped[List["Verdict"]] = relationship("Verdict", back_populates="claim", cascade="all, delete-orphan")


class Verdict(Base):
    """A verdict comparing a claim against retrieved evidence."""

    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), nullable=False, index=True)
    status: Mapped[VerdictStatus] = mapped_column(Enum(VerdictStatus), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    claim: Mapped[Claim] = relationship("Claim", back_populates="verdicts")
