"""Database utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings


engine = create_engine(settings.db_url, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=Session, future=True)


def init_db() -> None:
    """Create database tables if they do not exist."""
    from .models import Base  # Imported lazily to avoid circular import

    Base.metadata.create_all(bind=engine)
    _ensure_verdict_explanation_column()
    _ensure_document_author_column()


def reset_db() -> None:
    """Drop and recreate all tables."""
    from .models import Base  # Imported lazily to avoid circular import

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    _ensure_verdict_explanation_column()
    _ensure_document_author_column()


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager yielding a database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_verdict_explanation_column() -> None:
    """Add the verdict.explanation column if missing (for backward compatibility)."""
    try:
        with engine.begin() as connection:
            inspector = inspect(connection)
            if "verdicts" not in inspector.get_table_names():
                return
            column_names = {col["name"] for col in inspector.get_columns("verdicts")}
            if "explanation" in column_names:
                return
            connection.execute(text("ALTER TABLE verdicts ADD COLUMN explanation TEXT"))
    except Exception:
        # Failing here should not block the app; legacy databases may require manual migration.
        pass


def _ensure_document_author_column() -> None:
    """Add the document.author column if missing."""
    try:
        with engine.begin() as connection:
            inspector = inspect(connection)
            if "documents" not in inspector.get_table_names():
                return
            column_names = {col["name"] for col in inspector.get_columns("documents")}
            if "author" in column_names:
                return
            connection.execute(text("ALTER TABLE documents ADD COLUMN author VARCHAR(256)"))
    except Exception:
        pass
