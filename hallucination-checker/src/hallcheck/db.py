"""Database utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings


engine = create_engine(settings.db_url, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=Session, future=True)


def init_db() -> None:
    """Create database tables if they do not exist."""
    from .models import Base  # Imported lazily to avoid circular import

    Base.metadata.create_all(bind=engine)


def reset_db() -> None:
    """Drop and recreate all tables."""
    from .models import Base  # Imported lazily to avoid circular import

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


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
