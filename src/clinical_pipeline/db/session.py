"""SQLAlchemy session factory."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from clinical_pipeline.config import get_settings


def get_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    kwargs = {"pool_pre_ping": True}
    if not url.startswith("sqlite"):
        kwargs["pool_size"] = 5
    return create_engine(url, **kwargs)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session with automatic commit/rollback."""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
