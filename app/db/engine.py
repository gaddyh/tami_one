"""Database engine and session helpers (PostgreSQL or SQLite)."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    db_path = os.getenv("DATABASE_PATH", "tami.db")
    return f"sqlite:///{db_path}"


def _is_sqlite() -> bool:
    return _database_url().startswith("sqlite")


def _connect_args() -> dict:
    if _is_sqlite():
        return {"check_same_thread": False}
    return {}


engine = create_engine(_database_url(), echo=False, connect_args=_connect_args())


def init_db(*, overwrite: bool = False) -> None:
    """Create tables if they don't exist.

    If *overwrite* is True, drops all tables first (SQLite deletes the file,
    PostgreSQL drops via DROP TABLE ... CASCADE).
    """
    import app.db.models  # noqa: F401 — registers tables

    if _is_sqlite():
        db_path = os.getenv("DATABASE_PATH", "tami.db")

        if overwrite and os.path.isfile(db_path):
            os.remove(db_path)
            logger.info("Removed existing database for overwrite: %s", db_path)

        db_existed = os.path.isfile(db_path)

        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        SQLModel.metadata.create_all(engine)

        if db_existed:
            logger.info("Loaded existing database: %s", db_path)
        else:
            logger.info("Created new database: %s", db_path)
    else:
        if overwrite:
            SQLModel.metadata.drop_all(engine)
            logger.info("Dropped all tables for overwrite in PostgreSQL")

        SQLModel.metadata.create_all(engine)
        logger.info("Ensured tables exist in PostgreSQL")


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLModel session."""
    with Session(engine) as session:
        yield session
