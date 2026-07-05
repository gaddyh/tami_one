"""SQLite database engine and session helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)


def _db_path() -> str:
    return os.getenv("DATABASE_PATH", "tami.db")


def _database_url() -> str:
    return f"sqlite:///{_db_path()}"


def _connect_args() -> dict:
    return {"check_same_thread": False}


engine = create_engine(_database_url(), echo=False, connect_args=_connect_args())


def init_db(*, overwrite: bool = False) -> None:
    """Create the SQLite database file and tables if they don't exist.

    If the database file already exists, loads it as-is. If *overwrite* is
    True, deletes the existing file and creates a fresh one with all tables.
    """
    db_path = _db_path()

    if overwrite and os.path.isfile(db_path):
        os.remove(db_path)
        logger.info("Removed existing database for overwrite: %s", db_path)

    db_existed = os.path.isfile(db_path)

    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    import app.db.models  # noqa: F401 — registers tables

    SQLModel.metadata.create_all(engine)

    if db_existed:
        logger.info("Loaded existing database: %s", db_path)
    else:
        logger.info("Created new database: %s", db_path)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLModel session."""
    with Session(engine) as session:
        yield session
