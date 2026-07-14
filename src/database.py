"""Database engine and transaction ownership."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session

from src.config import Settings, normalize_sqlite_url
from src.database_listeners.pragma_listeners import apply_pragmas

logger = logging.getLogger(__name__)

type SessionBind = Engine | Connection


def _attach_sqlite_listeners(db_engine: Engine) -> None:
    """Apply the app's SQLite pragmas to every new connection."""
    event.listen(db_engine, "connect", apply_pragmas)


def _create_engine_impl(database_url: str | None = None) -> Engine:
    """Create one engine for a configured database URL."""
    configured_url = Settings().db_url if database_url is None else database_url
    url = normalize_sqlite_url(configured_url)
    kwargs: dict[str, Any] = {
        "connect_args": {"check_same_thread": False},
        "pool_pre_ping": True,
    }
    if url in {"sqlite://", "sqlite:///:memory:"}:
        kwargs["poolclass"] = StaticPool
    db_engine = create_engine(url, **kwargs)
    _attach_sqlite_listeners(db_engine)

    logger.info(
        "Database engine created for %s",
        db_engine.url.render_as_string(hide_password=True),
    )
    return db_engine


@cache
def get_engine(database_url: str | None = None) -> Engine:
    """Return the process-owned engine, created lazily on first use."""
    return _create_engine_impl(database_url)


def get_session(bind: SessionBind | None = None) -> Session:
    """Open a session against an explicit bind or the process engine."""
    return Session(bind=bind or get_engine(), expire_on_commit=False)


@contextmanager
def db_session(bind: SessionBind | None = None) -> Generator[Session, None, None]:
    """Commit one transaction or roll it back on failure."""
    with get_session(bind) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


@contextmanager
def db_session_no_autocommit(
    bind: SessionBind | None = None,
) -> Generator[Session, None, None]:
    """Open a session whose transaction is controlled by the caller."""
    with get_session(bind) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            if session.in_transaction():
                session.rollback()


def _create_tables_impl(db_engine: Engine) -> None:
    """Create every registered SQLModel table on an engine."""
    from src.database_models import AppSQLModel

    AppSQLModel.metadata.create_all(db_engine)


def create_db_and_tables(bind: Engine | None = None) -> None:
    """Create application tables on an explicit or process-owned engine."""
    _create_tables_impl(bind or get_engine())


def _pool_value(pool: Any, name: str, default: int) -> int:
    value = getattr(pool, name, None)
    return int(value()) if callable(value) else default


def get_connection_pool_status(bind: Engine | None = None) -> dict[str, Any]:
    """Return bounded diagnostics for an engine's connection pool."""
    db_engine = bind or get_engine()
    try:
        pool = db_engine.pool
        return {
            "pool_size": _pool_value(pool, "size", 1),
            "checked_out": _pool_value(pool, "checkedout", 0),
            "overflow": _pool_value(pool, "overflow", 0),
            "invalid": _pool_value(pool, "invalid", 0),
            "pool_type": type(pool).__name__,
            "engine_url": db_engine.url.render_as_string(hide_password=True),
        }
    except Exception as error:
        logger.warning("Could not inspect database connection pool: %s", error)
        return {
            "pool_size": "unknown",
            "checked_out": "unknown",
            "overflow": "unknown",
            "invalid": "unknown",
            "pool_type": "unknown",
            "error": str(error),
        }
