"""Database connection and session management for the AI Job Scraper.

This module provides optimized database connectivity using SQLAlchemy
and SQLModel with thread-safe configuration for background tasks.
It handles database engine creation, session management, table creation,
and SQLite optimization for concurrent access patterns.
"""

import logging
import time

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from .config import Settings

settings = Settings()
logger = logging.getLogger(__name__)

# Performance monitoring for slow queries
SLOW_QUERY_THRESHOLD = 1.0  # Log queries taking longer than 1 second


def _apply_pragmas_handler(conn, _):
    """Apply SQLite pragmas on each new connection."""
    cursor = conn.cursor()
    for pragma in settings.sqlite_pragmas:
        try:
            cursor.execute(pragma)
            logger.debug(f"Applied SQLite pragma: {pragma}")
        except Exception as e:
            logger.warning(f"Failed to apply pragma '{pragma}': {e}")
    cursor.close()


def _start_timer_handler(conn, cursor, stmt, params, ctx, many):
    """Start timing for query execution."""
    ctx._query_start = time.time()


def _log_slow_handler(conn, cursor, stmt, params, ctx, many):
    """Log slow queries and performance metrics."""
    dt = time.time() - ctx._query_start
    if dt > SLOW_QUERY_THRESHOLD:
        preview = (stmt[:200] + "...") if len(stmt) > 200 else stmt
        logger.warning(f"Slow query {dt:.3f}s – {preview}")
    elif dt > 0.1:
        logger.debug(f"Query took {dt:.3f}s")


def _attach_sqlite_listeners(engine):
    """Attach SQLite event listeners for pragmas and optional performance monitoring.

    This consolidated function handles both pragma application and optional
    performance monitoring based on configuration settings.

    Args:
        engine: SQLAlchemy engine instance to attach listeners to.
    """
    # Always attach pragma handler
    event.listen(engine, "connect", _apply_pragmas_handler)

    # Only attach performance monitoring if enabled in settings
    if settings.db_monitoring:
        event.listen(engine, "before_cursor_execute", _start_timer_handler)
        event.listen(engine, "after_cursor_execute", _log_slow_handler)


# Create thread-safe SQLAlchemy engine with optimized configuration
if settings.db_url.startswith("sqlite"):
    # SQLite-specific configuration for thread safety and performance
    engine = create_engine(
        settings.db_url,
        echo=False,
        connect_args={
            "check_same_thread": False,  # Allow cross-thread access
        },
        poolclass=StaticPool,  # Single connection reused safely
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=3600,  # Refresh connections hourly
    )
    # Configure SQLite optimizations and optional performance monitoring
    _attach_sqlite_listeners(engine)
else:
    # PostgreSQL or other database configuration
    engine = create_engine(
        settings.db_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

# Create session factory with optimized settings
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # Prevent lazy loading issues in background threads
)


def create_db_and_tables() -> None:
    """Create database tables from SQLModel definitions.

    This function creates all tables defined in the SQLModel metadata.
    It should be called once during application initialization to ensure
    all required database tables exist.
    """
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Create a new database session.

    Returns:
        Session: A new SQLAlchemy session for database operations.

    Note:
        The caller is responsible for closing the session when done.
        Consider using a context manager or try/finally block.
    """
    return SessionLocal()
