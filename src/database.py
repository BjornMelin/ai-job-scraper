"""Database connection and session management for the AI Job Scraper.

This module provides optimized database connectivity using SQLAlchemy
and SQLModel with thread-safe configuration for background tasks.
It handles database engine creation, session management, table creation,
and SQLite optimization for concurrent access patterns.
"""

import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.config import Settings
from src.database_listeners.monitoring_listeners import log_slow, start_timer
from src.database_listeners.pragma_listeners import apply_pragmas

settings = Settings()
logger = logging.getLogger(__name__)


def _attach_sqlite_listeners(engine):
    """Attach SQLite event listeners for pragmas and optional performance monitoring.

    This function uses modular event listeners organized by responsibility:
    - Pragma listeners: Handle SQLite optimization settings
    - Monitoring listeners: Track query performance and log slow queries

    Args:
        engine: SQLAlchemy engine instance to attach listeners to.
    """
    # Always attach pragma handler for SQLite optimization
    event.listen(engine, "connect", apply_pragmas)

    # Only attach performance monitoring if enabled in settings
    if settings.db_monitoring:
        event.listen(engine, "before_cursor_execute", start_timer)
        event.listen(engine, "after_cursor_execute", log_slow)


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
        Session: A new SQLModel session for database operations.

    Note:
        The caller is responsible for closing the session when done.
        Consider using a context manager or try/finally block.
    """
    return Session(engine)


def get_connection_pool_status() -> dict:
    """Get current database connection pool status for monitoring.

    Returns:
        Dictionary with connection pool statistics including:
        - pool_size: Current pool size
        - checked_out: Number of connections currently in use
        - overflow: Number of overflow connections
        - invalid: Number of invalid connections

    Note:
        This function is useful for monitoring database connection usage
        and identifying potential connection pool exhaustion issues.
    """
    try:
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "engine_url": str(engine.url).split("@")[-1]
            if "@" in str(engine.url)
            else str(engine.url),
        }
    except Exception as e:
        logger.warning(f"Could not get connection pool status: {e}")
        return {
            "pool_size": "unknown",
            "checked_out": "unknown",
            "overflow": "unknown",
            "invalid": "unknown",
            "error": str(e),
        }
