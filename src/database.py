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

# SQLite optimization pragmas for performance and concurrency
SQLITE_PRAGMAS = [
    "PRAGMA journal_mode = WAL",  # Write-Ahead Logging for better concurrency
    "PRAGMA synchronous = NORMAL",  # Balanced safety/performance
    "PRAGMA cache_size = 64000",  # 64MB cache (default is 2MB)
    "PRAGMA temp_store = MEMORY",  # Store temp tables in memory
    "PRAGMA mmap_size = 134217728",  # 128MB memory-mapped I/O
    "PRAGMA foreign_keys = ON",  # Enable foreign key constraints
    "PRAGMA optimize",  # Auto-optimize indexes
]


def _configure_sqlite_engine() -> None:
    """Configure SQLite engine with performance and threading optimizations.

    Applies SQLite pragmas for:
    - WAL mode for better concurrent access
    - Optimized cache and memory settings
    - Thread-safe configuration for background tasks
    """

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Apply SQLite pragmas on each new connection."""
        cursor = dbapi_connection.cursor()
        for pragma in SQLITE_PRAGMAS:
            try:
                cursor.execute(pragma)
                logger.debug(f"Applied SQLite pragma: {pragma}")
            except Exception as e:
                logger.warning(f"Failed to apply pragma '{pragma}': {e}")
        cursor.close()


def _configure_performance_monitoring() -> None:
    """Configure performance monitoring for database operations."""

    @event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        """Start timing for query execution."""
        context._query_start_time = time.time()

    @event.listens_for(engine, "after_cursor_execute")
    def receive_after_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        """Log slow queries and performance metrics."""
        total_time = time.time() - context._query_start_time

        if total_time > SLOW_QUERY_THRESHOLD:
            # Truncate long statements for logging
            statement_preview = (
                statement[:200] + "..." if len(statement) > 200 else statement
            )
            logger.warning(
                f"Slow query detected: {total_time:.3f}s - {statement_preview}"
            )
        elif total_time > 0.1:  # Log moderately slow queries at debug level
            logger.debug(f"Query took {total_time:.3f}s")


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
    # Configure SQLite optimizations
    _configure_sqlite_engine()
    # Configure performance monitoring (optional, can be disabled in production)
    _configure_performance_monitoring()
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
