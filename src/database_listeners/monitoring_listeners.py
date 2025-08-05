"""SQLite performance monitoring event listeners.

This module contains event listeners for tracking database query performance,
logging slow queries, and providing performance insights for optimization.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Performance monitoring threshold for slow queries
SLOW_QUERY_THRESHOLD = 1.0  # Log queries taking longer than 1 second


def start_timer(conn, cursor, stmt, params, ctx, many):
    """Start timing for query execution.

    This function is called before each SQL query execution to record
    the start time for performance monitoring.

    Args:
        conn: Database connection
        cursor: Database cursor
        stmt: SQL statement being executed
        params: Query parameters
        ctx: Execution context (used to store timing info)
        many: Whether this is a bulk operation

    Note:
        The start time is stored in ctx._query_start for later retrieval
        by the log_slow function.
    """
    ctx._query_start = time.time()


def log_slow(conn, cursor, stmt, params, ctx, many):
    """Log slow queries and performance metrics.

    This function is called after each SQL query execution to calculate
    execution time and log performance warnings for slow queries.

    Args:
        conn: Database connection
        cursor: Database cursor
        stmt: SQL statement that was executed
        params: Query parameters
        ctx: Execution context (contains timing info)
        many: Whether this was a bulk operation

    Note:
        Queries exceeding SLOW_QUERY_THRESHOLD are logged as warnings,
        while queries over 100ms are logged as debug information.
    """
    dt = time.time() - ctx._query_start
    if dt > SLOW_QUERY_THRESHOLD:
        preview = f"{stmt[:200]}..." if len(stmt) > 200 else stmt
        logger.warning(f"Slow query {dt:.3f}s – {preview}")
    elif dt > 0.1:  # Log queries over 100ms as debug
        logger.debug(f"Query took {dt:.3f}s")
