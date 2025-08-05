"""SQLite performance monitoring event listeners.

This module contains event listeners for tracking database query performance,
logging slow queries, and providing performance insights for optimization.
"""

import functools
import logging
import time

logger = logging.getLogger(__name__)

# Performance monitoring threshold for slow queries
SLOW_QUERY_THRESHOLD = 1.0  # Log queries taking longer than 1 second


def performance_monitor(func):
    """Decorator to monitor database operation performance.

    This decorator logs the execution time of database service methods,
    providing insights into query performance and helping identify
    performance bottlenecks.

    Args:
        func: The function to monitor.

    Returns:
        Wrapped function with performance monitoring.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__qualname__}"

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            if execution_time > SLOW_QUERY_THRESHOLD:
                logger.warning(
                    "Slow database operation: %s took %s",
                    func_name,
                    execution_time,
                )
            elif execution_time > 0.1:  # Log operations over 100ms as debug
                logger.debug(
                    "Database operation: %s took %s",
                    func_name,
                    execution_time,
                )

        except Exception:
            execution_time = time.time() - start_time
            logger.exception(
                "Database operation failed: %s failed after %s",
                func_name,
                execution_time,
            )
            raise
        else:
            return result

    return wrapper


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
        logger.warning("Slow query %s - %s", dt, preview)
    elif dt > 0.1:  # Log queries over 100ms as debug
        logger.debug("Query took %s", dt)
