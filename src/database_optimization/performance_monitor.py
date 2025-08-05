"""Database performance monitoring utilities.

This module provides tools for monitoring database query performance,
connection pool usage, and identifying optimization opportunities.
"""

import contextlib
import logging
import time

from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from src.database import get_connection_pool_status

logger = logging.getLogger(__name__)


@contextmanager
def query_timer(operation_name: str) -> Generator[dict, None, None]:
    """Context manager to time database operations and log performance.

    Args:
        operation_name: Description of the database operation being timed.

    Yields:
        Dictionary to store additional performance metrics.

    Example:
        with query_timer("get_all_companies") as metrics:
            companies = CompanyService.get_all_companies()
            metrics["record_count"] = len(companies)
    """
    start_time = time.time()
    pool_status_before = get_connection_pool_status()
    metrics = {}

    try:
        yield metrics

    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        pool_status_after = get_connection_pool_status()

        # Log performance metrics
        log_msg = f"🔍 {operation_name}: {execution_time:.3f}s"

        if "record_count" in metrics:
            log_msg += f" | {metrics['record_count']} records"

        if execution_time > 1.0:
            logger.warning(f"⚠️  SLOW QUERY: {log_msg}")
        elif execution_time > 0.5:
            logger.info(f"📊 {log_msg}")
        else:
            logger.debug(f"⚡ {log_msg}")

        # Log connection pool changes if significant
        pool_before = pool_status_before.get("checked_out", 0)
        pool_after = pool_status_after.get("checked_out", 0)

        if (
            isinstance(pool_before, int)
            and isinstance(pool_after, int)
            and abs(pool_after - pool_before) > 2
        ):
            logger.info(f"🔄 Connection pool: {pool_before} -> {pool_after}")


def performance_monitor(operation_name: str = None):
    """Decorator to monitor the performance of database service methods.

    Args:
        operation_name: Optional name for the operation. If not provided,
                       uses the function name.

    Example:
        @performance_monitor("bulk_company_updates")
        def bulk_update_scrape_stats(self, updates):
            # method implementation
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            with query_timer(op_name) as metrics:
                result = func(*args, **kwargs)

                # Try to extract record count from common return types
                if isinstance(result, list):
                    metrics["record_count"] = len(result)
                elif isinstance(result, dict) and "record_count" in str(result):
                    # Handle dictionary results with embedded counts
                    pass
                elif hasattr(result, "__len__"):
                    with contextlib.suppress(Exception):
                        metrics["record_count"] = len(result)

                return result

        return wrapper

    return decorator


def analyze_query_patterns():
    """Analyze and report on database query patterns and performance.

    This function examines the current database state and provides
    recommendations for further optimization.
    """
    from sqlalchemy import text

    from src.database import engine

    logger.info("📈 Analyzing Database Query Patterns")
    logger.info("=" * 50)

    with engine.connect() as conn:
        try:
            # Get table row counts
            count = conn.execute(text("SELECT COUNT(*) FROM companysql")).scalar()
            logger.info(f"📊 companysql: {count:,} records")

            count = conn.execute(text("SELECT COUNT(*) FROM jobsql")).scalar()
            logger.info(f"📊 jobsql: {count:,} records")

            # Get index usage statistics (SQLite specific)
            index_sql = text("""
                SELECT name, tbl_name 
                FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY tbl_name, name
            """)

            indexes = conn.execute(index_sql).fetchall()
            logger.info(f"\n🔍 Database Indexes: {len(indexes)} total")

            current_table = None
            for index_name, table_name in indexes:
                if table_name != current_table:
                    current_table = table_name
                    logger.info(f"\n  {table_name.upper()}:")
                logger.info(f"    ✓ {index_name}")

        except Exception as e:
            logger.error(f"Error analyzing query patterns: {e}")

    # Get current connection pool status
    pool_status = get_connection_pool_status()
    logger.info("\n🔄 Connection Pool Status:")
    for key, value in pool_status.items():
        if key != "error":
            logger.info(f"  {key}: {value}")

    logger.info("\n💡 Performance Optimization Tips:")
    logger.info("  • Use get_companies_with_job_counts() for UI displays")
    logger.info("  • Use bulk_update_scrape_stats() for batch operations")
    logger.info("  • Monitor query times with performance_monitor decorator")
    logger.info("  • Enable db_monitoring in settings for detailed logging")


def benchmark_operations():
    """Run benchmarks on common database operations.

    This function performs timing tests on key database operations
    to measure the impact of optimizations.
    """
    from src.services.company_service import CompanyService

    logger.info("🏃 Running Database Performance Benchmarks")
    logger.info("=" * 50)

    operations = [
        ("get_all_companies", lambda: CompanyService.get_all_companies()),
        ("get_active_companies", lambda: CompanyService.get_active_companies()),
    ]

    # Check if optimized method exists
    if hasattr(CompanyService, "get_companies_with_job_counts"):
        operations.append(
            (
                "get_companies_with_job_counts",
                lambda: CompanyService.get_companies_with_job_counts(),
            )
        )

    for op_name, op_func in operations:
        try:
            # Run operation multiple times and average
            times = []
            for _ in range(3):
                with query_timer(f"benchmark_{op_name}") as metrics:
                    result = op_func()
                    if isinstance(result, list):
                        metrics["record_count"] = len(result)

                # Extract timing from the last log entry (simplified approach)
                times.append(time.time())

            logger.info(f"✅ {op_name}: Completed successfully")

        except Exception as e:
            logger.error(f"❌ {op_name}: Failed - {e}")

    logger.info("\n🎯 Benchmark Complete!")
    logger.info("Check logs above for detailed timing information.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("🔧 Database Performance Monitor")

    try:
        analyze_query_patterns()
        print("\n" + "=" * 50 + "\n")
        benchmark_operations()

    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        raise
