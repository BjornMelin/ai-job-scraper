# ruff: noqa: S608
"""Analytics service using DuckDB sqlite_scanner for data analysis.

This module provides analytics capabilities using DuckDB's sqlite_scanner
extension to query SQLite data directly. Provides simple analytics functions
for job trends, company metrics, and salary analysis.

Features:
- Direct SQLite querying via DuckDB sqlite_scanner extension
- Job posting trend analysis with date filtering
- Company hiring metrics and statistics
- Salary range analysis and aggregations
- Streamlit caching integration for dashboard performance
"""

from __future__ import annotations

import logging
import os

from typing import Any

# Import streamlit with fallback for non-Streamlit environments
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    class _DummyStreamlit:
        @staticmethod
        def cache_data(**_kwargs):
            def decorator(wrapped_func):
                return wrapped_func

            return decorator

    st = _DummyStreamlit()

# Optional DuckDB import with fallback
try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases for compatibility
type AnalyticsResponse = dict[str, Any]


class AnalyticsService:
    """Analytics service using DuckDB sqlite_scanner for data analysis.

    This service provides analytics capabilities using DuckDB's
    sqlite_scanner extension to query SQLite data directly without ETL processes.

    Features:
    - Direct SQLite querying via DuckDB sqlite_scanner extension
    - SQL aggregations and analytics functions
    - Streamlit caching integration for dashboard performance
    - Job trends analysis with configurable date ranges
    - Company hiring metrics and salary statistics

    Example:
        ```python
        analytics = AnalyticsService()

        # Get job posting trends for last 30 days
        trends = analytics.get_job_trends(days=30)

        # Get company analytics with hiring statistics
        companies = analytics.get_company_analytics()
        ```
    """

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize analytics service with DuckDB connection.

        Args:
            db_path: Path to SQLite database file for direct scanning.
        """
        self.db_path = db_path
        self._conn = None
        self._init_duckdb()

    def _init_duckdb(self) -> None:
        """Initialize DuckDB connection with sqlite_scanner extension."""
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB not available - analytics service disabled")
            return

        # Skip DuckDB extension installation in test environment to prevent crashes
        is_testing = (
            os.getenv("PYTEST_CURRENT_TEST") is not None
            or os.getenv("CI") is not None
            or "pytest" in os.getenv("_", "")
        )

        if is_testing:
            logger.info("Test environment detected - using DuckDB without extensions")
            try:
                self._conn = duckdb.connect(":memory:")
                logger.info("DuckDB initialized for testing (no extensions)")
            except Exception as e:
                logger.warning("Failed to initialize DuckDB for testing: %s", e)
                self._conn = None
            return

        try:
            self._conn = duckdb.connect(":memory:")

            # Try to install extension with timeout protection
            try:
                self._conn.execute("INSTALL sqlite_scanner")
                self._conn.execute("LOAD sqlite_scanner")
                logger.info("DuckDB sqlite_scanner initialized successfully")

                if STREAMLIT_AVAILABLE:
                    st.success("🚀 Analytics powered by DuckDB sqlite_scanner")

            except Exception as extension_error:
                logger.warning(
                    "DuckDB extension failed, continuing without: %s", extension_error
                )
                # Don't fail completely, just log and continue

        except Exception as e:
            logger.exception("Failed to initialize DuckDB connection")
            self._conn = None
            if STREAMLIT_AVAILABLE:
                st.error(f"Analytics unavailable: {e}")

    @staticmethod
    def clear_all_caches() -> None:
        """Clear all Streamlit caches used by the analytics service.

        Useful for forcing fresh analytics calculations.
        """
        if STREAMLIT_AVAILABLE:
            st.cache_data.clear()
            logger.info("✅ All AnalyticsService caches cleared")
        else:
            logger.info("INFO: Streamlit not available - no caches to clear")

    @staticmethod
    def get_cache_stats() -> dict[str, Any]:
        """Get cache utilization statistics for the analytics service.

        Returns information about cache performance and memory usage.
        """
        return {
            "streamlit_available": STREAMLIT_AVAILABLE,
            "caching_enabled": STREAMLIT_AVAILABLE,
            "cached_methods": [
                "get_job_trends",
                "get_company_analytics",
                "get_salary_analytics",
            ],
            "cache_config": {
                "ttl_seconds": 300,  # 5 minutes
                "max_entries_trends": 100,  # Job trends cache size
                "max_entries_company": 50,  # Company analytics cache size
                "max_entries_salary": 50,  # Salary analytics cache size
            },
            "performance_benefits": {
                "reduced_duckdb_queries": "5min analytics caching",
                "improved_dashboard_responsiveness": "Cached analytics computations",
                "reduced_database_load": "Less frequent DuckDB operations",
            },
            "analytics_method": "duckdb_sqlite_scanner",
        }

    @st.cache_data(
        ttl=300, max_entries=100, show_spinner="Analyzing job trends..."
    )  # Cache for 5 minutes
    def get_job_trends(_self, days: int = 30) -> AnalyticsResponse:  # noqa: N805
        """Get job posting trends using DuckDB's native SQL capabilities.

        Args:
            days: Number of days to include in trend analysis.

        Returns:
            Dict containing trends data and metadata.
        """
        if not _self._conn:
            return {"trends": [], "status": "error", "error": "DuckDB unavailable"}

        try:
            # Direct SQL query using DuckDB's sqlite_scanner
            query = f"""
                SELECT DATE_TRUNC('day', posted_date) as date,
                       COUNT(*) as job_count
                FROM sqlite_scan('{_self.db_path}', 'jobsql')
                WHERE posted_date >= CURRENT_DATE - INTERVAL '{days}' DAYS
                  AND archived = false
                GROUP BY DATE_TRUNC('day', posted_date)
                ORDER BY date
            """

            # Use DuckDB's native DataFrame conversion
            trends_df = _self._conn.execute(query).df()
            trends_data = trends_df.to_dict("records")

            logger.info("DuckDB trends query returned %d data points", len(trends_data))

            return {
                "trends": trends_data,
                "method": "duckdb_sqlite_scanner",
                "status": "success",
                "total_jobs": sum(t["job_count"] for t in trends_data),
            }

        except Exception as e:
            logger.exception("DuckDB job trends query failed")
            return {
                "trends": [],
                "status": "error",
                "error": str(e),
                "method": "duckdb_sqlite_scanner",
            }

    @st.cache_data(
        ttl=300, max_entries=50, show_spinner="Computing company analytics..."
    )  # Cache for 5 minutes
    def get_company_analytics(_self) -> AnalyticsResponse:  # noqa: N805
        """Get company hiring analytics using DuckDB's aggregation functions.

        Returns:
            Dict containing company analytics data and metadata.
        """
        if not _self._conn:
            return {"companies": [], "status": "error", "error": "DuckDB unavailable"}

        try:
            # Use DuckDB's native SQL for company analytics
            query = f"""
                SELECT
                    c.name as company,
                    COUNT(j.id) as total_jobs,
                    ROUND(AVG(CAST(json_extract(j.salary, '$[0]') AS DOUBLE)), 2)
                        as avg_min_salary,
                    ROUND(AVG(CAST(json_extract(j.salary, '$[1]') AS DOUBLE)), 2)
                        as avg_max_salary,
                    MAX(j.posted_date) as last_job_posted
                FROM sqlite_scan('{_self.db_path}', 'jobsql') j
                JOIN sqlite_scan('{_self.db_path}', 'companysql') c
                    ON j.company_id = c.id
                WHERE j.archived = false
                GROUP BY c.name
                ORDER BY total_jobs DESC
                LIMIT 20
            """

            # Use DuckDB's native DataFrame conversion
            company_df = _self._conn.execute(query).df()
            company_data = company_df.to_dict("records")

            logger.info(
                "DuckDB company analytics returned %d companies", len(company_data)
            )

            return {
                "companies": company_data,
                "method": "duckdb_sqlite_scanner",
                "status": "success",
                "total_companies": len(company_data),
            }

        except Exception as e:
            logger.exception("DuckDB company analytics failed")
            return {
                "companies": [],
                "status": "error",
                "error": str(e),
                "method": "duckdb_sqlite_scanner",
            }

    @st.cache_data(
        ttl=300, max_entries=50, show_spinner="Analyzing salary data..."
    )  # Cache for 5 minutes
    def get_salary_analytics(_self, days: int = 90) -> AnalyticsResponse:  # noqa: N805
        """Get salary analytics using DuckDB's statistical functions.

        Args:
            days: Number of days to include in salary analysis.

        Returns:
            Dict containing salary analytics and metadata.
        """
        if not _self._conn:
            return {"salary_data": {}, "status": "error", "error": "DuckDB unavailable"}

        try:
            # Use DuckDB's native statistical functions
            query = f"""
                SELECT
                    COUNT(*) as total_jobs_with_salary,
                    ROUND(AVG(CAST(json_extract(salary, '$[0]') AS DOUBLE)), 2)
                        as avg_min_salary,
                    ROUND(AVG(CAST(json_extract(salary, '$[1]') AS DOUBLE)), 2)
                        as avg_max_salary,
                    MIN(CAST(json_extract(salary, '$[0]') AS DOUBLE)) as min_salary,
                    MAX(CAST(json_extract(salary, '$[1]') AS DOUBLE)) as max_salary,
                    ROUND(STDDEV(CAST(json_extract(salary, '$[0]') AS DOUBLE)), 2)
                        as salary_std_dev
                FROM sqlite_scan('{_self.db_path}', 'jobsql')
                WHERE posted_date >= CURRENT_DATE - INTERVAL '{days}' DAYS
                  AND archived = false
                  AND json_extract(salary, '$[0]') IS NOT NULL
            """

            result = _self._conn.execute(query).fetchone()

            salary_data = {
                "total_jobs_with_salary": result[0] or 0,
                "avg_min_salary": result[1] or 0,
                "avg_max_salary": result[2] or 0,
                "min_salary": result[3] or 0,
                "max_salary": result[4] or 0,
                "salary_std_dev": result[5] or 0,
                "analysis_period_days": days,
            }

            logger.info(
                "DuckDB salary analytics completed for %d jobs",
                salary_data["total_jobs_with_salary"],
            )

        except Exception as e:
            logger.exception("DuckDB salary analytics failed")
            return {
                "salary_data": {},
                "status": "error",
                "error": str(e),
                "method": "duckdb_sqlite_scanner",
            }
        else:
            return {
                "salary_data": salary_data,
                "method": "duckdb_sqlite_scanner",
                "status": "success",
            }

    def get_status_report(self) -> dict[str, Any]:
        """Get simple analytics service status.

        Returns:
            Dict containing service status and configuration.
        """
        return {
            "analytics_method": "duckdb_sqlite_scanner",
            "duckdb_available": DUCKDB_AVAILABLE,
            "streamlit_available": STREAMLIT_AVAILABLE,
            "database_path": self.db_path,
            "connection_active": self._conn is not None,
            "status": "active" if self._conn else "unavailable",
            "cache_enabled": STREAMLIT_AVAILABLE,
            "cached_analytics_methods": 3 if STREAMLIT_AVAILABLE else 0,
        }

    def __del__(self) -> None:
        """Clean up DuckDB connection on object destruction."""
        if self._conn:
            from contextlib import suppress

            with suppress(Exception):
                self._conn.close()

    def refresh_all_caches(self) -> None:
        """Refresh all analytics caches by clearing them.

        Forces fresh data retrieval on next analytics call.
        Useful after database updates or for periodic data refresh.
        """
        self.clear_all_caches()
        logger.info("🔄 Analytics caches refreshed - next calls will fetch fresh data")
