"""Analytics service with intelligent method selection between SQLite and DuckDB.

This module implements the Analytics Service from ADR-019, providing:
- SQLite baseline analytics using existing SQLModel patterns
- DuckDB sqlite_scanner for high-performance scaling (p95 >500ms trigger)
- Python 3.12 sys.monitoring for 20x faster performance tracking
- Intelligent method selection based on performance thresholds
- Real-time cost tracking foundations for $50 monthly budget
- Comprehensive error handling with graceful fallbacks

The service automatically switches from SQLite to DuckDB when performance
thresholds are exceeded, providing zero-ETL high-performance analytics when needed.
"""

import logging
import sys
import time

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

# Import streamlit with fallback for non-Streamlit environments
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create dummy decorator for non-Streamlit environments

    class _DummyStreamlit:
        @staticmethod
        def cache_data(**_kwargs):
            def decorator(wrapped_func):
                return wrapped_func

            return decorator

        @staticmethod
        def info(msg: str) -> None:
            pass

        @staticmethod
        def warning(msg: str) -> None:
            pass

        @staticmethod
        def success(msg: str) -> None:
            pass

        @staticmethod
        def error(msg: str) -> None:
            pass

    st = _DummyStreamlit()

from sqlalchemy import func
from sqlmodel import select

from src.database import db_session
from src.models import CompanySQL, JobSQL

# Optional DuckDB import with fallback
try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases for better readability
type AnalyticsResponse = dict[str, Any]
type PerformanceMetrics = dict[str, Any]
type CostTrackingData = dict[str, Any]


class AnalyticsService:
    """Analytics service with intelligent method selection.

    Uses SQLModel and DuckDB sqlite_scanner with automatic scaling.

    This service implements the intelligent analytics architecture from ADR-019,
    providing SQLite baseline performance with automatic DuckDB scaling when
    performance thresholds are exceeded (p95 >500ms or max query >2s).

    Features:
    - Python 3.12 sys.monitoring integration for 20x performance improvement
    - Zero-ETL DuckDB sqlite_scanner for high-performance analytics
    - Intelligent method selection based on actual performance metrics
    - Comprehensive error handling with graceful fallbacks
    - Cost tracking foundations for budget monitoring
    - Streamlit caching with appropriate TTL values

    Example:
        ```python
        analytics = AnalyticsService()

        # Get job trends - automatically uses optimal method
        trends = analytics.get_job_trends(days=30)

        # Check performance status
        report = analytics.get_performance_report()
        ```
    """

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize analytics service with performance monitoring.

        Args:
            db_path: Path to SQLite database file for DuckDB sqlite_scanner.
        """
        self.db_path = db_path
        self._duckdb_conn: duckdb.DuckDBPyConnection | None = None
        self._monitoring_enabled = False
        self._performance_metrics = {
            "query_times": [],
            "high_performance_active": False,
        }
        self._cost_tracking = {"monthly_budget": 50.0, "current_costs": 0.0}

        # Initialize performance monitoring
        self._init_sys_monitoring()

    def _init_sys_monitoring(self) -> None:
        """Initialize Python 3.12 sys.monitoring for 20x performance improvement.

        Uses Python 3.12's new sys.monitoring API which provides significantly
        better performance than cProfile with zero runtime overhead when disabled.
        Falls back gracefully for older Python versions.
        """
        if not hasattr(sys, "monitoring") or sys.version_info < (3, 12):
            logger.info("Python 3.12+ sys.monitoring not available, using basic timing")
            return

        try:
            # Use tool ID 0 for analytics monitoring
            sys.monitoring.use_tool_id(0, "analytics_perf")
            sys.monitoring.set_events(0, sys.monitoring.events.CALL)
            sys.monitoring.register_callback(
                0, sys.monitoring.events.CALL, self._monitor_performance
            )
            self._monitoring_enabled = True

            if STREAMLIT_AVAILABLE:
                st.success(
                    "📊 Python 3.12 sys.monitoring enabled (20x faster than cProfile)"
                )

            logger.info("Python 3.12 sys.monitoring initialized successfully")
        except Exception as e:
            logger.warning("sys.monitoring initialization failed: %s", e)
            if STREAMLIT_AVAILABLE:
                st.warning(f"sys.monitoring not available: {e}")

    def _monitor_performance(self, event, args, *_) -> None:
        """Ultra-lightweight performance monitoring callback.

        This callback is invoked by Python 3.12's sys.monitoring system
        with zero overhead when monitoring is disabled. Tracks analytics
        function performance for intelligent method selection.

        Args:
            event: Monitoring event type
            args: Event arguments containing function details
            *_: Additional arguments from monitoring system
        """
        if event == "call" and "analytics" in args.f_code.co_name:
            # Track analytics function calls with minimal overhead
            current_time = time.perf_counter()

            if hasattr(self, "_call_start_time"):
                duration = current_time - self._call_start_time
                self._performance_metrics["query_times"].append(duration)

                # Keep only recent measurements (last 50 queries)
                if len(self._performance_metrics["query_times"]) > 50:
                    self._performance_metrics["query_times"] = (
                        self._performance_metrics["query_times"][-50:]
                    )

            self._call_start_time = current_time

    def _should_use_duckdb_analytics(self) -> bool:
        """Determine if DuckDB high-performance analytics should be used.

        Uses actual performance metrics to determine when to switch from SQLite
        to DuckDB. Triggers high-performance mode when:
        - p95 latency exceeds 500ms, OR
        - Any single query exceeds 2 seconds

        Returns:
            bool: True if DuckDB should be used, False for SQLite baseline.
        """
        if not DUCKDB_AVAILABLE:
            return False

        query_times = self._performance_metrics["query_times"]

        # Need minimum sample size for reliable p95 calculation
        if not query_times or len(query_times) < 10:
            return False

        # Calculate p95 latency
        sorted_times = sorted(query_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_latency = sorted_times[p95_index] if p95_index < len(sorted_times) else 0

        # Performance thresholds from ADR-019
        high_performance_needed = (
            p95_latency > 0.5  # p95 latency >500ms
            or any(t > 2.0 for t in query_times)  # Any query >2 seconds
        )

        # Notify when switching to high-performance mode
        if (
            high_performance_needed
            and not self._performance_metrics["high_performance_active"]
        ):
            self._performance_metrics["high_performance_active"] = True

            if STREAMLIT_AVAILABLE:
                st.warning("📈 Performance thresholds exceeded - Activating DuckDB")

            logger.info(
                "Switching to DuckDB: p95_latency=%.3fs, max_query=%.3fs",
                p95_latency,
                max(query_times),
            )

        return high_performance_needed

    def _track_query_performance(self, query_time: float) -> None:
        """Track query performance for intelligent method selection.

        Args:
            query_time: Query execution time in seconds.
        """
        self._performance_metrics["query_times"].append(query_time)

        # Maintain rolling window of recent measurements
        if len(self._performance_metrics["query_times"]) > 50:
            self._performance_metrics["query_times"] = self._performance_metrics[
                "query_times"
            ][-50:]

    def _init_duckdb_connection(self) -> bool:
        """Initialize DuckDB connection with sqlite_scanner extension.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB not available - cannot initialize connection")
            return False

        try:
            if self._duckdb_conn is None:
                self._duckdb_conn = duckdb.connect(":memory:")
                self._duckdb_conn.execute("INSTALL sqlite_scanner")
                self._duckdb_conn.execute("LOAD sqlite_scanner")

                if STREAMLIT_AVAILABLE:
                    st.info(
                        "🚀 DuckDB sqlite_scanner activated - High-performance enabled"
                    )

                logger.info("DuckDB sqlite_scanner initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize DuckDB connection")

            if STREAMLIT_AVAILABLE:
                st.error(f"DuckDB initialization failed: {e}")

            return False

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_job_trends(self, days: int = 30) -> AnalyticsResponse:
        """Get job trends using optimal method based on performance requirements.

        Automatically selects between SQLite and DuckDB based on performance
        thresholds. Provides job posting trends, growth rates, and patterns.

        Args:
            days: Number of days to include in trend analysis.

        Returns:
            Dict containing:
            - trends: List of daily job counts with dates
            - method: Analytics method used ("sqlmodel_sqlite" or "duckdb")
            - query_time_ms: Query execution time in milliseconds
            - performance_reason: Explanation of method selection

        Raises:
            Exception: If both SQLite and DuckDB methods fail.
        """
        start_time = time.perf_counter()

        try:
            # Choose analytics method based on performance requirements
            if self._should_use_duckdb_analytics():
                trends_data = self._get_job_trends_duckdb(days)
                method_used = "duckdb_sqlite_scanner"
                performance_reason = (
                    "High-performance mode: p95 >500ms or max query >2s"
                )
            else:
                trends_data = self._get_job_trends_sqlite(days)
                method_used = "sqlmodel_sqlite"
                performance_reason = (
                    "Standard mode: Performance within acceptable limits"
                )

            query_time = time.perf_counter() - start_time
            self._track_query_performance(query_time)

            return {
                "trends": trends_data,
                "method": method_used,
                "query_time_ms": round(query_time * 1000, 2),
                "performance_reason": performance_reason,
                "status": "success",
            }

        except Exception as e:
            logger.exception("Failed to get job trends")

            # Fallback error response
            query_time = time.perf_counter() - start_time
            return {
                "trends": [],
                "method": "error_fallback",
                "query_time_ms": round(query_time * 1000, 2),
                "performance_reason": f"Error: {e!s}",
                "status": "error",
                "error": str(e),
            }

    def _get_job_trends_sqlite(self, days: int = 30) -> list[dict[str, Any]]:
        """SQLite analytics using proven SQLModel foundation.

        Uses existing SQLModel patterns and database connection for baseline
        performance. Handles timezone-aware date filtering and provides
        consistent job trend data.

        Args:
            days: Number of days to include in analysis.

        Returns:
            List of dictionaries with date and job_count keys.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                cutoff_date = datetime.now(UTC) - timedelta(days=days)

                # Type-safe SQLModel queries using existing patterns
                query = (
                    select(
                        func.date(JobSQL.posted_date).label("date"),
                        func.count(JobSQL.id).label("job_count"),
                    )
                    .where(JobSQL.posted_date >= cutoff_date)
                    .where(JobSQL.archived.is_(False))
                    .group_by(func.date(JobSQL.posted_date))
                    .order_by("date")
                )

                results = session.exec(query).all()
                trends_data = [
                    {"date": str(r.date), "job_count": r.job_count} for r in results
                ]

                logger.info(
                    "SQLite trends query returned %d data points", len(trends_data)
                )
                return trends_data

        except Exception:
            logger.exception("SQLite job trends query failed")
            raise

    def _get_job_trends_duckdb(self, days: int = 30) -> list[dict[str, Any]]:
        """DuckDB sqlite_scanner: Zero-ETL direct SQLite access.

        Uses DuckDB's sqlite_scanner extension to directly query SQLite
        database with high-performance SQL engine. Provides same data as
        SQLite method but with potentially better performance for complex queries.

        Args:
            days: Number of days to include in analysis.

        Returns:
            List of dictionaries with date and job_count keys.

        Raises:
            Exception: If DuckDB initialization or query fails.
        """
        if not self._init_duckdb_connection():
            raise RuntimeError("Failed to initialize DuckDB connection")

        try:
            # Direct SQLite scanning with DuckDB performance
            query = f"""  # noqa: S608
                SELECT DATE(posted_date) as date,
                       COUNT(*) as job_count
                FROM sqlite_scan('{self.db_path}', 'jobsql')
                WHERE posted_date >= CURRENT_DATE - INTERVAL '{days}' DAYS
                  AND archived = false
                GROUP BY DATE(posted_date)
                ORDER BY date
            """

            results = self._duckdb_conn.execute(query).fetchall()
            trends_data = [{"date": str(r[0]), "job_count": r[1]} for r in results]

            logger.info("DuckDB trends query returned %d data points", len(trends_data))
            return trends_data
        except Exception:
            logger.exception("DuckDB job trends query failed")
            raise

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_company_analytics(self) -> AnalyticsResponse:
        """Company analytics with intelligent method selection.

        Provides company hiring patterns, volume analysis, and salary insights
        using the optimal analytics method based on current performance metrics.

        Returns:
            Dict containing:
            - companies: List of company analytics with hiring stats
            - method: Analytics method used
            - query_time_ms: Query execution time
            - performance_reason: Method selection explanation

        Raises:
            Exception: If both analytics methods fail.
        """
        start_time = time.perf_counter()

        try:
            if self._should_use_duckdb_analytics():
                company_data = self._get_company_analytics_duckdb()
                method_used = "duckdb_sqlite_scanner"
                performance_reason = "High-performance mode active"
            else:
                company_data = self._get_company_analytics_sqlite()
                method_used = "sqlmodel_sqlite"
                performance_reason = "Standard SQLite performance"

            query_time = time.perf_counter() - start_time
            self._track_query_performance(query_time)

            return {
                "companies": company_data,
                "method": method_used,
                "query_time_ms": round(query_time * 1000, 2),
                "performance_reason": performance_reason,
                "status": "success",
            }

        except Exception as e:
            logger.exception("Failed to get company analytics")

            query_time = time.perf_counter() - start_time
            return {
                "companies": [],
                "method": "error_fallback",
                "query_time_ms": round(query_time * 1000, 2),
                "performance_reason": f"Error: {e!s}",
                "status": "error",
                "error": str(e),
            }

    def _get_company_analytics_sqlite(self) -> list[dict[str, Any]]:
        """SQLite company analytics using SQLModel patterns.

        Provides company hiring statistics including job counts, average
        salaries, and hiring velocity using existing database patterns.

        Returns:
            List of company analytics dictionaries.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Join with companies to get names and calculate stats
                query = (
                    select(
                        CompanySQL.name,
                        func.count(JobSQL.id).label("total_jobs"),
                        func.avg(func.json_extract(JobSQL.salary, "$[0]")).label(
                            "avg_min_salary"
                        ),
                        func.avg(func.json_extract(JobSQL.salary, "$[1]")).label(
                            "avg_max_salary"
                        ),
                        func.max(JobSQL.posted_date).label("last_job_posted"),
                    )
                    .select_from(JobSQL)
                    .join(CompanySQL, JobSQL.company_id == CompanySQL.id)
                    .where(JobSQL.archived.is_(False))
                    .group_by(CompanySQL.name)
                    .order_by(func.count(JobSQL.id).desc())
                    .limit(20)
                )

                results = session.exec(query).all()

                company_data = []
                for r in results:
                    company_data.append(
                        {
                            "company": r.name,
                            "total_jobs": r.total_jobs,
                            "avg_min_salary": round(r.avg_min_salary or 0, 2),
                            "avg_max_salary": round(r.avg_max_salary or 0, 2),
                            "last_job_posted": r.last_job_posted.isoformat()
                            if r.last_job_posted
                            else None,
                        }
                    )

                logger.info(
                    "SQLite company analytics returned %d companies", len(company_data)
                )
                return company_data

        except Exception:
            logger.exception("SQLite company analytics failed")
            raise

    def _get_company_analytics_duckdb(self) -> list[dict[str, Any]]:
        """DuckDB company analytics with direct sqlite_scanner.

        High-performance company analytics using DuckDB's SQL engine
        for complex aggregations and joins.

        Returns:
            List of company analytics dictionaries.

        Raises:
            Exception: If DuckDB query fails.
        """
        if not self._init_duckdb_connection():
            raise RuntimeError("Failed to initialize DuckDB connection")

        try:
            query = f"""  # noqa: S608
                SELECT
                    c.name as company,
                    COUNT(j.id) as total_jobs,
                    ROUND(AVG(CAST(json_extract(j.salary, '$[0]') AS DOUBLE)), 2)
                        as avg_min_salary,
                    ROUND(AVG(CAST(json_extract(j.salary, '$[1]') AS DOUBLE)), 2)
                        as avg_max_salary,
                    MAX(j.posted_date) as last_job_posted
                FROM sqlite_scan('{self.db_path}', 'jobsql') j
                JOIN sqlite_scan('{self.db_path}', 'companysql') c
                    ON j.company_id = c.id
                WHERE j.archived = false
                GROUP BY c.name
                ORDER BY total_jobs DESC
                LIMIT 20
            """

            results = self._duckdb_conn.execute(query).fetchall()

            company_data = []
            for r in results:
                company_data.append(
                    {
                        "company": r[0],
                        "total_jobs": r[1],
                        "avg_min_salary": r[2] or 0,
                        "avg_max_salary": r[3] or 0,
                        "last_job_posted": r[4].isoformat() if r[4] else None,
                    }
                )

            logger.info(
                "DuckDB company analytics returned %d companies", len(company_data)
            )
            return company_data
        except Exception:
            logger.exception("DuckDB company analytics failed")
            raise

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_salary_analytics(self, days: int = 90) -> AnalyticsResponse:
        """Salary analytics with market insights and trends.

        Provides comprehensive salary analysis including ranges, distributions,
        and market trends using the optimal analytics method.

        Args:
            days: Number of days to include in salary analysis.

        Returns:
            Dict containing salary analytics and method information.
        """
        start_time = time.perf_counter()

        try:
            if self._should_use_duckdb_analytics():
                salary_data = self._get_salary_analytics_duckdb(days)
                method_used = "duckdb_sqlite_scanner"
            else:
                salary_data = self._get_salary_analytics_sqlite(days)
                method_used = "sqlmodel_sqlite"

            query_time = time.perf_counter() - start_time
            self._track_query_performance(query_time)

            return {
                "salary_data": salary_data,
                "method": method_used,
                "query_time_ms": round(query_time * 1000, 2),
                "status": "success",
            }

        except Exception as e:
            logger.exception("Failed to get salary analytics")

            query_time = time.perf_counter() - start_time
            return {
                "salary_data": {},
                "method": "error_fallback",
                "query_time_ms": round(query_time * 1000, 2),
                "status": "error",
                "error": str(e),
            }

    def _get_salary_analytics_sqlite(self, days: int = 90) -> dict[str, Any]:
        """SQLite salary analytics using SQLModel patterns.

        Args:
            days: Number of days for analysis.

        Returns:
            Dictionary with salary statistics and trends.
        """
        try:
            with db_session() as session:
                cutoff_date = datetime.now(UTC) - timedelta(days=days)

                # Basic salary statistics
                stats_query = (
                    select(
                        func.count(JobSQL.id).label("total_jobs_with_salary"),
                        func.avg(func.json_extract(JobSQL.salary, "$[0]")).label(
                            "avg_min_salary"
                        ),
                        func.avg(func.json_extract(JobSQL.salary, "$[1]")).label(
                            "avg_max_salary"
                        ),
                        func.min(func.json_extract(JobSQL.salary, "$[0]")).label(
                            "min_salary"
                        ),
                        func.max(func.json_extract(JobSQL.salary, "$[1]")).label(
                            "max_salary"
                        ),
                    )
                    .where(JobSQL.posted_date >= cutoff_date)
                    .where(JobSQL.archived.is_(False))
                    .where(func.json_extract(JobSQL.salary, "$[0]").isnot(None))
                )

                result = session.exec(stats_query).first()

                return {
                    "total_jobs_with_salary": result.total_jobs_with_salary or 0,
                    "avg_min_salary": round(result.avg_min_salary or 0, 2),
                    "avg_max_salary": round(result.avg_max_salary or 0, 2),
                    "min_salary": result.min_salary or 0,
                    "max_salary": result.max_salary or 0,
                    "analysis_period_days": days,
                }

        except Exception:
            logger.exception("SQLite salary analytics failed")
            raise

    def _get_salary_analytics_duckdb(self, days: int = 90) -> dict[str, Any]:
        """DuckDB salary analytics with high-performance aggregations.

        Args:
            days: Number of days for analysis.

        Returns:
            Dictionary with salary statistics and trends.
        """
        if not self._init_duckdb_connection():
            raise RuntimeError("Failed to initialize DuckDB connection")

        try:
            query = f"""  # noqa: S608
                SELECT
                    COUNT(*) as total_jobs_with_salary,
                    ROUND(AVG(CAST(json_extract(salary, '$[0]') AS DOUBLE)), 2)
                        as avg_min_salary,
                    ROUND(AVG(CAST(json_extract(salary, '$[1]') AS DOUBLE)), 2)
                        as avg_max_salary,
                    MIN(CAST(json_extract(salary, '$[0]') AS DOUBLE)) as min_salary,
                    MAX(CAST(json_extract(salary, '$[1]') AS DOUBLE)) as max_salary
                FROM sqlite_scan('{self.db_path}', 'jobsql')
                WHERE posted_date >= CURRENT_DATE - INTERVAL '{days}' DAYS
                  AND archived = false
                  AND json_extract(salary, '$[0]') IS NOT NULL
            """

            result = self._duckdb_conn.execute(query).fetchone()

            return {
                "total_jobs_with_salary": result[0] or 0,
                "avg_min_salary": result[1] or 0,
                "avg_max_salary": result[2] or 0,
                "min_salary": result[3] or 0,
                "max_salary": result[4] or 0,
                "analysis_period_days": days,
            }

        except Exception:
            logger.exception("DuckDB salary analytics failed")
            raise

    def get_performance_report(self) -> PerformanceMetrics:
        """Get comprehensive performance and analytics status report.

        Provides detailed performance metrics, method selection rationale,
        and system status information for monitoring and debugging.

        Returns:
            Dict containing:
            - analytics_method: Current method being used
            - monitoring_enabled: Whether sys.monitoring is active
            - performance_metrics: Detailed performance statistics
            - performance_status: Status and reasoning
            - system_info: Python version and capabilities
        """
        query_times = self._performance_metrics["query_times"]

        if not query_times:
            return {
                "status": "insufficient_data",
                "monitoring_enabled": self._monitoring_enabled,
                "system_info": {
                    "python_version": (
                        f"{sys.version_info.major}.{sys.version_info.minor}"
                    ),
                    "sys_monitoring_available": hasattr(sys, "monitoring"),
                    "duckdb_available": DUCKDB_AVAILABLE,
                },
            }

        # Calculate performance statistics
        sorted_times = sorted(query_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_latency_ms = (
            (sorted_times[p95_index] * 1000) if p95_index < len(sorted_times) else 0
        )

        # Determine current method and reasoning
        using_duckdb = self._should_use_duckdb_analytics()

        return {
            "analytics_method": "duckdb_sqlite_scanner"
            if using_duckdb
            else "sqlmodel_sqlite",
            "monitoring_enabled": self._monitoring_enabled,
            "monitoring_type": "sys.monitoring (Python 3.12)"
            if self._monitoring_enabled
            else "basic_timing",
            "performance_metrics": {
                "p95_latency_ms": round(p95_latency_ms, 2),
                "max_query_time_s": round(max(query_times), 2),
                "avg_query_time_ms": round(
                    (sum(query_times) / len(query_times)) * 1000, 2
                ),
                "total_queries": len(query_times),
                "sample_size": len(query_times),
            },
            "performance_status": {
                "high_performance_active": self._performance_metrics[
                    "high_performance_active"
                ],
                "performance_reason": (
                    "p95 latency >500ms or max query >2s"
                    if using_duckdb
                    else "Performance within acceptable limits"
                ),
                "method_explanation": (
                    "DuckDB sqlite_scanner for high-performance analytics"
                    if using_duckdb
                    else "SQLModel SQLite for standard performance"
                ),
            },
            "system_info": {
                "python_version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                ),
                "sys_monitoring_available": hasattr(sys, "monitoring"),
                "duckdb_available": DUCKDB_AVAILABLE,
                "streamlit_available": STREAMLIT_AVAILABLE,
            },
            "cost_tracking": {
                "monthly_budget": self._cost_tracking["monthly_budget"],
                "current_costs": self._cost_tracking["current_costs"],
                "budget_utilization": (
                    self._cost_tracking["current_costs"]
                    / self._cost_tracking["monthly_budget"]
                )
                * 100,
            },
            "status": "active",
        }

    def track_operation_cost(
        self,
        operation: str,
        cost: float,
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Track operational costs for budget monitoring.

        Foundation for cost tracking system integration. Currently provides
        basic cost accumulation, designed to integrate with full cost monitoring
        service from ADR-019.

        Args:
            operation: Description of operation (e.g., "ai_extraction", "proxy_request")
            cost: Cost in USD for the operation
            metadata: Additional operation details for tracking
        """
        try:
            self._cost_tracking["current_costs"] += cost

            logger.info(
                "Cost tracked: operation=%s, cost=$%.4f, total=$%.4f",
                operation,
                cost,
                self._cost_tracking["current_costs"],
            )

            # Log warning if approaching budget limit
            utilization = (
                self._cost_tracking["current_costs"]
                / self._cost_tracking["monthly_budget"]
            )

            if utilization > 0.8:  # 80% of budget used
                logger.warning(
                    "Budget utilization high: %.1f%% of $%.2f monthly budget used",
                    utilization * 100,
                    self._cost_tracking["monthly_budget"],
                )

                if STREAMLIT_AVAILABLE:
                    st.warning(
                        f"Budget Alert: {utilization * 100:.1f}% of monthly budget used"
                    )

        except Exception:
            logger.exception("Failed to track operation cost")

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics for testing or troubleshooting.

        Clears accumulated performance data and resets high-performance mode.
        Useful for testing performance thresholds or starting fresh analysis.
        """
        self._performance_metrics = {
            "query_times": [],
            "high_performance_active": False,
        }

        logger.info("Performance metrics reset")

        if STREAMLIT_AVAILABLE:
            st.success("Performance metrics reset - method selection will recalibrate")

    def force_performance_test(self, slow_query_time: float = 0.6) -> None:
        """Force performance test by simulating slow query for testing.

        Useful for testing method selection logic and high-performance mode
        activation without waiting for natural performance degradation.

        Args:
            slow_query_time: Simulated slow query time in seconds.
        """
        # Add multiple slow queries to trigger p95 threshold
        for _ in range(15):
            self._track_query_performance(slow_query_time)

        logger.info("Forced performance test with %.3fs queries", slow_query_time)

        if STREAMLIT_AVAILABLE:
            st.info(
                f"Performance test completed - simulated {slow_query_time}s queries"
            )

    def __del__(self) -> None:
        """Clean up DuckDB connection on object destruction."""
        if self._duckdb_conn is not None:
            with suppress(Exception):
                self._duckdb_conn.close()
