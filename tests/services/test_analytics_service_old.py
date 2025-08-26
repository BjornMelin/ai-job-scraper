"""Comprehensive tests for the refactored DuckDB sqlite_scanner AnalyticsService.

This test suite validates the simplified analytics service implementation that uses
DuckDB's native sqlite_scanner for zero-ETL analytics queries against SQLite data.

Test coverage includes:
- DuckDB initialization and connection management
- DuckDB availability/unavailability scenarios
- Job trends analysis with date filtering
- Company analytics with aggregations
- Salary analytics with statistical functions
- Streamlit caching integration
- Error handling and graceful fallbacks
- Performance validation
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.models import CompanySQL, JobSQL
from src.services.analytics_service import (
    DUCKDB_AVAILABLE,
    STREAMLIT_AVAILABLE,
    AnalyticsService,
)

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def test_sqlite_db(tmp_path):
    """Create a test SQLite database with sample data for DuckDB querying."""
    db_path = tmp_path / "test_analytics.db"

    # Create SQLite engine and tables
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    # Sample data for testing
    base_date = datetime.now(UTC)
    companies_data = [
        {"id": 1, "name": "TechCorp", "url": "https://techcorp.com", "active": True},
        {
            "id": 2,
            "name": "AI Solutions",
            "url": "https://aisolutions.com",
            "active": True,
        },
        {
            "id": 3,
            "name": "DataFlow Inc",
            "url": "https://dataflow.com",
            "active": True,
        },
    ]

    jobs_data = [
        {
            "id": 1,
            "company_id": 1,
            "title": "Senior AI Engineer",
            "description": "Leading AI development",
            "link": "https://techcorp.com/job1",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=1),
            "salary": [150000, 200000],
            "archived": False,
            "content_hash": "hash1",
        },
        {
            "id": 2,
            "company_id": 2,
            "title": "Machine Learning Engineer",
            "description": "ML model development",
            "link": "https://aisolutions.com/job1",
            "location": "New York, NY",
            "posted_date": base_date - timedelta(days=2),
            "salary": [140000, 180000],
            "archived": False,
            "content_hash": "hash2",
        },
        {
            "id": 3,
            "company_id": 3,
            "title": "Data Scientist",
            "description": "Statistical modeling",
            "link": "https://dataflow.com/job1",
            "location": "Austin, TX",
            "posted_date": base_date - timedelta(days=5),
            "salary": [120000, 160000],
            "archived": False,
            "content_hash": "hash3",
        },
        {
            "id": 4,
            "company_id": 1,
            "title": "DevOps Engineer",
            "description": "Infrastructure management",
            "link": "https://techcorp.com/job2",
            "location": "Remote",
            "posted_date": base_date - timedelta(days=35),
            "salary": [110000, 150000],
            "archived": False,
            "content_hash": "hash4",
        },
        {
            "id": 5,
            "company_id": 2,
            "title": "Software Engineer",
            "description": "Full-stack development",
            "link": "https://aisolutions.com/job2",
            "location": "Seattle, WA",
            "posted_date": base_date - timedelta(days=3),
            "salary": None,
            "archived": True,
            "content_hash": "hash5",  # Archived job
        },
    ]

    # Insert test data
    with Session(engine) as session:
        # Add companies
        for company_data in companies_data:
            company = CompanySQL(**company_data)
            session.add(company)

        # Add jobs
        for job_data in jobs_data:
            job = JobSQL(**job_data)
            session.add(job)

        session.commit()

    return str(db_path)


class TestAnalyticsServiceInitialization:
    """Test AnalyticsService initialization and connection management."""

    def test_init_with_default_db_path(self):
        """Test initialization with default database path."""
        service = AnalyticsService()
        assert service.db_path == "jobs.db"
        assert (
            service._conn is None if not DUCKDB_AVAILABLE else service._conn is not None
        )

    def test_init_with_custom_db_path(self, test_sqlite_db):
        """Test initialization with custom database path."""
        service = AnalyticsService(db_path=test_sqlite_db)
        assert service.db_path == test_sqlite_db

    @patch("src.services.analytics_service.DUCKDB_AVAILABLE", False)
    def test_init_without_duckdb_available(self, test_sqlite_db):
        """Test initialization when DuckDB is not available."""
        service = AnalyticsService(db_path=test_sqlite_db)
        assert service._conn is None

    @patch("src.services.analytics_service.duckdb.connect")
    def test_init_duckdb_connection_error(self, mock_connect, test_sqlite_db):
        """Test DuckDB initialization with connection error."""
        mock_connect.side_effect = Exception("Connection failed")

        service = AnalyticsService(db_path=test_sqlite_db)
        assert service._conn is None

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_init_duckdb_success(self, test_sqlite_db):
        """Test successful DuckDB initialization with sqlite_scanner."""
        service = AnalyticsService(db_path=test_sqlite_db)

        if service._conn:
            # Verify sqlite_scanner is loaded
            result = service._conn.execute(
                "SELECT extension_name FROM duckdb_extensions() WHERE loaded"
            ).fetchall()
            extension_names = [row[0] for row in result]
            assert "sqlite_scanner" in extension_names


class TestAnalyticsServiceJobTrends:
    """Test job trends analysis functionality."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_job_trends_success(self, test_sqlite_db):
        """Test successful job trends retrieval."""
        service = AnalyticsService(db_path=test_sqlite_db)

        result = service.get_job_trends(days=30)

        assert result["status"] == "success"
        assert result["method"] == "duckdb_sqlite_scanner"
        assert isinstance(result["trends"], list)
        assert result["total_jobs"] >= 0
        assert len(result["trends"]) >= 0

        # Validate data structure
        if result["trends"]:
            trend = result["trends"][0]
            assert "date" in trend
            assert "job_count" in trend
            assert isinstance(trend["job_count"], (int, float))

    def test_get_job_trends_no_connection(self):
        """Test job trends when DuckDB connection is unavailable."""
        service = AnalyticsService()
        service._conn = None  # Simulate no connection

        result = service.get_job_trends(days=30)

        assert result["status"] == "error"
        assert result["error"] == "DuckDB unavailable"
        assert result["trends"] == []

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_job_trends_query_error(self, test_sqlite_db):
        """Test job trends with database query error."""
        service = AnalyticsService(db_path=test_sqlite_db)

        if service._conn:
            # Mock the connection to raise an error
            with patch.object(
                service._conn, "execute", side_effect=Exception("Query failed")
            ):
                result = service.get_job_trends(days=30)

                assert result["status"] == "error"
                assert "Query failed" in result["error"]
                assert result["trends"] == []
                assert result["method"] == "duckdb_sqlite_scanner"

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_job_trends_different_days(self, test_sqlite_db):
        """Test job trends with different day ranges."""
        service = AnalyticsService(db_path=test_sqlite_db)

        # Test various day ranges
        for days in [7, 14, 30, 90]:
            result = service.get_job_trends(days=days)
            assert result["status"] == "success"
            assert isinstance(result["trends"], list)


class TestAnalyticsServiceCompanyAnalytics:
    """Test company analytics functionality."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_company_analytics_success(self, test_sqlite_db):
        """Test successful company analytics retrieval."""
        service = AnalyticsService(db_path=test_sqlite_db)

        result = service.get_company_analytics()

        assert result["status"] == "success"
        assert result["method"] == "duckdb_sqlite_scanner"
        assert isinstance(result["companies"], list)
        assert result["total_companies"] >= 0

        # Validate data structure
        if result["companies"]:
            company = result["companies"][0]
            expected_fields = [
                "company",
                "total_jobs",
                "avg_min_salary",
                "avg_max_salary",
                "last_job_posted",
            ]
            for field in expected_fields:
                assert field in company

    def test_get_company_analytics_no_connection(self):
        """Test company analytics when DuckDB connection is unavailable."""
        service = AnalyticsService()
        service._conn = None

        result = service.get_company_analytics()

        assert result["status"] == "error"
        assert result["error"] == "DuckDB unavailable"
        assert result["companies"] == []

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_company_analytics_query_error(self, test_sqlite_db):
        """Test company analytics with database query error."""
        service = AnalyticsService(db_path=test_sqlite_db)

        if service._conn:
            with patch.object(
                service._conn, "execute", side_effect=Exception("Query failed")
            ):
                result = service.get_company_analytics()

                assert result["status"] == "error"
                assert "Query failed" in result["error"]
                assert result["companies"] == []


class TestAnalyticsServiceSalaryAnalytics:
    """Test salary analytics functionality."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_salary_analytics_success(self, test_sqlite_db):
        """Test successful salary analytics retrieval."""
        service = AnalyticsService(db_path=test_sqlite_db)

        result = service.get_salary_analytics(days=90)

        assert result["status"] == "success"
        assert result["method"] == "duckdb_sqlite_scanner"
        assert isinstance(result["salary_data"], dict)

        # Validate salary data structure
        salary_data = result["salary_data"]
        expected_fields = [
            "total_jobs_with_salary",
            "avg_min_salary",
            "avg_max_salary",
            "min_salary",
            "max_salary",
            "salary_std_dev",
            "analysis_period_days",
        ]
        for field in expected_fields:
            assert field in salary_data

        assert salary_data["analysis_period_days"] == 90

    def test_get_salary_analytics_no_connection(self):
        """Test salary analytics when DuckDB connection is unavailable."""
        service = AnalyticsService()
        service._conn = None

        result = service.get_salary_analytics(days=90)

        assert result["status"] == "error"
        assert result["error"] == "DuckDB unavailable"
        assert result["salary_data"] == {}

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_salary_analytics_query_error(self, test_sqlite_db):
        """Test salary analytics with database query error."""
        service = AnalyticsService(db_path=test_sqlite_db)

        if service._conn:
            with patch.object(
                service._conn, "execute", side_effect=Exception("Query failed")
            ):
                result = service.get_salary_analytics(days=90)

                assert result["status"] == "error"
                assert "Query failed" in result["error"]
                assert result["salary_data"] == {}

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_get_salary_analytics_different_periods(self, test_sqlite_db):
        """Test salary analytics with different time periods."""
        service = AnalyticsService(db_path=test_sqlite_db)

        for days in [30, 60, 90, 180]:
            result = service.get_salary_analytics(days=days)
            assert result["status"] == "success"
            assert result["salary_data"]["analysis_period_days"] == days


class TestAnalyticsServiceStatusAndHealth:
    """Test service status and health monitoring."""

    def test_get_status_report(self, test_sqlite_db):
        """Test status report generation."""
        service = AnalyticsService(db_path=test_sqlite_db)

        status = service.get_status_report()

        assert isinstance(status, dict)
        assert status["analytics_method"] == "duckdb_sqlite_scanner"
        assert status["duckdb_available"] == DUCKDB_AVAILABLE
        assert status["streamlit_available"] == STREAMLIT_AVAILABLE
        assert status["database_path"] == test_sqlite_db
        assert "connection_active" in status
        assert "status" in status

        # Status should be "active" if connection exists, "unavailable" otherwise
        expected_status = "active" if service._conn else "unavailable"
        assert status["status"] == expected_status

    def test_destructor_cleanup(self, test_sqlite_db):
        """Test connection cleanup in destructor."""
        service = AnalyticsService(db_path=test_sqlite_db)

        if service._conn:
            # Store original connection

            # Call destructor
            service.__del__()

            # Connection should be handled - this test mainly ensures no exceptions
            assert True  # Test passes if no exception is raised

    def test_destructor_no_connection(self):
        """Test destructor when no connection exists."""
        service = AnalyticsService()
        service._conn = None

        # Should not raise an exception
        service.__del__()


class TestAnalyticsServiceStreamlitIntegration:
    """Test Streamlit caching and integration features."""

    @patch("src.services.analytics_service.STREAMLIT_AVAILABLE", True)
    @patch("src.services.analytics_service.st")
    def test_streamlit_success_message(self, mock_st, test_sqlite_db):
        """Test Streamlit success message display on initialization."""
        mock_st.success = Mock()

        # This will trigger _init_duckdb which should call st.success
        if DUCKDB_AVAILABLE:
            AnalyticsService(db_path=test_sqlite_db)
            mock_st.success.assert_called_with(
                "🚀 Analytics powered by DuckDB sqlite_scanner"
            )

    @patch("src.services.analytics_service.STREAMLIT_AVAILABLE", True)
    @patch("src.services.analytics_service.st")
    @patch(
        "src.services.analytics_service.duckdb.connect",
        side_effect=Exception("Init failed"),
    )
    def test_streamlit_error_message(self, mock_connect, mock_st, test_sqlite_db):
        """Test Streamlit error message display on initialization failure."""
        mock_st.error = Mock()

        AnalyticsService(db_path=test_sqlite_db)
        mock_st.error.assert_called_with("Analytics unavailable: Init failed")

    @patch("src.services.analytics_service.st.cache_data")
    def test_streamlit_caching_decorators(self, mock_cache_data):
        """Test that Streamlit caching decorators are applied correctly."""
        # The @st.cache_data decorator should be applied to analytics methods
        # This is tested by checking that the decorator exists on methods

        service = AnalyticsService()

        # Check that methods have cache decorators (they're wrapped)
        assert hasattr(service.get_job_trends, "__wrapped__") or hasattr(
            service.get_job_trends, "clear"
        )
        assert hasattr(service.get_company_analytics, "__wrapped__") or hasattr(
            service.get_company_analytics, "clear"
        )
        assert hasattr(service.get_salary_analytics, "__wrapped__") or hasattr(
            service.get_salary_analytics, "clear"
        )


class TestAnalyticsServicePerformance:
    """Test performance characteristics of analytics service."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_analytics_service_performance(self, test_sqlite_db):
        """Test analytics service performance with reasonable response times."""
        import time

        service = AnalyticsService(db_path=test_sqlite_db)

        # Test job trends performance
        start_time = time.time()
        trends_result = service.get_job_trends(days=30)
        trends_time = time.time() - start_time

        assert trends_result["status"] == "success"
        assert trends_time < 2.0  # Should complete in under 2 seconds

        # Test company analytics performance
        start_time = time.time()
        company_result = service.get_company_analytics()
        company_time = time.time() - start_time

        assert company_result["status"] == "success"
        assert company_time < 2.0  # Should complete in under 2 seconds

        # Test salary analytics performance
        start_time = time.time()
        salary_result = service.get_salary_analytics(days=90)
        salary_time = time.time() - start_time

        assert salary_result["status"] == "success"
        assert salary_time < 2.0  # Should complete in under 2 seconds


class TestAnalyticsServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_empty_database(self, tmp_path):
        """Test analytics service with empty database."""
        # Create empty database
        empty_db = tmp_path / "empty.db"
        engine = create_engine(f"sqlite:///{empty_db}")
        SQLModel.metadata.create_all(engine)

        service = AnalyticsService(db_path=str(empty_db))

        # All methods should handle empty database gracefully
        trends = service.get_job_trends(days=30)
        assert trends["status"] == "success"
        assert trends["trends"] == []
        assert trends["total_jobs"] == 0

        companies = service.get_company_analytics()
        assert companies["status"] == "success"
        assert companies["companies"] == []
        assert companies["total_companies"] == 0

        salary = service.get_salary_analytics(days=90)
        assert salary["status"] == "success"
        # Empty database should return zeros for salary stats
        assert salary["salary_data"]["total_jobs_with_salary"] == 0

    def test_invalid_database_path(self):
        """Test analytics service with invalid database path."""
        service = AnalyticsService(db_path="/nonexistent/path/database.db")

        # Methods should handle invalid path gracefully
        if service._conn:
            # If connection was somehow established, queries should still work
            # or fail gracefully
            result = service.get_job_trends(days=30)
            assert result["status"] in ["success", "error"]
        else:
            # If no connection, should return error
            result = service.get_job_trends(days=30)
            assert result["status"] == "error"

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_extreme_date_ranges(self, test_sqlite_db):
        """Test analytics with extreme date ranges."""
        service = AnalyticsService(db_path=test_sqlite_db)

        # Test very small date range
        result = service.get_job_trends(days=1)
        assert result["status"] == "success"

        # Test very large date range
        result = service.get_job_trends(days=3650)  # 10 years
        assert result["status"] == "success"

        # Test salary analytics with extreme ranges
        result = service.get_salary_analytics(days=1)
        assert result["status"] == "success"

        result = service.get_salary_analytics(days=3650)
        assert result["status"] == "success"
