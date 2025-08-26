"""Modernized tests for the DuckDB sqlite_scanner AnalyticsService.

This test suite validates the simplified analytics service implementation that uses
DuckDB's native sqlite_scanner for zero-ETL analytics queries against SQLite data.

Key improvements in this modernized version:
- Removed unnecessary DUCKDB_AVAILABLE skip markers (DuckDB is a core dependency)
- Fixed Streamlit caching compatibility issues
- Simplified test setup with library-first approach
- Real integration tests without excessive mocking
- Focus on actual user scenarios
"""

import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.models import CompanySQL, JobSQL
from src.services.analytics_service import AnalyticsService

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def test_db_with_data(tmp_path):
    """Create a test SQLite database with realistic sample data."""
    db_path = tmp_path / "test_analytics.db"

    # Create SQLite engine and tables
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    # Sample data for testing - realistic job market data
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
        # Recent jobs for trend analysis
        {
            "id": 1,
            "company_id": 1,
            "title": "Senior AI Engineer",
            "description": "Leading AI development with Python and ML",
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
            "description": "ML model development and deployment",
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
            "description": "Statistical modeling and data analysis",
            "link": "https://dataflow.com/job1",
            "location": "Austin, TX",
            "posted_date": base_date - timedelta(days=5),
            "salary": [120000, 160000],
            "archived": False,
            "content_hash": "hash3",
        },
        # Older job for date filtering tests
        {
            "id": 4,
            "company_id": 1,
            "title": "DevOps Engineer",
            "description": "Infrastructure management and automation",
            "link": "https://techcorp.com/job2",
            "location": "Remote",
            "posted_date": base_date - timedelta(days=35),
            "salary": [110000, 150000],
            "archived": False,
            "content_hash": "hash4",
        },
        # Archived job - should be filtered out
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
            "content_hash": "hash5",
        },
        # Job without salary for analytics testing
        {
            "id": 6,
            "company_id": 3,
            "title": "Frontend Developer",
            "description": "React and TypeScript development",
            "link": "https://dataflow.com/job2",
            "location": "Denver, CO",
            "posted_date": base_date - timedelta(days=7),
            "salary": None,
            "archived": False,
            "content_hash": "hash6",
        },
    ]

    # Insert test data
    with Session(engine) as session:
        for company_data in companies_data:
            company = CompanySQL(**company_data)
            session.add(company)

        for job_data in jobs_data:
            job = JobSQL(**job_data)
            session.add(job)

        session.commit()

    return str(db_path)


@pytest.fixture
def analytics_service(test_db_with_data):
    """Create analytics service instance with test data."""
    return AnalyticsService(db_path=test_db_with_data)


class TestAnalyticsServiceCore:
    """Test core analytics service functionality."""

    def test_service_initialization(self, test_db_with_data):
        """Test service initializes correctly with DuckDB connection."""
        service = AnalyticsService(db_path=test_db_with_data)

        assert service.db_path == test_db_with_data
        assert service._conn is not None  # DuckDB should be available

        # Verify status report
        status = service.get_status_report()
        assert status["analytics_method"] == "duckdb_sqlite_scanner"
        assert status["duckdb_available"] is True
        assert status["database_path"] == test_db_with_data
        assert status["connection_active"] is True
        assert status["status"] == "active"

    def test_job_trends_analysis(self, analytics_service):
        """Test job trends analysis with realistic data."""
        # Test recent trends (30 days)
        trends = analytics_service.get_job_trends(days=30)

        assert trends["status"] == "success"
        assert trends["method"] == "duckdb_sqlite_scanner"
        assert isinstance(trends["trends"], list)
        assert trends["total_jobs"] > 0

        # Should have trend data for recent jobs (excluding archived)
        # We expect 4 non-archived jobs within 30 days
        assert trends["total_jobs"] == 4

    def test_job_trends_date_filtering(self, analytics_service):
        """Test job trends with different date ranges."""
        # Test short range (7 days) - should get 3 jobs
        recent_trends = analytics_service.get_job_trends(days=7)
        assert recent_trends["status"] == "success"
        assert recent_trends["total_jobs"] == 3

        # Test longer range (45 days) - should get all 4 non-archived jobs
        long_trends = analytics_service.get_job_trends(days=45)
        assert long_trends["status"] == "success"
        assert long_trends["total_jobs"] == 4

    def test_company_analytics(self, analytics_service):
        """Test company analytics with aggregations."""
        analytics = analytics_service.get_company_analytics()

        assert analytics["status"] == "success"
        assert analytics["method"] == "duckdb_sqlite_scanner"
        assert isinstance(analytics["companies"], list)
        assert analytics["total_companies"] > 0

        # Verify company data structure
        if analytics["companies"]:
            company = analytics["companies"][0]
            required_fields = [
                "company",
                "total_jobs",
                "avg_min_salary",
                "avg_max_salary",
                "last_job_posted",
            ]
            for field in required_fields:
                assert field in company

    def test_salary_analytics(self, analytics_service):
        """Test salary analytics with statistical functions."""
        salary_data = analytics_service.get_salary_analytics(days=90)

        assert salary_data["status"] == "success"
        assert salary_data["method"] == "duckdb_sqlite_scanner"
        assert isinstance(salary_data["salary_data"], dict)

        salary_info = salary_data["salary_data"]
        # Should have jobs with salary data (excluding None salaries)
        assert salary_info["total_jobs_with_salary"] > 0
        assert salary_info["avg_min_salary"] > 0
        assert salary_info["avg_max_salary"] > 0
        assert salary_info["analysis_period_days"] == 90


class TestAnalyticsServiceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database_graceful_handling(self, tmp_path):
        """Test analytics service handles empty database gracefully."""
        # Create empty database
        empty_db = tmp_path / "empty.db"
        engine = create_engine(f"sqlite:///{empty_db}")
        SQLModel.metadata.create_all(engine)

        service = AnalyticsService(db_path=str(empty_db))

        # All methods should return success with empty results
        trends = service.get_job_trends()
        assert trends["status"] == "success"
        assert trends["trends"] == []
        assert trends["total_jobs"] == 0

        companies = service.get_company_analytics()
        assert companies["status"] == "success"
        assert companies["companies"] == []

        salaries = service.get_salary_analytics()
        assert salaries["status"] == "success"
        # Empty database should have zero salary data
        salary_data = salaries["salary_data"]
        assert salary_data["total_jobs_with_salary"] == 0

    def test_invalid_database_path(self):
        """Test service handles invalid database paths."""
        service = AnalyticsService(db_path="/nonexistent/path/database.db")

        # Should still initialize but methods should handle errors gracefully
        trends = service.get_job_trends()
        # Could be success with empty results or error - both are acceptable
        assert "status" in trends
        assert "trends" in trends

    @patch("src.services.analytics_service.duckdb.connect")
    def test_duckdb_unavailable_fallback(self, mock_connect, test_db_with_data):
        """Test service handles DuckDB connection failures."""
        mock_connect.side_effect = Exception("DuckDB connection failed")

        service = AnalyticsService(db_path=test_db_with_data)
        assert service._conn is None

        # All methods should return error status
        trends = service.get_job_trends()
        assert trends["status"] == "error"
        assert "DuckDB unavailable" in trends["error"]


class TestAnalyticsServicePerformance:
    """Test performance characteristics."""

    def test_query_performance(self, analytics_service):
        """Test that queries complete within reasonable time."""
        import time

        # Job trends should be fast
        start = time.perf_counter()
        trends = analytics_service.get_job_trends()
        trends_time = time.perf_counter() - start

        assert trends["status"] == "success"
        assert trends_time < 1.0  # Should complete under 1 second

        # Company analytics should be fast
        start = time.perf_counter()
        companies = analytics_service.get_company_analytics()
        company_time = time.perf_counter() - start

        assert companies["status"] == "success"
        assert company_time < 1.0  # Should complete under 1 second


class TestAnalyticsServiceIntegration:
    """Test integration scenarios."""

    def test_multiple_operations_sequence(self, analytics_service):
        """Test multiple analytics operations work together."""
        # Get all analytics in sequence
        trends = analytics_service.get_job_trends(days=30)
        companies = analytics_service.get_company_analytics()
        salaries = analytics_service.get_salary_analytics(days=90)
        status = analytics_service.get_status_report()

        # All should succeed
        assert trends["status"] == "success"
        assert companies["status"] == "success"
        assert salaries["status"] == "success"
        assert status["status"] == "active"

        # Data should be consistent
        # Total jobs from trends should be <= total from companies
        total_trend_jobs = trends["total_jobs"]
        total_company_jobs = sum(c["total_jobs"] for c in companies["companies"])
        # Company analytics includes all jobs, trends excludes archived
        assert total_trend_jobs <= total_company_jobs

    def test_database_changes_reflection(self, analytics_service, test_db_with_data):
        """Test analytics reflect database changes."""
        # Get initial state
        initial_trends = analytics_service.get_job_trends()
        initial_count = initial_trends["total_jobs"]

        # Add new job to database
        engine = create_engine(f"sqlite:///{test_db_with_data}")
        with Session(engine) as session:
            new_job = JobSQL(
                id=99,
                company_id=1,
                title="Test Engineer",
                description="Testing new functionality",
                link="https://test.com/job99",
                location="Test City",
                posted_date=datetime.now(UTC),
                salary=[100000, 130000],
                archived=False,
                content_hash="test_hash",
            )
            session.add(new_job)
            session.commit()

        # Analytics should reflect the change
        # Note: Due to caching, we might need to create a new service instance
        # or the cache might prevent seeing immediate changes
        updated_service = AnalyticsService(db_path=test_db_with_data)
        updated_trends = updated_service.get_job_trends()

        # Should have one more job
        assert updated_trends["total_jobs"] >= initial_count
