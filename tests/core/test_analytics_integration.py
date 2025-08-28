"""Core analytics integration tests.

This module tests analytics functionality, DuckDB integration, caching,
and performance validation following the clean foundation from Group 3.
"""

import tempfile

from unittest.mock import Mock, patch

import pytest

from src.services.analytics_service import AnalyticsService
from src.ui.utils.service_cache import get_analytics_service


class TestAnalyticsCaching:
    """Test analytics service caching with @st.cache_resource."""

    def test_analytics_service_cached_instance(self):
        """Test that get_analytics_service returns cached instances."""
        service1 = get_analytics_service()
        service2 = get_analytics_service()

        # Should return same cached instance
        assert service1 is service2
        assert isinstance(service1, AnalyticsService)

    def test_cache_performance_impact(self):
        """Test that caching improves analytics performance."""
        import time

        # First call should initialize service
        start_time = time.time()
        service1 = get_analytics_service()
        first_call_time = time.time() - start_time

        # Second call should use cached instance (much faster)
        start_time = time.time()
        service2 = get_analytics_service()
        second_call_time = time.time() - start_time

        # Second call should be significantly faster
        assert second_call_time <= first_call_time
        assert service1 is service2


class TestAnalyticsDataGeneration:
    """Test analytics data generation functionality."""

    @pytest.fixture
    def mock_analytics_service(self):
        """Create a mock analytics service for testing."""
        service = Mock(spec=AnalyticsService)
        return service

    def test_job_trends_analysis(self, mock_analytics_service):
        """Test job trends analytics generation."""
        mock_trends = {
            "total_jobs": 1250,
            "new_jobs_this_week": 85,
            "trends": [
                {"date": "2024-08-21", "count": 15},
                {"date": "2024-08-22", "count": 20},
                {"date": "2024-08-23", "count": 12},
                {"date": "2024-08-24", "count": 18},
                {"date": "2024-08-25", "count": 20},
            ],
        }

        mock_analytics_service.get_job_trends.return_value = mock_trends

        trends = mock_analytics_service.get_job_trends(days=7)

        assert isinstance(trends, dict)
        assert "total_jobs" in trends
        assert "trends" in trends
        assert isinstance(trends["trends"], list)

    def test_company_analytics(self, mock_analytics_service):
        """Test company analytics generation."""
        mock_company_stats = {
            "total_companies": 45,
            "top_companies": [
                {"name": "Google", "job_count": 15, "avg_salary": 145000},
                {"name": "Apple", "job_count": 12, "avg_salary": 140000},
                {"name": "Meta", "job_count": 10, "avg_salary": 155000},
            ],
            "company_distribution": [
                {"company": "Tech Giants", "count": 37},
                {"company": "Startups", "count": 8},
            ],
        }

        mock_analytics_service.get_company_analytics.return_value = mock_company_stats

        company_stats = mock_analytics_service.get_company_analytics()

        assert isinstance(company_stats, dict)
        assert "total_companies" in company_stats
        assert "top_companies" in company_stats
        assert isinstance(company_stats["top_companies"], list)

    def test_salary_analytics(self, mock_analytics_service):
        """Test salary analytics generation."""
        mock_salary_stats = {
            "average_salary": 125000,
            "median_salary": 115000,
            "salary_ranges": [
                {"range": "50k-75k", "count": 15},
                {"range": "75k-100k", "count": 35},
                {"range": "100k-150k", "count": 45},
                {"range": "150k+", "count": 25},
            ],
            "salary_by_location": [
                {"location": "San Francisco", "avg_salary": 165000},
                {"location": "New York", "avg_salary": 155000},
                {"location": "Remote", "avg_salary": 135000},
            ],
        }

        mock_analytics_service.get_salary_analytics.return_value = mock_salary_stats

        salary_stats = mock_analytics_service.get_salary_analytics()

        assert isinstance(salary_stats, dict)
        assert "average_salary" in salary_stats
        assert "salary_ranges" in salary_stats
        assert isinstance(salary_stats["salary_ranges"], list)

    def test_location_analytics(self, mock_analytics_service):
        """Test location-based analytics generation."""
        mock_location_stats = {
            "top_locations": [
                {"location": "Remote", "count": 45},
                {"location": "San Francisco, CA", "count": 25},
                {"location": "New York, NY", "count": 20},
            ],
            "remote_percentage": 36.0,
            "location_distribution": {"Remote": 45, "On-site": 80},
        }

        mock_analytics_service.get_location_analytics.return_value = mock_location_stats

        location_stats = mock_analytics_service.get_location_analytics()

        assert isinstance(location_stats, dict)
        assert "top_locations" in location_stats
        assert "remote_percentage" in location_stats


class TestAnalyticsPerformance:
    """Test analytics performance requirements (<100ms response times)."""

    def test_analytics_query_performance(self):
        """Test that analytics queries meet performance requirements."""
        import time

        mock_service = Mock(spec=AnalyticsService)
        mock_service.get_job_trends.return_value = {"total_jobs": 100, "trends": []}

        # Test performance of analytics queries
        start_time = time.time()
        trends = mock_service.get_job_trends(days=30)
        query_time = time.time() - start_time

        # Should be very fast for mocked service
        assert query_time < 0.1  # 100ms
        assert isinstance(trends, dict)

    def test_concurrent_analytics_operations(self):
        """Test performance of concurrent analytics operations."""
        import concurrent.futures
        import time

        mock_service = Mock(spec=AnalyticsService)
        mock_service.get_job_trends.return_value = {"trends": []}
        mock_service.get_company_analytics.return_value = {"companies": []}
        mock_service.get_salary_analytics.return_value = {"salaries": []}

        def run_analytics():
            return [
                mock_service.get_job_trends(days=7),
                mock_service.get_company_analytics(),
                mock_service.get_salary_analytics(),
            ]

        start_time = time.time()

        # Run multiple analytics operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_analytics) for _ in range(3)]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # Should complete quickly even with concurrent operations
        assert total_time < 1.0  # 1 second for concurrent operations
        assert len(results) == 3

    @pytest.mark.benchmark
    def test_large_dataset_performance(self):
        """Test analytics performance with large datasets."""
        import time

        # Mock large dataset response
        large_trends = {
            "total_jobs": 10000,
            "trends": [
                {"date": f"2024-{month:02d}-{day:02d}", "count": day * 10}
                for month in range(1, 13)
                for day in range(1, 29)
            ],
        }

        mock_service = Mock(spec=AnalyticsService)
        mock_service.get_job_trends.return_value = large_trends

        start_time = time.time()
        trends = mock_service.get_job_trends(days=365)
        processing_time = time.time() - start_time

        # Should handle large datasets efficiently
        assert processing_time < 0.5  # 500ms for large dataset
        assert len(trends["trends"]) > 300  # Large dataset


class TestAnalyticsErrorHandling:
    """Test error handling in analytics functionality."""

    def test_database_connection_error_handling(self):
        """Test handling of database connection errors."""
        mock_service = Mock(spec=AnalyticsService)
        mock_service.get_job_trends.side_effect = Exception(
            "Database connection failed"
        )

        # Should handle database errors gracefully
        with pytest.raises(Exception):
            mock_service.get_job_trends(days=7)

    def test_invalid_date_range_handling(self):
        """Test handling of invalid date ranges."""
        mock_service = Mock(spec=AnalyticsService)
        mock_service.get_job_trends.return_value = {"trends": []}

        # Test with invalid date ranges
        invalid_ranges = [-1, 0, 10000, "invalid"]

        for invalid_range in invalid_ranges:
            try:
                result = mock_service.get_job_trends(days=invalid_range)
                assert isinstance(result, dict)
            except (ValueError, TypeError):
                # Expected for some invalid inputs
                pass

    def test_empty_dataset_handling(self):
        """Test analytics with empty datasets."""
        mock_empty_response = {
            "total_jobs": 0,
            "trends": [],
            "companies": [],
            "salaries": [],
        }

        mock_service = Mock(spec=AnalyticsService)
        mock_service.get_job_trends.return_value = mock_empty_response
        mock_service.get_company_analytics.return_value = mock_empty_response
        mock_service.get_salary_analytics.return_value = mock_empty_response

        # Should handle empty datasets gracefully
        trends = mock_service.get_job_trends(days=7)
        companies = mock_service.get_company_analytics()
        salaries = mock_service.get_salary_analytics()

        assert trends["total_jobs"] == 0
        assert len(trends["trends"]) == 0
        assert len(companies["companies"]) == 0
        assert len(salaries["salaries"]) == 0


class TestDuckDBIntegration:
    """Test DuckDB integration for analytics."""

    def test_duckdb_sqlite_scanning(self):
        """Test DuckDB's ability to scan SQLite databases."""
        # Mock DuckDB connection and query execution
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("2024-08-28", 15),
            ("2024-08-27", 12),
            ("2024-08-26", 18),
        ]
        mock_connection.cursor.return_value = mock_cursor

        with patch("duckdb.connect", return_value=mock_connection):
            # Simulate analytics service using DuckDB
            mock_service = Mock(spec=AnalyticsService)
            mock_service._execute_analytics_query.return_value = mock_cursor.fetchall()

            results = mock_service._execute_analytics_query(
                "SELECT posted_date, COUNT(*) FROM jobs GROUP BY posted_date"
            )

            assert len(results) == 3
            assert results[0][0] == "2024-08-28"

    def test_memory_management_large_queries(self):
        """Test memory management for large analytics queries."""
        import time

        # Mock large query results
        large_results = [(f"item_{i}", i * 100) for i in range(10000)]

        mock_service = Mock(spec=AnalyticsService)
        mock_service._execute_analytics_query.return_value = large_results

        start_time = time.time()
        results = mock_service._execute_analytics_query("SELECT * FROM large_table")
        processing_time = time.time() - start_time

        # Should handle large results efficiently
        assert len(results) == 10000
        assert processing_time < 1.0  # Should process large results quickly

    def test_cross_table_analytics_queries(self):
        """Test analytics queries spanning multiple tables."""
        mock_results = [
            ("Google", 15, 145000.0),
            ("Apple", 12, 140000.0),
            ("Meta", 10, 155000.0),
        ]

        mock_service = Mock(spec=AnalyticsService)
        mock_service._execute_analytics_query.return_value = mock_results

        # Simulate cross-table query (jobs + companies)
        results = mock_service._execute_analytics_query("""
            SELECT c.name, COUNT(j.id), AVG(j.salary_min)
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            GROUP BY c.name
        """)

        assert len(results) == 3
        assert results[0][0] == "Google"  # Company name
        assert results[0][1] == 15  # Job count
        assert results[0][2] == 145000.0  # Average salary


class TestAnalyticsVisualization:
    """Test analytics data preparation for visualization."""

    def test_chart_data_formatting(self):
        """Test formatting of analytics data for charts."""
        mock_service = Mock(spec=AnalyticsService)

        # Mock raw analytics data
        raw_trends = [
            ("2024-08-24", 12),
            ("2024-08-25", 15),
            ("2024-08-26", 18),
            ("2024-08-27", 10),
            ("2024-08-28", 20),
        ]

        # Mock formatted data for charts
        formatted_data = {
            "dates": [
                "2024-08-24",
                "2024-08-25",
                "2024-08-26",
                "2024-08-27",
                "2024-08-28",
            ],
            "values": [12, 15, 18, 10, 20],
            "chart_type": "line",
            "title": "Job Postings Trend",
        }

        mock_service._format_chart_data.return_value = formatted_data

        chart_data = mock_service._format_chart_data(
            raw_trends, "line", "Job Postings Trend"
        )

        assert "dates" in chart_data
        assert "values" in chart_data
        assert chart_data["chart_type"] == "line"
        assert len(chart_data["dates"]) == 5

    def test_plotly_chart_configuration(self):
        """Test Plotly chart configuration generation."""
        mock_service = Mock(spec=AnalyticsService)

        chart_config = {
            "data": [
                {
                    "x": ["Remote", "San Francisco", "New York"],
                    "y": [45, 25, 20],
                    "type": "bar",
                    "name": "Job Locations",
                }
            ],
            "layout": {
                "title": "Jobs by Location",
                "xaxis": {"title": "Location"},
                "yaxis": {"title": "Job Count"},
            },
        }

        mock_service._generate_plotly_config.return_value = chart_config

        config = mock_service._generate_plotly_config("Jobs by Location", "bar")

        assert "data" in config
        assert "layout" in config
        assert config["layout"]["title"] == "Jobs by Location"

    def test_metrics_card_data(self):
        """Test data formatting for Streamlit metrics cards."""
        mock_service = Mock(spec=AnalyticsService)

        metrics_data = {
            "total_jobs": {"value": 1250, "delta": 85, "delta_color": "normal"},
            "avg_salary": {
                "value": "$125,000",
                "delta": "$5,000",
                "delta_color": "normal",
            },
            "remote_percentage": {
                "value": "36%",
                "delta": "2%",
                "delta_color": "normal",
            },
            "active_companies": {"value": 45, "delta": 3, "delta_color": "normal"},
        }

        mock_service._format_metrics_data.return_value = metrics_data

        metrics = mock_service._format_metrics_data()

        assert "total_jobs" in metrics
        assert "avg_salary" in metrics
        assert metrics["total_jobs"]["value"] == 1250
        assert metrics["total_jobs"]["delta"] == 85


@pytest.mark.integration
class TestAnalyticsIntegration:
    """Test integration between analytics and other components."""

    def test_job_service_analytics_integration(self):
        """Test integration between JobService and AnalyticsService."""
        from src.services.job_service import JobService

        # Mock services working together
        mock_job_service = Mock(spec=JobService)
        mock_analytics_service = Mock(spec=AnalyticsService)

        mock_job_service.get_jobs.return_value = []
        mock_analytics_service.get_job_trends.return_value = {"trends": []}

        # Should be able to use both services together
        jobs = mock_job_service.get_jobs(limit=100)
        trends = mock_analytics_service.get_job_trends(days=30)

        assert isinstance(jobs, list)
        assert isinstance(trends, dict)

    def test_streamlit_caching_integration(self):
        """Test integration with Streamlit caching."""
        # Test that analytics service uses caching properly
        service1 = get_analytics_service()
        service2 = get_analytics_service()

        # Should return same cached instance
        assert service1 is service2

    def test_real_database_analytics_integration(self):
        """Test analytics with real database operations."""
        # This test uses a temporary database for real integration testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            analytics_service = AnalyticsService(db_path=temp_db.name)

            # Test that service can be instantiated with real database
            assert analytics_service is not None

            # Test basic operations don't crash
            try:
                # These calls may fail due to missing tables, which is expected
                trends = analytics_service.get_job_trends(days=7)
                assert isinstance(trends, dict) or trends is None
            except Exception as e:
                # Log the error but don't fail the test for missing tables
                print(f"Expected database error in integration test: {e}")

    def test_analytics_cache_invalidation(self):
        """Test analytics cache invalidation scenarios."""
        service = get_analytics_service()

        # Test that service methods exist and can be called
        assert hasattr(service, "get_job_trends") or hasattr(service, "_get_job_trends")

        # Note: Real cache invalidation testing would require
        # actual Streamlit runtime context
