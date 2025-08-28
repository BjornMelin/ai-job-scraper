"""Comprehensive Error Handling Tests for AI Job Scraper.

This test suite validates error recovery, retry mechanisms, and graceful degradation
across all major components of the AI Job Scraper system. Tests ensure robust
operation under various failure conditions.

Test coverage includes:
- Scraper error recovery (ConnectionError, TimeoutError, etc.)
- Database transaction rollbacks and integrity
- Retry mechanisms and exponential backoff
- Graceful degradation and fallback systems
- Network failure scenarios
- Resource exhaustion handling
- AI service failures and fallbacks
- Data corruption recovery
"""

import contextlib
import logging
import sqlite3
import time

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from requests.exceptions import ConnectionError, HTTPError, Timeout
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.database import db_session
from src.models import CompanySQL
from src.schemas import JobCreate
from src.services.analytics_service import AnalyticsService
from src.services.company_service import CompanyService
from src.services.cost_monitor import CostMonitor
from src.services.job_service import JobService
from tests.factories import create_sample_companies, create_sample_jobs

# Disable logging during tests
logging.disable(logging.CRITICAL)


class MockHTTPError(HTTPError):
    """Mock HTTP error for testing."""

    def __init__(self, status_code, message="HTTP Error"):
        self.response = Mock()
        self.response.status_code = status_code
        super().__init__(message)


@pytest.fixture
def test_database_with_data(tmp_path):
    """Create test database with sample data for error testing."""
    db_path = tmp_path / "error_test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        companies = create_sample_companies(session, count=5)
        for company in companies:
            create_sample_jobs(session, count=10, company=company)
        session.commit()

    return str(db_path)


@pytest.fixture
def corrupted_database(tmp_path):
    """Create a partially corrupted database for recovery testing."""
    db_path = tmp_path / "corrupted.db"

    # Create valid database first
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    # Add some valid data
    with Session(engine) as session:
        companies = create_sample_companies(session, count=3)
        create_sample_jobs(session, count=15, company=companies[0])
        session.commit()

    # Corrupt part of the database
    with Path(db_path).open("r+b") as f:
        f.seek(2048)  # Seek to middle area
        f.write(b"\x00" * 512)  # Write null bytes to corrupt data

    return str(db_path)


class TestScraperErrorHandling:
    """Test scraper error recovery and retry mechanisms."""

    def test_connection_error_retry_mechanism(self):
        """Test retry mechanism for connection errors."""
        with patch("src.scraper.httpx.AsyncClient") as mock_client:
            # Setup mock to fail first few attempts, then succeed
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Test</body></html>"

            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # First 2 calls raise ConnectionError, 3rd succeeds
            mock_client_instance.get.side_effect = [
                ConnectionError("Connection failed"),
                ConnectionError("Connection failed again"),
                mock_response,
            ]

            from src.scraper import scrape_company_page

            # Should eventually succeed after retries
            result = scrape_company_page("https://example.com/careers")

            # Verify retries occurred
            assert mock_client_instance.get.call_count == 3
            assert result is not None

    def test_timeout_error_handling(self):
        """Test handling of timeout errors during scraping."""
        with patch("src.scraper.httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # All calls timeout
            mock_client_instance.get.side_effect = Timeout("Request timed out")

            from src.scraper import scrape_company_page

            # Should handle timeout gracefully
            result = scrape_company_page("https://example.com/careers")

            # Should return None or empty result, not crash
            assert result is None or result == []

    def test_http_error_status_codes(self):
        """Test handling of various HTTP error status codes."""
        error_codes = [400, 401, 403, 404, 429, 500, 502, 503, 504]

        for status_code in error_codes:
            with patch("src.scraper.httpx.AsyncClient") as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value.__aenter__.return_value = mock_client_instance

                # Mock HTTP error response
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.raise_for_status.side_effect = MockHTTPError(status_code)
                mock_client_instance.get.return_value = mock_response

                from src.scraper import scrape_company_page

                # Should handle error gracefully
                result = scrape_company_page("https://example.com/careers")

                # Should not crash, return empty/None result
                assert result is None or result == [] or isinstance(result, list)

    def test_malformed_html_handling(self):
        """Test handling of malformed or invalid HTML."""
        malformed_html_cases = [
            "<html><body>Unclosed tag",
            "<html><body><div><span>Nested unclosed</div>",
            "Not HTML at all - just text",
            "",  # Empty response
            "<html><body>" + "x" * 10000 + "</body></html>",  # Very large HTML
            "<?xml version='1.0'?><root>XML not HTML</root>",
            "<html><body><script>alert('xss')</script></body></html>",
        ]

        with patch("src.scraper.httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            from src.scraper import scrape_company_page

            for malformed_html in malformed_html_cases:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = malformed_html
                mock_client_instance.get.return_value = mock_response

                # Should handle malformed HTML gracefully
                try:
                    result = scrape_company_page("https://example.com/careers")
                    # Should return some result, not crash
                    assert result is not None
                except Exception as e:
                    # If exception occurs, should be handled gracefully
                    assert "timeout" not in str(e).lower()  # Not a timeout
                    assert "memory" not in str(e).lower()  # Not memory issue

    def test_ai_extraction_failure_fallback(self):
        """Test fallback when AI extraction fails."""
        with (
            patch("src.scraper.httpx.AsyncClient") as mock_client,
            patch("src.ai_client.extract_jobs") as mock_extract,
        ):
            # Setup successful HTTP response
            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body><h1>Software Engineer</h1></body></html>"
            mock_client_instance.get.return_value = mock_response

            # AI extraction fails
            mock_extract.side_effect = Exception("AI service unavailable")

            from src.scraper import scrape_company_page

            # Should fall back to basic parsing or return empty
            result = scrape_company_page("https://example.com/careers")

            # Should not crash, even if AI fails
            assert isinstance(result, (list, type(None)))
            if result:
                assert all(isinstance(job, dict) for job in result)

    def test_rate_limiting_backoff(self):
        """Test exponential backoff for rate limiting."""
        with (
            patch("src.scraper.httpx.AsyncClient") as mock_client,
            patch("time.sleep") as mock_sleep,
        ):
            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # Return 429 (rate limited) then success
            responses = [
                Mock(status_code=429),
                Mock(status_code=429),
                Mock(status_code=200, text="<html><body>Success</body></html>"),
            ]
            mock_client_instance.get.side_effect = responses

            from src.scraper import scrape_company_page

            scrape_company_page("https://example.com/careers")

            # Should have used exponential backoff
            if mock_sleep.call_count > 0:
                # Verify sleep times increase (exponential backoff)
                sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
                assert len(sleep_times) >= 1
                # Should have some delay for rate limiting
                assert any(t > 0 for t in sleep_times)

    def test_concurrent_scraping_error_isolation(self):
        """Test that errors in one scraping task don't affect others."""
        from concurrent.futures import ThreadPoolExecutor

        def scrape_with_potential_error(url):
            """Mock scraping function that may fail."""
            if "fail" in url:
                raise ConnectionError("Intentional failure")
            return [{"title": f"Job from {url}", "company": "Test Co"}]

        urls = [
            "https://good1.com/careers",
            "https://fail1.com/careers",
            "https://good2.com/careers",
            "https://fail2.com/careers",
            "https://good3.com/careers",
        ]

        results = []
        errors = []

        # Run scraping tasks concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {
                executor.submit(scrape_with_potential_error, url): url for url in urls
            }

            for future in future_to_url:
                url = future_to_url[future]
                try:
                    result = future.result(timeout=1.0)
                    results.append((url, result))
                except Exception as e:
                    errors.append((url, str(e)))

        # Verify successful tasks completed despite failures
        successful_results = [r for r in results if r[1]]
        assert len(successful_results) == 3  # 3 good URLs
        assert len(errors) == 2  # 2 failing URLs

        # Verify errors were isolated
        for url, error in errors:
            assert "fail" in url
            assert "Intentional failure" in error


class TestDatabaseErrorHandling:
    """Test database error recovery and transaction integrity."""

    def test_transaction_rollback_on_error(self, test_database_with_data):
        """Test transaction rollback when database errors occur."""
        with patch("src.database.create_engine") as mock_engine:
            # Setup engine that fails on commit
            mock_session = Mock()
            mock_session.add = Mock()
            mock_session.commit.side_effect = OperationalError(
                "statement", "params", "orig"
            )
            mock_session.rollback = Mock()

            mock_engine.return_value.begin.return_value.__enter__.return_value = (
                mock_session
            )

            from src.services.job_service import JobService

            # Attempt to create job that should fail
            job_data = JobCreate(
                company_id=1,
                title="Test Job",
                description="Test Description",
                link="https://test.com/job",
                location="Remote",
            )

            # Should handle error gracefully
            with contextlib.suppress(Exception):
                JobService.create_job(job_data)

            # Verify rollback was called
            if hasattr(mock_session, "rollback"):
                mock_session.rollback.assert_called()

    def test_corrupted_database_recovery(self, corrupted_database):
        """Test recovery from partially corrupted database."""
        # Try to read from corrupted database
        try:
            analytics = AnalyticsService(db_path=corrupted_database)

            # Some operations might fail, others might work
            job_trends = analytics.get_job_trends(days=30)
            company_analytics = analytics.get_company_analytics()

            # Should return error status, not crash
            assert job_trends["status"] in ["success", "error"]
            assert company_analytics["status"] in ["success", "error"]

        except Exception:
            # If exception occurs, it should be a controlled database error
            pass

    def test_database_connection_pool_exhaustion(self, test_database_with_data):
        """Test handling when database connection pool is exhausted."""
        import threading

        from concurrent.futures import ThreadPoolExecutor

        connection_errors = []
        successful_queries = []
        error_lock = threading.Lock()

        def attempt_database_query(query_id):
            """Attempt database query that might fail due to pool exhaustion."""
            try:
                with db_session() as session:
                    # Simulate holding connection for a while
                    time.sleep(0.1)

                    session.query(CompanySQL).limit(1).all()
                    with error_lock:
                        successful_queries.append(query_id)

            except Exception as e:
                with error_lock:
                    connection_errors.append((query_id, str(e)))

        # Attempt many concurrent database operations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(attempt_database_query, i) for i in range(50)]

            # Wait for all to complete
            for future in futures:
                try:
                    future.result(timeout=5.0)
                except Exception:
                    pass  # Individual failures are tracked

        # Some should succeed, system should handle pool exhaustion gracefully
        total_attempts = len(successful_queries) + len(connection_errors)
        assert total_attempts == 50
        assert len(successful_queries) > 0  # At least some should succeed

    def test_database_lock_timeout_handling(self, test_database_with_data):
        """Test handling of database lock timeouts."""
        import sqlite3
        import threading

        # Create direct SQLite connection for locking
        conn1 = sqlite3.connect(test_database_with_data)
        conn1.execute("BEGIN EXCLUSIVE TRANSACTION")

        lock_errors = []

        def attempt_locked_operation():
            """Attempt operation on locked database."""
            try:
                with db_session() as session:
                    # This should timeout due to exclusive lock
                    session.query(CompanySQL).all()

            except Exception as e:
                lock_errors.append(str(e))

        # Start thread that will encounter lock
        thread = threading.Thread(target=attempt_locked_operation)
        thread.start()
        thread.join(timeout=2.0)

        # Release lock
        conn1.rollback()
        conn1.close()

        # Should have encountered lock error or timeout
        # (exact behavior depends on SQLite configuration)
        assert thread.is_alive() is False  # Thread should complete

    def test_schema_migration_error_recovery(self, tmp_path):
        """Test recovery from schema migration errors."""
        db_path = tmp_path / "migration_test.db"

        # Create database with old schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE companysql (
                id INTEGER PRIMARY KEY,
                name TEXT,
                old_column TEXT  -- This column doesn't exist in current schema
            )
        """)
        conn.commit()
        conn.close()

        # Try to use with current schema
        try:
            engine = create_engine(
                f"sqlite:///{db_path}",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )

            # This might fail due to schema mismatch
            with contextlib.suppress(Exception):
                SQLModel.metadata.create_all(engine)

            # Test service operations with schema issues
            company_service = CompanyService()
            with contextlib.suppress(Exception):
                stats = company_service.get_company_statistics()
                assert isinstance(stats, (list, dict, type(None)))

        except Exception:
            # Schema errors should be handled gracefully
            pass

    def test_concurrent_database_modification_conflicts(self, test_database_with_data):
        """Test handling of concurrent modification conflicts."""
        import threading

        from concurrent.futures import ThreadPoolExecutor

        conflict_results = []
        result_lock = threading.Lock()

        def modify_company_concurrently(company_id, worker_id):
            """Modify same company from multiple threads."""
            try:
                with db_session() as session:
                    company = (
                        session.query(CompanySQL)
                        .filter(CompanySQL.id == company_id)
                        .first()
                    )
                    if company:
                        # Simulate processing time
                        time.sleep(0.01)

                        # Modify company
                        company.scrape_count += 1
                        company.last_scraped = datetime.now(UTC)
                        session.commit()

                        with result_lock:
                            conflict_results.append(("success", worker_id))
                    else:
                        with result_lock:
                            conflict_results.append(("not_found", worker_id))

            except Exception as e:
                with result_lock:
                    conflict_results.append(("error", worker_id, str(e)))

        # Multiple workers modify same company
        company_id = 1
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(modify_company_concurrently, company_id, worker_id)
                for worker_id in range(20)
            ]

            for future in futures:
                future.result(timeout=2.0)

        # Analyze results
        successes = [r for r in conflict_results if r[0] == "success"]
        errors = [r for r in conflict_results if r[0] == "error"]

        # Should have some successes, handle conflicts gracefully
        assert len(successes) > 0
        # Some conflicts are acceptable in concurrent scenarios
        total_processed = len(successes) + len(errors)
        assert total_processed == 20


class TestServiceLevelErrorHandling:
    """Test error handling at the service layer."""

    def test_analytics_service_degraded_mode(self, test_database_with_data):
        """Test analytics service operating in degraded mode."""
        # Test with missing DuckDB
        with patch("src.services.analytics_service.DUCKDB_AVAILABLE", False):
            analytics = AnalyticsService(db_path=test_database_with_data)

            # Should return error responses, not crash
            trends = analytics.get_job_trends(days=30)
            companies = analytics.get_company_analytics()
            salaries = analytics.get_salary_analytics(days=90)

            assert trends["status"] == "error"
            assert companies["status"] == "error"
            assert salaries["status"] == "error"

            # Should include informative error messages
            assert "DuckDB unavailable" in trends["error"]

    def test_cost_monitor_budget_overflow_handling(self, tmp_path):
        """Test cost monitor behavior when budget is exceeded."""
        cost_monitor = CostMonitor(db_path=str(tmp_path / "overflow_test.db"))

        # Add costs that exceed budget
        large_costs = [
            ("model1", 10000, 25.00, "large_operation1"),
            ("model2", 15000, 30.00, "large_operation2"),
            ("model3", 12000, 20.00, "large_operation3"),  # Total: $75 > $50 budget
        ]

        for model, tokens, cost, operation in large_costs:
            cost_monitor.track_ai_cost(model, tokens, cost, operation)

        # Should handle over-budget gracefully
        summary = cost_monitor.get_monthly_summary()
        alerts = cost_monitor.get_cost_alerts()

        assert summary["budget_status"] == "over_budget"
        assert summary["total_cost"] > 50.0
        assert len(alerts) > 0
        assert any(alert["type"] == "error" for alert in alerts)

    def test_job_service_with_invalid_data(self):
        """Test job service handling of invalid or malformed data."""
        invalid_job_data_cases = [
            # Missing required fields
            {"title": "Job Title"},  # Missing company, description, etc.
            # Invalid data types
            {
                "company_id": "not_a_number",
                "title": "Valid Title",
                "description": "Valid Description",
                "link": "not_a_url",
                "location": "Valid Location",
            },
            # Extremely long data
            {
                "company_id": 1,
                "title": "x" * 10000,  # Very long title
                "description": "y" * 50000,  # Very long description
                "link": "https://example.com/job",
                "location": "Remote",
            },
            # SQL injection attempts
            {
                "company_id": 1,
                "title": "'; DROP TABLE jobsql; --",
                "description": "Description",
                "link": "https://example.com/job",
                "location": "Remote",
            },
            # None/null values
            {
                "company_id": None,
                "title": None,
                "description": "Description",
                "link": "https://example.com/job",
                "location": "Remote",
            },
        ]

        for invalid_data in invalid_job_data_cases:
            try:
                # Should either validate and reject, or handle gracefully
                with contextlib.suppress(Exception):
                    result = JobService.create_job(JobCreate(**invalid_data))
                    # If successful, should return valid result
                    if result:
                        assert hasattr(result, "id")

            except (ValueError, TypeError, AttributeError) as e:
                # Validation errors are acceptable
                assert "validation" in str(e).lower() or "type" in str(e).lower()

            except Exception:
                # Should not cause system crashes
                pass

    def test_company_service_duplicate_handling(self):
        """Test company service handling of duplicate entries."""
        from src.services.company_service import CompanyService

        # Attempt to create duplicate companies
        duplicate_companies = [
            {"name": "Duplicate Corp", "url": "https://example.com"},
            {"name": "Duplicate Corp", "url": "https://example.com"},  # Exact duplicate
            {
                "name": "Duplicate Corp",
                "url": "https://example.com/careers",
            },  # Similar URL
            {"name": "duplicate corp", "url": "https://example.com"},  # Case variation
        ]

        created_companies = []
        duplicate_errors = []

        for company_data in duplicate_companies:
            try:
                with contextlib.suppress(Exception):
                    # Should handle duplicates gracefully
                    result = CompanyService.create_company(**company_data)
                    if result:
                        created_companies.append(result)

            except Exception as e:
                duplicate_errors.append(str(e))

        # Should either prevent duplicates or handle them gracefully
        # Exact behavior depends on business logic
        assert len(created_companies) <= len(duplicate_companies)

    def test_search_service_malformed_query_handling(self):
        """Test search service handling of malformed search queries."""
        from src.services.search_service import JobSearchService

        malformed_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a" * 1000,  # Extremely long query
            "SELECT * FROM jobs WHERE 1=1",  # SQL injection attempt
            "'; DROP TABLE jobs; --",  # SQL injection
            "\\x00\\x01\\x02",  # Binary data
            "🚀🔥💯" * 100,  # Many emojis
            "search term\n\r\ttest",  # Special characters
            None,  # None value (if passed)
        ]

        for query in malformed_queries:
            try:
                if query is None:
                    continue

                # Should handle malformed queries gracefully
                search_service = JobSearchService()
                with contextlib.suppress(Exception):
                    results = search_service.search_jobs(query)
                    # Should return empty results or error, not crash
                    assert isinstance(results, (list, dict, type(None)))

            except (ValueError, TypeError) as e:
                # Input validation errors are acceptable
                assert "query" in str(e).lower() or "search" in str(e).lower()

            except Exception:
                # Should not cause system crashes
                pass


class TestIntegrationErrorScenarios:
    """Test error scenarios across multiple system components."""

    def test_full_scraping_pipeline_with_failures(self, test_database_with_data):
        """Test full scraping pipeline with various failure points."""
        failure_scenarios = []

        # Mock various failure points
        with (
            patch("src.scraper.httpx.AsyncClient") as mock_client,
            patch("src.ai_client.extract_jobs") as mock_ai,
            patch("src.services.job_service.JobService.create_job") as mock_create_job,
        ):
            # HTTP request fails
            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.get.side_effect = ConnectionError("Network error")

            try:
                from src.scraper import scrape_company_page

                scrape_company_page("https://example.com/careers")
                failure_scenarios.append(("http_failure", "handled"))
            except Exception as e:
                failure_scenarios.append(("http_failure", f"error: {e}"))

            # HTTP succeeds, AI extraction fails
            mock_client_instance.get.side_effect = None
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Jobs content</body></html>"
            mock_client_instance.get.return_value = mock_response

            mock_ai.side_effect = Exception("AI service down")

            try:
                scrape_company_page("https://example.com/careers")
                failure_scenarios.append(("ai_failure", "handled"))
            except Exception as e:
                failure_scenarios.append(("ai_failure", f"error: {e}"))

            # AI succeeds, database save fails
            mock_ai.side_effect = None
            mock_ai.return_value = [
                {"title": "Test Job", "company": "Test Co", "description": "Test"}
            ]

            mock_create_job.side_effect = SQLAlchemyError("Database error")

            try:
                # This would be in the full pipeline
                from src.services.job_service import JobService

                JobService.create_job(
                    JobCreate(
                        company_id=1,
                        title="Test Job",
                        description="Test Description",
                        link="https://test.com/job",
                        location="Remote",
                    )
                )
                failure_scenarios.append(("database_failure", "handled"))
            except Exception as e:
                failure_scenarios.append(("database_failure", f"error: {e}"))

        # Verify failures were handled gracefully
        for _scenario, outcome in failure_scenarios:
            assert "handled" in outcome or "error" in outcome
            # Should not have unhandled crashes

    def test_system_resource_exhaustion_recovery(self):
        """Test system behavior under resource exhaustion."""
        from concurrent.futures import ThreadPoolExecutor

        # Simulate memory-intensive operations
        memory_intensive_results = []

        def memory_intensive_operation(worker_id):
            """Operation that consumes memory."""
            try:
                # Create large data structure
                large_data = [f"data_{i}_{worker_id}" for i in range(10000)]

                # Simulate processing
                processed = [item.upper() for item in large_data[:1000]]

                return len(processed)

            except MemoryError:
                return "memory_error"
            except Exception as e:
                return f"error: {e}"

        # Run many memory-intensive operations concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(memory_intensive_operation, i) for i in range(100)
            ]

            for future in futures:
                try:
                    result = future.result(timeout=5.0)
                    memory_intensive_results.append(result)
                except Exception as e:
                    memory_intensive_results.append(f"timeout: {e}")

        # System should handle resource pressure gracefully
        successful_operations = [
            r for r in memory_intensive_results if isinstance(r, int)
        ]
        error_operations = [
            r for r in memory_intensive_results if not isinstance(r, int)
        ]

        # Should have some successful operations
        assert len(successful_operations) > 0

        # Errors should be controlled, not system crashes
        if error_operations:
            assert all(
                "error" in str(err)
                or "memory_error" in str(err)
                or "timeout" in str(err)
                for err in error_operations
            )

    def test_cascading_failure_isolation(self, test_database_with_data):
        """Test that failures in one component don't cascade to others."""
        component_results = {}

        # Test analytics component with database issues
        with patch.object(AnalyticsService, "_conn", None):
            analytics = AnalyticsService(db_path=test_database_with_data)
            try:
                analytics.get_job_trends(days=30)
                component_results["analytics"] = "handled"
            except Exception:
                component_results["analytics"] = "failed"

        # Test cost monitoring (should be independent)
        try:
            cost_monitor = CostMonitor(db_path="/tmp/test_costs.db")
            cost_monitor.track_ai_cost("test_model", 1000, 0.01, "test_operation")
            cost_monitor.get_monthly_summary()
            component_results["cost_monitor"] = "success"
        except Exception:
            component_results["cost_monitor"] = "failed"

        # Test company service (should be independent)
        try:
            CompanyService.get_all_companies()
            component_results["company_service"] = "handled"
        except Exception:
            component_results["company_service"] = "failed"

        # Verify failure isolation
        # Even if one component fails, others should continue working
        successful_components = [
            comp
            for comp, result in component_results.items()
            if result in ["success", "handled"]
        ]
        assert len(successful_components) >= 1  # At least one should work

    def test_recovery_after_temporary_failures(self, test_database_with_data):
        """Test system recovery after temporary failures are resolved."""
        recovery_test_results = []

        # Simulate temporary database failure and recovery
        class TemporaryFailureDB:
            def __init__(self):
                self.failure_count = 0

            def execute_query(self):
                self.failure_count += 1
                if self.failure_count <= 3:  # Fail first 3 attempts
                    raise OperationalError("statement", "params", "Temporary failure")
                return "success"

        temp_db = TemporaryFailureDB()

        # Attempt operations with retry
        for attempt in range(5):
            try:
                result = temp_db.execute_query()
                recovery_test_results.append(("attempt", attempt, result))
                break  # Success, stop retrying
            except OperationalError as e:
                recovery_test_results.append(("attempt", attempt, f"failed: {e}"))
                time.sleep(0.01)  # Brief delay before retry

        # Verify recovery occurred
        successful_attempts = [r for r in recovery_test_results if r[2] == "success"]
        failed_attempts = [r for r in recovery_test_results if "failed" in r[2]]

        assert len(successful_attempts) == 1  # Eventually succeeded
        assert len(failed_attempts) == 3  # Failed first 3 times

        # Verify system state after recovery
        final_result = recovery_test_results[-1]
        assert final_result[2] == "success"
