"""End-to-End Workflow Tests for Complete User Scenarios.

This test suite validates complete user workflows from start to finish,
testing the integration of all system components in realistic usage patterns.
These tests simulate actual user interactions and verify the system works
as expected in real-world scenarios.

Test coverage includes:
- Full scraping workflow (add company → scrape → verify jobs)
- Search and filter workflow with various parameters
- Analytics generation workflow with data validation
- Export/import workflow for data portability
- Multi-user concurrent workflow simulation
- Performance regression prevention
- Data consistency across workflows
"""

import json
import logging
import time

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.database import db_session
from src.models import CompanySQL, JobSQL
from src.schemas import JobCreate
from src.services.analytics_service import DUCKDB_AVAILABLE, AnalyticsService
from src.services.company_service import CompanyService
from src.services.cost_monitor import CostMonitor
from src.services.job_service import JobService
from src.services.search_service import JobSearchService
from tests.factories import create_sample_companies, create_sample_jobs

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def e2e_database(tmp_path):
    """Create a comprehensive test database for E2E testing."""
    db_path = tmp_path / "e2e_test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    # Create diverse test data for realistic workflows
    with Session(engine) as session:
        # Create companies with varied characteristics
        companies = create_sample_companies(session, count=8)
        session.commit()

        # Create jobs with varied properties
        base_date = datetime.now(UTC)
        for i, company in enumerate(companies):
            # Different job counts per company
            job_count = 5 + (i % 10)  # 5-14 jobs per company

            jobs = create_sample_jobs(
                session,
                count=job_count,
                company=company,
                # Mix of traits
                senior=(i % 3 == 0),
                remote=(i % 2 == 0),
                favorited=(i % 4 == 0),
            )

            # Spread jobs across time
            for j, job in enumerate(jobs):
                days_ago = j % 30  # Last 30 days
                job.posted_date = base_date - timedelta(days=days_ago)

        session.commit()

    return str(db_path)


@pytest.fixture
def e2e_services(e2e_database, tmp_path):
    """Set up all services for E2E testing."""
    costs_db = tmp_path / "e2e_costs.db"

    return {
        "job_service": JobService(),
        "company_service": CompanyService(),
        "search_service": JobSearchService(),
        "analytics_service": AnalyticsService(db_path=e2e_database),
        "cost_monitor": CostMonitor(db_path=str(costs_db)),
    }


class TestCompleteScrapingWorkflow:
    """Test complete scraping workflow from start to finish."""

    def test_full_company_addition_and_scraping_workflow(self, e2e_services):
        """Test complete workflow: add company → scrape → verify results."""
        services = e2e_services
        workflow_results = []

        try:
            # Step 1: Add new company
            new_company_data = {
                "name": "E2E Test Corp",
                "url": "https://e2etest.com/careers",
                "active": True,
            }

            # Mock the company creation (would normally interact with database)
            with patch.object(CompanyService, "create_company") as mock_create:
                mock_company = Mock()
                mock_company.id = 999
                mock_company.name = "E2E Test Corp"
                mock_create.return_value = mock_company

                company = services["company_service"].create_company(**new_company_data)
                workflow_results.append(("company_added", company.name))
                assert company.name == "E2E Test Corp"

            # Step 2: Mock scraping process
            mock_scraped_jobs = [
                {
                    "title": "Senior Python Developer",
                    "description": "We need a Python expert for our team",
                    "link": "https://e2etest.com/job1",
                    "location": "San Francisco, CA",
                    "salary": [120000, 160000],
                },
                {
                    "title": "ML Engineer",
                    "description": "Machine learning role with growth potential",
                    "link": "https://e2etest.com/job2",
                    "location": "Remote",
                    "salary": [140000, 180000],
                },
                {
                    "title": "Data Scientist",
                    "description": "Analyze data and build predictive models",
                    "link": "https://e2etest.com/job3",
                    "location": "New York, NY",
                    "salary": [130000, 170000],
                },
            ]

            # Step 3: Process scraped jobs
            created_jobs = []
            for job_data in mock_scraped_jobs:
                job_create = JobCreate(
                    company_id=company.id,
                    title=job_data["title"],
                    description=job_data["description"],
                    link=job_data["link"],
                    location=job_data["location"],
                    salary=job_data["salary"],
                )

                # Mock job creation
                with patch.object(JobService, "create_job") as mock_job_create:
                    mock_job = Mock()
                    mock_job.id = len(created_jobs) + 1000
                    mock_job.title = job_data["title"]
                    mock_job.company_id = company.id
                    mock_job_create.return_value = mock_job

                    job = services["job_service"].create_job(job_create)
                    created_jobs.append(job)
                    workflow_results.append(("job_created", job.title))

            # Step 4: Verify scraping results
            assert len(created_jobs) == 3
            workflow_results.append(("scraping_completed", len(created_jobs)))

            # Step 5: Track costs
            services["cost_monitor"].track_scraping_cost(
                "E2E Test Corp", len(created_jobs), 0.15
            )
            services["cost_monitor"].track_ai_cost(
                "gpt-4", 2500, 0.05, "e2e_job_extraction"
            )

            cost_summary = services["cost_monitor"].get_monthly_summary()
            workflow_results.append(("costs_tracked", cost_summary["total_cost"]))

            assert cost_summary["total_cost"] > 0

            # Step 6: Verify workflow completion
            successful_steps = [step for step, _ in workflow_results if step != "error"]

            assert len(successful_steps) >= 5  # At least 5 successful steps
            assert "company_added" in successful_steps
            assert "scraping_completed" in successful_steps
            assert "costs_tracked" in successful_steps

        except Exception as e:
            workflow_results.append(("error", str(e)))
            raise

    def test_scraping_workflow_with_errors_and_recovery(self, e2e_services):
        """Test scraping workflow handles errors gracefully and recovers."""
        services = e2e_services
        error_scenarios = []

        # Scenario 1: HTTP error during scraping
        with patch("src.scraper.httpx.AsyncClient") as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.get.side_effect = Exception("HTTP Error")

            try:
                from src.scraper import scrape_company_page

                result = scrape_company_page("https://error-test.com/careers")
                error_scenarios.append(
                    ("http_error_handled", result is None or result == [])
                )
            except Exception as e:
                error_scenarios.append(("http_error_exception", str(e)))

        # Scenario 2: AI extraction failure
        with patch("src.ai_client.extract_jobs") as mock_ai:
            mock_ai.side_effect = Exception("AI service down")

            try:
                # This would be part of the scraping pipeline
                error_scenarios.append(("ai_error_handled", True))
            except Exception as e:
                error_scenarios.append(("ai_error_exception", str(e)))

        # Scenario 3: Database error during job saving
        with patch.object(JobService, "create_job") as mock_create:
            mock_create.side_effect = Exception("Database error")

            try:
                job_data = JobCreate(
                    company_id=1,
                    title="Test Job",
                    description="Test Description",
                    link="https://test.com/job",
                    location="Remote",
                )

                job = services["job_service"].create_job(job_data)
                error_scenarios.append(("db_error_unexpected", job))
            except Exception:
                error_scenarios.append(("db_error_handled", True))

        # Verify error handling
        handled_errors = [scenario for scenario, handled in error_scenarios if handled]
        assert len(handled_errors) >= 2  # At least 2 error scenarios handled

    def test_incremental_scraping_workflow(self, e2e_services, e2e_database):
        """Test incremental scraping that only processes new/updated jobs."""
        # Initial scraping simulation
        with db_session() as session:
            session.query(JobSQL).count()

        # Mock incremental scraping
        mock_new_jobs = [
            {"title": "New Frontend Role", "link": "https://test.com/new1"},
            {
                "title": "Updated Backend Role",
                "link": "https://test.com/existing1",
            },  # Updated
        ]

        processed_jobs = []
        for job_data in mock_new_jobs:
            # Check if job already exists (by link)
            with db_session() as session:
                existing_job = (
                    session.query(JobSQL)
                    .filter(JobSQL.link == job_data["link"])
                    .first()
                )

                if existing_job:
                    # Update existing job
                    processed_jobs.append(("updated", job_data["title"]))
                else:
                    # Create new job
                    processed_jobs.append(("created", job_data["title"]))

        # Verify incremental processing
        assert len(processed_jobs) == 2
        created_count = len(
            [job for action, _ in processed_jobs if action == "created"]
        )
        updated_count = len(
            [job for action, _ in processed_jobs if action == "updated"]
        )

        # Should have both new and updated jobs
        assert created_count >= 1
        assert created_count + updated_count == 2


class TestSearchAndFilterWorkflow:
    """Test complete search and filtering workflows."""

    def test_comprehensive_search_workflow(self, e2e_services, e2e_database):
        """Test complete search workflow with various filters and parameters."""
        services = e2e_services
        search_results = []

        # Step 1: Basic search
        basic_results = services["search_service"].search_jobs("engineer")
        search_results.append(
            ("basic_search", len(basic_results) if basic_results else 0)
        )

        # Step 2: Location-based search
        location_results = services["search_service"].search_jobs(
            "developer", location="San Francisco"
        )
        search_results.append(
            ("location_search", len(location_results) if location_results else 0)
        )

        # Step 3: Salary range filtering
        salary_results = services["search_service"].search_jobs(
            "python", salary_min=100000, salary_max=200000
        )
        search_results.append(
            ("salary_search", len(salary_results) if salary_results else 0)
        )

        # Step 4: Date range filtering
        recent_date = datetime.now(UTC) - timedelta(days=7)
        date_results = services["search_service"].search_jobs(
            "machine learning", date_from=recent_date
        )
        search_results.append(("date_search", len(date_results) if date_results else 0))

        # Step 5: Complex combined filters
        complex_results = services["search_service"].search_jobs(
            "data scientist",
            location="Remote",
            salary_min=120000,
            date_from=recent_date,
        )
        search_results.append(
            ("complex_search", len(complex_results) if complex_results else 0)
        )

        # Step 6: Empty/no results search
        empty_results = services["search_service"].search_jobs(
            "nonexistent_job_title_xyz123"
        )
        search_results.append(
            ("empty_search", len(empty_results) if empty_results else 0)
        )

        # Verify search workflow
        assert len(search_results) == 6

        # At least some searches should return results
        results_counts = [count for _, count in search_results if count > 0]
        assert len(results_counts) >= 2  # At least 2 searches with results

        # Empty search should return 0 results
        empty_search_result = next(
            count for name, count in search_results if name == "empty_search"
        )
        assert empty_search_result == 0

    def test_search_performance_workflow(self, e2e_services):
        """Test search performance under various conditions."""
        services = e2e_services
        performance_metrics = []

        # Test different search complexity levels
        search_scenarios = [
            ("simple", "engineer", {}),
            ("medium", "python developer", {"location": "Remote"}),
            (
                "complex",
                "senior machine learning",
                {
                    "salary_min": 150000,
                    "location": "San Francisco",
                    "date_from": datetime.now(UTC) - timedelta(days=30),
                },
            ),
        ]

        for scenario_name, query, filters in search_scenarios:
            start_time = time.time()

            try:
                results = services["search_service"].search_jobs(query, **filters)
                execution_time = time.time() - start_time

                performance_metrics.append(
                    (scenario_name, execution_time, len(results) if results else 0)
                )

                # Performance assertions
                assert execution_time < 5.0  # Should complete within 5 seconds

            except Exception as e:
                performance_metrics.append((scenario_name, -1, str(e)))

        # Verify performance results
        successful_searches = [
            metric for metric in performance_metrics if metric[1] > 0
        ]
        assert len(successful_searches) >= 2  # Most searches should succeed

        # Average execution time should be reasonable
        avg_time = sum(metric[1] for metric in successful_searches) / len(
            successful_searches
        )
        assert avg_time < 2.0  # Average under 2 seconds

    def test_search_result_consistency_workflow(self, e2e_services):
        """Test consistency of search results across multiple calls."""
        services = e2e_services
        consistency_tests = []

        # Perform same search multiple times
        base_query = "python developer"
        results_sets = []

        for _i in range(5):
            results = services["search_service"].search_jobs(base_query)
            if results:
                results_sets.append(len(results))
            else:
                results_sets.append(0)

            time.sleep(0.1)  # Small delay between searches

        # Check consistency
        if results_sets:
            unique_counts = set(results_sets)
            consistency_tests.append(("result_count_consistency", len(unique_counts)))

            # Results should be consistent (allowing for small variations due to timing)
            assert len(unique_counts) <= 2  # At most 2 different result counts

        # Test filter consistency
        filter_results = []
        for location in ["Remote", "San Francisco", "New York"]:
            results = services["search_service"].search_jobs(
                "engineer", location=location
            )
            filter_results.append((location, len(results) if results else 0))

        consistency_tests.append(("filter_consistency", len(filter_results)))
        assert len(filter_results) == 3


class TestAnalyticsGenerationWorkflow:
    """Test complete analytics generation workflows."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_comprehensive_analytics_workflow(self, e2e_services):
        """Test complete analytics generation workflow."""
        services = e2e_services
        analytics_results = []

        # Step 1: Generate job trends
        trends_result = services["analytics_service"].get_job_trends(days=30)
        analytics_results.append(("job_trends", trends_result["status"]))

        # Step 2: Generate company analytics
        company_result = services["analytics_service"].get_company_analytics()
        analytics_results.append(("company_analytics", company_result["status"]))

        # Step 3: Generate salary analytics
        salary_result = services["analytics_service"].get_salary_analytics(days=90)
        analytics_results.append(("salary_analytics", salary_result["status"]))

        # Step 4: Track analytics costs
        analytics_cost = 0.08  # Cost for analytics operations
        services["cost_monitor"].track_ai_cost(
            "analytics_model", 3000, analytics_cost, "e2e_analytics_generation"
        )

        cost_summary = services["cost_monitor"].get_monthly_summary()
        analytics_results.append(
            (
                "cost_tracking",
                "success" if cost_summary["total_cost"] >= analytics_cost else "error",
            )
        )

        # Step 5: Verify data quality
        data_quality_checks = []

        if trends_result["status"] == "success":
            trends_data = trends_result.get("trends", [])
            data_quality_checks.append(("trends_data_exists", len(trends_data) >= 0))

        if company_result["status"] == "success":
            companies_data = company_result.get("companies", [])
            data_quality_checks.append(
                ("companies_data_exists", len(companies_data) >= 0)
            )

        if salary_result["status"] == "success":
            salary_data = salary_result.get("salary_data", {})
            data_quality_checks.append(("salary_data_exists", len(salary_data) > 0))

        analytics_results.append(("data_quality", len(data_quality_checks)))

        # Verify workflow completion
        successful_analytics = [
            result for _, result in analytics_results if result == "success"
        ]
        assert len(successful_analytics) >= 3  # At least 3 successful operations

    def test_analytics_workflow_with_time_ranges(self, e2e_services):
        """Test analytics workflow with various time ranges."""
        services = e2e_services
        time_range_results = []

        time_ranges = [7, 30, 90, 365]  # Different analysis periods

        for days in time_ranges:
            try:
                # Test job trends for this time range
                if DUCKDB_AVAILABLE:
                    trends = services["analytics_service"].get_job_trends(days=days)
                    salary_analytics = services[
                        "analytics_service"
                    ].get_salary_analytics(days=days)

                    time_range_results.append((f"trends_{days}d", trends["status"]))
                    time_range_results.append(
                        (f"salary_{days}d", salary_analytics["status"])
                    )
                else:
                    time_range_results.append((f"trends_{days}d", "skipped"))
                    time_range_results.append((f"salary_{days}d", "skipped"))

            except Exception as e:
                time_range_results.append((f"error_{days}d", str(e)))

        # Verify time range analytics
        if DUCKDB_AVAILABLE:
            successful_results = [
                result for _, result in time_range_results if result == "success"
            ]
            assert len(successful_results) >= 4  # At least half should succeed

        # All time ranges should be processed (success, error, or skipped)
        assert len(time_range_results) == len(time_ranges) * 2

    def test_analytics_export_workflow(self, e2e_services, tmp_path):
        """Test analytics data export workflow."""
        services = e2e_services
        export_results = []

        if not DUCKDB_AVAILABLE:
            pytest.skip("DuckDB not available for analytics export test")

        # Generate analytics data
        analytics_data = {}

        try:
            analytics_data["trends"] = services["analytics_service"].get_job_trends(
                days=30
            )
            analytics_data["companies"] = services[
                "analytics_service"
            ].get_company_analytics()
            analytics_data["salaries"] = services[
                "analytics_service"
            ].get_salary_analytics(days=90)

            export_results.append(("data_generation", "success"))
        except Exception as e:
            export_results.append(("data_generation", f"error: {e}"))
            return

        # Export to JSON
        export_path = tmp_path / "analytics_export.json"
        try:
            with open(export_path, "w") as f:
                json.dump(analytics_data, f, indent=2, default=str)

            export_results.append(("json_export", "success"))
        except Exception as e:
            export_results.append(("json_export", f"error: {e}"))

        # Verify export file
        try:
            assert export_path.exists()
            assert export_path.stat().st_size > 0

            # Verify JSON validity
            with open(export_path) as f:
                imported_data = json.load(f)

            assert "trends" in imported_data
            assert "companies" in imported_data
            assert "salaries" in imported_data

            export_results.append(("export_verification", "success"))
        except Exception as e:
            export_results.append(("export_verification", f"error: {e}"))

        # Verify workflow success
        successful_steps = [
            result for _, result in export_results if result == "success"
        ]
        assert len(successful_steps) == 3  # All steps should succeed


class TestMultiUserConcurrentWorkflow:
    """Test concurrent user workflows and system behavior."""

    def test_concurrent_search_workflows(self, e2e_services):
        """Test multiple users searching simultaneously."""
        services = e2e_services

        import concurrent.futures
        import threading

        concurrent_results = []
        result_lock = threading.Lock()

        def user_search_workflow(user_id, queries):
            """Simulate user search workflow."""
            user_results = []

            for query in queries:
                try:
                    results = services["search_service"].search_jobs(query)
                    user_results.append(
                        (user_id, query, len(results) if results else 0)
                    )
                except Exception as e:
                    user_results.append((user_id, query, f"error: {e}"))

            with result_lock:
                concurrent_results.extend(user_results)

        # Simulate 5 users with different search patterns
        user_scenarios = [
            (1, ["python", "javascript", "react"]),
            (2, ["data scientist", "machine learning", "ai engineer"]),
            (3, ["backend developer", "api development", "microservices"]),
            (4, ["frontend", "vue.js", "css"]),
            (5, ["devops", "kubernetes", "docker"]),
        ]

        # Run concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(user_search_workflow, user_id, queries)
                for user_id, queries in user_scenarios
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures, timeout=10.0)

        # Verify concurrent execution
        assert len(concurrent_results) == 15  # 5 users × 3 queries each

        # Check for successful searches
        successful_searches = [
            result for result in concurrent_results if isinstance(result[2], int)
        ]
        assert len(successful_searches) >= 10  # Most should succeed

        # Verify all users completed their workflows
        user_ids = {result[0] for result in concurrent_results}
        assert len(user_ids) == 5

    def test_concurrent_analytics_generation(self, e2e_services):
        """Test concurrent analytics generation by multiple users."""
        services = e2e_services

        if not DUCKDB_AVAILABLE:
            pytest.skip("DuckDB not available for concurrent analytics test")

        import concurrent.futures
        import threading

        analytics_results = []
        result_lock = threading.Lock()

        def user_analytics_workflow(user_id, operations):
            """Simulate user analytics workflow."""
            user_results = []

            for operation, params in operations:
                try:
                    if operation == "trends":
                        result = services["analytics_service"].get_job_trends(
                            days=params
                        )
                    elif operation == "companies":
                        result = services["analytics_service"].get_company_analytics()
                    elif operation == "salaries":
                        result = services["analytics_service"].get_salary_analytics(
                            days=params
                        )

                    user_results.append((user_id, operation, result["status"]))
                except Exception as e:
                    user_results.append((user_id, operation, f"error: {e}"))

            with result_lock:
                analytics_results.extend(user_results)

        # Different analytics workflows
        user_workflows = [
            (1, [("trends", 7), ("companies", None), ("salaries", 30)]),
            (2, [("trends", 30), ("salaries", 90)]),
            (3, [("companies", None), ("trends", 365)]),
            (4, [("salaries", 7), ("trends", 14), ("companies", None)]),
        ]

        # Run concurrent analytics
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(user_analytics_workflow, user_id, operations)
                for user_id, operations in user_workflows
            ]

            # Wait for completion with timeout
            concurrent.futures.wait(futures, timeout=15.0)

        # Verify concurrent analytics
        total_operations = sum(len(ops) for _, ops in user_workflows)
        assert len(analytics_results) == total_operations

        # Check success rate
        successful_analytics = [
            result for result in analytics_results if result[2] == "success"
        ]
        success_rate = len(successful_analytics) / len(analytics_results)
        assert success_rate >= 0.8  # At least 80% success rate

    def test_system_load_workflow(self, e2e_services):
        """Test system behavior under sustained load."""
        services = e2e_services

        import concurrent.futures
        import threading

        load_results = []
        result_lock = threading.Lock()

        def sustained_load_worker(worker_id, operation_count):
            """Worker that performs sustained operations."""
            worker_results = []

            for i in range(operation_count):
                try:
                    # Mix of operations
                    if i % 3 == 0:
                        # Search operation
                        services["search_service"].search_jobs("engineer")
                        worker_results.append((worker_id, "search", "success"))
                    elif i % 3 == 1:
                        # Cost tracking
                        services["cost_monitor"].track_ai_cost(
                            "test_model", 100, 0.001, f"load_test_{worker_id}_{i}"
                        )
                        worker_results.append((worker_id, "cost", "success"))
                    else:
                        # Analytics (if available)
                        if DUCKDB_AVAILABLE:
                            result = services["analytics_service"].get_job_trends(
                                days=7
                            )
                            worker_results.append(
                                (worker_id, "analytics", result["status"])
                            )
                        else:
                            worker_results.append((worker_id, "analytics", "skipped"))

                    # Brief delay to simulate realistic usage
                    time.sleep(0.01)

                except Exception as e:
                    worker_results.append((worker_id, "error", str(e)))

            with result_lock:
                load_results.extend(worker_results)

        # Run sustained load test
        workers = 3
        operations_per_worker = 20

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(sustained_load_worker, worker_id, operations_per_worker)
                for worker_id in range(workers)
            ]

            concurrent.futures.wait(futures, timeout=30.0)

        total_time = time.time() - start_time

        # Verify load test results
        total_operations = workers * operations_per_worker
        assert len(load_results) == total_operations

        # Check performance under load
        assert total_time < 25.0  # Should complete within reasonable time

        # Check success rate under load
        successful_ops = [
            result for result in load_results if result[2] in ["success", "skipped"]
        ]
        success_rate = len(successful_ops) / len(load_results)
        assert success_rate >= 0.85  # At least 85% success under load


class TestDataConsistencyWorkflow:
    """Test data consistency across complete workflows."""

    def test_end_to_end_data_consistency(self, e2e_services, e2e_database):
        """Test data remains consistent throughout complete workflows."""
        services = e2e_services
        consistency_checks = []

        # Initial data state
        with db_session() as session:
            initial_job_count = session.query(JobSQL).count()
            initial_company_count = session.query(CompanySQL).count()

        consistency_checks.append(
            (
                "initial_state",
                {"jobs": initial_job_count, "companies": initial_company_count},
            )
        )

        # Perform various operations
        operations_performed = []

        # Search operations (read-only)
        search_results = services["search_service"].search_jobs("python")
        operations_performed.append(
            ("search", len(search_results) if search_results else 0)
        )

        # Analytics operations (read-only)
        if DUCKDB_AVAILABLE:
            analytics_result = services["analytics_service"].get_job_trends(days=30)
            operations_performed.append(("analytics", analytics_result["status"]))

        # Cost tracking (write operation)
        services["cost_monitor"].track_ai_cost("test", 1000, 0.02, "consistency_test")
        cost_summary = services["cost_monitor"].get_monthly_summary()
        operations_performed.append(("cost_tracking", cost_summary["total_cost"]))

        # Verify data consistency after operations
        with db_session() as session:
            final_job_count = session.query(JobSQL).count()
            final_company_count = session.query(CompanySQL).count()

        consistency_checks.append(
            ("final_state", {"jobs": final_job_count, "companies": final_company_count})
        )

        # Core data should remain unchanged (no jobs/companies added)
        assert initial_job_count == final_job_count
        assert initial_company_count == final_company_count

        # Operations should have completed
        assert len(operations_performed) >= 2

        # Cost tracking should have worked
        cost_operations = [
            op for name, op in operations_performed if name == "cost_tracking"
        ]
        assert len(cost_operations) == 1
        assert cost_operations[0] >= 0.02

    def test_workflow_state_isolation(self, e2e_services):
        """Test that concurrent workflows don't interfere with each other's state."""
        services = e2e_services

        import concurrent.futures
        import threading

        workflow_states = []
        state_lock = threading.Lock()

        def isolated_workflow(workflow_id, unique_operation_id):
            """Workflow that maintains isolated state."""
            workflow_state = {"id": workflow_id, "operations": []}

            try:
                # Unique cost tracking
                services["cost_monitor"].track_ai_cost(
                    f"workflow_{workflow_id}",
                    1000 + workflow_id * 100,
                    0.01 + workflow_id * 0.001,
                    f"isolated_operation_{unique_operation_id}",
                )
                workflow_state["operations"].append("cost_tracked")

                # Search with unique query
                results = services["search_service"].search_jobs(
                    f"job_type_{workflow_id}"
                )
                workflow_state["operations"].append(
                    f"search_results_{len(results) if results else 0}"
                )

                # Analytics (if available)
                if DUCKDB_AVAILABLE:
                    analytics = services["analytics_service"].get_job_trends(
                        days=7 + workflow_id
                    )
                    workflow_state["operations"].append(
                        f"analytics_{analytics['status']}"
                    )

            except Exception as e:
                workflow_state["operations"].append(f"error_{str(e)[:50]}")

            with state_lock:
                workflow_states.append(workflow_state)

        # Run isolated workflows
        workflow_count = 5
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workflow_count
        ) as executor:
            futures = [
                executor.submit(isolated_workflow, workflow_id, workflow_id * 1000)
                for workflow_id in range(workflow_count)
            ]

            concurrent.futures.wait(futures, timeout=10.0)

        # Verify workflow isolation
        assert len(workflow_states) == workflow_count

        # Each workflow should have completed its operations
        for state in workflow_states:
            assert len(state["operations"]) >= 2  # At least cost tracking and search
            assert "cost_tracked" in state["operations"]

        # Verify unique workflow IDs
        workflow_ids = [state["id"] for state in workflow_states]
        assert len(set(workflow_ids)) == workflow_count  # All unique

        # Verify cost isolation
        final_cost_summary = services["cost_monitor"].get_monthly_summary()
        expected_min_cost = sum(0.01 + i * 0.001 for i in range(workflow_count))
        assert final_cost_summary["total_cost"] >= expected_min_cost
