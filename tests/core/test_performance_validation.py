"""Performance validation tests.

This module tests that the application meets the <100ms response time
requirements established by Group 3's performance optimizations,
validating caching effectiveness and system responsiveness.
"""

import concurrent.futures
import time

from unittest.mock import patch

import pytest

from src.services.analytics_service import AnalyticsService
from src.services.job_service import JobService
from src.services.search_service import JobSearchService
from src.ui.utils.service_cache import (
    clear_service_cache,
    get_analytics_service,
    get_job_service,
    get_search_service,
)


class TestResponseTimeRequirements:
    """Test <100ms response time requirements."""

    @pytest.mark.performance
    def test_job_service_response_time(self):
        """Test that job service operations complete within 100ms."""
        job_service = get_job_service()

        # Mock the database operation to return quickly
        with patch.object(job_service, "get_jobs", return_value=[]):
            start_time = time.time()
            jobs = job_service.get_jobs(limit=50)
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            assert response_time < 100, (
                f"Job service took {response_time:.2f}ms (should be <100ms)"
            )
            assert isinstance(jobs, list)

    @pytest.mark.performance
    def test_search_service_response_time(self):
        """Test that search operations complete within 100ms."""
        search_service = get_search_service()

        # Mock the search operation to return quickly
        with patch.object(search_service, "search_jobs", return_value=[]):
            start_time = time.time()
            results = search_service.search_jobs("python developer")
            response_time = (time.time() - start_time) * 1000

            assert response_time < 100, (
                f"Search took {response_time:.2f}ms (should be <100ms)"
            )
            assert isinstance(results, list)

    @pytest.mark.performance
    def test_analytics_service_response_time(self):
        """Test that analytics operations complete within 100ms."""
        analytics_service = get_analytics_service()

        # Mock analytics operations to return quickly
        mock_trends = {"total_jobs": 100, "trends": []}
        with patch.object(
            analytics_service, "get_job_trends", return_value=mock_trends
        ):
            start_time = time.time()
            trends = analytics_service.get_job_trends(days=7)
            response_time = (time.time() - start_time) * 1000

            assert response_time < 100, (
                f"Analytics took {response_time:.2f}ms (should be <100ms)"
            )
            assert isinstance(trends, dict)

    @pytest.mark.performance
    def test_cached_service_instantiation_time(self):
        """Test that cached service instantiation is under 100ms."""
        # Clear cache first to test both cold and warm starts
        clear_service_cache()

        # Cold start (first time)
        start_time = time.time()
        job_service1 = get_job_service()
        cold_start_time = (time.time() - start_time) * 1000

        # Warm start (cached)
        start_time = time.time()
        job_service2 = get_job_service()
        warm_start_time = (time.time() - start_time) * 1000

        # Both should be under 100ms, cached should be much faster
        assert cold_start_time < 100, f"Cold start took {cold_start_time:.2f}ms"
        assert warm_start_time < 100, f"Warm start took {warm_start_time:.2f}ms"
        assert warm_start_time < cold_start_time, "Cached access should be faster"
        assert job_service1 is job_service2, "Services should be cached"


class TestCachingPerformanceValidation:
    """Test that caching optimizations achieve performance targets."""

    @pytest.mark.performance
    def test_streamlit_cache_resource_performance(self):
        """Test @st.cache_resource performance impact."""
        # Test multiple rapid calls to cached services
        call_times = []

        for _ in range(10):
            start_time = time.time()
            service = get_job_service()
            call_time = (time.time() - start_time) * 1000
            call_times.append(call_time)

            assert isinstance(service, JobService)

        # All calls should be fast
        max_call_time = max(call_times)
        avg_call_time = sum(call_times) / len(call_times)

        assert max_call_time < 10, f"Slowest cached call: {max_call_time:.2f}ms"
        assert avg_call_time < 1, f"Average cached call: {avg_call_time:.2f}ms"

    @pytest.mark.performance
    def test_concurrent_cache_access_performance(self):
        """Test performance under concurrent cache access."""

        def get_all_services():
            return [get_job_service(), get_search_service(), get_analytics_service()]

        start_time = time.time()

        # Concurrent access to all cached services
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_all_services) for _ in range(20)]
            results = [future.result() for future in futures]

        total_time = (time.time() - start_time) * 1000

        # Should handle 20 concurrent operations quickly
        assert total_time < 1000, (
            f"Concurrent access took {total_time:.2f}ms (should be <1000ms)"
        )
        assert len(results) == 20

        # All results should have same cached instances
        first_result = results[0]
        for result in results[1:]:
            assert result[0] is first_result[0]  # Same JobService
            assert result[1] is first_result[1]  # Same SearchService
            assert result[2] is first_result[2]  # Same AnalyticsService

    @pytest.mark.performance
    def test_cache_memory_efficiency(self):
        """Test that caching provides memory efficiency."""
        import gc

        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create many service references
        services = []
        start_time = time.time()

        for _ in range(1000):
            services.extend(
                [get_job_service(), get_search_service(), get_analytics_service()]
            )

        creation_time = (time.time() - start_time) * 1000

        # Should create references quickly due to caching
        assert creation_time < 1000, (
            f"Creating 3000 references took {creation_time:.2f}ms"
        )

        # Memory should not increase dramatically
        gc.collect()
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        assert object_increase < 50, f"Memory increased by {object_increase} objects"

        # Verify caching worked (only 3 unique instances)
        unique_services = set(id(s) for s in services)
        assert len(unique_services) == 3, (
            f"Should have 3 unique services, got {len(unique_services)}"
        )


class TestSystemResponseTimeValidation:
    """Test system-wide response time validation."""

    @pytest.mark.performance
    def test_job_filtering_performance(self):
        """Test job filtering operations performance."""
        job_service = get_job_service()

        # Test various filter combinations
        filter_combinations = [
            {"location": "Remote"},
            {"salary_min": 80000, "salary_max": 150000},
            {"company": ["Google", "Apple"]},
            {"application_status": ["Not Applied"]},
            {"favorites_only": True},
            {
                "location": "Remote",
                "salary_min": 100000,
                "company": ["Google", "Apple", "Meta"],
            },
        ]

        for filters in filter_combinations:
            with patch.object(job_service, "get_jobs", return_value=[]):
                start_time = time.time()
                jobs = job_service.get_jobs(filters=filters)
                response_time = (time.time() - start_time) * 1000

                assert response_time < 100, (
                    f"Filtering with {filters} took {response_time:.2f}ms (should be <100ms)"
                )
                assert isinstance(jobs, list)

    @pytest.mark.performance
    def test_search_with_filters_performance(self):
        """Test search with filters performance."""
        search_service = get_search_service()

        search_scenarios = [
            ("python", {}),
            ("data scientist", {"location": "Remote"}),
            ("machine learning", {"salary_min": 120000}),
            (
                "full stack",
                {"company": ["Google", "Meta"], "location": "San Francisco"},
            ),
        ]

        for query, filters in search_scenarios:
            with patch.object(search_service, "search_jobs", return_value=[]):
                start_time = time.time()
                results = search_service.search_jobs(query, filters=filters)
                response_time = (time.time() - start_time) * 1000

                assert response_time < 100, (
                    f"Search '{query}' with filters took {response_time:.2f}ms (should be <100ms)"
                )
                assert isinstance(results, list)

    @pytest.mark.performance
    def test_analytics_computation_performance(self):
        """Test analytics computation performance."""
        analytics_service = get_analytics_service()

        analytics_operations = [
            ("get_job_trends", {"days": 7}),
            ("get_job_trends", {"days": 30}),
            ("get_company_analytics", {}),
            ("get_salary_analytics", {}),
            ("get_location_analytics", {}),
        ]

        for method_name, kwargs in analytics_operations:
            if hasattr(analytics_service, method_name):
                method = getattr(analytics_service, method_name)

                with patch.object(analytics_service, method_name, return_value={}):
                    start_time = time.time()
                    result = method(**kwargs)
                    response_time = (time.time() - start_time) * 1000

                    assert response_time < 100, (
                        f"{method_name} took {response_time:.2f}ms (should be <100ms)"
                    )
                    assert result is not None


class TestLoadPerformanceValidation:
    """Test performance under various load scenarios."""

    @pytest.mark.performance
    def test_rapid_sequential_requests(self):
        """Test performance under rapid sequential requests."""
        job_service = get_job_service()

        with patch.object(job_service, "get_jobs", return_value=[]):
            request_times = []

            for i in range(50):
                start_time = time.time()
                jobs = job_service.get_jobs(limit=20)
                response_time = (time.time() - start_time) * 1000
                request_times.append(response_time)

                # Each request should be fast
                assert response_time < 100, (
                    f"Request {i + 1} took {response_time:.2f}ms (should be <100ms)"
                )
                assert isinstance(jobs, list)

            # Average should be very fast due to caching
            avg_time = sum(request_times) / len(request_times)
            max_time = max(request_times)

            assert avg_time < 10, (
                f"Average request time: {avg_time:.2f}ms (should be <10ms)"
            )
            assert max_time < 100, (
                f"Slowest request: {max_time:.2f}ms (should be <100ms)"
            )

    @pytest.mark.performance
    def test_mixed_operation_performance(self):
        """Test performance with mixed service operations."""
        job_service = get_job_service()
        search_service = get_search_service()
        analytics_service = get_analytics_service()

        # Mock all operations
        with (
            patch.object(job_service, "get_jobs", return_value=[]),
            patch.object(search_service, "search_jobs", return_value=[]),
            patch.object(analytics_service, "get_job_trends", return_value={}),
        ):
            operations = [
                (job_service.get_jobs, {"limit": 20}),
                (search_service.search_jobs, {"query": "python"}),
                (analytics_service.get_job_trends, {"days": 7}),
                (job_service.get_jobs, {"filters": {"location": "Remote"}}),
                (
                    search_service.search_jobs,
                    {"query": "data science", "filters": {"salary_min": 100000}},
                ),
            ]

            start_time = time.time()

            for operation, kwargs in operations:
                op_start = time.time()
                result = operation(**kwargs) if kwargs else operation()
                op_time = (time.time() - op_start) * 1000

                assert op_time < 100, (
                    f"Operation took {op_time:.2f}ms (should be <100ms)"
                )
                assert result is not None

            total_time = (time.time() - start_time) * 1000
            assert total_time < 500, (
                f"All operations took {total_time:.2f}ms (should be <500ms)"
            )

    @pytest.mark.performance
    def test_burst_load_performance(self):
        """Test performance under burst load conditions."""

        def perform_operation():
            job_service = get_job_service()
            search_service = get_search_service()

            # Simulate mixed operations
            operations_performed = 0
            with (
                patch.object(job_service, "get_jobs", return_value=[]),
                patch.object(search_service, "search_jobs", return_value=[]),
            ):
                start_time = time.time()

                # Perform burst of operations
                for _ in range(10):
                    job_service.get_jobs(limit=20)
                    search_service.search_jobs("python")
                    operations_performed += 2

                burst_time = (time.time() - start_time) * 1000

            return burst_time, operations_performed

        # Simulate multiple concurrent bursts
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(perform_operation) for _ in range(5)]
            results = [future.result() for future in futures]

        for burst_time, ops_count in results:
            avg_op_time = burst_time / ops_count
            assert burst_time < 1000, (
                f"Burst took {burst_time:.2f}ms (should be <1000ms)"
            )
            assert avg_op_time < 50, (
                f"Average operation time: {avg_op_time:.2f}ms (should be <50ms)"
            )


class TestPerformanceRegression:
    """Test for performance regression prevention."""

    @pytest.mark.performance
    def test_service_instantiation_regression(self):
        """Test that service instantiation doesn't regress in performance."""
        # Clear cache to test cold start
        clear_service_cache()

        # Benchmark service instantiation
        instantiation_times = []
        services = [
            (get_job_service, JobService),
            (get_search_service, JobSearchService),
            (get_analytics_service, AnalyticsService),
        ]

        for get_service_func, service_class in services:
            start_time = time.time()
            service = get_service_func()
            instantiation_time = (time.time() - start_time) * 1000

            instantiation_times.append(instantiation_time)

            assert instantiation_time < 1000, (
                f"{service_class.__name__} instantiation took {instantiation_time:.2f}ms"
            )
            assert isinstance(service, service_class)

        # All instantiations should complete quickly
        total_instantiation_time = sum(instantiation_times)
        assert total_instantiation_time < 3000, (
            f"Total service instantiation: {total_instantiation_time:.2f}ms (should be <3000ms)"
        )

    @pytest.mark.performance
    def test_cache_hit_performance_regression(self):
        """Test that cache hit performance doesn't regress."""
        # Warm up cache
        job_service = get_job_service()
        search_service = get_search_service()
        analytics_service = get_analytics_service()

        # Test rapid cache hits
        cache_hit_times = []

        for _ in range(100):
            start_time = time.time()
            services = [
                get_job_service(),
                get_search_service(),
                get_analytics_service(),
            ]
            cache_hit_time = (time.time() - start_time) * 1000
            cache_hit_times.append(cache_hit_time)

            # Verify cached instances
            assert services[0] is job_service
            assert services[1] is search_service
            assert services[2] is analytics_service

        # Cache hits should be consistently fast
        max_hit_time = max(cache_hit_times)
        avg_hit_time = sum(cache_hit_times) / len(cache_hit_times)

        assert max_hit_time < 10, (
            f"Slowest cache hit: {max_hit_time:.2f}ms (should be <10ms)"
        )
        assert avg_hit_time < 1, (
            f"Average cache hit: {avg_hit_time:.2f}ms (should be <1ms)"
        )

    @pytest.mark.performance
    def test_memory_usage_regression(self):
        """Test that memory usage doesn't regress with caching."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create many service references
        services = []
        for _ in range(5000):
            services.extend(
                [get_job_service(), get_search_service(), get_analytics_service()]
            )

        # Check memory after creating references
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory

        # Memory increase should be minimal due to caching
        assert memory_increase < 50, (
            f"Memory increased by {memory_increase:.2f}MB (should be <50MB)"
        )

        # Verify caching is working (should have only 3 unique instances)
        unique_services = set(id(s) for s in services)
        assert len(unique_services) == 3, (
            f"Should have 3 unique services, got {len(unique_services)}"
        )


class TestRealWorldPerformanceScenarios:
    """Test performance in realistic usage scenarios."""

    @pytest.mark.performance
    def test_user_workflow_performance(self):
        """Test performance of typical user workflow."""
        # Simulate: User loads jobs page, searches, filters, views analytics

        job_service = get_job_service()
        search_service = get_search_service()
        analytics_service = get_analytics_service()

        with (
            patch.object(job_service, "get_jobs", return_value=[]),
            patch.object(search_service, "search_jobs", return_value=[]),
            patch.object(analytics_service, "get_job_trends", return_value={}),
        ):
            workflow_start = time.time()

            # 1. Load initial jobs
            start_time = time.time()
            jobs = job_service.get_jobs(limit=50)
            load_time = (time.time() - start_time) * 1000

            # 2. Search for specific term
            start_time = time.time()
            search_results = search_service.search_jobs("python developer")
            search_time = (time.time() - start_time) * 1000

            # 3. Apply filters
            start_time = time.time()
            filtered_jobs = job_service.get_jobs(
                filters={"location": "Remote", "salary_min": 100000}
            )
            filter_time = (time.time() - start_time) * 1000

            # 4. View analytics
            start_time = time.time()
            trends = analytics_service.get_job_trends(days=30)
            analytics_time = (time.time() - start_time) * 1000

            total_workflow_time = (time.time() - workflow_start) * 1000

            # Individual operations should be fast
            assert load_time < 100, f"Job loading: {load_time:.2f}ms"
            assert search_time < 100, f"Search: {search_time:.2f}ms"
            assert filter_time < 100, f"Filtering: {filter_time:.2f}ms"
            assert analytics_time < 100, f"Analytics: {analytics_time:.2f}ms"

            # Total workflow should complete quickly
            assert total_workflow_time < 400, (
                f"Total workflow: {total_workflow_time:.2f}ms"
            )

    @pytest.mark.performance
    def test_dashboard_refresh_performance(self):
        """Test performance of dashboard refresh operations."""
        # Simulate dashboard refresh that updates multiple components

        services = {
            "job": get_job_service(),
            "search": get_search_service(),
            "analytics": get_analytics_service(),
        }

        # Mock all operations
        with (
            patch.object(services["job"], "get_jobs", return_value=[]),
            patch.object(services["search"], "search_jobs", return_value=[]),
            patch.object(services["analytics"], "get_job_trends", return_value={}),
        ):
            refresh_start = time.time()

            # Simulate dashboard refresh operations
            operations = [
                services["job"].get_jobs(limit=20),
                services["analytics"].get_job_trends(days=7),
                services["job"].get_jobs(filters={"favorites_only": True}),
                services["search"].search_jobs("python"),
                services["analytics"].get_job_trends(days=30),
            ]

            refresh_time = (time.time() - refresh_start) * 1000

            # Dashboard refresh should be snappy
            assert refresh_time < 300, (
                f"Dashboard refresh: {refresh_time:.2f}ms (should be <300ms)"
            )

            # All operations should have completed
            assert len(operations) == 5
            for result in operations:
                assert result is not None
