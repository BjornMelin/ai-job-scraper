"""Service caching validation tests.

This module tests @st.cache_resource for service objects,
performance optimization, and proper cache invalidation patterns
established in the clean foundation.
"""

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


class TestServiceCachingFunctionality:
    """Test @st.cache_resource functionality for services."""

    def test_job_service_caching(self):
        """Test that JobService is properly cached."""
        # Get service instances multiple times
        service1 = get_job_service()
        service2 = get_job_service()
        service3 = get_job_service()

        # All should be the same cached instance
        assert service1 is service2
        assert service2 is service3
        assert isinstance(service1, JobService)

    def test_search_service_caching(self):
        """Test that JobSearchService is properly cached."""
        service1 = get_search_service()
        service2 = get_search_service()
        service3 = get_search_service()

        # All should be the same cached instance
        assert service1 is service2
        assert service2 is service3
        assert isinstance(service1, JobSearchService)

    def test_analytics_service_caching(self):
        """Test that AnalyticsService is properly cached."""
        service1 = get_analytics_service()
        service2 = get_analytics_service()
        service3 = get_analytics_service()

        # All should be the same cached instance
        assert service1 is service2
        assert service2 is service3
        assert isinstance(service1, AnalyticsService)

    def test_different_services_are_separate(self):
        """Test that different service types are separate instances."""
        job_service = get_job_service()
        search_service = get_search_service()
        analytics_service = get_analytics_service()

        # Different service types should be different instances
        assert job_service is not search_service
        assert search_service is not analytics_service
        assert job_service is not analytics_service

        # But correct types
        assert isinstance(job_service, JobService)
        assert isinstance(search_service, JobSearchService)
        assert isinstance(analytics_service, AnalyticsService)


class TestCachePerformanceOptimization:
    """Test performance optimizations from service caching."""

    def test_service_initialization_performance(self):
        """Test that cached services improve initialization performance."""
        # First call (cache miss) - may be slower
        start_time = time.time()
        service1 = get_job_service()
        first_call_time = time.time() - start_time

        # Subsequent calls (cache hits) - should be faster
        start_time = time.time()
        service2 = get_job_service()
        second_call_time = time.time() - start_time

        start_time = time.time()
        service3 = get_job_service()
        third_call_time = time.time() - start_time

        # Cache hits should be significantly faster
        assert service1 is service2 is service3
        assert second_call_time <= first_call_time
        assert third_call_time <= first_call_time

        # Cache hits should be very fast
        assert second_call_time < 0.01  # 10ms
        assert third_call_time < 0.01  # 10ms

    def test_concurrent_service_access_performance(self):
        """Test performance of concurrent service access."""
        import concurrent.futures

        def get_services():
            return [get_job_service(), get_search_service(), get_analytics_service()]

        start_time = time.time()

        # Concurrent access to cached services
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_services) for _ in range(10)]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # Should complete quickly with caching
        assert total_time < 1.0  # 1 second for 10 concurrent operations
        assert len(results) == 10

        # All results should have the same cached instances
        first_result = results[0]
        for result in results[1:]:
            assert result[0] is first_result[0]  # Same JobService
            assert result[1] is first_result[1]  # Same SearchService
            assert result[2] is first_result[2]  # Same AnalyticsService

    def test_memory_efficiency_from_caching(self):
        """Test memory efficiency from service caching."""
        # Get multiple references to services
        services = []
        for _ in range(100):
            services.extend(
                [get_job_service(), get_search_service(), get_analytics_service()]
            )

        # Despite 300 references, should only have 3 actual service instances
        unique_job_services = set(
            id(s) for s in services[::3]
        )  # Every 3rd is JobService
        unique_search_services = set(
            id(s) for s in services[1::3]
        )  # Every 3rd+1 is SearchService
        unique_analytics_services = set(
            id(s) for s in services[2::3]
        )  # Every 3rd+2 is AnalyticsService

        # Should have exactly one instance of each service type
        assert len(unique_job_services) == 1
        assert len(unique_search_services) == 1
        assert len(unique_analytics_services) == 1


class TestCacheInvalidationPatterns:
    """Test cache invalidation and refresh patterns."""

    def test_service_cache_clearing(self):
        """Test that service cache can be cleared."""
        # Get initial services
        initial_job_service = get_job_service()
        initial_search_service = get_search_service()
        initial_analytics_service = get_analytics_service()

        # Clear cache
        clear_service_cache()

        # Get new services after cache clear
        new_job_service = get_job_service()
        new_search_service = get_search_service()
        new_analytics_service = get_analytics_service()

        # New services should be different instances (cache was cleared)
        assert new_job_service is not initial_job_service
        assert new_search_service is not initial_search_service
        assert new_analytics_service is not initial_analytics_service

        # But still correct types
        assert isinstance(new_job_service, JobService)
        assert isinstance(new_search_service, JobSearchService)
        assert isinstance(new_analytics_service, AnalyticsService)

    def test_cache_invalidation_on_configuration_change(self):
        """Test cache invalidation when configuration changes."""
        # This test simulates configuration changes that should invalidate cache

        # Get initial service
        initial_service = get_job_service()

        # Simulate configuration change by clearing cache
        clear_service_cache()

        # Get service after configuration change
        new_service = get_job_service()

        # Should be a new instance
        assert new_service is not initial_service
        assert isinstance(new_service, JobService)

    def test_selective_cache_operations(self):
        """Test that cache operations work correctly."""
        # Get all services to populate cache
        job_service1 = get_job_service()
        search_service1 = get_search_service()
        analytics_service1 = get_analytics_service()

        # Clear all caches
        clear_service_cache()

        # Get services again - should be new instances
        job_service2 = get_job_service()
        search_service2 = get_search_service()
        analytics_service2 = get_analytics_service()

        # All should be new instances
        assert job_service1 is not job_service2
        assert search_service1 is not search_service2
        assert analytics_service1 is not analytics_service2

        # But if we get them again without clearing, should be same
        job_service3 = get_job_service()
        search_service3 = get_search_service()
        analytics_service3 = get_analytics_service()

        assert job_service2 is job_service3
        assert search_service2 is search_service3
        assert analytics_service2 is analytics_service3


class TestCacheErrorHandling:
    """Test error handling in cached services."""

    def test_service_initialization_error_handling(self):
        """Test error handling during service initialization."""
        # Mock service initialization failure
        with patch(
            "src.services.job_service.JobService.__init__",
            side_effect=Exception("DB Connection failed"),
        ):
            # Service cache should handle initialization errors
            with pytest.raises(Exception):
                get_job_service()

    def test_cached_service_method_error_handling(self):
        """Test that cached services handle method call errors gracefully."""
        # Get a cached service
        job_service = get_job_service()

        # Mock a method to raise an error
        with patch.object(
            job_service, "get_jobs", side_effect=Exception("Database error")
        ):
            # Service methods should be callable but may raise expected errors
            with pytest.raises(Exception):
                job_service.get_jobs()

    def test_cache_corruption_recovery(self):
        """Test recovery from cache corruption."""
        # Get initial service
        service1 = get_job_service()
        assert isinstance(service1, JobService)

        # Clear cache to simulate corruption recovery
        clear_service_cache()

        # Should be able to get new service instance
        service2 = get_job_service()
        assert isinstance(service2, JobService)
        assert service1 is not service2  # New instance after cache clear


class TestCacheIntegrationPatterns:
    """Test cache integration with application patterns."""

    def test_service_cache_with_database_operations(self):
        """Test that cached services work with database operations."""
        # Get cached services
        job_service = get_job_service()
        search_service = get_search_service()

        # Services should be ready for database operations
        assert hasattr(job_service, "get_jobs")
        assert hasattr(search_service, "search_jobs")

        # Mock database operations to test caching doesn't interfere
        with patch.object(job_service, "get_jobs", return_value=[]) as mock_get_jobs:
            jobs = job_service.get_jobs(limit=10)
            mock_get_jobs.assert_called_once_with(limit=10)
            assert isinstance(jobs, list)

    def test_service_cache_thread_safety(self):
        """Test that service caching is thread-safe."""
        import threading

        services_collected = []

        def collect_services():
            services_collected.extend(
                [get_job_service(), get_search_service(), get_analytics_service()]
            )

        # Start multiple threads accessing cached services
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=collect_services)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have collected 30 service references (10 threads × 3 services)
        assert len(services_collected) == 30

        # All JobService references should be the same instance
        job_services = services_collected[::3]  # Every 3rd item
        assert all(s is job_services[0] for s in job_services)

        # All SearchService references should be the same instance
        search_services = services_collected[1::3]  # Every 3rd+1 item
        assert all(s is search_services[0] for s in search_services)

        # All AnalyticsService references should be the same instance
        analytics_services = services_collected[2::3]  # Every 3rd+2 item
        assert all(s is analytics_services[0] for s in analytics_services)

    def test_cache_with_streamlit_lifecycle(self):
        """Test cache behavior with Streamlit app lifecycle."""
        # This test simulates Streamlit app lifecycle events

        # Initial app load
        initial_services = [
            get_job_service(),
            get_search_service(),
            get_analytics_service(),
        ]

        # Simulate app rerun (services should remain cached)
        rerun_services = [
            get_job_service(),
            get_search_service(),
            get_analytics_service(),
        ]

        # Services should be the same instances across reruns
        for initial, rerun in zip(initial_services, rerun_services, strict=False):
            assert initial is rerun

        # Simulate app restart (clear cache)
        clear_service_cache()

        # New app session
        new_session_services = [
            get_job_service(),
            get_search_service(),
            get_analytics_service(),
        ]

        # Services should be new instances after restart
        for initial, new_session in zip(
            initial_services, new_session_services, strict=False
        ):
            assert initial is not new_session
            assert type(initial) is type(new_session)  # Same type, different instance


class TestCachePerformanceMetrics:
    """Test cache performance metrics and monitoring."""

    def test_cache_hit_rate_optimization(self):
        """Test that cache hit rate is optimized."""
        # Clear cache to start fresh
        clear_service_cache()

        # First calls (cache misses)
        start_time = time.time()
        service1 = get_job_service()
        first_call_time = time.time() - start_time

        # Subsequent calls (cache hits)
        hit_times = []
        for _ in range(10):
            start_time = time.time()
            service = get_job_service()
            hit_time = time.time() - start_time
            hit_times.append(hit_time)
            assert service is service1  # Same cached instance

        # Cache hits should be consistently fast
        avg_hit_time = sum(hit_times) / len(hit_times)
        assert avg_hit_time < 0.001  # Sub-millisecond cache hits

        # All hit times should be faster than first call
        for hit_time in hit_times:
            assert hit_time <= first_call_time

    def test_memory_usage_monitoring(self):
        """Test memory usage patterns with service caching."""
        import gc

        # Force garbage collection to get clean baseline
        gc.collect()

        # Get initial memory usage
        initial_objects = len(gc.get_objects())

        # Create many references to cached services
        service_references = []
        for _ in range(1000):
            service_references.extend(
                [get_job_service(), get_search_service(), get_analytics_service()]
            )

        # Force garbage collection
        gc.collect()

        # Check memory usage after creating references
        final_objects = len(gc.get_objects())

        # Object count should not increase dramatically due to caching
        # (Allow some increase for test overhead)
        object_increase = final_objects - initial_objects
        assert object_increase < 100, (
            f"Memory usage increased by {object_increase} objects"
        )

        # Verify we still have the same cached instances
        unique_job_services = set(id(ref) for ref in service_references[::3])
        unique_search_services = set(id(ref) for ref in service_references[1::3])
        unique_analytics_services = set(id(ref) for ref in service_references[2::3])

        assert len(unique_job_services) == 1
        assert len(unique_search_services) == 1
        assert len(unique_analytics_services) == 1
