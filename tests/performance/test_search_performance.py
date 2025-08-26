"""Performance benchmark tests for FTS5 search implementation.

This module provides performance benchmark tests to validate FTS5 search engine
performance characteristics:
- Search latency <10ms for typical queries on small datasets
- Search latency <50ms for typical queries on large datasets (1000+ jobs)
- Index creation and maintenance performance benchmarks
- Concurrent search handling under load
- Memory usage validation for sustained operations

Tests validate both performance thresholds and result correctness.
"""

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None

from tests.fixtures.search_fixtures import SearchTestUtils


class TestSearchPerformanceBenchmarks:
    """Performance benchmark tests for search functionality."""

    @pytest.mark.performance
    def test_basic_search_performance_small_dataset(self, search_test_database):
        """Test search performance on small dataset (8 jobs)."""
        # Warm up the database
        search_test_database.search_jobs("python")

        # Measure search performance
        queries = ["python", "developer", "machine learning", "remote"]

        for query in queries:
            start_time = time.perf_counter()
            results = search_test_database.search_jobs(query)
            end_time = time.perf_counter()

            search_time_ms = (end_time - start_time) * 1000

            # Should complete well under 10ms for small dataset
            assert search_time_ms < 10.0, (
                f"Search for '{query}' took {search_time_ms:.2f}ms, should be <10ms"
            )

            # Results should be valid
            SearchTestUtils.assert_search_results_valid(results)

    @pytest.mark.performance
    def test_large_dataset_search_performance(self, performance_test_database):
        """Test search performance on large dataset (1000 jobs)."""
        # Warm up
        performance_test_database.search_jobs("developer")

        # Test various query types
        test_queries = [
            "python",  # Simple term
            "developer",  # Common term (high frequency)
            '"python developer"',  # Phrase query
            "python AND django",  # Boolean query
        ]

        for query in test_queries:
            # Measure multiple runs for consistency
            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                results = performance_test_database.search_jobs(query)
                end_time = time.perf_counter()

                search_time_ms = (end_time - start_time) * 1000
                times.append(search_time_ms)

                # Validate results format
                SearchTestUtils.assert_search_results_valid(results)

            # Average time should be reasonable for large dataset
            avg_time = sum(times) / len(times)
            assert avg_time < 50.0, (
                f"Average search time for '{query}' was {avg_time:.2f}ms, "
                f"should be <50ms"
            )

            # No individual search should take too long
            max_time = max(times)
            assert max_time < 100.0, (
                f"Slowest search for '{query}' took {max_time:.2f}ms, should be <100ms"
            )

    @pytest.mark.performance
    def test_concurrent_search_performance(self, performance_test_database):
        """Test search performance under concurrent load."""
        queries = [
            "python developer",
            "machine learning",
            "data scientist",
            "full stack",
            "devops engineer",
        ]

        def search_worker(query: str) -> tuple[str, float, int]:
            """Worker function for concurrent search testing."""
            start_time = time.perf_counter()
            results = performance_test_database.search_jobs(query)
            end_time = time.perf_counter()

            search_time_ms = (end_time - start_time) * 1000
            return query, search_time_ms, len(results)

        # Run concurrent searches
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit multiple searches per query
            futures = []
            for _ in range(3):  # 3 rounds
                for query in queries:
                    future = executor.submit(search_worker, query)
                    futures.append(future)

            # Collect results
            results = []
            for future in as_completed(futures):
                query, search_time, result_count = future.result()
                results.append((query, search_time, result_count))

        # Analyze concurrent performance
        search_times = [time_ms for _, time_ms, _ in results]
        avg_concurrent_time = sum(search_times) / len(search_times)
        max_concurrent_time = max(search_times)

        # Concurrent searches should still be reasonably fast
        assert avg_concurrent_time < 100.0, (
            f"Average concurrent search time {avg_concurrent_time:.2f}ms, "
            "should be <100ms"
        )
        assert max_concurrent_time < 200.0, (
            f"Slowest concurrent search {max_concurrent_time:.2f}ms, should be <200ms"
        )

        # All searches should return results
        result_counts = [count for _, _, count in results]
        assert all(count >= 0 for count in result_counts), (
            "All searches should return valid results"
        )

    @pytest.mark.performance
    def test_filter_performance_impact(self, performance_test_database):
        """Test performance impact of various filter combinations."""
        base_query = "python developer"

        # Test different filter combinations
        filter_scenarios = [
            {},  # No filters
            {"salary_min": 100000},  # Simple filter
            {"location": "Remote"},  # Text filter
            {"favorites_only": True},  # Boolean filter
            {"application_status": ["New", "Applied"]},  # List filter
            {  # Complex filter combination
                "salary_min": 80000,
                "salary_max": 150000,
                "location": "Remote",
                "favorites_only": True,
            },
        ]

        performance_results = []

        for filters in filter_scenarios:
            # Measure multiple runs
            times = []
            for _ in range(3):
                start_time = time.perf_counter()
                performance_test_database.search_jobs(base_query, filters)
                end_time = time.perf_counter()

                search_time_ms = (end_time - start_time) * 1000
                times.append(search_time_ms)

            avg_time = sum(times) / len(times)
            filter_count = len(filters)
            performance_results.append((filter_count, avg_time))

        # Performance should not degrade significantly with filters
        base_performance = performance_results[0][1]  # No filters

        for filter_count, avg_time in performance_results[1:]:
            performance_ratio = avg_time / base_performance

            # Filters should not cause more than 3x performance degradation
            assert performance_ratio < 3.0, (
                f"Filter performance degradation {performance_ratio:.2f}x too high "
                f"for {filter_count} filters"
            )

    @pytest.mark.performance
    def test_index_rebuild_performance(self, performance_test_database):
        """Test FTS5 index rebuild performance."""
        start_time = time.perf_counter()
        result = performance_test_database.rebuild_search_index()
        end_time = time.perf_counter()

        rebuild_time_ms = (end_time - start_time) * 1000

        if result:  # Only test if FTS5 is available
            # Index rebuild should complete in reasonable time for 1000 jobs
            assert rebuild_time_ms < 5000.0, (
                f"Index rebuild took {rebuild_time_ms:.2f}ms, should be <5000ms"
            )

            # Verify search still works after rebuild
            results = performance_test_database.search_jobs("python")
            SearchTestUtils.assert_search_results_valid(results)

    @pytest.mark.performance
    def test_search_result_limits_performance(self, performance_test_database):
        """Test performance with different result limits."""
        query = "developer"  # Common term that returns many results

        limits = [10, 25, 50, 100]

        for limit in limits:
            start_time = time.perf_counter()
            results = performance_test_database.search_jobs(query, limit=limit)
            end_time = time.perf_counter()

            search_time_ms = (end_time - start_time) * 1000

            # Performance should not degrade significantly with higher limits
            assert search_time_ms < 100.0, (
                f"Search with limit {limit} took {search_time_ms:.2f}ms, "
                "should be <100ms"
            )

            # Should respect the limit
            assert len(results) <= limit, f"Results exceed limit of {limit}"

    @pytest.mark.performance
    def test_search_statistics_performance(self, performance_test_database):
        """Test performance of search statistics retrieval."""
        start_time = time.perf_counter()
        stats = performance_test_database.get_search_stats()
        end_time = time.perf_counter()

        stats_time_ms = (end_time - start_time) * 1000

        # Statistics retrieval should be very fast
        assert stats_time_ms < 50.0, (
            f"Search stats retrieval took {stats_time_ms:.2f}ms, should be <50ms"
        )

        # Verify stats format
        required_keys = ["fts_enabled", "indexed_jobs", "total_jobs", "index_coverage"]
        for key in required_keys:
            assert key in stats, f"Missing required stat: {key}"

    @pytest.mark.performance
    def test_fallback_search_performance(self, performance_test_database):
        """Test fallback search performance when FTS5 is disabled."""
        # Force fallback mode
        performance_test_database._fts_enabled = False

        # Test fallback performance
        queries = ["python", "developer", "data scientist"]

        for query in queries:
            start_time = time.perf_counter()
            results = performance_test_database.search_jobs(query)
            end_time = time.perf_counter()

            search_time_ms = (end_time - start_time) * 1000

            # Fallback should still be reasonably fast
            assert search_time_ms < 200.0, (
                f"Fallback search for '{query}' took {search_time_ms:.2f}ms, "
                f"should be <200ms"
            )

            # Results should still be valid
            SearchTestUtils.assert_search_results_valid(results)


class TestSearchMemoryUsage:
    """Memory usage tests for search functionality."""

    @pytest.mark.performance
    def test_search_memory_efficiency(self, performance_test_database):
        """Test that search operations don't cause memory leaks."""
        if not _HAS_PSUTIL:
            pytest.skip("psutil is not installed; skipping memory efficiency test.")

        import gc
        import os

        process = psutil.Process(os.getpid())

        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss

        # Perform many search operations
        queries = ["python", "developer", "machine learning", "remote", "senior"]
        for _ in range(100):  # 100 iterations
            for query in queries:
                results = performance_test_database.search_jobs(query)
                # Process results to ensure they're loaded
                for result in results:
                    _ = result["title"]

        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss

        # Memory growth should be minimal (less than 50MB)
        memory_growth = (final_memory - baseline_memory) / 1024 / 1024  # MB
        assert memory_growth < 50.0, (
            f"Memory grew by {memory_growth:.1f}MB during search operations, "
            f"should be <50MB"
        )

    @pytest.mark.performance
    def test_large_result_set_memory(self, performance_test_database):
        """Test memory usage with large result sets."""
        # Search for very common term to get many results
        results = performance_test_database.search_jobs(
            "", limit=500
        )  # Empty query to get many results

        # Should handle large result sets without issues
        assert isinstance(results, list)

        # Process all results to ensure memory is allocated
        total_chars = sum(len(str(result)) for result in results)
        assert total_chars > 0, "Results should contain data"


class TestSearchStressTest:
    """Stress tests for search functionality."""

    @pytest.mark.performance
    @pytest.mark.slow  # Mark as slow test
    def test_sustained_search_load(self, performance_test_database):
        """Test search performance under sustained load."""
        queries = [
            "python developer",
            "machine learning",
            "data scientist",
            "full stack",
            "backend",
            "frontend",
            "devops",
            "senior engineer",
            "junior developer",
            "remote work",
        ]

        total_searches = 0
        total_time = 0
        max_time = 0

        # Run searches for 30 seconds or 1000 searches, whichever comes first
        start_test_time = time.perf_counter()

        while total_searches < 1000:
            current_time = time.perf_counter()
            if current_time - start_test_time > 30:  # 30 second limit
                break

            query = queries[total_searches % len(queries)]

            start_time = time.perf_counter()
            results = performance_test_database.search_jobs(query)
            end_time = time.perf_counter()

            search_time = (end_time - start_time) * 1000
            total_time += search_time
            max_time = max(max_time, search_time)
            total_searches += 1

            # Verify results are still valid
            SearchTestUtils.assert_search_results_valid(results)

        # Calculate performance metrics
        avg_time = total_time / total_searches
        searches_per_second = total_searches / (time.perf_counter() - start_test_time)

        # Performance should remain stable
        assert avg_time < 100.0, (
            f"Average search time {avg_time:.2f}ms degraded under load, "
            "should be <100ms"
        )
        assert max_time < 500.0, (
            f"Maximum search time {max_time:.2f}ms too high, should be <500ms"
        )
        assert searches_per_second > 10.0, (
            f"Search throughput {searches_per_second:.1f}/sec too low, "
            "should be >10/sec"
        )

    @pytest.mark.performance
    def test_rapid_filter_changes(self, performance_test_database):
        """Test performance when filters change rapidly."""
        base_query = "python"

        # Simulate rapid filter changes (like user typing/changing filters)
        filter_combinations = [
            {},
            {"salary_min": 80000},
            {"salary_min": 80000, "location": "Remote"},
            {"salary_min": 80000, "location": "Remote", "favorites_only": True},
            {"salary_min": 90000, "location": "Remote", "favorites_only": True},
            {"salary_min": 90000, "location": "San Francisco", "favorites_only": True},
            {"salary_min": 100000, "location": "San Francisco"},
            {"salary_min": 100000},
            {},
        ]

        times = []
        for filters in filter_combinations:
            start_time = time.perf_counter()
            results = performance_test_database.search_jobs(base_query, filters)
            end_time = time.perf_counter()

            search_time_ms = (end_time - start_time) * 1000
            times.append(search_time_ms)

            # Verify results
            SearchTestUtils.assert_search_results_valid(results)

        # All searches should complete quickly
        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 75.0, (
            f"Average time for rapid filter changes {avg_time:.2f}ms, should be <75ms"
        )
        assert max_time < 150.0, (
            f"Maximum time for rapid filter changes {max_time:.2f}ms, should be <150ms"
        )


# Performance test configuration and utilities
class PerformanceTestConfig:
    """Configuration for performance tests."""

    # Performance thresholds (in milliseconds)
    SMALL_DATASET_THRESHOLD = 10.0  # <10ms for small datasets
    LARGE_DATASET_THRESHOLD = 50.0  # <50ms for large datasets
    CONCURRENT_THRESHOLD = 100.0  # <100ms average for concurrent searches
    FILTER_THRESHOLD = 75.0  # <75ms for filtered searches
    INDEX_REBUILD_THRESHOLD = 5000.0  # <5s for index rebuild
    MEMORY_GROWTH_THRESHOLD = 50.0  # <50MB memory growth

    @classmethod
    def validate_performance(cls, operation: str, duration_ms: float) -> None:
        """Validate performance against thresholds."""
        thresholds = {
            "small_dataset": cls.SMALL_DATASET_THRESHOLD,
            "large_dataset": cls.LARGE_DATASET_THRESHOLD,
            "concurrent": cls.CONCURRENT_THRESHOLD,
            "filter": cls.FILTER_THRESHOLD,
            "index_rebuild": cls.INDEX_REBUILD_THRESHOLD,
        }

        threshold = thresholds.get(operation, 100.0)
        assert duration_ms < threshold, (
            f"{operation} took {duration_ms:.2f}ms, should be <{threshold}ms"
        )
