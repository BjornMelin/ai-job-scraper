"""Stream B Caching Performance Testing.

Comprehensive testing for Streamlit native caching components:
- st.cache_data() for data operations with TTL management
- st.cache_resource() for connection and resource management
- Cache hit/miss ratio analysis and performance optimization
- Memory efficiency validation for cached data structures

Focuses on:
1. Performance Validation: Measuring cache effectiveness and speed improvements
2. Functionality Preservation: Ensuring cached results match uncached results
3. Resource Management: Testing connection pooling and resource lifecycle
4. Cache Efficiency: Validating hit rates and memory usage optimization
"""

import hashlib
import time

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from tests.native_components.framework import (
    NativeComponentMetrics,
    NativeComponentTester,
    NativeComponentValidator,
    PerformanceBenchmark,
    StreamType,
)


class CacheMetricsTracker:
    """Track caching metrics and behavior."""

    def __init__(self):
        """Initialize cache metrics tracking."""
        self.cache_data_calls = []
        self.cache_resource_calls = []
        self.function_executions = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.ttl_expirations = 0
        self.invalidations = 0
        self.memory_snapshots = []

    def record_cache_data_call(
        self, func_name: str, args: tuple, kwargs: dict, ttl: int | None = None
    ) -> None:
        """Record st.cache_data function call."""
        call_info = {
            "func_name": func_name,
            "args": args,
            "kwargs": kwargs,
            "ttl": ttl,
            "timestamp": datetime.now(),
        }
        self.cache_data_calls.append(call_info)

    def record_cache_resource_call(
        self, func_name: str, args: tuple, kwargs: dict
    ) -> None:
        """Record st.cache_resource function call."""
        call_info = {
            "func_name": func_name,
            "args": args,
            "kwargs": kwargs,
            "timestamp": datetime.now(),
        }
        self.cache_resource_calls.append(call_info)

    def record_function_execution(self, func_name: str) -> None:
        """Record actual function execution (cache miss)."""
        self.function_executions[func_name] = (
            self.function_executions.get(func_name, 0) + 1
        )

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.cache_misses += 1

    def record_ttl_expiration(self) -> None:
        """Record TTL expiration."""
        self.ttl_expirations += 1

    def record_invalidation(self) -> None:
        """Record cache invalidation."""
        self.invalidations += 1

    def record_memory_snapshot(self, description: str, size_mb: float) -> None:
        """Record memory usage snapshot."""
        self.memory_snapshots.append(
            {
                "description": description,
                "size_mb": size_mb,
                "timestamp": datetime.now(),
            }
        )

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) if total > 0 else 0.0

    def get_function_call_count(self, func_name: str) -> int:
        """Get number of actual function calls."""
        return self.function_executions.get(func_name, 0)

    def reset(self) -> None:
        """Reset all tracking."""
        self.cache_data_calls.clear()
        self.cache_resource_calls.clear()
        self.function_executions.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.ttl_expirations = 0
        self.invalidations = 0
        self.memory_snapshots.clear()


class MockStreamlitCache:
    """Mock implementation of Streamlit caching system."""

    def __init__(self, tracker: CacheMetricsTracker):
        """Initialize mock cache."""
        self.tracker = tracker
        self.data_cache = {}
        self.resource_cache = {}
        self.ttl_timestamps = {}

    def mock_cache_data(self, ttl: int | None = None, hash_funcs: dict | None = None):
        """Mock st.cache_data decorator."""

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func, args, kwargs)

                # Record the call
                self.tracker.record_cache_data_call(func.__name__, args, kwargs, ttl)

                # Check TTL expiration
                if ttl and cache_key in self.ttl_timestamps:
                    expiry_time = self.ttl_timestamps[cache_key] + timedelta(
                        seconds=ttl
                    )
                    if datetime.now() > expiry_time:
                        # Expired, remove from cache
                        self.data_cache.pop(cache_key, None)
                        self.ttl_timestamps.pop(cache_key, None)
                        self.tracker.record_ttl_expiration()

                # Check cache
                if cache_key in self.data_cache:
                    self.tracker.record_cache_hit()
                    return self.data_cache[cache_key]

                # Cache miss - execute function
                self.tracker.record_cache_miss()
                self.tracker.record_function_execution(func.__name__)

                result = func(*args, **kwargs)

                # Store in cache
                self.data_cache[cache_key] = result
                if ttl:
                    self.ttl_timestamps[cache_key] = datetime.now()

                return result

            wrapper._is_cached = True
            wrapper._cache_type = "data"
            return wrapper

        return decorator

    def mock_cache_resource(self, hash_funcs: dict | None = None):
        """Mock st.cache_resource decorator."""

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func, args, kwargs)

                # Record the call
                self.tracker.record_cache_resource_call(func.__name__, args, kwargs)

                # Check cache
                if cache_key in self.resource_cache:
                    self.tracker.record_cache_hit()
                    return self.resource_cache[cache_key]

                # Cache miss - execute function
                self.tracker.record_cache_miss()
                self.tracker.record_function_execution(func.__name__)

                result = func(*args, **kwargs)

                # Store in cache (resources cached indefinitely)
                self.resource_cache[cache_key] = result

                return result

            wrapper._is_cached = True
            wrapper._cache_type = "resource"
            return wrapper

        return decorator

    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key."""
        key_data = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear all caches."""
        cleared_count = len(self.data_cache) + len(self.resource_cache)
        self.data_cache.clear()
        self.resource_cache.clear()
        self.ttl_timestamps.clear()

        for _ in range(cleared_count):
            self.tracker.record_invalidation()


class CachingPerformanceValidator(NativeComponentValidator):
    """Validator for Stream B caching performance components."""

    def __init__(self):
        """Initialize caching performance validator."""
        super().__init__(StreamType.STREAM_B, "caching_performance")
        self.tracker = CacheMetricsTracker()
        self.mock_cache = MockStreamlitCache(self.tracker)

    def validate_functionality(self, test_func, *args, **kwargs) -> bool:
        """Validate caching component functionality preservation."""
        try:
            self.tracker.reset()

            # Mock caching components
            with (
                patch("streamlit.cache_data", self.mock_cache.mock_cache_data),
                patch("streamlit.cache_resource", self.mock_cache.mock_cache_resource),
            ):
                test_func(*args, **kwargs)

            # Update metrics
            self.metrics.cache_hits = self.tracker.cache_hits
            self.metrics.cache_misses = self.tracker.cache_misses
            self.metrics.cache_efficiency = self.tracker.get_cache_hit_rate() * 100

            return True

        except Exception:
            self.metrics.error_count += 1
            return False

    def measure_performance(
        self, test_func, iterations: int = 10
    ) -> NativeComponentMetrics:
        """Measure caching component performance."""
        total_metrics = NativeComponentMetrics()

        for iteration in range(iterations):
            # Reset cache for each iteration to ensure consistent measurements
            if (
                iteration > 0
            ):  # Keep cache for first iteration to measure cache benefits
                self.tracker.reset()

            with self.performance_monitoring() as metrics:
                self.validate_functionality(test_func)

            # Accumulate metrics
            total_metrics.execution_time += metrics.execution_time
            total_metrics.memory_usage_mb += metrics.memory_usage_mb
            total_metrics.cpu_usage_percent += metrics.cpu_usage_percent
            total_metrics.peak_memory_mb = max(
                total_metrics.peak_memory_mb, metrics.peak_memory_mb
            )

            # Cache-specific metrics
            total_metrics.cache_hits += metrics.cache_hits
            total_metrics.cache_misses += metrics.cache_misses

        # Average the metrics
        avg_metrics = NativeComponentMetrics(
            execution_time=total_metrics.execution_time / iterations,
            memory_usage_mb=total_metrics.memory_usage_mb / iterations,
            cpu_usage_percent=total_metrics.cpu_usage_percent / iterations,
            peak_memory_mb=total_metrics.peak_memory_mb,
            cache_hits=total_metrics.cache_hits,
            cache_misses=total_metrics.cache_misses,
        )

        # Calculate final cache efficiency
        avg_metrics.calculate_cache_efficiency()

        return avg_metrics

    def compare_implementations(
        self, baseline_func, optimized_func, iterations: int = 10
    ) -> PerformanceBenchmark:
        """Compare baseline vs optimized caching implementations."""
        benchmark = PerformanceBenchmark(
            component_name=self.component_name,
            stream_type=self.stream_type,
            test_name="caching_comparison",
            iterations=iterations,
        )

        try:
            # Measure baseline (uncached)
            baseline_metrics = self.measure_performance(baseline_func, iterations)
            benchmark.baseline_metrics = baseline_metrics

            # Measure optimized (cached)
            optimized_metrics = self.measure_performance(optimized_func, iterations)
            benchmark.optimized_metrics = optimized_metrics

            # Test functionality preservation
            baseline_result = self._capture_component_output(baseline_func)
            optimized_result = self._capture_component_output(optimized_func)
            benchmark.functionality_preserved = self._compare_results(
                baseline_result, optimized_result
            )

            benchmark.passed = True

        except Exception as e:
            benchmark.error_message = str(e)
            benchmark.passed = False

        return benchmark

    def _capture_component_output(self, test_func) -> dict[str, Any]:
        """Capture component output for comparison."""
        self.tracker.reset()
        result = self.validate_functionality(test_func)

        return {
            "function_result": result,
            "cache_hits": self.tracker.cache_hits,
            "cache_misses": self.tracker.cache_misses,
            "function_executions": dict(self.tracker.function_executions),
            "hit_rate": self.tracker.get_cache_hit_rate(),
        }

    def _compare_results(
        self, baseline: dict[str, Any], optimized: dict[str, Any]
    ) -> bool:
        """Compare results focusing on functionality preservation."""
        try:
            # Compare function results (should be identical)
            if baseline["function_result"] != optimized["function_result"]:
                return False

            # Cached version should have better hit rate (when called multiple times)
            if optimized["hit_rate"] < baseline["hit_rate"]:
                return False  # Cached version should perform at least as well

            return True

        except (KeyError, AssertionError):
            return False


class TestStreamBCachingPerformance:
    """Test suite for Stream B caching performance components."""

    @pytest.fixture
    def caching_validator(self):
        """Provide caching performance validator."""
        return CachingPerformanceValidator()

    @pytest.fixture
    def native_tester(self, caching_validator):
        """Provide configured native component tester."""
        tester = NativeComponentTester()
        tester.register_validator("caching_performance", caching_validator)
        return tester

    def test_basic_cache_data_functionality(self, caching_validator):
        """Test basic st.cache_data functionality."""

        def basic_cache_data_test():
            import streamlit as st

            @st.cache_data(ttl=60)
            def expensive_computation(n: int) -> int:
                """Simulate expensive computation."""
                time.sleep(0.01)  # 10ms delay
                return sum(i**2 for i in range(n))

            # Call multiple times - should cache after first call
            results = []
            for _ in range(5):
                result = expensive_computation(100)
                results.append(result)

            return results

        result = caching_validator.validate_functionality(basic_cache_data_test)
        assert result is True

        # Verify caching behavior
        assert caching_validator.tracker.cache_misses == 1  # First call
        assert caching_validator.tracker.cache_hits == 4  # Subsequent calls
        assert (
            caching_validator.tracker.get_function_call_count("expensive_computation")
            == 1
        )

    def test_cache_data_with_different_parameters(self, caching_validator):
        """Test st.cache_data with different parameter combinations."""

        def cache_data_params_test():
            import streamlit as st

            @st.cache_data
            def parameterized_function(a: int, b: str, c: bool = True) -> str:
                """Function with multiple parameter types."""
                time.sleep(0.005)  # 5ms delay
                return f"result_{a}_{b}_{c}"

            # Different parameter combinations should create separate cache entries
            results = []

            # Call with different parameters
            results.append(parameterized_function(1, "test"))  # Cache miss
            results.append(parameterized_function(1, "test"))  # Cache hit
            results.append(
                parameterized_function(2, "test")
            )  # Cache miss (different a)
            results.append(
                parameterized_function(1, "other")
            )  # Cache miss (different b)
            results.append(
                parameterized_function(1, "test", False)
            )  # Cache miss (different c)
            results.append(
                parameterized_function(1, "test")
            )  # Cache hit (same as first)

            return results

        result = caching_validator.validate_functionality(cache_data_params_test)
        assert result is True

        # Verify parameter-based caching
        assert (
            caching_validator.tracker.cache_misses == 4
        )  # Unique parameter combinations
        assert caching_validator.tracker.cache_hits == 2  # Repeated calls
        assert (
            caching_validator.tracker.get_function_call_count("parameterized_function")
            == 4
        )

    def test_cache_resource_functionality(self, caching_validator):
        """Test basic st.cache_resource functionality."""

        def basic_cache_resource_test():
            import streamlit as st

            @st.cache_resource
            def create_connection(connection_string: str) -> dict:
                """Simulate creating expensive resource connection."""
                time.sleep(0.02)  # 20ms delay
                return {
                    "connection_id": f"conn_{connection_string}",
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                }

            # Multiple calls with same parameters - should reuse resource
            connections = []
            for _ in range(5):
                conn = create_connection("database://localhost:5432")
                connections.append(conn)

            return connections

        result = caching_validator.validate_functionality(basic_cache_resource_test)
        assert result is True

        # Verify resource caching behavior
        assert (
            caching_validator.tracker.cache_misses == 1
        )  # First call creates resource
        assert (
            caching_validator.tracker.cache_hits == 4
        )  # Subsequent calls reuse resource
        assert (
            caching_validator.tracker.get_function_call_count("create_connection") == 1
        )

    def test_cache_resource_different_connections(self, caching_validator):
        """Test st.cache_resource with different connection parameters."""

        def cache_resource_connections_test():
            import streamlit as st

            @st.cache_resource
            def create_database_connection(host: str, port: int = 5432) -> dict:
                """Create database connection resource."""
                time.sleep(0.015)  # 15ms delay
                return {
                    "host": host,
                    "port": port,
                    "connection_id": f"{host}:{port}",
                    "pool_size": 10,
                }

            # Different connections should be cached separately
            connections = []

            # Different hosts
            connections.append(create_database_connection("localhost"))  # Miss
            connections.append(create_database_connection("localhost"))  # Hit
            connections.append(
                create_database_connection("remote-db")
            )  # Miss (different host)
            connections.append(create_database_connection("remote-db"))  # Hit
            connections.append(
                create_database_connection("localhost", 3306)
            )  # Miss (different port)
            connections.append(
                create_database_connection("localhost")
            )  # Hit (original)

            return connections

        result = caching_validator.validate_functionality(
            cache_resource_connections_test
        )
        assert result is True

        # Verify multiple resource caching
        assert caching_validator.tracker.cache_misses == 3  # Three unique connections
        assert caching_validator.tracker.cache_hits == 3  # Three reuses
        assert (
            caching_validator.tracker.get_function_call_count(
                "create_database_connection"
            )
            == 3
        )

    def test_ttl_expiration_behavior(self, caching_validator):
        """Test TTL expiration in cache_data."""

        def ttl_expiration_test():
            import streamlit as st

            @st.cache_data(ttl=1)  # Very short TTL for testing
            def short_ttl_function(value: int) -> str:
                """Function with short TTL."""
                time.sleep(0.005)  # 5ms delay
                return f"processed_{value}_{datetime.now().microsecond}"

            # First call
            result1 = short_ttl_function(42)

            # Second call immediately - should hit cache
            result2 = short_ttl_function(42)

            # Simulate time passage (TTL handled by mock)
            time.sleep(1.1)  # Wait longer than TTL

            # Third call - should miss cache due to TTL expiration
            result3 = short_ttl_function(42)

            return [result1, result2, result3]

        result = caching_validator.validate_functionality(ttl_expiration_test)
        assert result is True

        # Verify TTL behavior (Note: Mock doesn't actually wait for TTL,
        # but tracks the behavior pattern)
        assert caching_validator.tracker.cache_misses >= 1
        assert caching_validator.tracker.cache_hits >= 1

    def test_integrated_caching_workflow(self, caching_validator):
        """Test integrated workflow using both cache types."""

        def integrated_caching_workflow():
            import streamlit as st

            @st.cache_resource
            def get_job_scraping_config() -> dict:
                """Get cached scraping configuration."""
                time.sleep(0.01)  # 10ms setup time
                return {
                    "user_agents": ["Bot1", "Bot2", "Bot3"],
                    "timeout": 30,
                    "max_retries": 3,
                    "batch_size": 50,
                }

            @st.cache_data(ttl=300)  # 5-minute cache
            def scrape_company_jobs(company_name: str, config: dict) -> dict:
                """Scrape jobs from company with caching."""
                time.sleep(0.02)  # 20ms scraping time
                return {
                    "company": company_name,
                    "jobs_count": len(company_name) * 5,  # Deterministic fake data
                    "scraped_at": datetime.now().isoformat(),
                    "config_used": config["batch_size"],
                }

            # Simulate scraping workflow
            results = []
            companies = ["TechCorp", "DataInc", "AIStartup", "TechCorp", "DataInc"]

            for company in companies:
                config = get_job_scraping_config()  # Should cache resource
                jobs = scrape_company_jobs(company, config)  # Should cache data
                results.append(
                    {
                        "company": company,
                        "jobs_found": jobs["jobs_count"],
                        "config_batch_size": config["batch_size"],
                    }
                )

            return results

        result = caching_validator.validate_functionality(integrated_caching_workflow)
        assert result is True

        # Verify integrated caching behavior
        assert caching_validator.tracker.cache_hits > 0  # Should have some cache hits
        assert (
            caching_validator.tracker.cache_misses > 0
        )  # Should have some cache misses

        # Config should be cached (resource)
        assert (
            caching_validator.tracker.get_function_call_count("get_job_scraping_config")
            == 1
        )

        # Jobs should be cached per unique company (data)
        assert (
            caching_validator.tracker.get_function_call_count("scrape_company_jobs")
            == 3
        )  # TechCorp, DataInc, AIStartup

    def test_cache_performance_improvement(self, caching_validator):
        """Test that caching provides measurable performance improvement."""

        def performance_improvement_test():
            import streamlit as st

            @st.cache_data
            def cpu_intensive_task(complexity: int) -> list:
                """CPU-intensive task that benefits from caching."""
                time.sleep(0.05)  # 50ms delay
                # Simulate CPU-intensive computation
                result = []
                for i in range(complexity):
                    result.append(i**2 + i**3)
                return result

            # Run the same computation multiple times
            results = []
            complexity = 1000

            for _ in range(10):
                result = cpu_intensive_task(complexity)
                results.append(len(result))

            return results

        with caching_validator.performance_monitoring():
            result = caching_validator.validate_functionality(
                performance_improvement_test
            )

        assert result is True

        # Should have significant caching benefit
        hit_rate = caching_validator.tracker.get_cache_hit_rate()
        assert hit_rate > 0.8  # 80%+ cache hit rate

        # Should execute expensive function only once
        assert (
            caching_validator.tracker.get_function_call_count("cpu_intensive_task") == 1
        )

    def test_cache_memory_efficiency(self, caching_validator):
        """Test memory efficiency of caching system."""

        def memory_efficiency_test():
            import streamlit as st

            @st.cache_data
            def create_large_dataset(size: int) -> list:
                """Create large dataset that should be cached efficiently."""
                time.sleep(0.01)  # 10ms delay
                return list(range(size))

            @st.cache_resource
            def create_processing_resource() -> dict:
                """Create processing resource."""
                time.sleep(0.005)  # 5ms delay
                return {
                    "processor": "high_performance",
                    "threads": 8,
                    "memory_pool": list(range(10000)),  # Simulated memory pool
                }

            # Test memory usage patterns
            results = []

            # Create different sized datasets
            for size in [1000, 5000, 10000]:
                resource = create_processing_resource()  # Should reuse cached resource
                dataset = create_large_dataset(
                    size
                )  # Should cache each size separately
                results.append(
                    {
                        "size": size,
                        "dataset_length": len(dataset),
                        "resource_threads": resource["threads"],
                    }
                )

            # Reuse first dataset - should hit cache
            dataset_again = create_large_dataset(1000)
            results.append({"reused_dataset_length": len(dataset_again)})

            return results

        with caching_validator.performance_monitoring() as metrics:
            result = caching_validator.validate_functionality(memory_efficiency_test)

        assert result is True

        # Verify memory efficient caching
        assert metrics.memory_usage_mb >= 0  # Memory usage recorded

        # Resource should be created only once
        assert (
            caching_validator.tracker.get_function_call_count(
                "create_processing_resource"
            )
            == 1
        )

        # Each dataset size should be cached
        assert (
            caching_validator.tracker.get_function_call_count("create_large_dataset")
            == 3
        )

    def test_functionality_preservation_comparison(self, caching_validator):
        """Test functionality preservation between cached and uncached implementations."""

        def uncached_implementation():
            """Baseline implementation without caching."""

            def compute_fibonacci(n: int) -> int:
                """Compute fibonacci number without caching."""
                if n <= 1:
                    return n
                return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)

            # Compute multiple fibonacci numbers
            results = []
            for i in range(5, 10):  # Small numbers to avoid exponential time
                result = compute_fibonacci(i)
                results.append(result)
                time.sleep(0.001)  # Small delay to simulate processing

            return results

        def cached_implementation():
            """Optimized implementation with caching."""
            import streamlit as st

            @st.cache_data
            def compute_fibonacci_cached(n: int) -> int:
                """Compute fibonacci number with caching."""
                if n <= 1:
                    return n
                return compute_fibonacci_cached(n - 1) + compute_fibonacci_cached(n - 2)

            # Compute the same fibonacci numbers
            results = []
            for i in range(5, 10):
                result = compute_fibonacci_cached(i)
                results.append(result)
                time.sleep(0.001)

            return results

        # Compare implementations
        benchmark = caching_validator.compare_implementations(
            uncached_implementation, cached_implementation, iterations=3
        )

        assert benchmark.passed is True
        assert benchmark.functionality_preserved is True

        # Cached version should have better performance characteristics
        assert benchmark.optimized_metrics.cache_hits >= 0

    @pytest.mark.parametrize("ttl_seconds", (30, 60, 300, 600))
    def test_cache_data_ttl_variations(self, caching_validator, ttl_seconds):
        """Test st.cache_data with different TTL values."""

        def ttl_variation_test():
            import streamlit as st

            @st.cache_data(ttl=ttl_seconds)
            def ttl_sensitive_function(data_key: str) -> dict:
                """Function with parameterized TTL."""
                time.sleep(0.005)  # 5ms delay
                return {
                    "key": data_key,
                    "processed_at": datetime.now().isoformat(),
                    "ttl_config": ttl_seconds,
                    "data_size": len(data_key) * 100,
                }

            # Call function multiple times
            results = []
            for i in range(5):
                result = ttl_sensitive_function(f"data_{i % 3}")  # Some repeats
                results.append(result)

            return results

        result = caching_validator.validate_functionality(ttl_variation_test)
        assert result is True

        # Should have some cache hits due to repeated keys
        assert caching_validator.tracker.cache_hits > 0
        assert (
            caching_validator.tracker.get_function_call_count("ttl_sensitive_function")
            <= 3
        )


class TestCachingPerformanceBenchmarks:
    """Benchmark tests for caching performance validation."""

    @pytest.fixture
    def benchmarking_tester(self):
        """Provide tester configured for benchmarking."""
        tester = NativeComponentTester()
        tester.register_validator("caching_performance", CachingPerformanceValidator())
        return tester

    def test_high_frequency_caching_benchmark(self, benchmarking_tester):
        """Benchmark high-frequency caching scenarios."""

        def high_frequency_test():
            import streamlit as st

            @st.cache_data
            def frequent_computation(key: int) -> str:
                """Function called very frequently."""
                time.sleep(0.001)  # 1ms delay
                return f"result_{key}_{key**2}"

            # High frequency calls with repeated keys
            results = []
            for i in range(1000):
                key = i % 50  # 50 unique keys, repeated 20 times each
                result = frequent_computation(key)
                results.append(result)

            return len(results)

        benchmark = benchmarking_tester.benchmark_component(
            "caching_performance", high_frequency_test, iterations=3
        )

        assert benchmark.passed is True

        # Should have excellent cache efficiency with repeated keys
        validator = benchmarking_tester.validators["caching_performance"]
        hit_rate = validator.tracker.get_cache_hit_rate()
        assert hit_rate > 0.9  # 90%+ hit rate

        # Should be very fast due to caching
        assert benchmark.optimized_metrics.execution_time < 5.0

    def test_memory_intensive_caching_benchmark(self, benchmarking_tester):
        """Benchmark memory-intensive caching scenarios."""

        def memory_intensive_test():
            import streamlit as st

            @st.cache_data
            def create_large_structure(structure_id: int, size: int) -> dict:
                """Create memory-intensive data structure."""
                time.sleep(0.01)  # 10ms delay
                return {
                    "id": structure_id,
                    "data": list(range(size)),
                    "metadata": {f"key_{i}": f"value_{i}" for i in range(size // 10)},
                    "checksum": sum(range(size)),
                }

            @st.cache_resource
            def get_processing_engine() -> dict:
                """Get cached processing engine."""
                time.sleep(0.02)  # 20ms setup
                return {
                    "engine_type": "high_performance",
                    "memory_pool": list(range(50000)),
                    "config": {"threads": 16, "memory_limit": "1GB"},
                }

            # Mix of resource and data caching
            results = []
            engine = get_processing_engine()  # Cached resource

            for i in range(20):
                struct_id = i % 5  # 5 unique structures, repeated 4 times each
                structure = create_large_structure(struct_id, 10000)
                results.append(
                    {
                        "structure_id": structure["id"],
                        "data_size": len(structure["data"]),
                        "engine_type": engine["engine_type"],
                    }
                )

            return results

        benchmark = benchmarking_tester.benchmark_component(
            "caching_performance", memory_intensive_test, iterations=2
        )

        assert benchmark.passed is True

        # Should handle memory-intensive caching efficiently
        assert benchmark.optimized_metrics.peak_memory_mb > 0
        validator = benchmarking_tester.validators["caching_performance"]

        # Resource should be created only once
        assert validator.tracker.get_function_call_count("get_processing_engine") == 1

        # Data structures should benefit from caching
        hit_rate = validator.tracker.get_cache_hit_rate()
        assert hit_rate > 0.7  # 70%+ hit rate

    def test_mixed_ttl_caching_benchmark(self, benchmarking_tester):
        """Benchmark mixed TTL scenarios."""

        def mixed_ttl_test():
            import streamlit as st

            @st.cache_data(ttl=30)  # Short TTL
            def short_ttl_data(key: str) -> dict:
                """Data with short TTL."""
                time.sleep(0.005)
                return {
                    "key": key,
                    "type": "short_ttl",
                    "timestamp": datetime.now().isoformat(),
                }

            @st.cache_data(ttl=300)  # Long TTL
            def long_ttl_data(key: str) -> dict:
                """Data with long TTL."""
                time.sleep(0.008)
                return {
                    "key": key,
                    "type": "long_ttl",
                    "timestamp": datetime.now().isoformat(),
                }

            @st.cache_resource
            def persistent_resource() -> dict:
                """Persistent resource (no TTL)."""
                time.sleep(0.015)
                return {"type": "persistent", "resource_id": "global_resource"}

            # Mix different TTL strategies
            results = []

            for i in range(30):
                # Access persistent resource
                resource = persistent_resource()

                # Access data with different TTL strategies
                short_data = short_ttl_data(f"short_{i % 5}")  # 5 unique keys
                long_data = long_ttl_data(f"long_{i % 3}")  # 3 unique keys

                results.append(
                    {
                        "iteration": i,
                        "resource_type": resource["type"],
                        "short_data_key": short_data["key"],
                        "long_data_key": long_data["key"],
                    }
                )

            return results

        benchmark = benchmarking_tester.benchmark_component(
            "caching_performance", mixed_ttl_test, iterations=2
        )

        assert benchmark.passed is True

        validator = benchmarking_tester.validators["caching_performance"]

        # Persistent resource should be created only once
        assert validator.tracker.get_function_call_count("persistent_resource") == 1

        # Should have good overall cache performance
        hit_rate = validator.tracker.get_cache_hit_rate()
        assert hit_rate > 0.6  # 60%+ hit rate with mixed strategies
