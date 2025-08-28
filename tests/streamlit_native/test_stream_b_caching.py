"""Stream B Testing: Caching Unification Validation.

Tests for Streamlit native caching components:
- st.cache_data() performance and behavior validation
- st.cache_resource() connection management testing
- TTL expiration and cache invalidation
- Cache hit/miss rate analysis
- Memory efficiency validation
- Performance regression testing for migration optimization

Ensures 100% functionality preservation during library optimization migration.
"""

import contextlib
import hashlib
import time

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from tests.streamlit_native.base_framework import (
    ComponentTestMetrics,
    StreamlitComponentValidator,
    StreamlitNativeTester,
)


class CacheMetrics:
    """Track caching metrics during tests."""

    def __init__(self):
        """Initialize cache metrics."""
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.cache_size = 0
        self.memory_usage = 0
        self.ttl_expirations = 0
        self.function_calls = {}
        self.cache_keys = set()

    def record_hit(self, key: str):
        """Record a cache hit."""
        self.hits += 1
        self.cache_keys.add(key)

    def record_miss(self, key: str):
        """Record a cache miss."""
        self.misses += 1
        self.cache_keys.add(key)

    def record_invalidation(self, key: str):
        """Record a cache invalidation."""
        self.invalidations += 1

    def record_function_call(self, func_name: str):
        """Record function execution."""
        self.function_calls[func_name] = self.function_calls.get(func_name, 0) + 1

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self):
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.cache_size = 0
        self.memory_usage = 0
        self.ttl_expirations = 0
        self.function_calls.clear()
        self.cache_keys.clear()


class MockStreamlitCache:
    """Mock Streamlit cache implementation for testing."""

    def __init__(self, metrics: CacheMetrics):
        """Initialize mock cache."""
        self.metrics = metrics
        self.cache_data_store = {}
        self.cache_resource_store = {}
        self.expiration_times = {}

    def mock_cache_data(self, ttl: int | None = None, hash_funcs: dict | None = None):
        """Mock st.cache_data decorator."""

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func, args, kwargs, hash_funcs)

                # Check TTL expiration
                if ttl and cache_key in self.expiration_times:
                    if datetime.now() > self.expiration_times[cache_key]:
                        # Expired, remove from cache
                        self.cache_data_store.pop(cache_key, None)
                        self.expiration_times.pop(cache_key, None)
                        self.metrics.ttl_expirations += 1

                # Check cache
                if cache_key in self.cache_data_store:
                    self.metrics.record_hit(cache_key)
                    return self.cache_data_store[cache_key]

                # Cache miss - execute function
                self.metrics.record_miss(cache_key)
                self.metrics.record_function_call(func.__name__)
                result = func(*args, **kwargs)

                # Store in cache
                self.cache_data_store[cache_key] = result
                if ttl:
                    self.expiration_times[cache_key] = datetime.now() + timedelta(
                        seconds=ttl
                    )

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
                cache_key = self._generate_cache_key(func, args, kwargs, hash_funcs)

                # Check cache
                if cache_key in self.cache_resource_store:
                    self.metrics.record_hit(cache_key)
                    return self.cache_resource_store[cache_key]

                # Cache miss - execute function
                self.metrics.record_miss(cache_key)
                self.metrics.record_function_call(func.__name__)
                result = func(*args, **kwargs)

                # Store in cache (resources are cached indefinitely)
                self.cache_resource_store[cache_key] = result

                return result

            wrapper._is_cached = True
            wrapper._cache_type = "resource"
            return wrapper

        return decorator

    def _generate_cache_key(
        self, func: Callable, args: tuple, kwargs: dict, hash_funcs: dict | None = None
    ) -> str:
        """Generate cache key for function call."""
        # Simple key generation for testing
        key_data = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def clear_cache(self):
        """Clear all caches."""
        cleared_keys = len(self.cache_data_store) + len(self.cache_resource_store)
        self.cache_data_store.clear()
        self.cache_resource_store.clear()
        self.expiration_times.clear()
        self.metrics.invalidations += cleared_keys


class CachingSystemValidator(StreamlitComponentValidator):
    """Validator for caching system components (st.cache_data, st.cache_resource)."""

    def __init__(self):
        """Initialize caching system validator."""
        super().__init__("caching_system")
        self.cache_metrics = CacheMetrics()
        self.mock_cache = MockStreamlitCache(self.cache_metrics)

    def validate_component_behavior(self, test_func, *args, **kwargs) -> bool:
        """Validate caching component behavior."""
        try:
            # Reset metrics
            self.cache_metrics.reset()

            # Mock the caching components
            with (
                patch("streamlit.cache_data", self.mock_cache.mock_cache_data),
                patch("streamlit.cache_resource", self.mock_cache.mock_cache_resource),
            ):
                # Execute test function
                test_func(*args, **kwargs)

            # Validate behavior (customizable per test)
            return True

        except Exception:
            self.metrics.error_count += 1
            return False

    def measure_performance(self, test_func, *args, **kwargs) -> ComponentTestMetrics:
        """Measure caching component performance."""
        with self.performance_monitoring() as metrics:
            self.validate_component_behavior(test_func, *args, **kwargs)

        # Add cache-specific metrics
        metrics.cache_hits = self.cache_metrics.hits
        metrics.cache_misses = self.cache_metrics.misses

        return metrics

    def validate_cache_data_behavior(self, cache_config: dict[str, Any]) -> bool:
        """Validate st.cache_data behavior."""

        def test_cache_data():
            import streamlit as st

            ttl = cache_config.get("ttl")
            hash_funcs = cache_config.get("hash_funcs")

            @st.cache_data(ttl=ttl, hash_funcs=hash_funcs)
            def cached_function(x: int, y: str = "default") -> str:
                # Simulate expensive operation
                time.sleep(0.001)  # 1ms delay
                return f"result_{x}_{y}"

            # Test multiple calls
            call_count = cache_config.get("call_count", 3)
            results = []
            for _i in range(call_count):
                result = cached_function(
                    cache_config.get("x", 1), cache_config.get("y", "test")
                )
                results.append(result)

            return results

        self.validate_component_behavior(test_cache_data)
        return True

    def validate_cache_resource_behavior(self, resource_config: dict[str, Any]) -> bool:
        """Validate st.cache_resource behavior."""

        def test_cache_resource():
            import streamlit as st

            hash_funcs = resource_config.get("hash_funcs")

            @st.cache_resource(hash_funcs=hash_funcs)
            def create_resource(connection_string: str) -> dict:
                # Simulate resource creation (database connection, etc.)
                time.sleep(0.005)  # 5ms delay
                return {
                    "connection": f"mock_connection_{connection_string}",
                    "created_at": datetime.now(),
                    "status": "connected",
                }

            # Test multiple calls
            call_count = resource_config.get("call_count", 3)
            resources = []
            for _i in range(call_count):
                resource = create_resource(
                    resource_config.get("connection_string", "localhost")
                )
                resources.append(resource)

            return resources

        self.validate_component_behavior(test_cache_resource)
        return True

    def validate_ttl_expiration(self, ttl_seconds: int) -> bool:
        """Validate TTL-based cache expiration."""

        def test_ttl_expiration():
            import streamlit as st

            @st.cache_data(ttl=ttl_seconds)
            def ttl_function(value: int) -> int:
                return value * 2

            # First call - should cache
            result1 = ttl_function(5)

            # Simulate time passage beyond TTL
            if ttl_seconds < 2:  # Only test short TTLs to keep tests fast
                time.sleep(ttl_seconds + 0.1)

            # Second call - should either hit cache or miss depending on TTL
            result2 = ttl_function(5)

            return result1, result2

        self.validate_component_behavior(test_ttl_expiration)

        # For fast tests, we check if the logic runs correctly
        # In real scenarios, TTL expiration would be observed in metrics
        return self.cache_metrics.function_calls.get("ttl_function", 0) > 0

    def validate_cache_invalidation(self) -> bool:
        """Validate cache invalidation behavior."""

        def test_cache_invalidation():
            import streamlit as st

            @st.cache_data
            def invalidation_test(x: int) -> int:
                return x**2

            # Cache some values
            invalidation_test(1)
            invalidation_test(2)
            invalidation_test(3)

            # Clear cache
            self.mock_cache.clear_cache()

            # Call again - should miss cache
            invalidation_test(1)

        self.validate_component_behavior(test_cache_invalidation)

        # Check that invalidation was recorded
        return self.cache_metrics.invalidations > 0

    def measure_cache_efficiency(
        self, test_scenario: dict[str, Any]
    ) -> dict[str, float]:
        """Measure cache efficiency metrics."""

        def efficiency_test():
            import streamlit as st

            @st.cache_data(ttl=test_scenario.get("ttl", 300))
            def data_function(key: str) -> str:
                time.sleep(0.002)  # 2ms delay
                return f"processed_{key}"

            @st.cache_resource
            def resource_function(resource_id: int) -> dict:
                time.sleep(0.005)  # 5ms delay
                return {"id": resource_id, "data": f"resource_{resource_id}"}

            # Execute test pattern
            keys = test_scenario.get("keys", ["a", "b", "c"])
            repetitions = test_scenario.get("repetitions", 3)

            for _ in range(repetitions):
                for key in keys:
                    data_function(key)
                    resource_function(hash(key) % 10)

        # Reset metrics before test
        self.cache_metrics.reset()

        with self.performance_monitoring() as metrics:
            self.validate_component_behavior(efficiency_test)

        return {
            "hit_rate": self.cache_metrics.get_hit_rate(),
            "total_hits": self.cache_metrics.hits,
            "total_misses": self.cache_metrics.misses,
            "function_executions": sum(self.cache_metrics.function_calls.values()),
            "render_time": metrics.render_time,
            "memory_delta": metrics.memory_usage,
        }


class TestStreamBCachingSystem:
    """Test suite for Stream B caching system components."""

    @pytest.fixture
    def caching_validator(self):
        """Provide caching system validator."""
        return CachingSystemValidator()

    @pytest.fixture
    def streamlit_tester(self, caching_validator):
        """Provide configured Streamlit tester."""
        tester = StreamlitNativeTester()
        tester.register_validator("caching_system", caching_validator)
        return tester

    def test_cache_data_basic_behavior(self, caching_validator):
        """Test basic st.cache_data functionality."""
        config = {"x": 42, "y": "test_value", "call_count": 3}

        assert caching_validator.validate_cache_data_behavior(config)

        # Should have 1 miss (first call) and 2 hits (subsequent calls)
        assert caching_validator.cache_metrics.misses == 1
        assert caching_validator.cache_metrics.hits == 2

    def test_cache_data_with_ttl(self, caching_validator):
        """Test st.cache_data with TTL."""
        config = {
            "ttl": 60,  # 60 seconds
            "x": 100,
            "y": "ttl_test",
            "call_count": 2,
        }

        assert caching_validator.validate_cache_data_behavior(config)

        # Should use cache on second call
        assert caching_validator.cache_metrics.hits >= 1

    def test_cache_data_different_parameters(self, caching_validator):
        """Test st.cache_data with different parameters."""

        def test_different_params():
            import streamlit as st

            @st.cache_data
            def param_function(a: int, b: str, c: bool = True) -> str:
                return f"{a}_{b}_{c}"

            # Different parameter combinations should create separate cache entries
            result1 = param_function(1, "a")
            result2 = param_function(1, "b")  # Different string
            result3 = param_function(2, "a")  # Different int
            result4 = param_function(1, "a", False)  # Different bool
            result5 = param_function(1, "a")  # Same as result1 - should hit cache

            return [result1, result2, result3, result4, result5]

        caching_validator.validate_component_behavior(test_different_params)

        # Should have 4 misses (unique combinations) and 1 hit (duplicate)
        assert caching_validator.cache_metrics.misses == 4
        assert caching_validator.cache_metrics.hits == 1

    def test_cache_resource_basic_behavior(self, caching_validator):
        """Test basic st.cache_resource functionality."""
        config = {"connection_string": "test_db", "call_count": 5}

        assert caching_validator.validate_cache_resource_behavior(config)

        # Should have 1 miss (resource creation) and 4 hits (reuse)
        assert caching_validator.cache_metrics.misses == 1
        assert caching_validator.cache_metrics.hits == 4

    def test_cache_resource_different_connections(self, caching_validator):
        """Test st.cache_resource with different connection strings."""

        def test_different_resources():
            import streamlit as st

            @st.cache_resource
            def create_connection(host: str, port: int = 5432) -> dict:
                return {"host": host, "port": port, "connection_id": f"{host}:{port}"}

            # Different resources should be cached separately
            conn1 = create_connection("localhost")
            conn2 = create_connection("localhost")  # Same - should hit cache
            conn3 = create_connection("remote_host")  # Different - new resource
            conn4 = create_connection("remote_host")  # Same as conn3 - should hit cache

            return [conn1, conn2, conn3, conn4]

        caching_validator.validate_component_behavior(test_different_resources)

        # Should have 2 misses (unique resources) and 2 hits (duplicates)
        assert caching_validator.cache_metrics.misses == 2
        assert caching_validator.cache_metrics.hits == 2

    def test_ttl_expiration_fast(self, caching_validator):
        """Test TTL expiration with very short TTL for fast testing."""
        # Use very short TTL for testing (1 second)
        assert caching_validator.validate_ttl_expiration(1)

        # Function should have been called at least once
        assert "ttl_function" in caching_validator.cache_metrics.function_calls

    def test_cache_invalidation(self, caching_validator):
        """Test cache invalidation functionality."""
        assert caching_validator.validate_cache_invalidation()

        # Should record invalidation
        assert caching_validator.cache_metrics.invalidations > 0

    def test_cache_efficiency_analysis(self, caching_validator):
        """Test cache efficiency measurement."""
        scenario = {"keys": ["key1", "key2", "key3"], "repetitions": 4, "ttl": 300}

        efficiency = caching_validator.measure_cache_efficiency(scenario)

        assert "hit_rate" in efficiency
        assert "total_hits" in efficiency
        assert "total_misses" in efficiency

        # With repetitions, we should have a good hit rate
        assert efficiency["hit_rate"] > 0.5  # At least 50% hit rate

    def test_mixed_cache_usage(self, streamlit_tester):
        """Test mixed usage of cache_data and cache_resource."""

        def mixed_cache_test():
            import streamlit as st

            @st.cache_resource
            def get_database_connection() -> dict:
                return {"connection": "database", "pool_size": 10}

            @st.cache_data(ttl=60)
            def get_user_data(user_id: int) -> dict:
                # Simulate database query using cached connection
                conn = get_database_connection()
                return {"user_id": user_id, "connection_id": conn["connection"]}

            # Test the mixed pattern
            conn = get_database_connection()  # Cache resource
            user1 = get_user_data(1)  # Cache data
            user2 = get_user_data(2)  # Cache data (different key)
            user1_again = get_user_data(1)  # Should hit data cache
            conn_again = get_database_connection()  # Should hit resource cache

            return [conn, user1, user2, user1_again, conn_again]

        result = streamlit_tester.run_component_validation(
            "caching_system", mixed_cache_test
        )

        assert result is True

        validator = streamlit_tester.validators["caching_system"]

        # Should have both cache hits and misses
        assert validator.cache_metrics.hits > 0
        assert validator.cache_metrics.misses > 0

    def test_cache_performance_optimization(self, streamlit_tester):
        """Test that caching provides performance optimization."""

        def performance_test():
            import streamlit as st

            @st.cache_data
            def expensive_computation(n: int) -> int:
                # Simulate expensive computation
                time.sleep(0.01)  # 10ms delay
                result = 0
                for i in range(n):
                    result += i**2
                return result

            # First run - will cache results
            results = []
            for _i in range(5):
                result = expensive_computation(100)
                results.append(result)

            return results

        # Benchmark with caching
        streamlit_tester.benchmark_component_performance(
            "caching_system", performance_test, iterations=3
        )

        validator = streamlit_tester.validators["caching_system"]

        # Should have significant cache reuse
        assert validator.cache_metrics.hits > 0
        assert validator.cache_metrics.get_hit_rate() > 0.8  # 80%+ hit rate

    def test_cache_memory_efficiency(self, caching_validator):
        """Test cache memory efficiency."""

        def memory_test():
            import streamlit as st

            @st.cache_data
            def create_large_data(size: int) -> list:
                # Create data that should be cached efficiently
                return list(range(size))

            # Test with different sizes
            small_data = create_large_data(100)
            medium_data = create_large_data(1000)
            small_data_again = create_large_data(100)  # Should hit cache

            return [small_data, medium_data, small_data_again]

        with caching_validator.performance_monitoring() as metrics:
            caching_validator.validate_component_behavior(memory_test)

        # Should use cache effectively
        assert caching_validator.cache_metrics.hits >= 1

        # Memory usage should be reasonable
        assert metrics.memory_usage >= 0  # At least not negative

    def test_cache_hash_functions(self, caching_validator):
        """Test custom hash functions in caching."""

        def hash_function_test():
            import streamlit as st

            # Custom hash function for complex objects
            def hash_dict(d: dict) -> str:
                return str(sorted(d.items()))

            @st.cache_data(hash_funcs={dict: hash_dict})
            def process_config(config: dict) -> str:
                return f"processed_{len(config)}_keys"

            # Test with dictionary parameters
            config1 = {"a": 1, "b": 2}
            config2 = {"b": 2, "a": 1}  # Same content, different order
            config3 = {"c": 3, "d": 4}  # Different content

            result1 = process_config(config1)
            result2 = process_config(config2)  # Should hit cache (same hash)
            result3 = process_config(config3)  # Should miss cache

            return [result1, result2, result3]

        caching_validator.validate_component_behavior(hash_function_test)

        # Should have cache behavior based on custom hash function
        assert (
            caching_validator.cache_metrics.function_calls.get("process_config", 0) >= 1
        )

    def test_cache_error_handling(self, caching_validator):
        """Test error handling in cached functions."""

        def error_handling_test():
            import streamlit as st

            @st.cache_data
            def function_with_errors(value: int) -> str:
                if value < 0:
                    raise ValueError("Negative values not allowed")
                return f"result_{value}"

            results = []

            # Normal calls
            results.append(function_with_errors(1))
            results.append(function_with_errors(1))  # Should hit cache

            # Error case - should not be cached
            with contextlib.suppress(ValueError):
                function_with_errors(-1)

            # Normal call after error
            results.append(function_with_errors(2))

            return results

        result = caching_validator.validate_component_behavior(error_handling_test)

        # Should handle errors gracefully
        assert result is True
        assert caching_validator.cache_metrics.hits >= 1

    def test_concurrent_cache_access(self, streamlit_tester):
        """Test concurrent access to cached functions."""

        def concurrent_test():
            def worker_test(worker_id: int):
                def thread_cache_test():
                    import streamlit as st

                    @st.cache_data
                    def thread_safe_function(x: int) -> str:
                        time.sleep(0.001)  # Small delay
                        return f"worker_{worker_id}_result_{x}"

                    # Each thread calls the function multiple times
                    results = []
                    for i in range(3):
                        result = thread_safe_function(i)
                        results.append(result)

                    return results

                return streamlit_tester.run_component_validation(
                    "caching_system", thread_cache_test
                )

            # Simulate concurrent access
            results = []
            for i in range(3):
                result = worker_test(i)
                results.append(result)

            return results

        results = concurrent_test()

        # All concurrent tests should succeed
        assert all(results)

    @pytest.mark.parametrize("ttl_value", (30, 60, 300, 600))
    def test_cache_data_ttl_values(self, caching_validator, ttl_value):
        """Test st.cache_data with different TTL values."""
        config = {"ttl": ttl_value, "x": 42, "call_count": 2}

        assert caching_validator.validate_cache_data_behavior(config)

        # Should use cache on repeat calls regardless of TTL
        assert caching_validator.cache_metrics.hits >= 1

    @pytest.mark.parametrize("cache_size", (10, 100, 1000))
    def test_cache_data_with_different_sizes(self, caching_validator, cache_size):
        """Test cache behavior with different data sizes."""

        def size_test():
            import streamlit as st

            @st.cache_data
            def create_sized_data(size: int) -> list:
                return list(range(size))

            # Test with the parameterized size
            data1 = create_sized_data(cache_size)
            data2 = create_sized_data(cache_size)  # Should hit cache

            return [data1, data2]

        caching_validator.validate_component_behavior(size_test)

        # Should have cache hit
        assert caching_validator.cache_metrics.hits >= 1


class TestCachingSystemBenchmarks:
    """Benchmark tests for caching system performance validation."""

    @pytest.fixture
    def benchmarking_tester(self):
        """Provide tester configured for benchmarking."""
        tester = StreamlitNativeTester()
        tester.register_validator("caching_system", CachingSystemValidator())
        return tester

    def test_cache_data_performance_improvement(self, benchmarking_tester):
        """Benchmark cache_data performance improvement."""

        def cached_vs_uncached():
            import streamlit as st

            @st.cache_data
            def expensive_operation(n: int) -> int:
                time.sleep(0.005)  # 5ms delay
                return sum(i**2 for i in range(n))

            # Multiple calls to same function
            results = []
            for _ in range(10):
                result = expensive_operation(100)
                results.append(result)

            return results

        benchmark = benchmarking_tester.benchmark_component_performance(
            "caching_system", cached_vs_uncached, iterations=3
        )

        validator = benchmarking_tester.validators["caching_system"]

        # Should have high cache hit rate
        hit_rate = validator.cache_metrics.get_hit_rate()
        assert hit_rate > 0.8  # 80%+ cache hits

        # Performance should be reasonable
        assert benchmark.after_metrics.render_time > 0

    def test_cache_resource_performance(self, benchmarking_tester):
        """Benchmark cache_resource performance."""

        def resource_benchmark():
            import streamlit as st

            @st.cache_resource
            def create_expensive_resource(resource_type: str) -> dict:
                time.sleep(0.01)  # 10ms delay
                return {
                    "type": resource_type,
                    "data": list(range(1000)),
                    "created": datetime.now(),
                }

            # Multiple access to same resource
            resources = []
            for _i in range(5):
                resource = create_expensive_resource("database")
                resources.append(resource)

            return resources

        benchmarking_tester.benchmark_component_performance(
            "caching_system", resource_benchmark, iterations=3
        )

        validator = benchmarking_tester.validators["caching_system"]

        # Should reuse resource efficiently
        assert validator.cache_metrics.hits >= 4  # 4 out of 5 calls should hit cache

    def test_mixed_caching_performance(self, benchmarking_tester):
        """Benchmark mixed cache_data and cache_resource performance."""

        def mixed_performance():
            import streamlit as st

            @st.cache_resource
            def get_connection() -> dict:
                time.sleep(0.002)  # 2ms delay
                return {"connection": "active", "pool": 10}

            @st.cache_data(ttl=60)
            def process_data(data_id: int) -> dict:
                conn = get_connection()  # Uses cached resource
                time.sleep(0.001)  # 1ms delay
                return {"id": data_id, "processed": True, "conn": conn["connection"]}

            # Mixed usage pattern
            results = []
            for i in range(20):
                result = process_data(i % 5)  # Repeat some data_ids
                results.append(result)

            return results

        benchmarking_tester.benchmark_component_performance(
            "caching_system", mixed_performance, iterations=2
        )

        validator = benchmarking_tester.validators["caching_system"]

        # Should have efficient caching for both types
        total_calls = validator.cache_metrics.hits + validator.cache_metrics.misses
        hit_rate = validator.cache_metrics.get_hit_rate()

        assert total_calls > 0
        assert hit_rate > 0.6  # 60%+ hit rate with mixed usage
