"""Pytest configuration for Streamlit native component testing.

Provides fixtures, markers, and configuration specific to testing
Streamlit native components during library optimization migration.
"""

import json
import os
import tempfile

from datetime import datetime
from pathlib import Path

import pytest

from tests.streamlit_native.base_framework import (
    ComponentTestMetrics,
    PerformanceBenchmark,
    StreamlitNativeTester,
)
from tests.streamlit_native.test_integration_validation import IntegratedStreamValidator
from tests.streamlit_native.test_stream_a_progress import ProgressSystemValidator
from tests.streamlit_native.test_stream_b_caching import CachingSystemValidator
from tests.streamlit_native.test_stream_c_fragments import FragmentSystemValidator


@pytest.fixture(scope="session")
def test_results_directory() -> Path:
    """Create directory for test results and reports."""
    results_dir = Path("test_results/streamlit_native")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture(scope="session")
def test_session_metadata():
    """Provide test session metadata."""
    return {
        "session_id": f"streamlit_native_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": datetime.now().isoformat(),
        "environment": {
            "testing": os.getenv("TESTING", "true"),
            "ci": os.getenv("CI", "false"),
            "python_version": os.getenv("PYTHON_VERSION", "unknown"),
        },
    }


@pytest.fixture
def performance_baseline():
    """Provide performance baseline expectations."""
    return {
        "progress_components": {
            "max_render_time": 1.0,  # 1 second
            "max_memory_delta": 50_000_000,  # 50MB
        },
        "caching_components": {
            "min_hit_rate": 0.8,  # 80% cache hit rate
            "max_render_time": 2.0,  # 2 seconds
        },
        "fragment_components": {
            "max_auto_refresh_delay": 0.1,  # 100ms timing tolerance
            "max_render_time": 1.5,  # 1.5 seconds
        },
        "integration": {
            "max_render_time": 5.0,  # 5 seconds for full integration
            "min_cross_interactions": 1,  # At least 1 cross-component interaction
        },
    }


@pytest.fixture
def stream_a_validator():
    """Provide Stream A (Progress) validator."""
    return ProgressSystemValidator()


@pytest.fixture
def stream_b_validator():
    """Provide Stream B (Caching) validator."""
    return CachingSystemValidator()


@pytest.fixture
def stream_c_validator():
    """Provide Stream C (Fragments) validator."""
    return FragmentSystemValidator()


@pytest.fixture
def integration_validator():
    """Provide integrated streams validator."""
    return IntegratedStreamValidator()


@pytest.fixture
def comprehensive_tester(
    stream_a_validator, stream_b_validator, stream_c_validator, integration_validator
):
    """Provide comprehensive Streamlit native tester with all validators."""
    tester = StreamlitNativeTester()
    tester.register_validator("progress_system", stream_a_validator)
    tester.register_validator("caching_system", stream_b_validator)
    tester.register_validator("fragment_system", stream_c_validator)
    tester.register_validator("integrated_streams", integration_validator)
    return tester


@pytest.fixture
def migration_scenarios():
    """Provide migration validation scenarios."""
    return [
        {
            "name": "basic_progress_migration",
            "description": "Basic progress bar migration from manual to st.progress",
            "components": ["progress"],
            "complexity": "simple",
        },
        {
            "name": "status_container_migration",
            "description": "Status container migration from manual to st.status",
            "components": ["status", "progress"],
            "complexity": "moderate",
        },
        {
            "name": "caching_migration",
            "description": "Caching migration from manual to st.cache_data/resource",
            "components": ["cache_data", "cache_resource"],
            "complexity": "moderate",
        },
        {
            "name": "fragment_migration",
            "description": "Fragment migration from manual refresh to st.fragment",
            "components": ["fragment", "rerun"],
            "complexity": "complex",
        },
        {
            "name": "full_integration_migration",
            "description": "Full migration using all native components together",
            "components": [
                "progress",
                "status",
                "toast",
                "cache_data",
                "cache_resource",
                "fragment",
            ],
            "complexity": "complex",
        },
    ]


@pytest.fixture
def real_world_scenarios():
    """Provide real-world usage scenarios for testing."""
    return [
        {
            "name": "job_scraping_workflow",
            "description": "Complete job scraping workflow with progress tracking",
            "streams": ["A", "B", "C"],
            "expected_duration": 10.0,  # seconds
            "expected_components": {
                "progress_updates": "> 5",
                "cache_operations": "> 3",
                "fragment_executions": "> 2",
            },
        },
        {
            "name": "analytics_dashboard",
            "description": "Real-time analytics dashboard with auto-refresh",
            "streams": ["A", "B", "C"],
            "expected_duration": 8.0,
            "expected_components": {
                "progress_bars": "> 3",
                "cached_metrics": "> 5",
                "fragment_updates": "> 3",
            },
        },
        {
            "name": "data_processing_pipeline",
            "description": "Multi-stage data processing with caching and progress",
            "streams": ["A", "B"],
            "expected_duration": 6.0,
            "expected_components": {
                "processing_steps": "== 5",
                "cache_hits": "> 10",
                "status_updates": "> 3",
            },
        },
    ]


# Test result collection fixtures


class StreamlitNativeTestResults:
    """Collect and organize test results."""

    def __init__(self):
        """Initialize results collector."""
        self.stream_results = {
            "A": {"passed": 0, "failed": 0, "benchmarks": []},
            "B": {"passed": 0, "failed": 0, "benchmarks": []},
            "C": {"passed": 0, "failed": 0, "benchmarks": []},
            "integration": {"passed": 0, "failed": 0, "benchmarks": []},
        }
        self.performance_results = []
        self.migration_results = []
        self.real_world_results = []

    def record_test_result(
        self,
        stream: str,
        test_name: str,
        passed: bool,
        benchmark: PerformanceBenchmark = None,
    ):
        """Record a test result."""
        if stream in self.stream_results:
            if passed:
                self.stream_results[stream]["passed"] += 1
            else:
                self.stream_results[stream]["failed"] += 1

            if benchmark:
                self.stream_results[stream]["benchmarks"].append(
                    {"test_name": test_name, "benchmark": benchmark}
                )

    def record_performance_result(self, test_name: str, metrics: ComponentTestMetrics):
        """Record performance test result."""
        self.performance_results.append(
            {
                "test_name": test_name,
                "render_time": metrics.render_time,
                "memory_usage": metrics.memory_usage,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def record_migration_result(self, scenario_name: str, success: bool, details: dict):
        """Record migration validation result."""
        self.migration_results.append(
            {
                "scenario": scenario_name,
                "success": success,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def record_real_world_result(
        self, scenario_name: str, success: bool, metrics: dict
    ):
        """Record real-world scenario result."""
        self.real_world_results.append(
            {
                "scenario": scenario_name,
                "success": success,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def generate_summary(self) -> dict:
        """Generate test results summary."""
        total_passed = sum(stream["passed"] for stream in self.stream_results.values())
        total_failed = sum(stream["failed"] for stream in self.stream_results.values())

        return {
            "summary": {
                "total_tests": total_passed + total_failed,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": total_passed / (total_passed + total_failed)
                if (total_passed + total_failed) > 0
                else 0.0,
            },
            "by_stream": self.stream_results,
            "performance_tests": len(self.performance_results),
            "migration_tests": len(self.migration_results),
            "real_world_tests": len(self.real_world_results),
        }


@pytest.fixture(scope="session")
def test_results():
    """Provide test results collector."""
    return StreamlitNativeTestResults()


@pytest.fixture(autouse=True)
def record_test_outcome(request, test_results):
    """Automatically record test outcomes."""
    yield

    # Determine which stream this test belongs to based on test name
    test_name = request.node.name
    stream = "integration"  # default

    if "stream_a" in test_name.lower() or "progress" in test_name.lower():
        stream = "A"
    elif "stream_b" in test_name.lower() or "caching" in test_name.lower():
        stream = "B"
    elif "stream_c" in test_name.lower() or "fragment" in test_name.lower():
        stream = "C"

    # Record the result
    passed = (
        request.node.rep_call.passed if hasattr(request.node, "rep_call") else False
    )
    test_results.record_test_result(stream, test_name, passed)


# Performance testing fixtures


@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time

    import psutil

    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss
    process.cpu_percent()

    yield

    end_time = time.time()
    end_memory = process.memory_info().rss
    end_cpu = process.cpu_percent()

    performance_data = {
        "duration": end_time - start_time,
        "memory_delta": end_memory - start_memory,
        "memory_peak": max(start_memory, end_memory),
        "cpu_usage": end_cpu,
    }

    # Store in test node for later retrieval
    pytest.current_test_performance = performance_data


# Markers for organizing tests


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "stream_a: Stream A (Progress System) tests")
    config.addinivalue_line("markers", "stream_b: Stream B (Caching System) tests")
    config.addinivalue_line("markers", "stream_c: Stream C (Fragment System) tests")
    config.addinivalue_line("markers", "integration: Cross-stream integration tests")
    config.addinivalue_line(
        "markers", "performance: Performance and benchmarking tests"
    )
    config.addinivalue_line("markers", "migration: Migration validation tests")
    config.addinivalue_line("markers", "real_world: Real-world scenario tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take longer to run")


# Hooks for test reporting


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Create test reports with additional metadata."""
    outcome = yield
    rep = outcome.get_result()

    # Store the report in the test node for later access
    setattr(item, f"rep_{rep.when}", rep)

    return rep


def pytest_sessionfinish(session, exitstatus):
    """Generate comprehensive test report at session end."""
    if hasattr(session, "test_results"):
        results = session.test_results

        # Generate summary report
        summary = results.generate_summary()

        # Write results to file
        results_file = Path("test_results/streamlit_native/test_summary.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("\n\nStreamlit Native Component Test Summary:")
        print(f"Total tests: {summary['summary']['total_tests']}")
        print(f"Passed: {summary['summary']['passed']}")
        print(f"Failed: {summary['summary']['failed']}")
        print(f"Success rate: {summary['summary']['success_rate']:.2%}")
        print(f"\nDetailed results written to: {results_file}")


# Custom assertions for Streamlit components


def assert_component_performance(metrics: ComponentTestMetrics, baseline: dict):
    """Assert component meets performance baseline."""
    assert metrics.render_time <= baseline.get("max_render_time", 5.0), (
        f"Render time {metrics.render_time:.3f}s exceeds baseline {baseline.get('max_render_time', 5.0):.3f}s"
    )

    if "max_memory_delta" in baseline:
        assert metrics.memory_usage <= baseline["max_memory_delta"], (
            f"Memory usage {metrics.memory_usage} exceeds baseline {baseline['max_memory_delta']}"
        )


def assert_cache_performance(cache_metrics, baseline: dict):
    """Assert caching meets performance baseline."""
    if hasattr(cache_metrics, "get_hit_rate"):
        hit_rate = cache_metrics.get_hit_rate()
        min_hit_rate = baseline.get("min_hit_rate", 0.5)
        assert hit_rate >= min_hit_rate, (
            f"Cache hit rate {hit_rate:.2%} below baseline {min_hit_rate:.2%}"
        )


def assert_fragment_timing(
    fragment_metrics, expected_interval: float, tolerance: float = 0.1
):
    """Assert fragment timing accuracy."""
    for fragment_id in fragment_metrics.executions:
        if fragment_id in fragment_metrics.run_every_intervals:
            accuracy = fragment_metrics.get_timing_accuracy(fragment_id)
            assert accuracy >= (1.0 - tolerance), (
                f"Fragment {fragment_id} timing accuracy {accuracy:.2%} below {(1.0 - tolerance):.2%}"
            )


# Utility fixtures for complex test scenarios


@pytest.fixture
def temp_test_data():
    """Provide temporary test data for complex scenarios."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test data files
        test_files = {}

        # Create test configuration
        config_file = temp_path / "test_config.json"
        config_data = {
            "test_mode": True,
            "cache_ttl": 60,
            "fragment_interval": "1s",
            "progress_steps": 10,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        test_files["config"] = config_file

        # Create test data file
        data_file = temp_path / "test_data.json"
        data = {
            "companies": ["TechCorp", "DataInc", "AIStartup"],
            "job_counts": [25, 15, 30],
            "processing_steps": ["init", "scrape", "process", "analyze", "store"],
        }
        with open(data_file, "w") as f:
            json.dump(data, f)
        test_files["data"] = data_file

        yield test_files


@pytest.fixture
def mock_external_services():
    """Provide mocked external services for testing."""
    services = {
        "job_api": Mock(),
        "analytics_db": Mock(),
        "notification_service": Mock(),
    }

    # Configure mock responses
    services["job_api"].get_jobs.return_value = [
        {"id": i, "title": f"Job {i}", "company": f"Company {i % 3}"} for i in range(20)
    ]

    services["analytics_db"].query.return_value = {
        "total_jobs": 150,
        "active_applications": 25,
        "response_rate": 0.15,
    }

    services["notification_service"].send.return_value = {"status": "sent"}

    return services


# Environment configuration


@pytest.fixture(scope="session", autouse=True)
def configure_streamlit_native_environment():
    """Configure environment for Streamlit native component testing."""
    # Set testing environment variables
    os.environ["STREAMLIT_NATIVE_TESTING"] = "true"
    os.environ["STREAMLIT_TEST_MODE"] = "native_components"
    os.environ["CACHE_TESTING"] = "true"
    os.environ["FRAGMENT_TESTING"] = "true"

    yield

    # Cleanup
    test_env_vars = [
        "STREAMLIT_NATIVE_TESTING",
        "STREAMLIT_TEST_MODE",
        "CACHE_TESTING",
        "FRAGMENT_TESTING",
    ]

    for var in test_env_vars:
        os.environ.pop(var, None)
