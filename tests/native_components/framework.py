"""Core Native Component Testing Framework.

Provides foundational classes and utilities for testing Streamlit's native
components with comprehensive validation, performance benchmarking, and
functionality preservation testing.

This framework focuses on:
1. Functionality Preservation: Ensures native components behave identically to manual implementations
2. Performance Validation: Measures and validates performance improvements
3. Integration Testing: Tests cross-component functionality and interactions
4. Regression Prevention: Catches regressions during library optimizations
"""

import os
import time

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import Mock

import psutil
import pytest

from tests.utils.streamlit_utils import (
    MockStreamlitSession,
    mock_streamlit_app,
    mock_streamlit_context,
)


class StreamType(Enum):
    """Stream types for native component testing."""

    STREAM_A = "progress"  # Progress components (st.status, st.progress, st.toast)
    STREAM_B = "caching"  # Caching performance (st.cache_data, st.cache_resource)
    STREAM_C = "fragments"  # Fragment behavior (st.fragment, run_every)


class ValidationResult(Enum):
    """Validation result types."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class NativeComponentMetrics:
    """Comprehensive metrics for native component testing."""

    # Base performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    peak_memory_mb: float = 0.0

    # Component-specific metrics
    render_count: int = 0
    rerun_count: int = 0
    error_count: int = 0

    # Stream A metrics (Progress)
    progress_updates: int = 0
    status_state_changes: int = 0
    toast_notifications: int = 0

    # Stream B metrics (Caching)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalidations: int = 0
    cache_efficiency: float = 0.0

    # Stream C metrics (Fragments)
    fragment_executions: int = 0
    auto_refresh_count: int = 0
    timing_accuracy: float = 0.0

    # Integration metrics
    cross_component_interactions: int = 0

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def calculate_cache_efficiency(self) -> float:
        """Calculate cache efficiency percentage."""
        total_operations = self.cache_hits + self.cache_misses
        if total_operations == 0:
            return 0.0
        self.cache_efficiency = (self.cache_hits / total_operations) * 100
        return self.cache_efficiency

    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.end_time = datetime.now()


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data for native components."""

    component_name: str
    stream_type: StreamType
    test_name: str

    # Before/after comparison
    baseline_metrics: NativeComponentMetrics | None = None
    optimized_metrics: NativeComponentMetrics | None = None

    # Performance deltas
    time_improvement_percent: float = 0.0
    memory_improvement_percent: float = 0.0
    functionality_preserved: bool = False

    # Benchmark configuration
    iterations: int = 1
    warmup_iterations: int = 0

    # Results
    passed: bool = False
    error_message: str | None = None

    def calculate_improvements(self) -> None:
        """Calculate performance improvements."""
        if not (self.baseline_metrics and self.optimized_metrics):
            return

        # Time improvement
        baseline_time = self.baseline_metrics.execution_time
        optimized_time = self.optimized_metrics.execution_time

        if baseline_time > 0:
            self.time_improvement_percent = (
                (baseline_time - optimized_time) / baseline_time
            ) * 100

        # Memory improvement
        baseline_memory = self.baseline_metrics.memory_usage_mb
        optimized_memory = self.optimized_metrics.memory_usage_mb

        if baseline_memory > 0:
            self.memory_improvement_percent = (
                (baseline_memory - optimized_memory) / baseline_memory
            ) * 100


@dataclass
class StreamValidationResults:
    """Validation results for a complete stream."""

    stream_type: StreamType
    component_results: dict[str, ValidationResult] = field(default_factory=dict)
    benchmarks: list[PerformanceBenchmark] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0

    # Performance summary
    avg_time_improvement: float = 0.0
    avg_memory_improvement: float = 0.0
    functionality_preservation_rate: float = 0.0

    def add_result(self, component_name: str, result: ValidationResult) -> None:
        """Add a component validation result."""
        self.component_results[component_name] = result
        self.total_tests += 1

        if result == ValidationResult.PASSED:
            self.passed_tests += 1
        elif result == ValidationResult.FAILED:
            self.failed_tests += 1
        elif result == ValidationResult.ERROR:
            self.error_tests += 1
        elif result == ValidationResult.SKIPPED:
            self.skipped_tests += 1

    def add_benchmark(self, benchmark: PerformanceBenchmark) -> None:
        """Add a performance benchmark."""
        self.benchmarks.append(benchmark)
        self._update_performance_summary()

    def _update_performance_summary(self) -> None:
        """Update performance summary statistics."""
        if not self.benchmarks:
            return

        valid_benchmarks = [b for b in self.benchmarks if b.passed]
        if not valid_benchmarks:
            return

        # Calculate averages
        time_improvements = [b.time_improvement_percent for b in valid_benchmarks]
        memory_improvements = [b.memory_improvement_percent for b in valid_benchmarks]
        functionality_rates = [
            100.0 if b.functionality_preserved else 0.0 for b in valid_benchmarks
        ]

        self.avg_time_improvement = sum(time_improvements) / len(time_improvements)
        self.avg_memory_improvement = sum(memory_improvements) / len(
            memory_improvements
        )
        self.functionality_preservation_rate = sum(functionality_rates) / len(
            functionality_rates
        )

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        return (
            (self.passed_tests / self.total_tests * 100)
            if self.total_tests > 0
            else 0.0
        )


class NativeComponentValidator(ABC):
    """Abstract base class for native component validators."""

    def __init__(self, stream_type: StreamType, component_name: str):
        """Initialize validator.

        Args:
            stream_type: The stream this validator belongs to
            component_name: Name of the component being validated
        """
        self.stream_type = stream_type
        self.component_name = component_name
        self.metrics = NativeComponentMetrics()
        self.session = MockStreamlitSession()

    @abstractmethod
    def validate_functionality(self, test_func: Callable, *args, **kwargs) -> bool:
        """Validate component functionality preservation.

        Args:
            test_func: Test function to execute
            *args: Test function arguments
            **kwargs: Test function keyword arguments

        Returns:
            True if functionality is preserved
        """

    @abstractmethod
    def measure_performance(
        self, test_func: Callable, iterations: int = 10
    ) -> NativeComponentMetrics:
        """Measure component performance.

        Args:
            test_func: Test function to benchmark
            iterations: Number of iterations to run

        Returns:
            Performance metrics
        """

    @abstractmethod
    def compare_implementations(
        self,
        baseline_func: Callable,
        optimized_func: Callable,
        iterations: int = 10,
    ) -> PerformanceBenchmark:
        """Compare baseline vs optimized implementations.

        Args:
            baseline_func: Original implementation
            optimized_func: Optimized implementation
            iterations: Number of iterations to run

        Returns:
            Performance benchmark comparison
        """

    @contextmanager
    def performance_monitoring(self) -> Generator[NativeComponentMetrics, None, None]:
        """Context manager for performance monitoring."""
        process = psutil.Process(os.getpid())

        # Reset and start metrics
        self.metrics = NativeComponentMetrics()
        start_time = time.perf_counter()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield self.metrics
        finally:
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            self.metrics.execution_time = end_time - start_time
            self.metrics.memory_usage_mb = end_memory - start_memory
            self.metrics.peak_memory_mb = max(start_memory, end_memory)

            try:
                self.metrics.cpu_usage_percent = process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.metrics.cpu_usage_percent = 0.0

            self.metrics.finalize()

    def setup_test_environment(
        self, initial_state: dict[str, Any] | None = None
    ) -> None:
        """Set up test environment."""
        self.session.clear()
        if initial_state:
            self.session.update(initial_state)

    def teardown_test_environment(self) -> None:
        """Clean up test environment."""
        self.session.clear()
        self.metrics = NativeComponentMetrics()


class NativeComponentTester:
    """Main tester class for native component validation."""

    def __init__(self):
        """Initialize native component tester."""
        self.validators: dict[str, NativeComponentValidator] = {}
        self.stream_results: dict[StreamType, StreamValidationResults] = {}
        self.session = MockStreamlitSession()

        # Initialize stream results
        for stream_type in StreamType:
            self.stream_results[stream_type] = StreamValidationResults(stream_type)

    def register_validator(
        self, name: str, validator: NativeComponentValidator
    ) -> None:
        """Register a component validator.

        Args:
            name: Unique name for the validator
            validator: Validator instance
        """
        self.validators[name] = validator

    def validate_component(
        self,
        validator_name: str,
        test_func: Callable,
        *args,
        **kwargs,
    ) -> ValidationResult:
        """Validate a single component.

        Args:
            validator_name: Name of the validator to use
            test_func: Test function to execute
            *args: Test function arguments
            **kwargs: Test function keyword arguments

        Returns:
            Validation result
        """
        if validator_name not in self.validators:
            return ValidationResult.ERROR

        validator = self.validators[validator_name]

        try:
            validator.setup_test_environment(kwargs.get("initial_state"))

            with mock_streamlit_context(session=self.session):
                with mock_streamlit_app():
                    success = validator.validate_functionality(
                        test_func, *args, **kwargs
                    )

            result = ValidationResult.PASSED if success else ValidationResult.FAILED

        except Exception:
            result = ValidationResult.ERROR

        finally:
            validator.teardown_test_environment()

        # Record result
        self.stream_results[validator.stream_type].add_result(
            validator.component_name, result
        )

        return result

    def benchmark_component(
        self,
        validator_name: str,
        test_func: Callable,
        iterations: int = 10,
        warmup_iterations: int = 2,
    ) -> PerformanceBenchmark:
        """Benchmark a single component.

        Args:
            validator_name: Name of the validator to use
            test_func: Test function to benchmark
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Performance benchmark
        """
        if validator_name not in self.validators:
            return PerformanceBenchmark(
                component_name=validator_name,
                stream_type=StreamType.STREAM_A,  # Default
                test_name=getattr(test_func, "__name__", str(test_func)),
                error_message="Validator not found",
            )

        validator = self.validators[validator_name]

        benchmark = PerformanceBenchmark(
            component_name=validator.component_name,
            stream_type=validator.stream_type,
            test_name=getattr(test_func, "__name__", str(test_func)),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        )

        try:
            # Warmup runs
            for _ in range(warmup_iterations):
                validator.setup_test_environment()
                with mock_streamlit_context(session=self.session):
                    with mock_streamlit_app():
                        validator.validate_functionality(test_func)
                validator.teardown_test_environment()

            # Actual benchmark
            metrics = validator.measure_performance(test_func, iterations)
            benchmark.optimized_metrics = metrics
            benchmark.passed = True

        except Exception as e:
            benchmark.error_message = str(e)
            benchmark.passed = False

        # Record benchmark
        self.stream_results[validator.stream_type].add_benchmark(benchmark)

        return benchmark

    def compare_implementations(
        self,
        validator_name: str,
        baseline_func: Callable,
        optimized_func: Callable,
        iterations: int = 10,
    ) -> PerformanceBenchmark:
        """Compare baseline vs optimized implementations.

        Args:
            validator_name: Name of the validator to use
            baseline_func: Baseline implementation
            optimized_func: Optimized implementation
            iterations: Number of iterations

        Returns:
            Comparison benchmark
        """
        if validator_name not in self.validators:
            return PerformanceBenchmark(
                component_name=validator_name,
                stream_type=StreamType.STREAM_A,  # Default
                test_name="comparison",
                error_message="Validator not found",
            )

        validator = self.validators[validator_name]

        try:
            benchmark = validator.compare_implementations(
                baseline_func, optimized_func, iterations
            )
            benchmark.calculate_improvements()

        except Exception as e:
            benchmark = PerformanceBenchmark(
                component_name=validator.component_name,
                stream_type=validator.stream_type,
                test_name="comparison",
                error_message=str(e),
            )

        # Record benchmark
        self.stream_results[validator.stream_type].add_benchmark(benchmark)

        return benchmark

    def validate_stream(
        self,
        stream_type: StreamType,
        test_scenarios: list[dict[str, Any]],
    ) -> StreamValidationResults:
        """Validate an entire stream with multiple scenarios.

        Args:
            stream_type: Stream to validate
            test_scenarios: List of test scenario configurations

        Returns:
            Stream validation results
        """
        stream_results = self.stream_results[stream_type]

        for scenario in test_scenarios:
            validator_name = scenario.get("validator")
            test_func = scenario.get("test_func")

            if not validator_name or not test_func:
                continue

            # Validate functionality
            self.validate_component(
                validator_name, test_func, **scenario.get("kwargs", {})
            )

            # Benchmark performance if requested
            if scenario.get("benchmark", False):
                self.benchmark_component(
                    validator_name, test_func, iterations=scenario.get("iterations", 10)
                )

        return stream_results

    def run_integration_tests(
        self, integration_scenarios: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run cross-stream integration tests.

        Args:
            integration_scenarios: Integration test configurations

        Returns:
            Integration test results
        """
        results = {
            "total_scenarios": len(integration_scenarios),
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "scenario_results": {},
        }

        for scenario in integration_scenarios:
            scenario_name = scenario.get("name", "unnamed")
            test_func = scenario.get("test_func")
            scenario.get("streams", [])

            try:
                # Set up integrated test environment
                with mock_streamlit_context(session=self.session):
                    with mock_streamlit_app():
                        success = test_func()

                if success:
                    results["passed_scenarios"] += 1
                    results["scenario_results"][scenario_name] = ValidationResult.PASSED
                else:
                    results["failed_scenarios"] += 1
                    results["scenario_results"][scenario_name] = ValidationResult.FAILED

            except Exception:
                results["failed_scenarios"] += 1
                results["scenario_results"][scenario_name] = ValidationResult.ERROR

        return results

    def generate_comprehensive_report(self) -> dict[str, Any]:
        """Generate comprehensive test report.

        Returns:
            Detailed test report
        """
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_streams": len(StreamType),
                "total_validators": len(self.validators),
            },
            "streams": {},
            "overall_metrics": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0,
            },
            "performance_summary": {
                "avg_time_improvement": 0.0,
                "avg_memory_improvement": 0.0,
                "functionality_preservation": 0.0,
            },
        }

        # Aggregate stream results
        total_tests = 0
        passed_tests = 0
        time_improvements = []
        memory_improvements = []
        functionality_rates = []

        for stream_type, results in self.stream_results.items():
            stream_name = stream_type.value

            report["streams"][stream_name] = {
                "total_tests": results.total_tests,
                "passed_tests": results.passed_tests,
                "failed_tests": results.failed_tests,
                "success_rate": results.success_rate,
                "avg_time_improvement": results.avg_time_improvement,
                "avg_memory_improvement": results.avg_memory_improvement,
                "functionality_preservation_rate": results.functionality_preservation_rate,
                "benchmarks": len(results.benchmarks),
            }

            total_tests += results.total_tests
            passed_tests += results.passed_tests

            if results.benchmarks:
                time_improvements.append(results.avg_time_improvement)
                memory_improvements.append(results.avg_memory_improvement)
                functionality_rates.append(results.functionality_preservation_rate)

        # Overall metrics
        report["overall_metrics"]["total_tests"] = total_tests
        report["overall_metrics"]["passed_tests"] = passed_tests
        report["overall_metrics"]["success_rate"] = (
            (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
        )

        # Performance summary
        if time_improvements:
            report["performance_summary"]["avg_time_improvement"] = sum(
                time_improvements
            ) / len(time_improvements)
        if memory_improvements:
            report["performance_summary"]["avg_memory_improvement"] = sum(
                memory_improvements
            ) / len(memory_improvements)
        if functionality_rates:
            report["performance_summary"]["functionality_preservation"] = sum(
                functionality_rates
            ) / len(functionality_rates)

        return report


# Utility functions for native component testing


def create_mock_native_components() -> dict[str, Mock]:
    """Create mock implementations of native Streamlit components."""
    components = {}

    # Stream A components
    components["progress"] = Mock(return_value=Mock())
    components["status"] = Mock(return_value=Mock())
    components["toast"] = Mock(return_value=Mock())

    # Stream B components
    components["cache_data"] = Mock(
        side_effect=lambda ttl=None, hash_funcs=None: lambda f: f
    )
    components["cache_resource"] = Mock(side_effect=lambda hash_funcs=None: lambda f: f)

    # Stream C components
    components["fragment"] = Mock(side_effect=lambda run_every=None: lambda f: f)
    components["rerun"] = Mock()

    return components


@pytest.fixture
def native_component_tester():
    """Pytest fixture for native component tester."""
    return NativeComponentTester()


@pytest.fixture
def native_component_metrics():
    """Pytest fixture for component metrics."""
    return NativeComponentMetrics()


@pytest.fixture
def performance_benchmark():
    """Pytest fixture for performance benchmark."""
    return PerformanceBenchmark(
        component_name="test_component",
        stream_type=StreamType.STREAM_A,
        test_name="test_benchmark",
    )


# Custom assertions for native component testing


def assert_functionality_preserved(
    baseline_result: Any, optimized_result: Any, tolerance: float = 0.01
) -> None:
    """Assert that functionality is preserved between implementations.

    Args:
        baseline_result: Result from baseline implementation
        optimized_result: Result from optimized implementation
        tolerance: Tolerance for numerical comparisons
    """
    if isinstance(baseline_result, dict) and isinstance(optimized_result, dict):
        # Compare dictionary results
        assert set(baseline_result.keys()) == set(optimized_result.keys()), (
            f"Result keys differ: {baseline_result.keys()} vs {optimized_result.keys()}"
        )
        for key in baseline_result:
            assert_functionality_preserved(
                baseline_result[key], optimized_result[key], tolerance
            )
    elif isinstance(baseline_result, (int, float)) and isinstance(
        optimized_result, (int, float)
    ):
        # Numerical comparison with tolerance
        assert abs(baseline_result - optimized_result) <= tolerance, (
            f"Numerical values differ beyond tolerance: {baseline_result} vs {optimized_result}"
        )
    else:
        # Direct comparison
        assert baseline_result == optimized_result, (
            f"Results differ: {baseline_result} vs {optimized_result}"
        )


def assert_performance_improvement(
    benchmark: PerformanceBenchmark,
    min_time_improvement: float = 0.0,
    min_memory_improvement: float = 0.0,
) -> None:
    """Assert that performance improvements meet minimum thresholds.

    Args:
        benchmark: Performance benchmark results
        min_time_improvement: Minimum time improvement percentage
        min_memory_improvement: Minimum memory improvement percentage
    """
    assert benchmark.passed, f"Benchmark failed: {benchmark.error_message}"

    assert benchmark.time_improvement_percent >= min_time_improvement, (
        f"Time improvement {benchmark.time_improvement_percent:.2f}% below minimum {min_time_improvement:.2f}%"
    )

    assert benchmark.memory_improvement_percent >= min_memory_improvement, (
        f"Memory improvement {benchmark.memory_improvement_percent:.2f}% below minimum {min_memory_improvement:.2f}%"
    )

    assert benchmark.functionality_preserved, (
        "Functionality not preserved in optimized implementation"
    )


def assert_stream_validation_success(
    results: StreamValidationResults, min_success_rate: float = 95.0
) -> None:
    """Assert that stream validation meets success criteria.

    Args:
        results: Stream validation results
        min_success_rate: Minimum success rate percentage
    """
    assert results.success_rate >= min_success_rate, (
        f"Stream success rate {results.success_rate:.2f}% below minimum {min_success_rate:.2f}%"
    )

    assert results.functionality_preservation_rate >= min_success_rate, (
        f"Functionality preservation {results.functionality_preservation_rate:.2f}% below minimum {min_success_rate:.2f}%"
    )
