"""Base validation framework for Week 1 stream achievements.

This module provides the foundational classes and utilities for validating
the performance claims and functionality improvements of Week 1 streams.
"""

import time

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import psutil
import pytest


class StreamAchievement(Enum):
    """Week 1 stream achievement types."""

    STREAM_A_CODE_REDUCTION = "stream_a_code_reduction"  # 96% reduction
    STREAM_B_CACHE_PERFORMANCE = "stream_b_cache_performance"  # 100.8x improvement
    STREAM_C_FRAGMENT_PERFORMANCE = "stream_c_fragment_performance"  # 30% improvement


@dataclass
class ValidationMetrics:
    """Comprehensive metrics for stream validation."""

    # Performance metrics
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    peak_memory_mb: float = 0.0

    # Stream-specific metrics
    code_lines_before: int = 0
    code_lines_after: int = 0
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    fragment_update_frequency: float = 0.0
    page_rerun_count: int = 0

    # Quality metrics
    functionality_preserved: bool = False
    error_rate: float = 0.0
    test_coverage: float = 0.0

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def calculate_code_reduction_percent(self) -> float:
        """Calculate code reduction percentage."""
        if self.code_lines_before == 0:
            return 0.0
        reduction = self.code_lines_before - self.code_lines_after
        return (reduction / self.code_lines_before) * 100

    def calculate_performance_improvement(self, baseline_time_ms: float) -> float:
        """Calculate performance improvement multiplier."""
        if self.execution_time_ms == 0 or baseline_time_ms == 0:
            return 0.0
        return baseline_time_ms / self.execution_time_ms

    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.end_time = datetime.now()


@dataclass
class ValidationResult:
    """Result of stream validation test."""

    stream: StreamAchievement
    test_name: str
    passed: bool
    metrics: ValidationMetrics
    error_message: str | None = None
    baseline_metrics: ValidationMetrics | None = None
    improvement_factor: float = 0.0
    meets_target: bool = False

    def __post_init__(self):
        """Calculate improvement factors after initialization."""
        if self.baseline_metrics and self.metrics:
            if self.baseline_metrics.execution_time_ms > 0:
                self.improvement_factor = (
                    self.metrics.calculate_performance_improvement(
                        self.baseline_metrics.execution_time_ms
                    )
                )

            # Check if targets are met based on stream type
            if self.stream == StreamAchievement.STREAM_A_CODE_REDUCTION:
                self.meets_target = (
                    self.metrics.calculate_code_reduction_percent() >= 90.0
                )
            elif self.stream == StreamAchievement.STREAM_B_CACHE_PERFORMANCE:
                self.meets_target = self.improvement_factor >= 50.0  # 50x minimum
            elif self.stream == StreamAchievement.STREAM_C_FRAGMENT_PERFORMANCE:
                self.meets_target = self.improvement_factor >= 1.25  # 25% minimum


class BaseStreamValidator(ABC):
    """Base class for stream validation."""

    def __init__(self, stream: StreamAchievement, test_name: str):
        """Initialize validator.

        Args:
            stream: The stream achievement being validated
            test_name: Name of the test being run
        """
        self.stream = stream
        self.test_name = test_name
        self.metrics = ValidationMetrics()

    @abstractmethod
    def validate_functionality(self, *args, **kwargs) -> bool:
        """Validate that functionality is preserved.

        Returns:
            True if functionality is preserved
        """

    @abstractmethod
    def measure_performance(
        self, test_func: Callable, iterations: int = 10
    ) -> ValidationMetrics:
        """Measure performance of the implementation.

        Args:
            test_func: Function to benchmark
            iterations: Number of iterations to run

        Returns:
            Performance metrics
        """

    @abstractmethod
    def compare_with_baseline(
        self, baseline_func: Callable, optimized_func: Callable
    ) -> ValidationResult:
        """Compare optimized implementation with baseline.

        Args:
            baseline_func: Original implementation
            optimized_func: Optimized implementation

        Returns:
            Validation result with comparison metrics
        """

    @contextmanager
    def performance_monitoring(self):
        """Context manager for performance monitoring."""
        process = psutil.Process()

        # Reset metrics
        self.metrics = ValidationMetrics()
        start_time = time.perf_counter()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield self.metrics
        finally:
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            self.metrics.execution_time_ms = (end_time - start_time) * 1000
            self.metrics.memory_usage_mb = end_memory - start_memory
            self.metrics.peak_memory_mb = max(start_memory, end_memory)

            try:
                self.metrics.cpu_usage_percent = process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.metrics.cpu_usage_percent = 0.0

            self.metrics.finalize()


class MockStreamlitEnvironment:
    """Mock Streamlit environment for testing."""

    def __init__(self):
        """Initialize mock environment."""
        self.session_state = {}
        self.fragments = {}
        self.cache_data_calls = []
        self.cache_resource_calls = []
        self.progress_calls = []
        self.status_calls = []
        self.toast_calls = []
        self.rerun_calls = []

    def reset(self):
        """Reset the mock environment."""
        self.session_state.clear()
        self.fragments.clear()
        self.cache_data_calls.clear()
        self.cache_resource_calls.clear()
        self.progress_calls.clear()
        self.status_calls.clear()
        self.toast_calls.clear()
        self.rerun_calls.clear()

    def mock_cache_data(self, ttl=None, max_entries=None, show_spinner=None):
        """Mock st.cache_data decorator."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                call_info = {
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                    "ttl": ttl,
                    "timestamp": datetime.now(),
                }
                self.cache_data_calls.append(call_info)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def mock_cache_resource(self):
        """Mock st.cache_resource decorator."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                call_info = {
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                    "timestamp": datetime.now(),
                }
                self.cache_resource_calls.append(call_info)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def mock_fragment(self, run_every=None):
        """Mock st.fragment decorator."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                fragment_id = f"{func.__name__}_{id(func)}"
                self.fragments[fragment_id] = {
                    "function": func,
                    "run_every": run_every,
                    "last_run": datetime.now(),
                }
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def mock_progress(self, value, text=None):
        """Mock st.progress."""
        call_info = {"value": value, "text": text, "timestamp": datetime.now()}
        self.progress_calls.append(call_info)
        return MagicMock()

    def mock_status(self, label, state="running", expanded=True):
        """Mock st.status."""
        call_info = {
            "label": label,
            "state": state,
            "expanded": expanded,
            "timestamp": datetime.now(),
        }
        self.status_calls.append(call_info)
        return MagicMock()

    def mock_toast(self, message, icon=None):
        """Mock st.toast."""
        call_info = {"message": message, "icon": icon, "timestamp": datetime.now()}
        self.toast_calls.append(call_info)
        return MagicMock()

    def mock_rerun(self, scope="app"):
        """Mock st.rerun."""
        call_info = {"scope": scope, "timestamp": datetime.now()}
        self.rerun_calls.append(call_info)


def count_code_lines(
    file_path: Path, exclude_comments: bool = True, exclude_empty: bool = True
) -> int:
    """Count lines of code in a Python file.

    Args:
        file_path: Path to the Python file
        exclude_comments: Whether to exclude comment lines
        exclude_empty: Whether to exclude empty lines

    Returns:
        Number of code lines
    """
    if not file_path.exists():
        return 0

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    code_lines = 0
    in_multiline_string = False
    multiline_delimiter = None

    for line in lines:
        stripped = line.strip()

        # Skip empty lines if requested
        if exclude_empty and not stripped:
            continue

        # Handle multiline strings
        if not in_multiline_string:
            # Check for start of multiline string
            if '"""' in stripped or "'''" in stripped:
                if stripped.count('"""') % 2 == 1:
                    in_multiline_string = True
                    multiline_delimiter = '"""'
                elif stripped.count("'''") % 2 == 1:
                    in_multiline_string = True
                    multiline_delimiter = "'''"
        else:
            # Check for end of multiline string
            if multiline_delimiter in stripped:
                if stripped.count(multiline_delimiter) % 2 == 1:
                    in_multiline_string = False
                    multiline_delimiter = None
            continue

        # Skip comments if requested and not in multiline string
        if exclude_comments and not in_multiline_string and stripped.startswith("#"):
            continue

        # Count this as a code line
        if not in_multiline_string or not exclude_comments:
            code_lines += 1

    return code_lines


@pytest.fixture
def mock_streamlit():
    """Pytest fixture for mock Streamlit environment."""
    env = MockStreamlitEnvironment()

    with patch.multiple(
        "streamlit",
        cache_data=env.mock_cache_data,
        cache_resource=env.mock_cache_resource,
        fragment=env.mock_fragment,
        progress=env.mock_progress,
        status=env.mock_status,
        toast=env.mock_toast,
        rerun=env.mock_rerun,
    ):
        yield env


class Week1ValidationSuite:
    """Comprehensive validation suite for Week 1 achievements."""

    def __init__(self):
        """Initialize validation suite."""
        self.results: list[ValidationResult] = []
        self.mock_env = MockStreamlitEnvironment()

    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)

    def get_stream_results(self, stream: StreamAchievement) -> list[ValidationResult]:
        """Get results for specific stream."""
        return [r for r in self.results if r.stream == stream]

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "failed_tests": sum(1 for r in self.results if not r.passed),
            "targets_met": sum(1 for r in self.results if r.meets_target),
            "streams": {},
        }

        for stream in StreamAchievement:
            stream_results = self.get_stream_results(stream)
            if stream_results:
                report["streams"][stream.value] = {
                    "total_tests": len(stream_results),
                    "passed_tests": sum(1 for r in stream_results if r.passed),
                    "targets_met": sum(1 for r in stream_results if r.meets_target),
                    "avg_improvement_factor": sum(
                        r.improvement_factor for r in stream_results
                    )
                    / len(stream_results),
                    "test_results": [
                        {
                            "test_name": r.test_name,
                            "passed": r.passed,
                            "meets_target": r.meets_target,
                            "improvement_factor": r.improvement_factor,
                            "error_message": r.error_message,
                        }
                        for r in stream_results
                    ],
                }

        return report


# Utility functions for validation


def assert_code_reduction_target(
    before_lines: int, after_lines: int, target_percent: float = 90.0
):
    """Assert that code reduction meets target percentage."""
    if before_lines == 0:
        pytest.fail("No baseline code lines to compare against")

    reduction_percent = ((before_lines - after_lines) / before_lines) * 100
    assert reduction_percent >= target_percent, (
        f"Code reduction {reduction_percent:.1f}% below target {target_percent}%"
    )


def assert_performance_improvement(
    baseline_time_ms: float, optimized_time_ms: float, target_factor: float
):
    """Assert that performance improvement meets target factor."""
    if optimized_time_ms == 0:
        pytest.fail("Zero execution time - cannot calculate improvement")

    improvement_factor = baseline_time_ms / optimized_time_ms
    assert improvement_factor >= target_factor, (
        f"Performance improvement {improvement_factor:.1f}x below target {target_factor}x"
    )


def assert_functionality_preserved(
    baseline_result: Any, optimized_result: Any, tolerance: float = 0.01
):
    """Assert that functionality is preserved between implementations."""
    if isinstance(baseline_result, (int, float)) and isinstance(
        optimized_result, (int, float)
    ):
        assert abs(baseline_result - optimized_result) <= tolerance, (
            f"Numerical results differ beyond tolerance: {baseline_result} vs {optimized_result}"
        )
    elif isinstance(baseline_result, dict) and isinstance(optimized_result, dict):
        assert set(baseline_result.keys()) == set(optimized_result.keys()), (
            f"Result keys differ: {baseline_result.keys()} vs {optimized_result.keys()}"
        )
        for key in baseline_result:
            assert_functionality_preserved(
                baseline_result[key], optimized_result[key], tolerance
            )
    else:
        assert baseline_result == optimized_result, (
            f"Results differ: {baseline_result} vs {optimized_result}"
        )
