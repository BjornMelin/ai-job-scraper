"""Base framework for Streamlit native component testing.

Provides foundational classes and utilities for testing Streamlit's native
components with focus on validation during library optimization migration.
Ensures functionality preservation and performance validation.
"""

import os
import time

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import psutil
import pytest

from tests.utils.streamlit_utils import (
    MockStreamlitSession,
    mock_streamlit_app,
    mock_streamlit_context,
)


@dataclass
class ComponentTestMetrics:
    """Metrics collected during component testing."""

    render_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    rerun_count: int = 0
    fragment_executions: int = 0
    error_count: int = 0

    # Progress component specific metrics
    progress_updates: int = 0
    status_state_changes: int = 0
    toast_notifications: int = 0

    # Performance benchmarks
    initial_load_time: float = 0.0
    subsequent_load_time: float = 0.0
    memory_peak: float = 0.0

    def __post_init__(self):
        """Initialize timestamp."""
        self.timestamp = datetime.now()


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results for before/after comparison."""

    test_name: str
    component_type: str
    before_metrics: ComponentTestMetrics = None
    after_metrics: ComponentTestMetrics = None
    improvement_percent: float = 0.0
    functionality_preserved: bool = False

    def calculate_improvement(self) -> float:
        """Calculate performance improvement percentage."""
        if not self.before_metrics or not self.after_metrics:
            return 0.0

        before_time = self.before_metrics.render_time
        after_time = self.after_metrics.render_time

        if before_time == 0:
            return 0.0

        improvement = ((before_time - after_time) / before_time) * 100
        self.improvement_percent = improvement
        return improvement


@dataclass
class StreamlitComponentState:
    """Captures the state of Streamlit components during testing."""

    session_state: dict[str, Any] = field(default_factory=dict)
    widget_states: dict[str, Any] = field(default_factory=dict)
    fragment_states: dict[str, Any] = field(default_factory=dict)
    cache_states: dict[str, Any] = field(default_factory=dict)

    # Component-specific state
    progress_values: dict[str, float] = field(default_factory=dict)
    status_states: dict[str, str] = field(default_factory=dict)
    toast_messages: list[dict[str, Any]] = field(default_factory=list)

    def capture_progress_state(self, key: str, value: float, text: str | None = None):
        """Capture progress component state."""
        self.progress_values[key] = {
            "value": value,
            "text": text,
            "timestamp": datetime.now(),
        }

    def capture_status_state(self, key: str, state: str, expanded: bool | None = None):
        """Capture status component state."""
        self.status_states[key] = {
            "state": state,
            "expanded": expanded,
            "timestamp": datetime.now(),
        }

    def capture_toast_message(self, message: str, icon: str | None = None):
        """Capture toast notification."""
        self.toast_messages.append(
            {"message": message, "icon": icon, "timestamp": datetime.now()}
        )


class StreamlitComponentValidator(ABC):
    """Abstract base class for Streamlit component validation."""

    def __init__(self, component_name: str):
        """Initialize validator with component name."""
        self.component_name = component_name
        self.metrics = ComponentTestMetrics()
        self.state = StreamlitComponentState()
        self.session = MockStreamlitSession()

    @abstractmethod
    def validate_component_behavior(self, *args, **kwargs) -> bool:
        """Validate that component behaves as expected."""

    @abstractmethod
    def measure_performance(self, *args, **kwargs) -> ComponentTestMetrics:
        """Measure component performance."""

    def setup_test_environment(self, initial_state: dict[str, Any] | None = None):
        """Set up test environment with mock Streamlit context."""
        if initial_state:
            self.session.update(initial_state)
        self.state.session_state = dict(self.session.state)

    def teardown_test_environment(self):
        """Clean up test environment."""
        self.session.clear()
        self.state = StreamlitComponentState()

    @contextmanager
    def performance_monitoring(self) -> Generator[ComponentTestMetrics, None, None]:
        """Context manager for performance monitoring during tests."""
        process = psutil.Process(os.getpid())
        start_time = time.perf_counter()
        start_memory = process.memory_info().rss

        # Reset metrics
        self.metrics = ComponentTestMetrics()

        try:
            yield self.metrics
        finally:
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss

            self.metrics.render_time = end_time - start_time
            self.metrics.memory_usage = end_memory - start_memory
            self.metrics.memory_peak = max(start_memory, end_memory)
            self.metrics.cpu_usage = process.cpu_percent()


class StreamlitNativeTester:
    """Main tester class for Streamlit native components."""

    def __init__(self):
        """Initialize native tester."""
        self.validators: dict[str, StreamlitComponentValidator] = {}
        self.benchmarks: list[PerformanceBenchmark] = []
        self.session = MockStreamlitSession()

    def register_validator(self, name: str, validator: StreamlitComponentValidator):
        """Register a component validator."""
        self.validators[name] = validator

    def run_component_validation(
        self, component_name: str, test_func: Callable, *args, **kwargs
    ) -> bool:
        """Run validation for a specific component."""
        if component_name not in self.validators:
            raise ValueError(f"No validator registered for {component_name}")

        validator = self.validators[component_name]
        validator.setup_test_environment(kwargs.get("initial_state"))

        try:
            with mock_streamlit_context(session=self.session):
                with mock_streamlit_app():
                    return validator.validate_component_behavior(
                        test_func, *args, **kwargs
                    )
        finally:
            validator.teardown_test_environment()

    def benchmark_component_performance(
        self,
        component_name: str,
        test_func: Callable,
        iterations: int = 10,
        *args,
        **kwargs,
    ) -> PerformanceBenchmark:
        """Benchmark component performance."""
        if component_name not in self.validators:
            raise ValueError(f"No validator registered for {component_name}")

        validator = self.validators[component_name]
        benchmark = PerformanceBenchmark(
            test_name=test_func.__name__
            if hasattr(test_func, "__name__")
            else str(test_func),
            component_type=component_name,
        )

        # Run multiple iterations for more accurate benchmarks
        total_metrics = ComponentTestMetrics()

        for _ in range(iterations):
            validator.setup_test_environment(kwargs.get("initial_state"))

            try:
                with validator.performance_monitoring() as metrics:
                    with mock_streamlit_context(session=self.session):
                        with mock_streamlit_app():
                            validator.validate_component_behavior(
                                test_func, *args, **kwargs
                            )

                # Aggregate metrics
                total_metrics.render_time += metrics.render_time
                total_metrics.memory_usage += metrics.memory_usage
                total_metrics.cpu_usage += metrics.cpu_usage
                total_metrics.memory_peak = max(
                    total_metrics.memory_peak, metrics.memory_peak
                )

            finally:
                validator.teardown_test_environment()

        # Average the metrics
        avg_metrics = ComponentTestMetrics(
            render_time=total_metrics.render_time / iterations,
            memory_usage=total_metrics.memory_usage / iterations,
            cpu_usage=total_metrics.cpu_usage / iterations,
            memory_peak=total_metrics.memory_peak,
        )

        benchmark.after_metrics = avg_metrics
        self.benchmarks.append(benchmark)
        return benchmark

    def compare_implementation_performance(
        self,
        component_name: str,
        old_implementation: Callable,
        new_implementation: Callable,
        iterations: int = 10,
        *args,
        **kwargs,
    ) -> PerformanceBenchmark:
        """Compare performance between old and new implementations."""
        # Benchmark old implementation
        old_benchmark = self.benchmark_component_performance(
            component_name, old_implementation, iterations, *args, **kwargs
        )

        # Benchmark new implementation
        new_benchmark = self.benchmark_component_performance(
            component_name, new_implementation, iterations, *args, **kwargs
        )

        # Create comparison benchmark
        comparison = PerformanceBenchmark(
            test_name=f"{old_implementation.__name__}_vs_{new_implementation.__name__}",
            component_type=component_name,
            before_metrics=old_benchmark.after_metrics,
            after_metrics=new_benchmark.after_metrics,
        )

        comparison.calculate_improvement()
        self.benchmarks.append(comparison)
        return comparison

    def validate_functionality_preservation(
        self,
        component_name: str,
        old_implementation: Callable,
        new_implementation: Callable,
        test_scenarios: list[dict[str, Any]],
        *args,
        **kwargs,
    ) -> bool:
        """Validate that new implementation preserves all functionality."""
        validator = self.validators.get(component_name)
        if not validator:
            raise ValueError(f"No validator registered for {component_name}")

        all_scenarios_pass = True

        for scenario in test_scenarios:
            # Test old implementation
            old_state = self._capture_component_state(
                old_implementation, scenario, *args, **kwargs
            )

            # Test new implementation
            new_state = self._capture_component_state(
                new_implementation, scenario, *args, **kwargs
            )

            # Compare states
            if not self._states_equivalent(old_state, new_state):
                all_scenarios_pass = False
                print(f"Scenario {scenario} failed: State mismatch")

        return all_scenarios_pass

    def _capture_component_state(
        self, implementation: Callable, scenario: dict[str, Any], *args, **kwargs
    ) -> StreamlitComponentState:
        """Capture component state after execution."""
        state = StreamlitComponentState()

        with mock_streamlit_context(session=self.session):
            with mock_streamlit_app():
                # Set up scenario
                for key, value in scenario.items():
                    self.session[key] = value

                # Execute implementation
                implementation(*args, **kwargs)

                # Capture final state
                state.session_state = dict(self.session.state)
                state.widget_states = dict(self.session.widgets)

        return state

    def _states_equivalent(
        self, state1: StreamlitComponentState, state2: StreamlitComponentState
    ) -> bool:
        """Compare two component states for equivalence."""
        # Compare session states (excluding internal Streamlit keys)
        filtered_state1 = {
            k: v for k, v in state1.session_state.items() if not k.startswith("_")
        }
        filtered_state2 = {
            k: v for k, v in state2.session_state.items() if not k.startswith("_")
        }

        return filtered_state1 == filtered_state2

    def generate_test_report(self) -> dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "test_summary": {
                "total_benchmarks": len(self.benchmarks),
                "validators_registered": list(self.validators.keys()),
                "timestamp": datetime.now().isoformat(),
            },
            "performance_improvements": [],
            "functionality_validation": [],
            "recommendations": [],
        }

        # Analyze benchmarks
        for benchmark in self.benchmarks:
            if benchmark.before_metrics and benchmark.after_metrics:
                improvement_data = {
                    "test_name": benchmark.test_name,
                    "component_type": benchmark.component_type,
                    "render_time_improvement": benchmark.improvement_percent,
                    "memory_improvement": (
                        (
                            benchmark.before_metrics.memory_usage
                            - benchmark.after_metrics.memory_usage
                        )
                        / benchmark.before_metrics.memory_usage
                        * 100
                        if benchmark.before_metrics.memory_usage > 0
                        else 0
                    ),
                    "functionality_preserved": benchmark.functionality_preserved,
                }
                report["performance_improvements"].append(improvement_data)

        return report


# Utility functions for native component testing


def mock_streamlit_progress(progress_tracker: dict[str, Any] | None = None):
    """Mock st.progress with state tracking."""
    if progress_tracker is None:
        progress_tracker = {}

    def progress_mock(value: float, text: str | None = None):
        progress_tracker["value"] = value
        progress_tracker["text"] = text
        progress_tracker["timestamp"] = datetime.now()
        return Mock()

    return progress_mock


def mock_streamlit_status(status_tracker: dict[str, Any] | None = None):
    """Mock st.status with state tracking."""
    if status_tracker is None:
        status_tracker = {}

    class MockStatus:
        def __init__(
            self, label: str, expanded: bool | None = None, state: str = "running"
        ):
            status_tracker["label"] = label
            status_tracker["expanded"] = expanded
            status_tracker["state"] = state
            status_tracker["timestamp"] = datetime.now()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update(
            self,
            label: str | None = None,
            state: str | None = None,
            expanded: bool | None = None,
        ):
            if label:
                status_tracker["label"] = label
            if state:
                status_tracker["state"] = state
            if expanded is not None:
                status_tracker["expanded"] = expanded
            status_tracker["last_update"] = datetime.now()

    return MockStatus


def mock_streamlit_toast(toast_tracker: list[dict[str, Any]] | None = None):
    """Mock st.toast with message tracking."""
    if toast_tracker is None:
        toast_tracker = []

    def toast_mock(message: str, icon: str | None = None):
        toast_tracker.append(
            {"message": message, "icon": icon, "timestamp": datetime.now()}
        )
        return Mock()

    return toast_mock


def mock_streamlit_fragment(fragment_tracker: dict[str, Any] | None = None):
    """Mock st.fragment with execution tracking."""
    if fragment_tracker is None:
        fragment_tracker = {}

    def fragment_decorator(run_every: str | float | None = None):
        def decorator(func: Callable):
            fragment_tracker["function"] = func.__name__
            fragment_tracker["run_every"] = run_every
            fragment_tracker["executions"] = 0

            def wrapper(*args, **kwargs):
                fragment_tracker["executions"] += 1
                fragment_tracker["last_execution"] = datetime.now()
                return func(*args, **kwargs)

            wrapper.run_every = run_every
            return wrapper

        return decorator

    return fragment_decorator


@pytest.fixture
def streamlit_native_tester():
    """Pytest fixture for Streamlit native component tester."""
    return StreamlitNativeTester()


@pytest.fixture
def component_test_metrics():
    """Pytest fixture for component test metrics."""
    return ComponentTestMetrics()


@pytest.fixture
def performance_benchmark():
    """Pytest fixture for performance benchmark."""
    return PerformanceBenchmark("test", "component")
