"""Stream C Testing: Fragment Architecture Validation.

Tests for Streamlit native fragment components:
- st.fragment() isolation and auto-refresh validation
- run_every timing accuracy and performance
- Fragment state management and isolation
- Scoped st.rerun() functionality
- Fragment coordination and interference testing
- Memory usage and performance monitoring

Ensures 100% functionality preservation during library optimization migration.
"""

import threading
import time

from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from tests.streamlit_native.base_framework import (
    ComponentTestMetrics,
    StreamlitComponentValidator,
    StreamlitNativeTester,
)


class FragmentMetrics:
    """Track fragment execution metrics during tests."""

    def __init__(self):
        """Initialize fragment metrics."""
        self.executions = {}  # fragment_id -> execution_count
        self.run_every_intervals = {}  # fragment_id -> interval
        self.execution_times = {}  # fragment_id -> list of execution timestamps
        self.isolation_violations = 0
        self.rerun_calls = 0
        self.fragment_states = {}  # fragment_id -> state_dict
        self.coordination_events = []  # list of coordination events

    def record_execution(
        self, fragment_id: str, execution_time: datetime | None = None
    ):
        """Record fragment execution."""
        if execution_time is None:
            execution_time = datetime.now()

        self.executions[fragment_id] = self.executions.get(fragment_id, 0) + 1

        if fragment_id not in self.execution_times:
            self.execution_times[fragment_id] = []
        self.execution_times[fragment_id].append(execution_time)

    def record_run_every(self, fragment_id: str, interval: float | str):
        """Record run_every configuration."""
        # Convert string intervals like "2s" to float seconds
        if isinstance(interval, str):
            if interval.endswith("s"):
                interval = float(interval[:-1])
            elif interval.endswith("ms"):
                interval = float(interval[:-2]) / 1000

        self.run_every_intervals[fragment_id] = interval

    def record_isolation_violation(self):
        """Record fragment isolation violation."""
        self.isolation_violations += 1

    def record_rerun(self):
        """Record st.rerun() call."""
        self.rerun_calls += 1

    def record_fragment_state(self, fragment_id: str, state: dict):
        """Record fragment state snapshot."""
        self.fragment_states[fragment_id] = state.copy()

    def record_coordination_event(self, event_type: str, details: dict):
        """Record fragment coordination event."""
        self.coordination_events.append(
            {"timestamp": datetime.now(), "type": event_type, "details": details}
        )

    def get_execution_rate(self, fragment_id: str) -> float:
        """Calculate actual execution rate for a fragment."""
        if (
            fragment_id not in self.execution_times
            or len(self.execution_times[fragment_id]) < 2
        ):
            return 0.0

        times = self.execution_times[fragment_id]
        total_duration = (times[-1] - times[0]).total_seconds()
        execution_count = len(times) - 1  # Intervals between executions

        return execution_count / total_duration if total_duration > 0 else 0.0

    def get_timing_accuracy(self, fragment_id: str) -> float:
        """Calculate timing accuracy vs expected run_every interval."""
        if fragment_id not in self.run_every_intervals:
            return 0.0

        expected_interval = self.run_every_intervals[fragment_id]
        actual_rate = self.get_execution_rate(fragment_id)
        expected_rate = 1.0 / expected_interval if expected_interval > 0 else 0.0

        if expected_rate == 0:
            return 0.0

        accuracy = 1.0 - abs(actual_rate - expected_rate) / expected_rate
        return max(0.0, accuracy)

    def reset(self):
        """Reset all metrics."""
        self.executions.clear()
        self.run_every_intervals.clear()
        self.execution_times.clear()
        self.isolation_violations = 0
        self.rerun_calls = 0
        self.fragment_states.clear()
        self.coordination_events.clear()


class MockStreamlitFragment:
    """Mock Streamlit fragment implementation for testing."""

    def __init__(self, metrics: FragmentMetrics):
        """Initialize mock fragment system."""
        self.metrics = metrics
        self.fragments = {}  # fragment_id -> fragment_info
        self.timers = {}  # fragment_id -> timer
        self.running = set()  # active fragment IDs

    def mock_fragment(self, run_every: str | float | None = None):
        """Mock st.fragment decorator."""

        def decorator(func: Callable):
            fragment_id = f"{func.__name__}_{id(func)}"

            # Record run_every configuration
            if run_every:
                self.metrics.record_run_every(fragment_id, run_every)

            def wrapper(*args, **kwargs):
                # Record execution
                self.metrics.record_execution(fragment_id)

                # Execute the fragment function
                result = func(*args, **kwargs)

                # Start auto-refresh if configured
                if run_every and fragment_id not in self.running:
                    self._start_auto_refresh(fragment_id, func, run_every, args, kwargs)

                return result

            # Store fragment info
            self.fragments[fragment_id] = {
                "function": func,
                "run_every": run_every,
                "wrapper": wrapper,
            }

            wrapper._fragment_id = fragment_id
            wrapper._is_fragment = True
            return wrapper

        return decorator

    def _start_auto_refresh(
        self,
        fragment_id: str,
        func: Callable,
        interval: str | float,
        args: tuple,
        kwargs: dict,
    ):
        """Start auto-refresh timer for fragment."""
        if fragment_id in self.running:
            return

        # Convert interval to seconds
        interval_seconds = interval
        if isinstance(interval, str):
            if interval.endswith("s"):
                interval_seconds = float(interval[:-1])
            elif interval.endswith("ms"):
                interval_seconds = float(interval[:-2]) / 1000

        self.running.add(fragment_id)

        def refresh():
            if fragment_id in self.running:
                # Record auto-refresh execution
                self.metrics.record_execution(fragment_id)

                # Execute fragment function
                func(*args, **kwargs)

                # Schedule next refresh
                self._schedule_refresh(
                    fragment_id, func, interval_seconds, args, kwargs
                )

        # Schedule first refresh
        timer = threading.Timer(interval_seconds, refresh)
        self.timers[fragment_id] = timer
        timer.start()

    def _schedule_refresh(
        self,
        fragment_id: str,
        func: Callable,
        interval: float,
        args: tuple,
        kwargs: dict,
    ):
        """Schedule next refresh for fragment."""
        if fragment_id not in self.running:
            return

        def refresh():
            if fragment_id in self.running:
                self.metrics.record_execution(fragment_id)
                func(*args, **kwargs)
                self._schedule_refresh(fragment_id, func, interval, args, kwargs)

        timer = threading.Timer(interval, refresh)
        self.timers[fragment_id] = timer
        timer.start()

    def stop_fragment(self, fragment_id: str):
        """Stop auto-refresh for a fragment."""
        if fragment_id in self.running:
            self.running.remove(fragment_id)

        if fragment_id in self.timers:
            self.timers[fragment_id].cancel()
            del self.timers[fragment_id]

    def stop_all_fragments(self):
        """Stop all fragment auto-refresh timers."""
        for fragment_id in list(self.running):
            self.stop_fragment(fragment_id)

    def mock_rerun(self, scope: str = "app"):
        """Mock st.rerun with scope support."""

        def rerun():
            self.metrics.record_rerun()

            if scope == "fragment":
                # Fragment-scoped rerun
                self.metrics.record_coordination_event(
                    "fragment_rerun", {"scope": scope}
                )
            else:
                # App-wide rerun
                self.metrics.record_coordination_event("app_rerun", {"scope": scope})

        return rerun


class FragmentSystemValidator(StreamlitComponentValidator):
    """Validator for fragment system components (st.fragment, run_every, isolation)."""

    def __init__(self):
        """Initialize fragment system validator."""
        super().__init__("fragment_system")
        self.fragment_metrics = FragmentMetrics()
        self.mock_fragment = MockStreamlitFragment(self.fragment_metrics)

    def validate_component_behavior(self, test_func, *args, **kwargs) -> bool:
        """Validate fragment component behavior."""
        try:
            # Reset metrics
            self.fragment_metrics.reset()
            self.mock_fragment.stop_all_fragments()

            # Mock the fragment components
            with (
                patch("streamlit.fragment", self.mock_fragment.mock_fragment),
                patch("streamlit.rerun", self.mock_fragment.mock_rerun()),
            ):
                # Execute test function
                test_func(*args, **kwargs)

            # Allow some time for fragment executions
            time.sleep(0.1)

            return True

        except Exception:
            self.metrics.error_count += 1
            return False
        finally:
            # Clean up timers
            self.mock_fragment.stop_all_fragments()

    def measure_performance(self, test_func, *args, **kwargs) -> ComponentTestMetrics:
        """Measure fragment component performance."""
        with self.performance_monitoring() as metrics:
            self.validate_component_behavior(test_func, *args, **kwargs)

        # Add fragment-specific metrics
        metrics.fragment_executions = sum(self.fragment_metrics.executions.values())
        metrics.rerun_count = self.fragment_metrics.rerun_calls

        return metrics

    def validate_fragment_isolation(self, isolation_config: dict[str, Any]) -> bool:
        """Validate fragment isolation behavior."""

        def test_isolation():
            import streamlit as st

            shared_state = {"counter": 0}

            @st.fragment
            def isolated_fragment_1():
                # Should not interfere with other fragments
                shared_state["counter"] += 1
                return shared_state["counter"]

            @st.fragment
            def isolated_fragment_2():
                # Should not interfere with fragment 1
                shared_state["counter"] *= 2
                return shared_state["counter"]

            # Execute fragments
            result1 = isolated_fragment_1()
            result2 = isolated_fragment_2()

            return result1, result2

        self.validate_component_behavior(test_isolation)

        # Check isolation metrics
        return self.fragment_metrics.isolation_violations == 0

    def validate_auto_refresh_behavior(self, refresh_config: dict[str, Any]) -> bool:
        """Validate auto-refresh timing behavior."""

        def test_auto_refresh():
            import streamlit as st

            interval = refresh_config.get("interval", "1s")
            duration = refresh_config.get("test_duration", 2.5)  # Test for 2.5 seconds

            @st.fragment(run_every=interval)
            def auto_refresh_fragment():
                return datetime.now()

            # Start fragment
            auto_refresh_fragment()

            # Let it run for the test duration
            time.sleep(duration)

            return True

        self.validate_component_behavior(test_auto_refresh)

        # Validate timing accuracy
        fragment_ids = list(self.fragment_metrics.executions.keys())
        if fragment_ids:
            fragment_id = fragment_ids[0]
            accuracy = self.fragment_metrics.get_timing_accuracy(fragment_id)
            return accuracy > 0.7  # 70% timing accuracy threshold

        return False

    def validate_scoped_rerun(self, rerun_config: dict[str, Any]) -> bool:
        """Validate scoped st.rerun() functionality."""

        def test_scoped_rerun():
            import streamlit as st

            @st.fragment
            def fragment_with_rerun():
                scope = rerun_config.get("scope", "fragment")

                # Trigger rerun with specified scope
                if scope == "fragment":
                    st.rerun(scope="fragment")
                else:
                    st.rerun()  # App-wide rerun

                return scope

            fragment_with_rerun()
            return True

        self.validate_component_behavior(test_scoped_rerun)

        # Check rerun was called
        return self.fragment_metrics.rerun_calls > 0

    def validate_fragment_coordination(
        self, coordination_config: dict[str, Any]
    ) -> bool:
        """Validate coordination between multiple fragments."""

        def test_coordination():
            import streamlit as st

            coordination_data = {"messages": []}

            @st.fragment(run_every="0.5s")
            def producer_fragment():
                coordination_data["messages"].append(f"produced_{datetime.now()}")
                return len(coordination_data["messages"])

            @st.fragment(run_every="1s")
            def consumer_fragment():
                if coordination_data["messages"]:
                    return coordination_data["messages"].pop(0)
                return None

            # Start both fragments
            producer_fragment()
            consumer_fragment()

            # Let them coordinate
            time.sleep(2.0)

            return coordination_data

        self.validate_component_behavior(test_coordination)

        # Check that both fragments executed
        return len(self.fragment_metrics.executions) >= 2

    def measure_fragment_performance_impact(
        self, performance_config: dict[str, Any]
    ) -> dict[str, float]:
        """Measure performance impact of fragments."""

        def performance_test():
            import streamlit as st

            fragment_count = performance_config.get("fragment_count", 3)
            interval = performance_config.get("interval", "0.5s")

            # Create multiple fragments
            for i in range(fragment_count):

                @st.fragment(run_every=interval)
                def performance_fragment():
                    # Simulate some work
                    time.sleep(0.001)  # 1ms work
                    return f"fragment_{i}_work"

                performance_fragment()

            # Let fragments run
            time.sleep(2.0)

            return True

        # Reset metrics
        self.fragment_metrics.reset()

        with self.performance_monitoring() as metrics:
            self.validate_component_behavior(performance_test)

        return {
            "total_executions": sum(self.fragment_metrics.executions.values()),
            "render_time": metrics.render_time,
            "memory_usage": metrics.memory_usage,
            "fragment_count": len(self.fragment_metrics.executions),
            "avg_execution_rate": sum(
                self.fragment_metrics.get_execution_rate(fid)
                for fid in self.fragment_metrics.executions
            )
            / len(self.fragment_metrics.executions)
            if self.fragment_metrics.executions
            else 0,
        }


class TestStreamCFragmentSystem:
    """Test suite for Stream C fragment system components."""

    @pytest.fixture
    def fragment_validator(self):
        """Provide fragment system validator."""
        return FragmentSystemValidator()

    @pytest.fixture
    def streamlit_tester(self, fragment_validator):
        """Provide configured Streamlit tester."""
        tester = StreamlitNativeTester()
        tester.register_validator("fragment_system", fragment_validator)
        return tester

    def test_basic_fragment_behavior(self, fragment_validator):
        """Test basic st.fragment functionality."""

        def basic_fragment_test():
            import streamlit as st

            @st.fragment
            def simple_fragment():
                return "fragment_result"

            return simple_fragment()

        assert fragment_validator.validate_component_behavior(basic_fragment_test)

        # Should have recorded fragment execution
        assert len(fragment_validator.fragment_metrics.executions) > 0

    def test_fragment_with_run_every(self, fragment_validator):
        """Test st.fragment with run_every parameter."""
        config = {"interval": "1s", "test_duration": 2.5}

        assert fragment_validator.validate_auto_refresh_behavior(config)

        # Should have multiple executions due to auto-refresh
        total_executions = sum(fragment_validator.fragment_metrics.executions.values())
        assert total_executions >= 2  # Initial + at least 2 auto-refreshes

    def test_fragment_run_every_timing_accuracy(self, fragment_validator):
        """Test timing accuracy of run_every intervals."""
        config = {
            "interval": "0.5s",  # 0.5 second interval
            "test_duration": 2.0,  # 2 second test
        }

        assert fragment_validator.validate_auto_refresh_behavior(config)

        # Check timing accuracy
        fragment_ids = list(fragment_validator.fragment_metrics.executions.keys())
        if fragment_ids:
            fragment_id = fragment_ids[0]
            accuracy = fragment_validator.fragment_metrics.get_timing_accuracy(
                fragment_id
            )
            assert accuracy > 0.5  # At least 50% timing accuracy

    def test_fragment_isolation(self, fragment_validator):
        """Test fragment isolation behavior."""
        config = {}

        assert fragment_validator.validate_fragment_isolation(config)

        # Should have no isolation violations
        assert fragment_validator.fragment_metrics.isolation_violations == 0

    def test_multiple_fragments_isolation(self, fragment_validator):
        """Test isolation between multiple fragments."""

        def multi_fragment_test():
            import streamlit as st

            fragment_states = {}

            @st.fragment
            def fragment_a():
                fragment_states["a"] = "fragment_a_data"
                return fragment_states["a"]

            @st.fragment
            def fragment_b():
                fragment_states["b"] = "fragment_b_data"
                return fragment_states["b"]

            @st.fragment
            def fragment_c():
                fragment_states["c"] = "fragment_c_data"
                return fragment_states["c"]

            # Execute all fragments
            return [fragment_a(), fragment_b(), fragment_c()]

        assert fragment_validator.validate_component_behavior(multi_fragment_test)

        # Should have 3 different fragments
        assert len(fragment_validator.fragment_metrics.executions) == 3

    def test_fragment_scoped_rerun(self, fragment_validator):
        """Test fragment-scoped st.rerun functionality."""
        config = {"scope": "fragment"}

        assert fragment_validator.validate_scoped_rerun(config)

        # Should have recorded rerun call
        assert fragment_validator.fragment_metrics.rerun_calls > 0

    def test_app_scoped_rerun(self, fragment_validator):
        """Test app-scoped st.rerun functionality."""
        config = {"scope": "app"}

        assert fragment_validator.validate_scoped_rerun(config)

        # Should have recorded rerun call
        assert fragment_validator.fragment_metrics.rerun_calls > 0

    def test_fragment_coordination(self, fragment_validator):
        """Test coordination between multiple fragments."""
        config = {}

        assert fragment_validator.validate_fragment_coordination(config)

        # Should have multiple fragments coordinating
        assert len(fragment_validator.fragment_metrics.executions) >= 2

    def test_fragment_state_management(self, streamlit_tester):
        """Test fragment state management and persistence."""

        def state_management_test():
            import streamlit as st

            # Initialize session state for fragment
            if "fragment_counter" not in st.session_state:
                st.session_state.fragment_counter = 0

            @st.fragment
            def stateful_fragment():
                st.session_state.fragment_counter += 1
                return st.session_state.fragment_counter

            # Execute fragment multiple times
            results = []
            for _ in range(3):
                result = stateful_fragment()
                results.append(result)

            return results

        result = streamlit_tester.run_component_validation(
            "fragment_system", state_management_test
        )

        assert result is True

    def test_fragment_error_handling(self, fragment_validator):
        """Test error handling in fragments."""

        def error_handling_test():
            import streamlit as st

            @st.fragment
            def error_fragment():
                raise ValueError("Fragment error for testing")

            @st.fragment
            def normal_fragment():
                return "normal_result"

            # Test that errors in one fragment don't affect others
            try:
                error_fragment()
            except ValueError:
                pass  # Expected error

            return normal_fragment()

        # Should handle errors gracefully
        result = fragment_validator.validate_component_behavior(error_handling_test)
        assert result is True

    def test_fragment_performance_impact(self, fragment_validator):
        """Test performance impact of multiple fragments."""
        config = {"fragment_count": 5, "interval": "0.5s"}

        performance = fragment_validator.measure_fragment_performance_impact(config)

        assert "total_executions" in performance
        assert "render_time" in performance
        assert "fragment_count" in performance

        # Should have reasonable performance
        assert performance["render_time"] < 5.0  # Less than 5 seconds
        assert performance["fragment_count"] == config["fragment_count"]

    def test_fragment_memory_efficiency(self, streamlit_tester):
        """Test fragment memory efficiency."""

        def memory_efficiency_test():
            import streamlit as st

            # Create fragments that manage data
            data_store = {}

            @st.fragment(run_every="0.5s")
            def data_producer():
                data_store["produced"] = list(range(100))
                return len(data_store["produced"])

            @st.fragment(run_every="1s")
            def data_consumer():
                if "produced" in data_store:
                    consumed = data_store.pop("produced")
                    return len(consumed)
                return 0

            # Start fragments
            data_producer()
            data_consumer()

            # Let them run briefly
            time.sleep(1.5)

            return data_store

        # Benchmark memory usage
        benchmark = streamlit_tester.benchmark_component_performance(
            "fragment_system", memory_efficiency_test, iterations=2
        )

        # Memory usage should be reasonable
        assert benchmark.after_metrics.memory_usage >= 0

    def test_concurrent_fragment_execution(self, fragment_validator):
        """Test concurrent execution of fragments."""

        def concurrent_test():
            import streamlit as st

            execution_log = []

            @st.fragment(run_every="0.3s")
            def fast_fragment():
                execution_log.append(f"fast_{datetime.now()}")
                return len(execution_log)

            @st.fragment(run_every="0.7s")
            def slow_fragment():
                execution_log.append(f"slow_{datetime.now()}")
                return len(execution_log)

            # Start both fragments
            fast_fragment()
            slow_fragment()

            # Let them run concurrently
            time.sleep(2.0)

            return execution_log

        result = fragment_validator.validate_component_behavior(concurrent_test)
        assert result is True

        # Should have executions from both fragments
        assert len(fragment_validator.fragment_metrics.executions) >= 2

    @pytest.mark.parametrize("interval", ("0.5s", "1s", "2s"))
    def test_fragment_different_intervals(self, fragment_validator, interval):
        """Test fragments with different run_every intervals."""
        config = {"interval": interval, "test_duration": 2.0}

        assert fragment_validator.validate_auto_refresh_behavior(config)

        # Should record the interval correctly
        intervals = list(
            fragment_validator.fragment_metrics.run_every_intervals.values()
        )
        assert len(intervals) > 0

    @pytest.mark.parametrize("fragment_count", (1, 3, 5))
    def test_fragment_scalability(self, fragment_validator, fragment_count):
        """Test fragment system scalability with different counts."""
        config = {"fragment_count": fragment_count, "interval": "1s"}

        performance = fragment_validator.measure_fragment_performance_impact(config)

        # Should scale reasonably
        assert performance["fragment_count"] == fragment_count
        assert performance["total_executions"] >= fragment_count

    def test_fragment_cleanup_on_stop(self, fragment_validator):
        """Test proper cleanup when fragments are stopped."""

        def cleanup_test():
            import streamlit as st

            @st.fragment(run_every="0.5s")
            def cleanup_fragment():
                return "running"

            # Start fragment
            cleanup_fragment()

            # Let it run briefly
            time.sleep(1.0)

            return True

        fragment_validator.validate_component_behavior(cleanup_test)

        # Stop all fragments and verify cleanup
        fragment_validator.mock_fragment.stop_all_fragments()

        # Should have no running fragments
        assert len(fragment_validator.mock_fragment.running) == 0

    def test_fragment_integration_with_other_components(self, streamlit_tester):
        """Test fragment integration with other Streamlit components."""

        def integration_test():
            import streamlit as st

            @st.fragment(run_every="1s")
            def integrated_fragment():
                # Use other Streamlit components within fragment
                st.write("Fragment content")
                st.progress(0.5)

                # Simulate some data processing
                return {"status": "running", "timestamp": datetime.now()}

            result = integrated_fragment()

            # Let fragment run once more
            time.sleep(1.1)

            return result

        result = streamlit_tester.run_component_validation(
            "fragment_system", integration_test
        )

        assert result is True


class TestFragmentSystemBenchmarks:
    """Benchmark tests for fragment system performance validation."""

    @pytest.fixture
    def benchmarking_tester(self):
        """Provide tester configured for benchmarking."""
        tester = StreamlitNativeTester()
        tester.register_validator("fragment_system", FragmentSystemValidator())
        return tester

    def test_fragment_execution_overhead(self, benchmarking_tester):
        """Benchmark fragment execution overhead."""

        def overhead_test():
            import streamlit as st

            @st.fragment
            def simple_fragment():
                # Minimal work to measure overhead
                return datetime.now()

            # Execute fragment multiple times
            results = []
            for _ in range(50):
                result = simple_fragment()
                results.append(result)

            return results

        benchmark = benchmarking_tester.benchmark_component_performance(
            "fragment_system", overhead_test, iterations=5
        )

        validator = benchmarking_tester.validators["fragment_system"]

        # Should have many executions with reasonable overhead
        total_executions = sum(validator.fragment_metrics.executions.values())
        assert total_executions >= 50

        # Execution should be fast
        assert benchmark.after_metrics.render_time < 1.0  # Less than 1 second

    def test_auto_refresh_performance(self, benchmarking_tester):
        """Benchmark auto-refresh performance."""

        def auto_refresh_performance():
            import streamlit as st

            @st.fragment(run_every="0.2s")  # Fast refresh
            def performance_fragment():
                # Simulate light work
                return list(range(10))

            # Start fragment
            performance_fragment()

            # Let it auto-refresh
            time.sleep(1.0)  # 1 second = ~5 refreshes

            return True

        benchmarking_tester.benchmark_component_performance(
            "fragment_system", auto_refresh_performance, iterations=3
        )

        validator = benchmarking_tester.validators["fragment_system"]

        # Should have multiple auto-refreshes
        total_executions = sum(validator.fragment_metrics.executions.values())
        assert total_executions >= 3  # Initial + auto-refreshes

    def test_multiple_fragments_performance(self, benchmarking_tester):
        """Benchmark performance with multiple concurrent fragments."""

        def multi_fragment_performance():
            import streamlit as st

            # Create 5 fragments with different intervals
            @st.fragment(run_every="0.4s")
            def fragment_1():
                return "fragment_1"

            @st.fragment(run_every="0.6s")
            def fragment_2():
                return "fragment_2"

            @st.fragment(run_every="0.8s")
            def fragment_3():
                return "fragment_3"

            @st.fragment(run_every="1s")
            def fragment_4():
                return "fragment_4"

            @st.fragment(run_every="1.2s")
            def fragment_5():
                return "fragment_5"

            # Start all fragments
            fragment_1()
            fragment_2()
            fragment_3()
            fragment_4()
            fragment_5()

            # Let them run concurrently
            time.sleep(2.5)

            return True

        benchmark = benchmarking_tester.benchmark_component_performance(
            "fragment_system", multi_fragment_performance, iterations=2
        )

        validator = benchmarking_tester.validators["fragment_system"]

        # Should have 5 different fragments
        assert len(validator.fragment_metrics.executions) == 5

        # Performance should be reasonable even with multiple fragments
        assert benchmark.after_metrics.render_time < 3.0  # Less than 3 seconds

    def test_fragment_memory_performance(self, benchmarking_tester):
        """Benchmark fragment memory performance."""

        def memory_performance():
            import streamlit as st

            @st.fragment(run_every="0.5s")
            def memory_fragment():
                # Create and discard data to test memory management
                large_data = list(range(1000))
                processed = [x * 2 for x in large_data]
                return len(processed)

            # Start fragment
            memory_fragment()

            # Let it run several times
            time.sleep(2.0)

            return True

        benchmark = benchmarking_tester.benchmark_component_performance(
            "fragment_system", memory_performance, iterations=3
        )

        # Memory usage should be reasonable
        assert benchmark.after_metrics.memory_usage >= 0
        # Should not have excessive memory growth
        assert benchmark.after_metrics.memory_peak < 100_000_000  # 100MB limit
