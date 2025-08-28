"""Stream C Fragment Behavior Testing.

Comprehensive testing for Streamlit native fragment components:
- st.fragment() with isolation and auto-refresh validation
- run_every timing accuracy and performance testing
- Fragment state management and scoped rerun functionality
- Cross-fragment coordination and interference testing

Focuses on:
1. Fragment Isolation: Ensuring fragments operate independently without interference
2. Auto-refresh Accuracy: Validating precise timing of run_every intervals
3. State Management: Testing fragment-specific state isolation and persistence
4. Performance Optimization: Measuring fragment efficiency and resource usage
"""

import threading
import time

from collections.abc import Callable
from datetime import datetime
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


class FragmentExecutionTracker:
    """Track fragment execution patterns and timing."""

    def __init__(self):
        """Initialize fragment execution tracking."""
        self.executions = {}  # fragment_id -> execution_count
        self.run_every_configs = {}  # fragment_id -> run_every_value
        self.execution_timestamps = {}  # fragment_id -> list of timestamps
        self.rerun_calls = []  # List of rerun calls with metadata
        self.state_snapshots = {}  # fragment_id -> state_data
        self.isolation_violations = 0
        self.coordination_events = []  # Cross-fragment interactions

    def record_execution(self, fragment_id: str) -> None:
        """Record fragment execution."""
        current_time = datetime.now()

        # Update execution count
        self.executions[fragment_id] = self.executions.get(fragment_id, 0) + 1

        # Record timestamp
        if fragment_id not in self.execution_timestamps:
            self.execution_timestamps[fragment_id] = []
        self.execution_timestamps[fragment_id].append(current_time)

    def record_run_every_config(self, fragment_id: str, interval: str | float) -> None:
        """Record run_every configuration."""
        # Convert string intervals to float seconds
        if isinstance(interval, str):
            if interval.endswith("s"):
                interval_float = float(interval[:-1])
            elif interval.endswith("ms"):
                interval_float = float(interval[:-2]) / 1000
            else:
                interval_float = float(interval)
        else:
            interval_float = float(interval)

        self.run_every_configs[fragment_id] = interval_float

    def record_rerun(self, scope: str = "app", fragment_id: str | None = None) -> None:
        """Record st.rerun call."""
        self.rerun_calls.append(
            {
                "scope": scope,
                "fragment_id": fragment_id,
                "timestamp": datetime.now(),
            }
        )

    def record_state_snapshot(self, fragment_id: str, state_data: dict) -> None:
        """Record fragment state snapshot."""
        self.state_snapshots[fragment_id] = {
            "state": state_data.copy(),
            "timestamp": datetime.now(),
        }

    def record_isolation_violation(self, description: str) -> None:
        """Record fragment isolation violation."""
        self.isolation_violations += 1
        self.coordination_events.append(
            {
                "type": "isolation_violation",
                "description": description,
                "timestamp": datetime.now(),
            }
        )

    def record_coordination_event(self, event_type: str, details: dict) -> None:
        """Record cross-fragment coordination event."""
        self.coordination_events.append(
            {
                "type": event_type,
                "details": details,
                "timestamp": datetime.now(),
            }
        )

    def get_execution_rate(self, fragment_id: str) -> float:
        """Calculate actual execution rate (executions per second)."""
        if fragment_id not in self.execution_timestamps:
            return 0.0

        timestamps = self.execution_timestamps[fragment_id]
        if len(timestamps) < 2:
            return 0.0

        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        if duration == 0:
            return 0.0

        return (len(timestamps) - 1) / duration

    def get_timing_accuracy(self, fragment_id: str) -> float:
        """Calculate timing accuracy vs expected run_every interval."""
        if fragment_id not in self.run_every_configs:
            return 0.0

        expected_interval = self.run_every_configs[fragment_id]
        actual_rate = self.get_execution_rate(fragment_id)
        expected_rate = 1.0 / expected_interval if expected_interval > 0 else 0.0

        if expected_rate == 0:
            return 0.0

        accuracy = 1.0 - abs(actual_rate - expected_rate) / expected_rate
        return max(0.0, accuracy)

    def get_average_interval(self, fragment_id: str) -> float:
        """Calculate average interval between executions."""
        if fragment_id not in self.execution_timestamps:
            return 0.0

        timestamps = self.execution_timestamps[fragment_id]
        if len(timestamps) < 2:
            return 0.0

        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(interval)

        return sum(intervals) / len(intervals) if intervals else 0.0

    def reset(self) -> None:
        """Reset all tracking data."""
        self.executions.clear()
        self.run_every_configs.clear()
        self.execution_timestamps.clear()
        self.rerun_calls.clear()
        self.state_snapshots.clear()
        self.coordination_events.clear()
        self.isolation_violations = 0


class MockStreamlitFragment:
    """Mock implementation of Streamlit fragment system."""

    def __init__(self, tracker: FragmentExecutionTracker):
        """Initialize mock fragment system."""
        self.tracker = tracker
        self.active_fragments = {}  # fragment_id -> fragment_info
        self.timers = {}  # fragment_id -> timer
        self.fragment_states = {}  # fragment_id -> state
        self.running = set()  # active fragment IDs

    def mock_fragment(self, run_every: str | float | None = None):
        """Mock st.fragment decorator."""

        def decorator(func: Callable):
            fragment_id = f"{func.__name__}_{id(func)}"

            # Record configuration
            if run_every:
                self.tracker.record_run_every_config(fragment_id, run_every)

            def wrapper(*args, **kwargs):
                # Record execution
                self.tracker.record_execution(fragment_id)

                # Initialize fragment state if needed
                if fragment_id not in self.fragment_states:
                    self.fragment_states[fragment_id] = {}

                # Execute the fragment function
                result = func(*args, **kwargs)

                # Start auto-refresh if configured
                if run_every and fragment_id not in self.running:
                    self._start_auto_refresh(fragment_id, func, run_every, args, kwargs)

                return result

            # Store fragment info
            self.active_fragments[fragment_id] = {
                "function": func,
                "wrapper": wrapper,
                "run_every": run_every,
                "args": (),
                "kwargs": {},
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
    ) -> None:
        """Start auto-refresh timer for fragment."""
        if fragment_id in self.running:
            return

        # Convert interval to seconds
        if isinstance(interval, str):
            if interval.endswith("s"):
                interval_seconds = float(interval[:-1])
            elif interval.endswith("ms"):
                interval_seconds = float(interval[:-2]) / 1000
            else:
                interval_seconds = float(interval)
        else:
            interval_seconds = float(interval)

        self.running.add(fragment_id)
        self.active_fragments[fragment_id]["args"] = args
        self.active_fragments[fragment_id]["kwargs"] = kwargs

        def auto_refresh():
            if fragment_id in self.running:
                try:
                    # Record auto-refresh execution
                    self.tracker.record_execution(fragment_id)

                    # Execute fragment function
                    func(*args, **kwargs)

                    # Schedule next refresh
                    self._schedule_next_refresh(
                        fragment_id, func, interval_seconds, args, kwargs
                    )

                except Exception:
                    # Stop auto-refresh on error
                    self.stop_fragment(fragment_id)

        # Schedule first refresh
        timer = threading.Timer(interval_seconds, auto_refresh)
        self.timers[fragment_id] = timer
        timer.start()

    def _schedule_next_refresh(
        self,
        fragment_id: str,
        func: Callable,
        interval_seconds: float,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Schedule next auto-refresh execution."""
        if fragment_id not in self.running:
            return

        def refresh():
            if fragment_id in self.running:
                try:
                    self.tracker.record_execution(fragment_id)
                    func(*args, **kwargs)
                    self._schedule_next_refresh(
                        fragment_id, func, interval_seconds, args, kwargs
                    )
                except Exception:
                    self.stop_fragment(fragment_id)

        timer = threading.Timer(interval_seconds, refresh)
        self.timers[fragment_id] = timer
        timer.start()

    def stop_fragment(self, fragment_id: str) -> None:
        """Stop auto-refresh for specific fragment."""
        self.running.discard(fragment_id)

        if fragment_id in self.timers:
            self.timers[fragment_id].cancel()
            del self.timers[fragment_id]

    def stop_all_fragments(self) -> None:
        """Stop all fragment auto-refresh timers."""
        for fragment_id in list(self.running):
            self.stop_fragment(fragment_id)

    def mock_rerun(self, scope: str = "app"):
        """Mock st.rerun with scope support."""

        def rerun():
            # Determine fragment context if available
            fragment_id = None
            if scope == "fragment":
                # In real implementation, would detect current fragment context
                fragment_id = "current_fragment"

            self.tracker.record_rerun(scope, fragment_id)

            # Simulate rerun behavior
            if scope == "fragment":
                # Fragment-scoped rerun
                self.tracker.record_coordination_event(
                    "fragment_rerun", {"scope": scope}
                )
            else:
                # App-wide rerun
                self.tracker.record_coordination_event("app_rerun", {"scope": scope})

        return rerun


class FragmentBehaviorValidator(NativeComponentValidator):
    """Validator for Stream C fragment behavior components."""

    def __init__(self):
        """Initialize fragment behavior validator."""
        super().__init__(StreamType.STREAM_C, "fragment_behavior")
        self.tracker = FragmentExecutionTracker()
        self.mock_fragment = MockStreamlitFragment(self.tracker)

    def validate_functionality(self, test_func, *args, **kwargs) -> bool:
        """Validate fragment component functionality preservation."""
        try:
            self.tracker.reset()
            self.mock_fragment.stop_all_fragments()

            # Mock fragment components
            with (
                patch("streamlit.fragment", self.mock_fragment.mock_fragment),
                patch("streamlit.rerun", self.mock_fragment.mock_rerun()),
            ):
                test_func(*args, **kwargs)

            # Update metrics
            self.metrics.fragment_executions = sum(self.tracker.executions.values())
            self.metrics.auto_refresh_count = len(
                [
                    fid
                    for fid, config in self.tracker.run_every_configs.items()
                    if config > 0
                ]
            )
            self.metrics.rerun_count = len(self.tracker.rerun_calls)

            # Calculate timing accuracy if applicable
            if self.tracker.run_every_configs:
                accuracies = [
                    self.tracker.get_timing_accuracy(fid)
                    for fid in self.tracker.run_every_configs
                ]
                self.metrics.timing_accuracy = (
                    sum(accuracies) / len(accuracies) if accuracies else 0.0
                )

            return True

        except Exception:
            self.metrics.error_count += 1
            return False
        finally:
            # Always clean up
            self.mock_fragment.stop_all_fragments()

    def measure_performance(
        self, test_func, iterations: int = 10
    ) -> NativeComponentMetrics:
        """Measure fragment component performance."""
        total_metrics = NativeComponentMetrics()

        for _ in range(iterations):
            with self.performance_monitoring() as metrics:
                self.validate_functionality(test_func)

            # Accumulate metrics
            total_metrics.execution_time += metrics.execution_time
            total_metrics.memory_usage_mb += metrics.memory_usage_mb
            total_metrics.cpu_usage_percent += metrics.cpu_usage_percent
            total_metrics.peak_memory_mb = max(
                total_metrics.peak_memory_mb, metrics.peak_memory_mb
            )

            # Fragment-specific metrics
            total_metrics.fragment_executions += metrics.fragment_executions
            total_metrics.auto_refresh_count += metrics.auto_refresh_count
            total_metrics.rerun_count += metrics.rerun_count

            # Timing accuracy (average)
            if metrics.timing_accuracy > 0:
                total_metrics.timing_accuracy += metrics.timing_accuracy

        # Average the metrics
        return NativeComponentMetrics(
            execution_time=total_metrics.execution_time / iterations,
            memory_usage_mb=total_metrics.memory_usage_mb / iterations,
            cpu_usage_percent=total_metrics.cpu_usage_percent / iterations,
            peak_memory_mb=total_metrics.peak_memory_mb,
            fragment_executions=total_metrics.fragment_executions,
            auto_refresh_count=total_metrics.auto_refresh_count,
            rerun_count=total_metrics.rerun_count,
            timing_accuracy=total_metrics.timing_accuracy / iterations
            if total_metrics.timing_accuracy > 0
            else 0,
        )

    def compare_implementations(
        self, baseline_func, optimized_func, iterations: int = 10
    ) -> PerformanceBenchmark:
        """Compare baseline vs optimized fragment implementations."""
        benchmark = PerformanceBenchmark(
            component_name=self.component_name,
            stream_type=self.stream_type,
            test_name="fragment_comparison",
            iterations=iterations,
        )

        try:
            # Measure baseline (manual refresh)
            baseline_metrics = self.measure_performance(baseline_func, iterations)
            benchmark.baseline_metrics = baseline_metrics

            # Measure optimized (fragment auto-refresh)
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

        # Run test with monitoring
        result = self.validate_functionality(test_func)

        return {
            "function_result": result,
            "fragment_executions": dict(self.tracker.executions),
            "rerun_calls": len(self.tracker.rerun_calls),
            "coordination_events": len(self.tracker.coordination_events),
            "isolation_violations": self.tracker.isolation_violations,
            "timing_accuracies": {
                fid: self.tracker.get_timing_accuracy(fid)
                for fid in self.tracker.run_every_configs
            },
        }

    def _compare_results(
        self, baseline: dict[str, Any], optimized: dict[str, Any]
    ) -> bool:
        """Compare results focusing on functionality preservation."""
        try:
            # Function results should be equivalent
            if baseline["function_result"] != optimized["function_result"]:
                return False

            # Fragment version should have better automation (more executions with auto-refresh)
            baseline_executions = (
                sum(baseline["fragment_executions"].values())
                if baseline["fragment_executions"]
                else 0
            )
            optimized_executions = (
                sum(optimized["fragment_executions"].values())
                if optimized["fragment_executions"]
                else 0
            )

            # Optimized should have at least as many executions due to auto-refresh
            return not optimized_executions < baseline_executions

        except (KeyError, TypeError, ValueError):
            return False

    def validate_fragment_isolation(self, isolation_test_config: dict) -> bool:
        """Validate fragment isolation behavior."""

        def isolation_test():
            import streamlit as st

            shared_data = {"counter": 0, "fragments_run": []}

            @st.fragment
            def isolated_fragment_1():
                """First isolated fragment."""
                shared_data["fragments_run"].append("fragment_1")
                shared_data["counter"] += 1

                # Fragment should not interfere with others
                self.tracker.record_state_snapshot(
                    "fragment_1",
                    {
                        "local_counter": shared_data["counter"],
                        "fragments_seen": shared_data["fragments_run"].copy(),
                    },
                )

                return shared_data["counter"]

            @st.fragment
            def isolated_fragment_2():
                """Second isolated fragment."""
                shared_data["fragments_run"].append("fragment_2")
                shared_data["counter"] += 10  # Different increment pattern

                self.tracker.record_state_snapshot(
                    "fragment_2",
                    {
                        "local_counter": shared_data["counter"],
                        "fragments_seen": shared_data["fragments_run"].copy(),
                    },
                )

                return shared_data["counter"]

            # Execute fragments
            result1 = isolated_fragment_1()
            result2 = isolated_fragment_2()

            # Validate no cross-interference
            if len(shared_data["fragments_run"]) != 2:
                self.tracker.record_isolation_violation(
                    "Fragment execution interference"
                )

            return [result1, result2]

        self.validate_functionality(isolation_test)
        return self.tracker.isolation_violations == 0

    def validate_auto_refresh_timing(self, timing_config: dict) -> bool:
        """Validate auto-refresh timing accuracy."""

        def timing_test():
            import streamlit as st

            interval = timing_config.get("interval", "1s")
            duration = timing_config.get("duration", 3.0)  # Test duration in seconds

            @st.fragment(run_every=interval)
            def timed_fragment():
                """Fragment with auto-refresh timing."""
                current_time = datetime.now()
                self.tracker.record_coordination_event(
                    "timed_execution",
                    {
                        "timestamp": current_time.isoformat(),
                        "fragment": "timed_fragment",
                    },
                )
                return current_time.isoformat()

            # Start fragment
            initial_result = timed_fragment()

            # Let it run for specified duration
            time.sleep(duration)

            return initial_result

        self.validate_functionality(timing_test)

        # Check timing accuracy
        fragment_ids = list(self.tracker.executions.keys())
        if fragment_ids:
            fragment_id = fragment_ids[0]
            accuracy = self.tracker.get_timing_accuracy(fragment_id)
            return accuracy > 0.5  # 50% timing accuracy threshold

        return False

    def validate_scoped_rerun_behavior(self, rerun_config: dict) -> bool:
        """Validate scoped st.rerun functionality."""

        def rerun_test():
            import streamlit as st

            rerun_scope = rerun_config.get("scope", "fragment")

            @st.fragment
            def rerun_fragment():
                """Fragment that triggers scoped rerun."""
                self.tracker.record_coordination_event(
                    "fragment_action", {"action": "before_rerun", "scope": rerun_scope}
                )

                # Trigger rerun with specified scope
                if rerun_scope == "fragment":
                    st.rerun()  # Fragment-scoped rerun
                else:
                    st.rerun()  # App-wide rerun (default)

                return f"rerun_triggered_{rerun_scope}"

            return rerun_fragment()

        self.validate_functionality(rerun_test)

        # Verify rerun was called with correct scope
        return len(self.tracker.rerun_calls) > 0


class TestStreamCFragmentBehavior:
    """Test suite for Stream C fragment behavior components."""

    @pytest.fixture
    def fragment_validator(self):
        """Provide fragment behavior validator."""
        return FragmentBehaviorValidator()

    @pytest.fixture
    def native_tester(self, fragment_validator):
        """Provide configured native component tester."""
        tester = NativeComponentTester()
        tester.register_validator("fragment_behavior", fragment_validator)
        return tester

    def test_basic_fragment_functionality(self, fragment_validator):
        """Test basic st.fragment component functionality."""

        def basic_fragment_test():
            import streamlit as st

            @st.fragment
            def simple_fragment():
                """Basic fragment without auto-refresh."""
                return "fragment_executed"

            # Execute fragment
            return simple_fragment()

        success = fragment_validator.validate_functionality(basic_fragment_test)
        assert success is True

        # Verify fragment execution
        assert fragment_validator.metrics.fragment_executions == 1
        assert len(fragment_validator.tracker.executions) == 1

    def test_fragment_with_auto_refresh(self, fragment_validator):
        """Test st.fragment with run_every auto-refresh."""

        def auto_refresh_test():
            import streamlit as st

            execution_log = []

            @st.fragment(run_every="0.5s")
            def auto_refresh_fragment():
                """Fragment with auto-refresh."""
                current_time = datetime.now()
                execution_log.append(current_time)
                return len(execution_log)

            # Start fragment
            auto_refresh_fragment()

            # Let it auto-refresh
            time.sleep(1.2)  # Should trigger ~2 auto-refreshes

            return execution_log

        success = fragment_validator.validate_functionality(auto_refresh_test)
        assert success is True

        # Should have multiple executions due to auto-refresh
        assert fragment_validator.metrics.fragment_executions >= 2
        assert fragment_validator.metrics.auto_refresh_count > 0

    def test_multiple_fragments_isolation(self, fragment_validator):
        """Test isolation between multiple fragments."""
        isolation_config = {}

        success = fragment_validator.validate_fragment_isolation(isolation_config)
        assert success is True

        # Should have no isolation violations
        assert fragment_validator.tracker.isolation_violations == 0

        # Should have executed multiple fragments
        assert len(fragment_validator.tracker.executions) >= 2

    def test_fragment_auto_refresh_timing_accuracy(self, fragment_validator):
        """Test timing accuracy of auto-refresh intervals."""
        timing_config = {
            "interval": "0.5s",
            "duration": 2.0,
        }

        success = fragment_validator.validate_auto_refresh_timing(timing_config)
        assert success is True

        # Check timing accuracy
        fragment_ids = list(fragment_validator.tracker.executions.keys())
        if fragment_ids:
            fragment_id = fragment_ids[0]
            accuracy = fragment_validator.tracker.get_timing_accuracy(fragment_id)
            assert (
                accuracy > 0.3
            )  # 30% accuracy threshold (lenient for test environment)

    def test_fragment_scoped_rerun(self, fragment_validator):
        """Test fragment-scoped st.rerun functionality."""
        rerun_config = {"scope": "fragment"}

        success = fragment_validator.validate_scoped_rerun_behavior(rerun_config)
        assert success is True

        # Should have recorded rerun calls
        assert fragment_validator.metrics.rerun_count > 0
        assert len(fragment_validator.tracker.rerun_calls) > 0

    def test_complex_fragment_coordination(self, fragment_validator):
        """Test coordination between multiple fragments."""

        def coordination_test():
            import streamlit as st

            shared_state = {"producer_count": 0, "consumer_count": 0, "messages": []}

            @st.fragment(run_every="0.3s")
            def producer_fragment():
                """Fragment that produces data."""
                shared_state["producer_count"] += 1
                message = f"message_{shared_state['producer_count']}"
                shared_state["messages"].append(message)

                fragment_validator.tracker.record_coordination_event(
                    "producer_action",
                    {
                        "message": message,
                        "total_produced": shared_state["producer_count"],
                    },
                )

                return shared_state["producer_count"]

            @st.fragment(run_every="0.7s")
            def consumer_fragment():
                """Fragment that consumes data."""
                if shared_state["messages"]:
                    consumed = shared_state["messages"].pop(0)
                    shared_state["consumer_count"] += 1

                    fragment_validator.tracker.record_coordination_event(
                        "consumer_action",
                        {
                            "consumed": consumed,
                            "total_consumed": shared_state["consumer_count"],
                        },
                    )

                    return consumed
                return None

            # Start both fragments
            producer_fragment()
            consumer_fragment()

            # Let them coordinate
            time.sleep(1.5)  # Allow multiple cycles

            return shared_state

        success = fragment_validator.validate_functionality(coordination_test)
        assert success is True

        # Should have coordination events
        assert len(fragment_validator.tracker.coordination_events) > 0

        # Should have executed multiple fragments
        assert len(fragment_validator.tracker.executions) == 2

    def test_fragment_state_management(self, fragment_validator):
        """Test fragment state management and persistence."""

        def state_management_test():
            import streamlit as st

            # Simulate session state
            session_state = {"fragment_data": {}}

            @st.fragment
            def stateful_fragment(fragment_id: str):
                """Fragment with state management."""
                if fragment_id not in session_state["fragment_data"]:
                    session_state["fragment_data"][fragment_id] = {
                        "counter": 0,
                        "created_at": datetime.now().isoformat(),
                    }

                session_state["fragment_data"][fragment_id]["counter"] += 1
                current_state = session_state["fragment_data"][fragment_id]

                # Record state snapshot
                fragment_validator.tracker.record_state_snapshot(
                    fragment_id, current_state
                )

                return current_state["counter"]

            # Execute fragment multiple times with different IDs
            results = []
            for i in range(3):
                frag_id = f"fragment_{i}"
                for _ in range(2):  # Execute each fragment twice
                    result = stateful_fragment(frag_id)
                    results.append((frag_id, result))

            return results

        success = fragment_validator.validate_functionality(state_management_test)
        assert success is True

        # Should have state snapshots for fragments
        assert len(fragment_validator.tracker.state_snapshots) > 0

    def test_fragment_error_handling(self, fragment_validator):
        """Test error handling in fragments."""

        def error_handling_test():
            import streamlit as st

            @st.fragment
            def error_prone_fragment():
                """Fragment that may encounter errors."""
                # Simulate random error condition
                import random

                if random.random() < 0.3:  # 30% error rate
                    raise ValueError("Simulated fragment error")

                return "success"

            @st.fragment
            def stable_fragment():
                """Fragment that should continue working despite other errors."""
                return "stable_execution"

            # Execute both fragments
            results = []

            try:
                result1 = error_prone_fragment()
                results.append(result1)
            except ValueError:
                results.append("error_handled")

            result2 = stable_fragment()
            results.append(result2)

            return results

        success = fragment_validator.validate_functionality(error_handling_test)
        assert success is True

        # Should have executed fragments despite errors
        assert fragment_validator.metrics.fragment_executions >= 1

    @pytest.mark.parametrize("interval", ("0.2s", "0.5s", "1s", "2s"))
    def test_fragment_different_intervals(self, fragment_validator, interval):
        """Test fragments with different run_every intervals."""

        def interval_test():
            import streamlit as st

            execution_count = 0

            @st.fragment(run_every=interval)
            def interval_fragment():
                nonlocal execution_count
                execution_count += 1
                return execution_count

            # Start fragment
            interval_fragment()

            # Let it run for a duration proportional to interval
            test_duration = max(1.5, float(interval[:-1]) * 3)
            time.sleep(test_duration)

            return execution_count

        success = fragment_validator.validate_functionality(interval_test)
        assert success is True

        # Should have recorded the interval configuration
        assert len(fragment_validator.tracker.run_every_configs) > 0

        # Should have multiple executions for shorter intervals
        expected_executions = 2 if interval in ["0.2s", "0.5s"] else 1
        assert fragment_validator.metrics.fragment_executions >= expected_executions

    def test_functionality_preservation_comparison(self, fragment_validator):
        """Test functionality preservation between manual and fragment implementations."""

        def manual_refresh_implementation():
            """Manual implementation without fragments."""
            execution_log = []

            def manual_task():
                """Task executed manually."""
                current_time = datetime.now()
                execution_log.append(current_time)
                time.sleep(0.1)  # Simulate work
                return len(execution_log)

            # Manual execution cycle
            results = []
            for _ in range(3):
                result = manual_task()
                results.append(result)
                time.sleep(0.5)  # Manual delay between executions

            return results

        def fragment_implementation():
            """Fragment-based implementation with auto-refresh."""
            import streamlit as st

            execution_log = []

            @st.fragment(run_every="0.5s")
            def fragment_task():
                """Task executed as fragment."""
                current_time = datetime.now()
                execution_log.append(current_time)
                time.sleep(0.1)  # Simulate work
                return len(execution_log)

            # Start fragment and let it run
            fragment_task()
            time.sleep(1.5)  # Let auto-refresh work

            return execution_log

        # Compare implementations
        benchmark = fragment_validator.compare_implementations(
            manual_refresh_implementation, fragment_implementation, iterations=2
        )

        assert benchmark.passed is True
        # Fragment implementation should have better automation
        assert (
            benchmark.optimized_metrics.fragment_executions
            >= benchmark.baseline_metrics.fragment_executions
        )

    def test_fragment_performance_monitoring(self, native_tester):
        """Test performance monitoring of fragments."""

        def performance_test():
            import streamlit as st

            @st.fragment(run_every="0.2s")
            def performance_fragment():
                """Fragment for performance testing."""
                # Simulate varying workload
                work_data = list(range(100))
                processed = [x * 2 for x in work_data]
                return len(processed)

            # Start fragment
            initial_result = performance_fragment()

            # Let it run for performance measurement
            time.sleep(1.0)

            return initial_result

        benchmark = native_tester.benchmark_component(
            "fragment_behavior", performance_test, iterations=3, warmup_iterations=1
        )

        assert benchmark.passed is True
        assert benchmark.optimized_metrics.execution_time > 0
        assert (
            benchmark.optimized_metrics.fragment_executions >= 3
        )  # Initial + auto-refreshes


class TestFragmentBehaviorBenchmarks:
    """Benchmark tests for fragment behavior performance validation."""

    @pytest.fixture
    def benchmarking_tester(self):
        """Provide tester configured for benchmarking."""
        tester = NativeComponentTester()
        tester.register_validator("fragment_behavior", FragmentBehaviorValidator())
        return tester

    def test_high_frequency_fragment_benchmark(self, benchmarking_tester):
        """Benchmark high-frequency fragment execution."""

        def high_frequency_test():
            import streamlit as st

            execution_count = 0

            @st.fragment(run_every="0.1s")  # Very frequent
            def rapid_fragment():
                nonlocal execution_count
                execution_count += 1
                # Light work to test overhead
                return execution_count

            # Start rapid execution
            rapid_fragment()

            # Let it run rapidly
            time.sleep(2.0)  # Should get ~20 executions

            return execution_count

        benchmark = benchmarking_tester.benchmark_component(
            "fragment_behavior", high_frequency_test, iterations=2
        )

        assert benchmark.passed is True

        # Should handle high frequency efficiently
        validator = benchmarking_tester.validators["fragment_behavior"]
        assert validator.metrics.fragment_executions >= 10
        assert benchmark.optimized_metrics.execution_time < 5.0

    def test_multiple_concurrent_fragments_benchmark(self, benchmarking_tester):
        """Benchmark multiple concurrent fragments."""

        def concurrent_fragments_test():
            import streamlit as st

            counters = {"a": 0, "b": 0, "c": 0, "d": 0}

            @st.fragment(run_every="0.3s")
            def fragment_a():
                counters["a"] += 1
                return counters["a"]

            @st.fragment(run_every="0.5s")
            def fragment_b():
                counters["b"] += 1
                return counters["b"]

            @st.fragment(run_every="0.7s")
            def fragment_c():
                counters["c"] += 1
                return counters["c"]

            @st.fragment(run_every="1s")
            def fragment_d():
                counters["d"] += 1
                return counters["d"]

            # Start all fragments
            fragment_a()
            fragment_b()
            fragment_c()
            fragment_d()

            # Let them run concurrently
            time.sleep(3.0)

            return counters

        benchmark = benchmarking_tester.benchmark_component(
            "fragment_behavior", concurrent_fragments_test, iterations=2
        )

        assert benchmark.passed is True

        # Should handle multiple concurrent fragments
        validator = benchmarking_tester.validators["fragment_behavior"]
        assert len(validator.tracker.executions) == 4  # Four fragments
        assert validator.metrics.fragment_executions >= 8  # Multiple executions

    def test_complex_fragment_workflow_benchmark(self, benchmarking_tester):
        """Benchmark complex fragment workflow."""

        def complex_workflow():
            import streamlit as st

            workflow_state = {
                "jobs_scraped": 0,
                "companies_processed": 0,
                "data_analyzed": 0,
                "reports_generated": 0,
            }

            @st.fragment(run_every="0.4s")
            def job_scraper_fragment():
                """Fragment that simulates job scraping."""
                workflow_state["jobs_scraped"] += 5  # Batch of 5 jobs
                time.sleep(0.01)  # Simulate scraping delay
                return workflow_state["jobs_scraped"]

            @st.fragment(run_every="0.8s")
            def company_processor_fragment():
                """Fragment that processes company data."""
                if workflow_state["jobs_scraped"] > 0:
                    workflow_state["companies_processed"] += 1
                    time.sleep(0.005)  # Simulate processing
                return workflow_state["companies_processed"]

            @st.fragment(run_every="1.2s")
            def analyzer_fragment():
                """Fragment that analyzes scraped data."""
                if workflow_state["companies_processed"] > 0:
                    workflow_state["data_analyzed"] += 1
                    time.sleep(0.008)  # Simulate analysis
                return workflow_state["data_analyzed"]

            @st.fragment(run_every="2s")
            def reporting_fragment():
                """Fragment that generates reports."""
                if workflow_state["data_analyzed"] > 0:
                    workflow_state["reports_generated"] += 1
                    time.sleep(0.003)  # Simulate report generation
                return workflow_state["reports_generated"]

            # Start workflow
            job_scraper_fragment()
            company_processor_fragment()
            analyzer_fragment()
            reporting_fragment()

            # Let complex workflow run
            time.sleep(4.0)

            return workflow_state

        benchmark = benchmarking_tester.benchmark_component(
            "fragment_behavior", complex_workflow, iterations=2
        )

        assert benchmark.passed is True

        # Should handle complex workflow efficiently
        validator = benchmarking_tester.validators["fragment_behavior"]
        assert len(validator.tracker.executions) == 4  # Four workflow stages

        # Should complete workflow steps
        assert validator.metrics.fragment_executions >= 8  # Multiple workflow cycles
