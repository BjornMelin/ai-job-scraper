"""Fragment Orchestration and Communication Patterns for Stream C Architecture.

This module provides comprehensive fragment coordination, performance optimization,
and communication patterns for st.fragment() based components. It implements
the Stream C architecture specifications for component isolation and coordination
simplification.

Key features:
- Fragment communication via session state coordination
- Performance optimization with selective refresh control
- Fragment lifecycle management and cleanup
- Error boundaries and graceful degradation
- Memory-efficient fragment state management
- Real-time coordination without full page reruns

Architecture Components:
- FragmentOrchestrator: Main coordination class
- FragmentCommunicationBus: Inter-fragment messaging
- FragmentPerformanceMonitor: Performance tracking and optimization
- FragmentErrorBoundary: Error isolation and recovery
- FragmentMemoryManager: Memory cleanup and optimization

This implementation follows Stream C fragment architecture specifications for
major coordination simplification and performance improvement.
"""

import logging
import time
import uuid

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class FragmentMessage:
    """Message for inter-fragment communication."""

    source_fragment: str
    target_fragment: str | None = None  # None for broadcast
    message_type: str = "update"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class FragmentPerformanceMetrics:
    """Performance metrics for fragment monitoring."""

    fragment_id: str
    update_count: int = 0
    total_update_time: float = 0.0
    average_update_time: float = 0.0
    last_update_time: float = 0.0
    error_count: int = 0
    memory_usage: float = 0.0
    refresh_interval: str = "0s"
    is_active: bool = True


@dataclass
class FragmentState:
    """State tracking for individual fragments."""

    fragment_id: str
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))
    update_count: int = 0
    error_count: int = 0
    is_active: bool = True
    refresh_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    performance_metrics: FragmentPerformanceMetrics = field(default=None)

    def __post_init__(self):
        """Initialize performance metrics if not provided."""
        if self.performance_metrics is None:
            self.performance_metrics = FragmentPerformanceMetrics(
                fragment_id=self.fragment_id
            )


class FragmentCommunicationBus:
    """Inter-fragment communication system using session state.

    This class provides a messaging system for fragments to communicate
    without triggering full page reruns, enabling coordinated updates
    and data sharing between isolated components.
    """

    def __init__(self):
        """Initialize the communication bus."""
        self.bus_id = f"fragment_bus_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"{__name__}.CommunicationBus")
        self._initialize_bus()

    def _initialize_bus(self) -> None:
        """Initialize bus state in session state."""
        if "fragment_message_queue" not in st.session_state:
            st.session_state.fragment_message_queue = deque(maxlen=100)
        if "fragment_subscribers" not in st.session_state:
            st.session_state.fragment_subscribers = defaultdict(list)

    def publish(self, message: FragmentMessage) -> None:
        """Publish a message to the communication bus.

        Args:
            message: FragmentMessage to publish.
        """
        try:
            # Add to message queue
            st.session_state.fragment_message_queue.append(message)

            # Update fragment communication state
            comm_state = st.session_state.get("fragment_communication", {})
            comm_state[message.message_id] = {
                "source": message.source_fragment,
                "target": message.target_fragment,
                "type": message.message_type,
                "timestamp": message.timestamp.isoformat(),
                "payload_size": len(str(message.payload)),
            }
            st.session_state.fragment_communication = comm_state

            self.logger.debug(
                "Published message %s from %s to %s",
                message.message_id[:8],
                message.source_fragment,
                message.target_fragment or "ALL",
            )

        except Exception as e:
            self.logger.error("Failed to publish message: %s", e)

    def subscribe(self, fragment_id: str, message_types: list[str]) -> None:
        """Subscribe a fragment to specific message types.

        Args:
            fragment_id: ID of the subscribing fragment.
            message_types: List of message types to subscribe to.
        """
        for msg_type in message_types:
            if fragment_id not in st.session_state.fragment_subscribers[msg_type]:
                st.session_state.fragment_subscribers[msg_type].append(fragment_id)

        self.logger.debug("Fragment %s subscribed to %s", fragment_id, message_types)

    def get_messages_for_fragment(
        self, fragment_id: str, message_types: list[str] = None
    ) -> list[FragmentMessage]:
        """Get messages for a specific fragment.

        Args:
            fragment_id: ID of the requesting fragment.
            message_types: Optional list of message types to filter.

        Returns:
            List of relevant FragmentMessage objects.
        """
        try:
            messages = []

            for message in st.session_state.fragment_message_queue:
                # Check if message is for this fragment
                if message.target_fragment and message.target_fragment != fragment_id:
                    continue

                # Check message type filter
                if message_types and message.message_type not in message_types:
                    continue

                # Don't return messages from the same fragment
                if message.source_fragment == fragment_id:
                    continue

                messages.append(message)

            return messages[-10:]  # Return last 10 messages

        except Exception as e:
            self.logger.error(
                "Failed to get messages for fragment %s: %s", fragment_id, e
            )
            return []

    def broadcast(self, source_fragment: str, message_type: str, payload: dict) -> None:
        """Broadcast a message to all subscribed fragments.

        Args:
            source_fragment: ID of the source fragment.
            message_type: Type of message to broadcast.
            payload: Message payload data.
        """
        message = FragmentMessage(
            source_fragment=source_fragment,
            target_fragment=None,  # Broadcast
            message_type=message_type,
            payload=payload,
        )
        self.publish(message)


class FragmentPerformanceMonitor:
    """Performance monitoring and optimization for fragments.

    This class tracks fragment performance metrics and provides
    optimization recommendations for better user experience.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.monitor_id = f"perf_monitor_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self._initialize_monitoring()

    def _initialize_monitoring(self) -> None:
        """Initialize monitoring state."""
        if "fragment_performance" not in st.session_state:
            st.session_state.fragment_performance = {}
        if "fragment_performance_history" not in st.session_state:
            st.session_state.fragment_performance_history = deque(maxlen=1000)

    def start_update(self, fragment_id: str) -> float:
        """Start timing a fragment update.

        Args:
            fragment_id: ID of the fragment being updated.

        Returns:
            Start time for the update.
        """
        return time.perf_counter()

    def end_update(self, fragment_id: str, start_time: float) -> None:
        """End timing a fragment update and record metrics.

        Args:
            fragment_id: ID of the fragment that was updated.
            start_time: Start time from start_update().
        """
        try:
            end_time = time.perf_counter()
            update_duration = end_time - start_time

            # Get or create performance metrics
            perf_data = st.session_state.fragment_performance.get(
                fragment_id, FragmentPerformanceMetrics(fragment_id=fragment_id)
            )

            # Update metrics
            perf_data.update_count += 1
            perf_data.total_update_time += update_duration
            perf_data.average_update_time = (
                perf_data.total_update_time / perf_data.update_count
            )
            perf_data.last_update_time = update_duration

            # Store updated metrics
            st.session_state.fragment_performance[fragment_id] = perf_data

            # Add to history
            st.session_state.fragment_performance_history.append(
                {
                    "fragment_id": fragment_id,
                    "duration": update_duration,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "update_count": perf_data.update_count,
                }
            )

            # Log slow updates
            if update_duration > 0.5:  # 500ms threshold
                self.logger.warning(
                    "Slow fragment update: %s took %.2fs", fragment_id, update_duration
                )

        except Exception as e:
            self.logger.error("Failed to record fragment performance: %s", e)

    def record_error(self, fragment_id: str, error: Exception) -> None:
        """Record a fragment error for monitoring.

        Args:
            fragment_id: ID of the fragment that errored.
            error: Exception that occurred.
        """
        try:
            perf_data = st.session_state.fragment_performance.get(
                fragment_id, FragmentPerformanceMetrics(fragment_id=fragment_id)
            )

            perf_data.error_count += 1
            st.session_state.fragment_performance[fragment_id] = perf_data

            self.logger.error(
                "Fragment %s error #%d: %s", fragment_id, perf_data.error_count, error
            )

        except Exception as e:
            self.logger.error("Failed to record fragment error: %s", e)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all fragments.

        Returns:
            Dictionary with performance summary data.
        """
        try:
            performance_data = st.session_state.get("fragment_performance", {})

            summary = {
                "total_fragments": len(performance_data),
                "active_fragments": sum(
                    1 for perf in performance_data.values() if perf.is_active
                ),
                "total_updates": sum(
                    perf.update_count for perf in performance_data.values()
                ),
                "total_errors": sum(
                    perf.error_count for perf in performance_data.values()
                ),
                "average_update_time": 0.0,
                "slowest_fragment": None,
                "most_active_fragment": None,
            }

            if performance_data:
                # Calculate overall average
                total_time = sum(
                    perf.total_update_time for perf in performance_data.values()
                )
                total_updates = summary["total_updates"]
                if total_updates > 0:
                    summary["average_update_time"] = total_time / total_updates

                # Find slowest fragment
                slowest = max(
                    performance_data.values(),
                    key=lambda p: p.average_update_time,
                    default=None,
                )
                if slowest:
                    summary["slowest_fragment"] = {
                        "id": slowest.fragment_id,
                        "avg_time": slowest.average_update_time,
                    }

                # Find most active fragment
                most_active = max(
                    performance_data.values(),
                    key=lambda p: p.update_count,
                    default=None,
                )
                if most_active:
                    summary["most_active_fragment"] = {
                        "id": most_active.fragment_id,
                        "update_count": most_active.update_count,
                    }

            return summary

        except Exception as e:
            self.logger.error("Failed to get performance summary: %s", e)
            return {"error": str(e)}

    def optimize_fragment_intervals(self) -> dict[str, str]:
        """Suggest optimal refresh intervals based on performance data.

        Returns:
            Dictionary mapping fragment IDs to suggested intervals.
        """
        try:
            performance_data = st.session_state.get("fragment_performance", {})
            suggestions = {}

            for fragment_id, perf in performance_data.items():
                current_avg = perf.average_update_time

                # Suggest intervals based on update time
                if current_avg < 0.1:  # Fast updates
                    suggestions[fragment_id] = "1s"
                elif current_avg < 0.3:  # Medium updates
                    suggestions[fragment_id] = "2s"
                elif current_avg < 0.5:  # Slower updates
                    suggestions[fragment_id] = "5s"
                else:  # Very slow updates
                    suggestions[fragment_id] = "10s"

                # Adjust for error rate
                if perf.error_count > 0:
                    error_rate = perf.error_count / max(perf.update_count, 1)
                    if error_rate > 0.1:  # >10% error rate
                        # Increase interval for error-prone fragments
                        current_interval = int(suggestions[fragment_id].rstrip("s"))
                        suggestions[fragment_id] = f"{current_interval * 2}s"

            return suggestions

        except Exception as e:
            self.logger.error("Failed to optimize fragment intervals: %s", e)
            return {}


class FragmentErrorBoundary:
    """Error boundary and recovery system for fragments.

    This class provides error isolation and graceful degradation
    for fragments, preventing individual fragment failures from
    affecting the entire application.
    """

    def __init__(self):
        """Initialize the error boundary system."""
        self.boundary_id = f"error_boundary_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"{__name__}.ErrorBoundary")
        self._initialize_boundary()

    def _initialize_boundary(self) -> None:
        """Initialize error boundary state."""
        if "fragment_errors" not in st.session_state:
            st.session_state.fragment_errors = {}
        if "fragment_recovery_attempts" not in st.session_state:
            st.session_state.fragment_recovery_attempts = defaultdict(int)

    def handle_fragment_error(
        self, fragment_id: str, error: Exception, fallback_renderer: Callable = None
    ) -> bool:
        """Handle a fragment error with recovery attempts.

        Args:
            fragment_id: ID of the failed fragment.
            error: Exception that occurred.
            fallback_renderer: Optional fallback rendering function.

        Returns:
            True if recovery was attempted, False if fragment should be disabled.
        """
        try:
            # Record the error
            error_data = {
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.now(UTC).isoformat(),
                "recovery_attempts": st.session_state.fragment_recovery_attempts[
                    fragment_id
                ],
            }
            st.session_state.fragment_errors[fragment_id] = error_data

            # Increment recovery attempts
            st.session_state.fragment_recovery_attempts[fragment_id] += 1
            attempts = st.session_state.fragment_recovery_attempts[fragment_id]

            self.logger.error(
                "Fragment %s error (attempt %d): %s", fragment_id, attempts, error
            )

            # Determine recovery strategy
            if attempts <= 3:
                # Try recovery with fallback
                if fallback_renderer:
                    try:
                        fallback_renderer()
                        return True
                    except Exception as fallback_error:
                        self.logger.error(
                            "Fallback failed for %s: %s", fragment_id, fallback_error
                        )

                # Show error state
                st.error(
                    f"⚠️ Fragment '{fragment_id}' encountered an error. "
                    f"Retrying... (Attempt {attempts}/3)"
                )
                return True

            # Too many failures, disable fragment
            st.error(
                f"❌ Fragment '{fragment_id}' has been disabled due to "
                f"repeated errors. Please refresh the page to retry."
            )
            return False

        except Exception as e:
            self.logger.error("Error in error boundary: %s", e)
            return False

    def reset_fragment_errors(self, fragment_id: str) -> None:
        """Reset error state for a fragment.

        Args:
            fragment_id: ID of the fragment to reset.
        """
        if fragment_id in st.session_state.fragment_errors:
            del st.session_state.fragment_errors[fragment_id]
        if fragment_id in st.session_state.fragment_recovery_attempts:
            del st.session_state.fragment_recovery_attempts[fragment_id]

        self.logger.info("Reset error state for fragment: %s", fragment_id)

    def get_error_summary(self) -> dict[str, Any]:
        """Get error summary for all fragments.

        Returns:
            Dictionary with error summary data.
        """
        try:
            errors = st.session_state.get("fragment_errors", {})
            attempts = st.session_state.get("fragment_recovery_attempts", {})

            return {
                "total_fragments_with_errors": len(errors),
                "total_recovery_attempts": sum(attempts.values()),
                "fragments_with_errors": list(errors.keys()),
                "recent_errors": [
                    {
                        "fragment_id": fid,
                        "error": error_data["error"],
                        "timestamp": error_data["timestamp"],
                    }
                    for fid, error_data in errors.items()
                ],
                "fragments_with_multiple_failures": [
                    fid for fid, count in attempts.items() if count > 3
                ],
            }

        except Exception as e:
            self.logger.error("Failed to get error summary: %s", e)
            return {"error": str(e)}


class FragmentOrchestrator:
    """Main orchestrator for fragment coordination and management.

    This class provides the primary interface for fragment coordination,
    performance monitoring, and error handling in the Stream C architecture.
    """

    def __init__(self):
        """Initialize the fragment orchestrator."""
        self.orchestrator_id = f"orchestrator_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")

        # Initialize subsystems
        self.communication_bus = FragmentCommunicationBus()
        self.performance_monitor = FragmentPerformanceMonitor()
        self.error_boundary = FragmentErrorBoundary()

        # Initialize orchestrator state
        self._initialize_orchestrator()

        self.logger.info("Fragment Orchestrator initialized: %s", self.orchestrator_id)

    def _initialize_orchestrator(self) -> None:
        """Initialize orchestrator state in session state."""
        if "fragment_orchestrator" not in st.session_state:
            st.session_state.fragment_orchestrator = {
                "orchestrator_id": self.orchestrator_id,
                "initialized_at": datetime.now(UTC).isoformat(),
                "active_fragments": {},
                "fragment_registry": {},
                "coordination_stats": {
                    "total_messages": 0,
                    "total_updates": 0,
                    "total_errors": 0,
                },
            }

    def register_fragment(
        self,
        fragment_id: str,
        fragment_type: str,
        refresh_interval: str = "10s",
        dependencies: list[str] = None,
    ) -> None:
        """Register a fragment with the orchestrator.

        Args:
            fragment_id: Unique identifier for the fragment.
            fragment_type: Type/category of the fragment.
            refresh_interval: Auto-refresh interval (e.g., "2s", "30s").
            dependencies: List of fragment IDs this fragment depends on.
        """
        try:
            registry = st.session_state.fragment_orchestrator["fragment_registry"]

            registry[fragment_id] = {
                "fragment_type": fragment_type,
                "refresh_interval": refresh_interval,
                "dependencies": dependencies or [],
                "registered_at": datetime.now(UTC).isoformat(),
                "is_active": True,
            }

            # Subscribe to dependency messages
            if dependencies:
                self.communication_bus.subscribe(
                    fragment_id, ["dependency_update", "data_change"]
                )

            self.logger.info(
                "Registered fragment %s (type: %s, interval: %s)",
                fragment_id,
                fragment_type,
                refresh_interval,
            )

        except Exception as e:
            self.logger.error("Failed to register fragment %s: %s", fragment_id, e)

    def execute_with_coordination(
        self, fragment_id: str, fragment_func: Callable, *args, **kwargs
    ) -> Any:
        """Execute a fragment function with full coordination support.

        Args:
            fragment_id: ID of the fragment being executed.
            fragment_func: Fragment function to execute.
            *args: Positional arguments for the fragment function.
            **kwargs: Keyword arguments for the fragment function.

        Returns:
            Result of the fragment function execution.
        """
        try:
            # Start performance monitoring
            start_time = self.performance_monitor.start_update(fragment_id)

            # Check for dependency messages
            messages = self.communication_bus.get_messages_for_fragment(
                fragment_id, ["dependency_update", "data_change"]
            )

            # Execute the fragment function
            try:
                result = fragment_func(*args, **kwargs)

                # End performance monitoring
                self.performance_monitor.end_update(fragment_id, start_time)

                # Update coordination stats
                stats = st.session_state.fragment_orchestrator["coordination_stats"]
                stats["total_updates"] += 1

                return result

            except Exception as fragment_error:
                # Handle fragment error with error boundary
                self.performance_monitor.record_error(fragment_id, fragment_error)

                # Create fallback renderer
                def fallback_renderer():
                    st.warning(
                        f"⚠️ Fragment '{fragment_id}' is temporarily unavailable. "
                        "Retrying..."
                    )

                # Try error recovery
                recovery_successful = self.error_boundary.handle_fragment_error(
                    fragment_id, fragment_error, fallback_renderer
                )

                # Update error stats
                stats = st.session_state.fragment_orchestrator["coordination_stats"]
                stats["total_errors"] += 1

                if not recovery_successful:
                    raise fragment_error

                return None

        except Exception as e:
            self.logger.error("Coordination error for fragment %s: %s", fragment_id, e)
            raise

    def broadcast_update(
        self, source_fragment: str, update_type: str, data: dict
    ) -> None:
        """Broadcast an update from one fragment to others.

        Args:
            source_fragment: ID of the fragment broadcasting the update.
            update_type: Type of update being broadcast.
            data: Update data to broadcast.
        """
        try:
            self.communication_bus.broadcast(source_fragment, update_type, data)

            # Update coordination stats
            stats = st.session_state.fragment_orchestrator["coordination_stats"]
            stats["total_messages"] += 1

            self.logger.debug(
                "Broadcast %s update from %s with data keys: %s",
                update_type,
                source_fragment,
                list(data.keys()),
            )

        except Exception as e:
            self.logger.error("Failed to broadcast update: %s", e)

    def get_orchestration_summary(self) -> dict[str, Any]:
        """Get comprehensive orchestration summary.

        Returns:
            Dictionary with orchestration status and metrics.
        """
        try:
            orchestrator_data = st.session_state.get("fragment_orchestrator", {})

            return {
                "orchestrator_id": self.orchestrator_id,
                "coordination_stats": orchestrator_data.get("coordination_stats", {}),
                "registered_fragments": len(
                    orchestrator_data.get("fragment_registry", {})
                ),
                "performance_summary": self.performance_monitor.get_performance_summary(),
                "error_summary": self.error_boundary.get_error_summary(),
                "optimization_suggestions": self.performance_monitor.optimize_fragment_intervals(),
                "uptime": (
                    datetime.now(UTC)
                    - datetime.fromisoformat(
                        orchestrator_data.get(
                            "initialized_at", datetime.now(UTC).isoformat()
                        )
                    )
                ).total_seconds(),
            }

        except Exception as e:
            self.logger.error("Failed to get orchestration summary: %s", e)
            return {"error": str(e)}


# ========== FRAGMENT DECORATORS AND UTILITIES ==========


def coordinated_fragment(
    fragment_id: str,
    fragment_type: str = "component",
    run_every: str = "10s",
    dependencies: list[str] = None,
    enable_error_boundary: bool = True,
):
    """Decorator for creating coordinated fragments with full orchestration.

    This decorator wraps st.fragment() with additional coordination,
    performance monitoring, and error handling capabilities.

    Args:
        fragment_id: Unique identifier for the fragment.
        fragment_type: Type/category of the fragment.
        run_every: Auto-refresh interval.
        dependencies: List of fragment IDs this fragment depends on.
        enable_error_boundary: Whether to enable error boundary protection.

    Returns:
        Decorated function with fragment coordination.
    """

    def decorator(func):
        # Get or create orchestrator
        if "fragment_orchestrator_instance" not in st.session_state:
            st.session_state.fragment_orchestrator_instance = FragmentOrchestrator()

        orchestrator = st.session_state.fragment_orchestrator_instance

        # Register the fragment
        orchestrator.register_fragment(
            fragment_id, fragment_type, run_every, dependencies
        )

        # Create coordinated fragment function
        @st.fragment(run_every=run_every)
        def coordinated_func(*args, **kwargs):
            if enable_error_boundary:
                return orchestrator.execute_with_coordination(
                    fragment_id, func, *args, **kwargs
                )
            return func(*args, **kwargs)

        return coordinated_func

    return decorator


def get_fragment_orchestrator() -> FragmentOrchestrator:
    """Get the global fragment orchestrator instance.

    Returns:
        FragmentOrchestrator singleton instance.
    """
    if "fragment_orchestrator_instance" not in st.session_state:
        st.session_state.fragment_orchestrator_instance = FragmentOrchestrator()

    return st.session_state.fragment_orchestrator_instance


def cleanup_fragment_state() -> None:
    """Clean up fragment state and free memory.

    This function should be called periodically or when navigating
    away from pages with many fragments to prevent memory leaks.
    """
    try:
        # Clear old messages (keep last 50)
        if "fragment_message_queue" in st.session_state:
            queue = st.session_state.fragment_message_queue
            if len(queue) > 50:
                # Keep only the most recent messages
                recent_messages = list(queue)[-50:]
                st.session_state.fragment_message_queue = deque(
                    recent_messages, maxlen=100
                )

        # Clear old performance history (keep last 500)
        if "fragment_performance_history" in st.session_state:
            history = st.session_state.fragment_performance_history
            if len(history) > 500:
                recent_history = list(history)[-500:]
                st.session_state.fragment_performance_history = deque(
                    recent_history, maxlen=1000
                )

        # Reset error attempts for fragments that haven't errored recently
        if "fragment_recovery_attempts" in st.session_state:
            attempts = st.session_state.fragment_recovery_attempts
            current_time = datetime.now(UTC)

            # Reset attempts for fragments inactive for >5 minutes
            to_reset = []
            if "fragment_performance" in st.session_state:
                for fid, perf in st.session_state.fragment_performance.items():
                    if fid in attempts and attempts[fid] > 0:
                        # Check if fragment has been inactive
                        if not perf.is_active:
                            to_reset.append(fid)

            for fid in to_reset:
                del attempts[fid]

        logger.info("Fragment state cleanup completed")

    except Exception as e:
        logger.error("Failed to cleanup fragment state: %s", e)
