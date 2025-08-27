"""Fragment Performance Optimization Utilities.

This module provides comprehensive performance optimization utilities for st.fragment()
components, implementing intelligent refresh control, performance monitoring, and
automatic optimization based on runtime metrics.

Key features:
- Intelligent refresh control based on activity state
- Dynamic interval adjustment based on performance metrics
- Memory usage optimization and cleanup
- Performance-based fragment activation/deactivation
- Automatic optimization suggestions and implementation

This supports the 30% performance improvement target through intelligent
fragment management and optimization.
"""

import logging
import time

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any

import streamlit as st

from src.ui.utils.fragment_orchestrator import get_fragment_orchestrator

logger = logging.getLogger(__name__)


def optimize_fragment_refresh(
    fragment_id: str,
    activity_check_func: Callable[[], bool] = None,
    min_interval: int = 1,
    max_interval: int = 30,
    performance_threshold: float = 0.5,
) -> str:
    """Optimize fragment refresh interval based on activity and performance.

    Args:
        fragment_id: Unique identifier for the fragment.
        activity_check_func: Function to check if system is active.
        min_interval: Minimum refresh interval in seconds.
        max_interval: Maximum refresh interval in seconds.
        performance_threshold: Performance threshold for adjustment.

    Returns:
        Optimized refresh interval as string (e.g., "2s").
    """
    try:
        # Check for auto-optimizations in session state
        auto_opts = st.session_state.get("fragment_auto_optimizations", {})
        if auto_opts.get("applied") and fragment_id in auto_opts.get("suggestions", {}):
            suggested = auto_opts["suggestions"][fragment_id]
            logger.debug(f"Using auto-optimization for {fragment_id}: {suggested}")
            return suggested

        # Get orchestrator performance data
        orchestrator = get_fragment_orchestrator()
        perf_summary = orchestrator.get_orchestration_summary()

        # Get performance data for this fragment
        perf_data = perf_summary.get("performance_summary", {})

        # Default interval
        interval = min_interval + 1  # 2s default

        # Adjust based on activity
        if activity_check_func:
            try:
                is_active = activity_check_func()
                if not is_active:
                    # Increase interval when inactive
                    interval = min(max_interval, interval * 2)
                    logger.debug(
                        f"Fragment {fragment_id} inactive, interval: {interval}s"
                    )
            except Exception as e:
                logger.warning(f"Activity check failed for {fragment_id}: {e}")

        # Adjust based on performance metrics
        fragment_perf = st.session_state.get("fragment_performance", {}).get(
            fragment_id
        )
        if fragment_perf:
            avg_time = getattr(fragment_perf, "average_update_time", 0)
            error_count = getattr(fragment_perf, "error_count", 0)
            update_count = max(getattr(fragment_perf, "update_count", 1), 1)

            # Adjust for slow performance
            if avg_time > performance_threshold:
                interval = min(max_interval, interval + int(avg_time))
                logger.debug(
                    f"Fragment {fragment_id} slow ({avg_time:.2f}s), increased interval: {interval}s"
                )

            # Adjust for errors
            error_rate = error_count / update_count
            if error_rate > 0.1:  # >10% error rate
                interval = min(max_interval, interval * 2)
                logger.debug(
                    f"Fragment {fragment_id} high error rate ({error_rate:.1%}), increased interval: {interval}s"
                )

        return f"{interval}s"

    except Exception:
        logger.exception(f"Failed to optimize refresh for {fragment_id}")
        return f"{min_interval + 1}s"  # Safe fallback


def performance_aware_fragment(
    fragment_id: str,
    run_every: str = "2s",
    activity_check: Callable[[], bool] = None,
    enable_optimization: bool = True,
    performance_threshold: float = 0.5,
):
    """Decorator for creating performance-optimized fragments.

    This decorator wraps st.fragment() with intelligent performance optimizations
    including dynamic interval adjustment and activity-based refresh control.

    Args:
        fragment_id: Unique identifier for the fragment.
        run_every: Base refresh interval.
        activity_check: Function to check if system is active.
        enable_optimization: Whether to enable performance optimizations.
        performance_threshold: Performance threshold for optimization.

    Returns:
        Decorated fragment function with performance optimizations.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply performance optimizations if enabled
            if enable_optimization:
                optimized_interval = optimize_fragment_refresh(
                    fragment_id,
                    activity_check,
                    performance_threshold=performance_threshold,
                )
            else:
                optimized_interval = run_every

            # Create optimized fragment function
            @st.fragment(run_every=optimized_interval)
            def optimized_fragment():
                return _execute_with_performance_monitoring(
                    fragment_id, func, *args, **kwargs
                )

            return optimized_fragment()

        return wrapper

    return decorator


def _execute_with_performance_monitoring(
    fragment_id: str, func: Callable, *args, **kwargs
) -> Any:
    """Execute fragment function with performance monitoring.

    Args:
        fragment_id: Fragment identifier for monitoring.
        func: Fragment function to execute.
        *args: Function arguments.
        **kwargs: Function keyword arguments.

    Returns:
        Function result.
    """
    try:
        # Start performance monitoring
        start_time = time.perf_counter()

        # Execute fragment function
        result = func(*args, **kwargs)

        # Record performance metrics
        end_time = time.perf_counter()
        duration = end_time - start_time

        _record_fragment_performance(fragment_id, duration, success=True)

        return result

    except Exception as e:
        # Record error
        _record_fragment_performance(fragment_id, 0, success=False, error=e)
        raise


def _record_fragment_performance(
    fragment_id: str, duration: float, success: bool = True, error: Exception = None
) -> None:
    """Record fragment performance metrics.

    Args:
        fragment_id: Fragment identifier.
        duration: Execution duration in seconds.
        success: Whether execution was successful.
        error: Exception if execution failed.
    """
    try:
        # Get or create orchestrator
        orchestrator = get_fragment_orchestrator()

        if success:
            orchestrator.performance_monitor.end_update(
                fragment_id, time.perf_counter() - duration
            )
        else:
            orchestrator.performance_monitor.record_error(fragment_id, error)

    except Exception:
        # Silently fail to avoid disrupting fragment execution
        pass


def should_skip_fragment_update(
    fragment_id: str,
    last_update_key: str = None,
    min_update_interval: timedelta = timedelta(seconds=1),
) -> bool:
    """Determine if fragment update should be skipped for performance.

    Args:
        fragment_id: Fragment identifier.
        last_update_key: Session state key for last update time.
        min_update_interval: Minimum time between updates.

    Returns:
        True if update should be skipped.
    """
    try:
        if not last_update_key:
            last_update_key = f"last_update_{fragment_id}"

        last_update = st.session_state.get(last_update_key)
        if not last_update:
            # First update
            st.session_state[last_update_key] = datetime.now(UTC)
            return False

        # Check if enough time has passed
        now = datetime.now(UTC)
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update)

        if (now - last_update) < min_update_interval:
            return True  # Skip update

        # Update last update time
        st.session_state[last_update_key] = now
        return False  # Allow update

    except Exception:
        logger.exception(f"Error checking update skip for {fragment_id}")
        return False  # Allow update on error


def optimize_fragment_memory(
    fragment_id: str, max_history_items: int = 100, cleanup_threshold: int = 1000
) -> None:
    """Optimize memory usage for fragment state.

    Args:
        fragment_id: Fragment identifier.
        max_history_items: Maximum history items to keep.
        cleanup_threshold: Threshold for triggering cleanup.
    """
    try:
        # Clean up fragment-specific session state
        fragment_keys = [k for k in st.session_state.keys() if fragment_id in k]

        for key in fragment_keys:
            value = st.session_state[key]

            # Clean up list/deque objects
            if hasattr(value, "__len__") and hasattr(value, "pop"):
                if len(value) > max_history_items:
                    # Keep only recent items
                    while len(value) > max_history_items:
                        if hasattr(value, "popleft"):
                            value.popleft()  # deque
                        else:
                            value.pop(0)  # list
                    logger.debug(f"Cleaned up {key}: kept {len(value)} items")

        # Global cleanup if threshold exceeded
        total_keys = len(st.session_state.keys())
        if total_keys > cleanup_threshold:
            logger.warning(
                f"Session state size ({total_keys}) exceeds threshold, triggering cleanup"
            )
            _global_fragment_cleanup()

    except Exception:
        logger.exception(f"Error optimizing memory for {fragment_id}")


def _global_fragment_cleanup() -> None:
    """Perform global fragment state cleanup."""
    try:
        # Clean up old fragment performance data
        current_time = datetime.now(UTC)
        cutoff_time = current_time - timedelta(hours=1)  # Keep last hour

        # Clean up performance history
        history_key = "fragment_performance_history"
        if history_key in st.session_state:
            history = st.session_state[history_key]
            if hasattr(history, "__iter__"):
                cleaned_history = []
                for item in history:
                    if isinstance(item, dict) and "timestamp" in item:
                        try:
                            timestamp = datetime.fromisoformat(item["timestamp"])
                            if timestamp > cutoff_time:
                                cleaned_history.append(item)
                        except ValueError:
                            continue  # Skip invalid timestamps
                    else:
                        cleaned_history.append(item)

                st.session_state[history_key] = cleaned_history
                logger.debug(
                    f"Cleaned performance history: {len(cleaned_history)} items kept"
                )

        # Clean up old fragment messages
        msg_queue_key = "fragment_message_queue"
        if msg_queue_key in st.session_state:
            queue = st.session_state[msg_queue_key]
            if hasattr(queue, "__len__") and len(queue) > 50:
                # Keep only recent messages
                while len(queue) > 50:
                    if hasattr(queue, "popleft"):
                        queue.popleft()
                    else:
                        break
                logger.debug(f"Cleaned message queue: {len(queue)} messages kept")

    except Exception:
        logger.exception("Error in global fragment cleanup")


def get_fragment_performance_summary() -> dict[str, Any]:
    """Get comprehensive fragment performance summary.

    Returns:
        Dictionary with performance summary and optimization recommendations.
    """
    try:
        orchestrator = get_fragment_orchestrator()
        return orchestrator.get_orchestration_summary()
    except Exception as e:
        logger.exception("Failed to get fragment performance summary")
        return {"error": str(e)}


def apply_performance_optimizations(auto_optimize: bool = True) -> dict[str, Any]:
    """Apply performance optimizations across all fragments.

    Args:
        auto_optimize: Whether to automatically apply optimizations.

    Returns:
        Dictionary with optimization results.
    """
    try:
        orchestrator = get_fragment_orchestrator()
        perf_summary = orchestrator.get_orchestration_summary()

        # Get optimization suggestions
        suggestions = perf_summary.get("optimization_suggestions", {})

        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "suggestions_count": len(suggestions),
            "applied": False,
            "suggestions": suggestions,
        }

        if auto_optimize and suggestions:
            # Store optimizations in session state
            st.session_state.fragment_auto_optimizations = {
                "timestamp": datetime.now(UTC).isoformat(),
                "suggestions": suggestions,
                "applied": True,
            }

            results["applied"] = True
            logger.info(f"Applied {len(suggestions)} fragment optimizations")

        return results

    except Exception as e:
        logger.exception("Failed to apply performance optimizations")
        return {"error": str(e), "applied": False}


# Context managers for performance monitoring


class FragmentPerformanceContext:
    """Context manager for fragment performance monitoring."""

    def __init__(self, fragment_id: str):
        """Initialize performance context.

        Args:
            fragment_id: Fragment identifier.
        """
        self.fragment_id = fragment_id
        self.start_time = None

    def __enter__(self):
        """Enter performance monitoring context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit performance monitoring context."""
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            success = exc_type is None
            error = exc_val if exc_type else None

            _record_fragment_performance(
                self.fragment_id, duration, success=success, error=error
            )
