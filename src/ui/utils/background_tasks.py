"""Background task management utilities for Streamlit applications.

This module provides utilities for managing background scraping tasks in the AI Job
Scraper Streamlit UI, including progress tracking, real-time updates, and proper
session state management.

Key features:
- Non-blocking background task execution
- Real-time progress reporting via callbacks
- Thread-safe integration with Streamlit session state
- Proper error handling and task lifecycle management
- Support for both threading and asyncio patterns

Example usage:
    # Start a background scraping task
    task_id = start_background_scraping()

    # Check if scraping is active
    if is_scraping_active():
        progress = get_scraping_progress(task_id)
        if progress:
            st.progress(progress.progress / 100)
            st.write(progress.message)

    # Stop all scraping tasks
    stop_all_scraping()
"""

import atexit
import copy
import inspect
import logging
import threading

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import streamlit as st


# Import the scraper function we need to launch in background
from src.scraper import scrape_all
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Information about a background task."""

    id: str
    name: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: Any | None = None
    thread: threading.Thread | None = None
    future: Future | None = None


@dataclass
class ProgressInfo:
    """Progress information for UI updates."""

    progress: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanyProgress:
    """Progress information for individual company scraping."""

    name: str
    status: str = "Pending"  # Pending, Scraping, Completed, Error
    jobs_found: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None


class BackgroundTaskManager:
    """Base class for managing background tasks without blocking the UI.

    This class provides the core functionality for launching and managing
    background tasks, with support for progress callbacks and error handling.
    """

    def __init__(self, max_workers: int = 3, max_task_history: int = 100):
        """Initialize the task manager.

        Args:
            max_workers: Maximum number of concurrent background tasks.
            max_task_history: Maximum number of completed tasks to keep in memory.
        """
        self.active_tasks: dict[str, TaskInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self.max_task_history = max_task_history
        self._shutdown = False

        # Register cleanup handlers
        atexit.register(self.shutdown)

        logger.info(
            "BackgroundTaskManager initialized with %d max workers", max_workers
        )

    def start_task(
        self,
        task_func: Callable,
        task_name: str,
        progress_callback: Callable[[str, float, str], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
        **task_kwargs,
    ) -> str:
        """Start a background task.

        Args:
            task_func: Function to execute in background.
            task_name: Human-readable task name.
            progress_callback: Optional callback for progress updates.
                Signature: (task_id, progress_percentage, message)
            error_callback: Optional callback for error handling.
                Signature: (task_id, error_message)
            **task_kwargs: Additional arguments to pass to task_func.

        Returns:
            str: Unique task ID for tracking.
        """
        task_id = str(uuid4())

        # Create task info
        task_info = TaskInfo(
            id=task_id,
            name=task_name,
            status="pending",
            started_at=datetime.now(timezone.utc),
        )

        with self._lock:
            self.active_tasks[task_id] = task_info

        # Define wrapper function that handles progress and error callbacks
        def task_wrapper():
            try:
                # Update status to running
                self._update_task_status(task_id, "running", 0.0, "Starting task...")

                # Create a copy of task_kwargs to prevent concurrent modification
                safe_kwargs = copy.deepcopy(task_kwargs)

                # Execute the actual task function
                if progress_callback:
                    # Pass progress callback to task function if it accepts it
                    sig = inspect.signature(task_func)
                    if "progress_callback" in sig.parameters:
                        safe_kwargs["progress_callback"] = (
                            lambda p, msg="": progress_callback(task_id, p, msg)
                        )

                result = task_func(**safe_kwargs)

                # Mark as completed
                self._update_task_status(
                    task_id, "completed", 100.0, "Task completed successfully"
                )

                with self._lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id].result = result
                        self.active_tasks[task_id].completed_at = datetime.now(
                            timezone.utc
                        )

                logger.info("Task %s (%s) completed successfully", task_id, task_name)

            except Exception as e:
                error_msg = f"Task failed: {e!s}"
                logger.exception(
                    "Task %s (%s) failed: %s", task_id, task_name, error_msg
                )

                # Mark as failed
                self._update_task_status(task_id, "failed", None, error_msg)

                with self._lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id].error = error_msg
                        self.active_tasks[task_id].completed_at = datetime.now(
                            timezone.utc
                        )

                # Call error callback if provided
                if error_callback:
                    try:
                        error_callback(task_id, error_msg)
                    except Exception:
                        logger.exception("Error callback failed")

        # Submit task to thread pool
        future = self.executor.submit(task_wrapper)

        with self._lock:
            self.active_tasks[task_id].future = future

        logger.info("Started background task %s (%s)", task_id, task_name)
        return task_id

    def get_task_status(self, task_id: str) -> TaskInfo | None:
        """Get current status of a task.

        Args:
            task_id: Task ID to check.

        Returns:
            TaskInfo object if task exists, None otherwise.
        """
        with self._lock:
            return self.active_tasks.get(task_id)

    def get_all_tasks(self) -> dict[str, TaskInfo]:
        """Get status of all tasks.

        Returns:
            Dictionary mapping task IDs to TaskInfo objects.
        """
        with self._lock:
            return self.active_tasks.copy()

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel.

        Returns:
            True if task was cancelled, False if not found or already completed.
        """
        with self._lock:
            task_info = self.active_tasks.get(task_id)
            if not task_info or task_info.status in ("completed", "failed"):
                return False

            if task_info.future:
                cancelled = task_info.future.cancel()
                if cancelled:
                    task_info.status = "cancelled"
                    task_info.completed_at = datetime.now(timezone.utc)
                    logger.info("Cancelled task %s", task_id)
                return cancelled

        return False

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Remove old completed tasks to prevent memory leaks.

        Args:
            max_age_hours: Maximum age in hours for completed tasks.

        Returns:
            Number of tasks cleaned up.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        cleaned = 0

        with self._lock:
            # Get all completed tasks
            completed_tasks = [
                (task_id, task_info)
                for task_id, task_info in self.active_tasks.items()
                if task_info.status in ("completed", "failed", "cancelled")
            ]

            # Sort by completion time (newest first)
            completed_tasks.sort(
                key=lambda x: x[1].completed_at.timestamp() if x[1].completed_at else 0,
                reverse=True,
            )

            # Remove old tasks beyond max_task_history limit
            task_ids_to_remove = []

            # First, remove tasks older than max_age_hours
            for task_id, task_info in completed_tasks:
                if (
                    task_info.completed_at
                    and task_info.completed_at.timestamp() < cutoff
                ):
                    task_ids_to_remove.append(task_id)

            # Then, if we still have too many tasks, remove oldest ones
            if len(completed_tasks) > self.max_task_history:
                excess_tasks = completed_tasks[self.max_task_history :]
                task_ids_to_remove.extend([task_id for task_id, _ in excess_tasks])

            # Remove duplicates and actually delete tasks
            unique_ids_to_remove = list(set(task_ids_to_remove))
            for task_id in unique_ids_to_remove:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                    cleaned += 1

        if cleaned > 0:
            logger.info("Cleaned up %d old completed tasks", cleaned)

        return cleaned

    def _update_task_status(
        self,
        task_id: str,
        status: str,
        progress: float | None = None,
        message: str = "",
    ) -> None:
        """Update task status internally.

        Args:
            task_id: Task ID to update.
            status: New status.
            progress: Progress percentage (0-100).
            message: Progress message.
        """
        with self._lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.status = status
                if progress is not None:
                    task_info.progress = progress
                if message:
                    task_info.message = message

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task manager and cleanup resources.

        Args:
            wait: Whether to wait for running tasks to complete.
        """
        if self._shutdown:
            return

        logger.info("Shutting down BackgroundTaskManager")
        self._shutdown = True

        # Cancel all pending tasks
        with self._lock:
            pending_tasks = [
                task_id
                for task_id, task_info in self.active_tasks.items()
                if task_info.status in ("pending", "running")
            ]

        for task_id in pending_tasks:
            self.cancel_task(task_id)

        # Shutdown executor
        self.executor.shutdown(wait=wait)

        with self._lock:
            self.active_tasks.clear()

        logger.info("BackgroundTaskManager shutdown complete")

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        try:
            self.shutdown(wait=False)
        except Exception as e:
            # Don't raise exceptions in destructor
            logger.warning("Error during BackgroundTaskManager cleanup: %s", e)


class StreamlitTaskManager(BackgroundTaskManager):
    """Streamlit-specific task manager with session state integration.

    This class extends BackgroundTaskManager to provide seamless integration
    with Streamlit's session state and UI update mechanisms. It handles the
    ScriptRunContext properly to enable Streamlit commands from background threads.
    """

    def __init__(
        self,
        max_workers: int = 3,
        max_task_history: int = 100,
        max_progress_entries: int = 50,
    ):
        """Initialize the Streamlit task manager.

        Args:
            max_workers: Maximum number of concurrent background tasks.
            max_task_history: Maximum number of completed tasks to keep in memory.
            max_progress_entries: Maximum number of progress entries to keep in
                session state.
        """
        super().__init__(max_workers, max_task_history)
        self.max_progress_entries = max_progress_entries
        self._ensure_session_state_keys()

        # Register Streamlit cleanup callback if available
        try:
            # This will work in newer versions of Streamlit
            st.runtime.get_instance().add_script_run_ctx_cleanup_callback(
                self._streamlit_cleanup
            )
        except (AttributeError, Exception):
            # Fallback for older versions or if callback system is not available
            logger.debug("Streamlit cleanup callback not available, relying on atexit")

    def _ensure_session_state_keys(self) -> None:
        """Ensure required session state keys exist."""
        if "background_tasks" not in st.session_state:
            st.session_state.background_tasks = {}
        if "task_progress" not in st.session_state:
            st.session_state.task_progress = {}

    def start_scraping_task(
        self, company_ids: list[int] | None = None, update_ui: bool = True
    ) -> str:
        """Start a background scraping task.

        Args:
            company_ids: Optional list of company IDs to scrape. If None, scrapes all.
            update_ui: Whether to update UI with progress.

        Returns:
            str: Task ID for tracking progress.
        """
        # Get current script context for thread communication
        script_ctx = get_script_run_ctx()

        def progress_callback(task_id: str, progress: float, message: str = "") -> None:
            """Update progress in session state safely from background thread.

            Note: We do NOT call st.rerun() from this callback as it's not thread-safe.
            The main UI thread will poll session state and update automatically.
            """
            if update_ui:
                # Update session state (this is thread-safe)
                st.session_state.task_progress[task_id] = ProgressInfo(
                    progress=progress,
                    message=message,
                    timestamp=datetime.now(timezone.utc),
                )

                # DO NOT call st.rerun() from background threads - not thread-safe
                # The main UI will detect session state changes and update naturally

        def error_callback(task_id: str, error_message: str) -> None:
            """Handle task errors."""
            logger.error("Scraping task %s failed: %s", task_id, error_message)
            if update_ui:
                st.session_state.task_progress[task_id] = ProgressInfo(
                    progress=0.0,
                    message=f"Error: {error_message}",
                    timestamp=datetime.now(timezone.utc),
                )
                try:
                    st.rerun()
                except Exception as e:
                    logger.warning("Could not trigger UI rerun: %s", e)

        # Start the scraping task
        task_id = self.start_task(
            task_func=self._scraping_task_wrapper,
            task_name="Job Scraping",
            progress_callback=progress_callback,
            error_callback=error_callback,
            _company_ids=company_ids,
            script_ctx=script_ctx,
        )

        # Store task reference in session state
        st.session_state.background_tasks[task_id] = {
            "type": "scraping",
            "started_at": datetime.now(timezone.utc),
            "company_ids": company_ids,
        }

        logger.info("Started scraping task %s", task_id)
        return task_id

    def _scraping_task_wrapper(
        self,
        _company_ids: list[int] | None = None,
        script_ctx=None,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, int]:
        """Wrapper function for scraping task that handles Streamlit context.

        Args:
            _company_ids: Optional list of company IDs to scrape (unused - TODO).
            script_ctx: Streamlit script context for UI updates.
            progress_callback: Callback for progress updates.

        Returns:
            Dictionary with scraping statistics.
        """
        # Add script context to current thread for Streamlit commands
        if script_ctx:
            add_script_run_ctx(threading.current_thread(), script_ctx)

        try:
            # Report start
            if progress_callback:
                progress_callback("", 0.0, "Initializing scraping workflow...")

            # TODO: In a more sophisticated implementation, we could modify
            # scrape_all to accept a progress callback and company_ids filter.
            # For now, we'll call the existing scrape_all function and simulate
            # progress.

            # Report progress at key stages
            if progress_callback:
                progress_callback("", 10.0, "Starting company page scraping...")

            # Execute the actual scraping
            result = scrape_all()

            # Report completion
            if progress_callback:
                total_jobs = sum(result.values()) if result else 0
                progress_callback(
                    "", 100.0, f"Scraping completed! Found {total_jobs} total jobs."
                )

        except Exception as e:
            logger.exception("Scraping task failed")
            if progress_callback:
                progress_callback("", 0.0, "Scraping failed: %s", e)
            raise
        else:
            return result

    def get_scraping_progress(self, task_id: str) -> ProgressInfo | None:
        """Get progress information for a scraping task.

        Args:
            task_id: Task ID to check progress for.

        Returns:
            ProgressInfo object if task exists, None otherwise.
        """
        return st.session_state.task_progress.get(task_id)

    def is_scraping_active(self) -> bool:
        """Check if any scraping tasks are currently active.

        Returns:
            True if any scraping tasks are running.
        """
        return any(
            task_info.status in ("pending", "running")
            and st.session_state.background_tasks.get(task_id, {}).get("type")
            == "scraping"
            for task_id, task_info in self.get_all_tasks().items()
        )

    def stop_all_scraping(self) -> int:
        """Stop all active scraping tasks.

        Returns:
            Number of tasks that were cancelled.
        """
        cancelled = sum(
            1
            for task_id, task_info in self.get_all_tasks().items()
            if (
                task_info.status in ("pending", "running")
                and st.session_state.background_tasks.get(task_id, {}).get("type")
                == "scraping"
                and self.cancel_task(task_id)
            )
        )

        if cancelled > 0:
            logger.info("Cancelled %d scraping tasks", cancelled)

        return cancelled

    def cleanup_session_state(self, max_age_hours: int = 1) -> dict[str, int]:
        """Clean up old task data from session state to prevent memory bloat.

        Args:
            max_age_hours: Maximum age in hours for progress entries.

        Returns:
            Dictionary with cleanup statistics.
        """
        if (
            "task_progress" not in st.session_state
            or "background_tasks" not in st.session_state
        ):
            return {"progress_cleaned": 0, "tasks_cleaned": 0, "manager_cleaned": 0}

        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        progress_cleaned = 0
        tasks_cleaned = 0

        # Clean up old progress entries
        progress_items = list(st.session_state.task_progress.items())

        # Sort by timestamp (newest first)
        progress_items.sort(
            key=lambda x: x[1].timestamp.timestamp()
            if hasattr(x[1], "timestamp")
            else 0,
            reverse=True,
        )

        # Remove old entries and enforce max_progress_entries limit
        task_ids_to_remove = []

        for i, (task_id, progress_info) in enumerate(progress_items):
            # Remove if too old
            if (
                hasattr(progress_info, "timestamp")
                and progress_info.timestamp.timestamp() < cutoff
            ) or i >= self.max_progress_entries:
                task_ids_to_remove.append(task_id)

        # Actually remove the entries
        for task_id in task_ids_to_remove:
            if task_id in st.session_state.task_progress:
                del st.session_state.task_progress[task_id]
                progress_cleaned += 1
            if task_id in st.session_state.background_tasks:
                del st.session_state.background_tasks[task_id]
                tasks_cleaned += 1

        # Clean up completed tasks from manager
        manager_cleaned = self.cleanup_completed_tasks(max_age_hours=max_age_hours)

        cleanup_stats = {
            "progress_cleaned": progress_cleaned,
            "tasks_cleaned": tasks_cleaned,
            "manager_cleaned": manager_cleaned,
        }

        if sum(cleanup_stats.values()) > 0:
            logger.info("Session state cleanup completed: %s", cleanup_stats)

        return cleanup_stats

    def _streamlit_cleanup(self) -> None:
        """Cleanup callback for Streamlit context cleanup."""
        try:
            self.cleanup_session_state()
        except Exception as e:
            logger.warning("Error during Streamlit cleanup: %s", e)

    def get_memory_usage_stats(self) -> dict[str, int]:
        """Get current memory usage statistics for monitoring.

        Returns:
            Dictionary with current usage counts.
        """
        with self._lock:
            active_count = len(
                [
                    task
                    for task in self.active_tasks.values()
                    if task.status in ("pending", "running")
                ]
            )
            completed_count = len(
                [
                    task
                    for task in self.active_tasks.values()
                    if task.status in ("completed", "failed", "cancelled")
                ]
            )

        progress_count = len(st.session_state.get("task_progress", {}))
        session_tasks_count = len(st.session_state.get("background_tasks", {}))

        return {
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "total_tasks": len(self.active_tasks),
            "progress_entries": progress_count,
            "session_tasks": session_tasks_count,
        }


# Global task manager instance for the application
# This should be initialized once and reused across the Streamlit app
# Using WeakValueDictionary to allow proper cleanup
_task_managers: dict[str, StreamlitTaskManager] = {}
_task_manager_lock = threading.Lock()


def get_task_manager(session_id: str | None = None) -> StreamlitTaskManager:
    """Get the global task manager instance.

    Args:
        session_id: Optional session ID for session-specific managers.

    Returns:
        StreamlitTaskManager: Global task manager instance.
    """
    # Use default session if none provided
    if session_id is None:
        session_id = "default"

    with _task_manager_lock:
        if session_id not in _task_managers:
            _task_managers[session_id] = StreamlitTaskManager(
                max_workers=2, max_task_history=50, max_progress_entries=25
            )
            logger.info("Initialized StreamlitTaskManager for session: %s", session_id)

        return _task_managers[session_id]


def cleanup_all_task_managers() -> dict[str, dict[str, int]]:
    """Cleanup all task managers and return statistics.

    Returns:
        Dictionary mapping session IDs to their cleanup statistics.
    """
    cleanup_stats = {}

    with _task_manager_lock:
        for session_id, manager in _task_managers.items():
            try:
                stats = manager.cleanup_session_state()
                cleanup_stats[session_id] = stats
            except Exception as e:
                logger.exception("Error cleaning up task manager %s", session_id)
                cleanup_stats[session_id] = {"error": str(e)}

    return cleanup_stats


def shutdown_all_task_managers(wait: bool = True) -> None:
    """Shutdown all task managers.

    Args:
        wait: Whether to wait for running tasks to complete.
    """
    global _task_managers  # noqa: PLW0602

    with _task_manager_lock:
        for session_id, manager in _task_managers.items():
            try:
                manager.shutdown(wait=wait)
                logger.info("Shutdown task manager for session: %s", session_id)
            except Exception:
                logger.exception("Error shutting down task manager %s", session_id)

        _task_managers.clear()


# Register global cleanup
atexit.register(lambda: shutdown_all_task_managers(wait=False))


def start_background_scraping(
    company_ids: list[int] | None = None, update_ui: bool = True
) -> str:
    """Convenience function to start background scraping.

    Args:
        company_ids: Optional list of company IDs to scrape.
        update_ui: Whether to update UI with progress.

    Returns:
        str: Task ID for tracking progress.
    """
    task_manager = get_task_manager()
    return task_manager.start_scraping_task(
        company_ids=company_ids, update_ui=update_ui
    )


def get_scraping_progress(task_id: str) -> ProgressInfo | None:
    """Get progress for a scraping task.

    Args:
        task_id: Task ID to check.

    Returns:
        ProgressInfo object if found, None otherwise.
    """
    task_manager = get_task_manager()
    return task_manager.get_scraping_progress(task_id)


def is_scraping_active() -> bool:
    """Check if any scraping is currently running.

    Returns:
        True if scraping is active.
    """
    task_manager = get_task_manager()
    return task_manager.is_scraping_active()


def stop_all_scraping() -> int:
    """Stop all active scraping tasks.

    Returns:
        Number of tasks cancelled.
    """
    task_manager = get_task_manager()
    return task_manager.stop_all_scraping()
