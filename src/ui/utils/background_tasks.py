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

import logging
import threading
import traceback

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Import the scraper function we need to launch in background
try:
    from ...scraper import scrape_all
except ImportError:
    # Fallback import path if needed
    from src.scraper import scrape_all

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
    timestamp: datetime = field(default_factory=datetime.now)
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

    def __init__(self, max_workers: int = 3):
        """Initialize the task manager.

        Args:
            max_workers: Maximum number of concurrent background tasks.
        """
        self.active_tasks: dict[str, TaskInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        logger.info(f"BackgroundTaskManager initialized with {max_workers} max workers")

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
            id=task_id, name=task_name, status="pending", started_at=datetime.now()
        )

        with self._lock:
            self.active_tasks[task_id] = task_info

        # Define wrapper function that handles progress and error callbacks
        def task_wrapper():
            try:
                # Update status to running
                self._update_task_status(task_id, "running", 0.0, "Starting task...")

                # Create a copy of task_kwargs to prevent concurrent modification
                import copy

                safe_kwargs = copy.deepcopy(task_kwargs)

                # Execute the actual task function
                if progress_callback:
                    # Pass progress callback to task function if it accepts it
                    import inspect

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
                        self.active_tasks[task_id].completed_at = datetime.now()

                logger.info(f"Task {task_id} ({task_name}) completed successfully")

            except Exception as e:
                error_msg = f"Task failed: {str(e)}"
                logger.error(f"Task {task_id} ({task_name}) failed: {error_msg}")
                logger.error(traceback.format_exc())

                # Mark as failed
                self._update_task_status(task_id, "failed", None, error_msg)

                with self._lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id].error = error_msg
                        self.active_tasks[task_id].completed_at = datetime.now()

                # Call error callback if provided
                if error_callback:
                    try:
                        error_callback(task_id, error_msg)
                    except Exception as cb_error:
                        logger.error(f"Error callback failed: {cb_error}")

        # Submit task to thread pool
        future = self.executor.submit(task_wrapper)

        with self._lock:
            self.active_tasks[task_id].future = future

        logger.info(f"Started background task {task_id} ({task_name})")
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
                    task_info.completed_at = datetime.now()
                    logger.info(f"Cancelled task {task_id}")
                return cancelled

        return False

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Remove old completed tasks to prevent memory leaks.

        Args:
            max_age_hours: Maximum age in hours for completed tasks.

        Returns:
            Number of tasks cleaned up.
        """
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned = 0

        with self._lock:
            task_ids_to_remove = []
            for task_id, task_info in self.active_tasks.items():
                if (
                    task_info.status in ("completed", "failed", "cancelled")
                    and task_info.completed_at
                    and task_info.completed_at.timestamp() < cutoff
                ):
                    task_ids_to_remove.append(task_id)

            for task_id in task_ids_to_remove:
                del self.active_tasks[task_id]
                cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old completed tasks")

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

    def shutdown(self) -> None:
        """Shutdown the task manager and cleanup resources."""
        logger.info("Shutting down BackgroundTaskManager")
        self.executor.shutdown(wait=True)
        with self._lock:
            self.active_tasks.clear()


class StreamlitTaskManager(BackgroundTaskManager):
    """Streamlit-specific task manager with session state integration.

    This class extends BackgroundTaskManager to provide seamless integration
    with Streamlit's session state and UI update mechanisms. It handles the
    ScriptRunContext properly to enable Streamlit commands from background threads.
    """

    def __init__(self, max_workers: int = 3):
        """Initialize the Streamlit task manager.

        Args:
            max_workers: Maximum number of concurrent background tasks.
        """
        super().__init__(max_workers)
        self._ensure_session_state_keys()

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
                    progress=progress, message=message, timestamp=datetime.now()
                )

                # DO NOT call st.rerun() from background threads - not thread-safe
                # The main UI will detect session state changes and update naturally

        def error_callback(task_id: str, error_message: str) -> None:
            """Handle task errors."""
            logger.error(f"Scraping task {task_id} failed: {error_message}")
            if update_ui:
                st.session_state.task_progress[task_id] = ProgressInfo(
                    progress=0.0,
                    message=f"Error: {error_message}",
                    timestamp=datetime.now(),
                )
                try:
                    st.rerun()
                except Exception as e:
                    logger.warning(f"Could not trigger UI rerun: {e}")

        # Start the scraping task
        task_id = self.start_task(
            task_func=self._scraping_task_wrapper,
            task_name="Job Scraping",
            progress_callback=progress_callback,
            error_callback=error_callback,
            company_ids=company_ids,
            script_ctx=script_ctx,
        )

        # Store task reference in session state
        st.session_state.background_tasks[task_id] = {
            "type": "scraping",
            "started_at": datetime.now(),
            "company_ids": company_ids,
        }

        logger.info(f"Started scraping task {task_id}")
        return task_id

    def _scraping_task_wrapper(
        self,
        company_ids: list[int] | None = None,
        script_ctx=None,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, int]:
        """Wrapper function for scraping task that handles Streamlit context.

        Args:
            company_ids: Optional list of company IDs to scrape.
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

            return result

        except Exception as e:
            logger.error(f"Scraping task failed: {e}")
            if progress_callback:
                progress_callback("", 0.0, f"Scraping failed: {str(e)}")
            raise

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
        for task_id, task_info in self.get_all_tasks().items():
            if (
                task_info.status in ("pending", "running")
                and st.session_state.background_tasks.get(task_id, {}).get("type")
                == "scraping"
            ):
                return True
        return False

    def stop_all_scraping(self) -> int:
        """Stop all active scraping tasks.

        Returns:
            Number of tasks that were cancelled.
        """
        cancelled = 0
        for task_id, task_info in self.get_all_tasks().items():
            if (
                task_info.status in ("pending", "running")
                and st.session_state.background_tasks.get(task_id, {}).get("type")
                == "scraping"
            ):
                if self.cancel_task(task_id):
                    cancelled += 1

        if cancelled > 0:
            logger.info(f"Cancelled {cancelled} scraping tasks")

        return cancelled

    def cleanup_session_state(self) -> None:
        """Clean up old task data from session state."""
        # Clean up completed task progress data older than 1 hour
        cutoff = datetime.now().timestamp() - 3600

        task_ids_to_remove = []
        for task_id, progress_info in st.session_state.task_progress.items():
            if progress_info.timestamp.timestamp() < cutoff:
                task_ids_to_remove.append(task_id)

        for task_id in task_ids_to_remove:
            del st.session_state.task_progress[task_id]
            if task_id in st.session_state.background_tasks:
                del st.session_state.background_tasks[task_id]

        # Clean up completed tasks from manager
        self.cleanup_completed_tasks(max_age_hours=1)


# Global task manager instance for the application
# This should be initialized once and reused across the Streamlit app
_task_manager: StreamlitTaskManager | None = None


def get_task_manager() -> StreamlitTaskManager:
    """Get the global task manager instance.

    Returns:
        StreamlitTaskManager: Global task manager instance.
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = StreamlitTaskManager(max_workers=2)
        logger.info("Initialized global StreamlitTaskManager")
    return _task_manager


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
