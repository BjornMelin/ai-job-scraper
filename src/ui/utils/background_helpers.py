"""Background task management and refresh utilities.

Provides library-first utilities for:
- Streamlit background task handling
- Progress tracking
- Throttled reruns
- Task state management

Optimized for Streamlit's unique execution context.
"""

import logging
import sys
import threading
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# Constants and type hints from background tasks
DEFAULT_INTERVAL_SECONDS: float = 2.0


@dataclass
class CompanyProgress:
    """Individual company scraping progress tracking."""

    name: str
    status: str = "Pending"
    jobs_found: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None


@dataclass
class ProgressInfo:
    """Progress information for background tasks."""

    progress: float
    message: str
    timestamp: datetime


@dataclass
class TaskInfo:
    """Task information for background tasks."""

    task_id: str
    status: str
    progress: float
    message: str
    timestamp: datetime


# Atomic session state operations
_session_state_lock = threading.Lock()


def _atomic_check_and_set(key: str, check_value: Any, set_value: Any) -> bool:
    """Atomically check session state value and set new value if match."""
    with _session_state_lock:
        current_value = st.session_state.get(key)
        if current_value == check_value:
            st.session_state[key] = set_value
            return True
        return False


def _is_test_environment() -> bool:
    """Detect if we're running in a test environment."""
    return (
        "pytest" in sys.modules
        or "unittest" in sys.modules
        or hasattr(st.session_state, "_test_mode")
    )


# Consolidated background task manager
class BackgroundTaskManager:
    """Simple background task manager for compatibility."""

    def __init__(self):
        self.tasks = {}

    def add_task(self, task_id: str, task_info: TaskInfo) -> None:
        """Add a task to tracking."""
        self.tasks[task_id] = task_info

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information."""
        return self.tasks.get(task_id)

    def remove_task(self, task_id: str) -> None:
        """Remove a task from tracking."""
        self.tasks.pop(task_id, None)


# Throttled rerun utility from refresh.py
def throttled_rerun(
    session_key: str = "last_refresh",
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    *,
    should_rerun: bool = True,
) -> None:
    """Trigger `st.rerun()` at most once per interval when condition is true."""
    if not should_rerun:
        return

    now = time.time()
    last = float(st.session_state.get(session_key, 0.0) or 0.0)

    if (now - last) >= max(0.0, interval_seconds):
        st.session_state[session_key] = now
        st.rerun()


# Key functions from background_tasks.py
def is_scraping_active() -> bool:
    """Check if scraping is currently active."""
    return st.session_state.get("scraping_active", False)


def get_scraping_results() -> dict[str, Any]:
    """Get results from the last scraping operation."""
    return st.session_state.get("scraping_results", {})


def get_task_manager() -> BackgroundTaskManager:
    """Get or create the task manager instance."""
    if "task_manager" not in st.session_state:
        st.session_state.task_manager = BackgroundTaskManager()
    return st.session_state.task_manager


def start_background_scraping(stay_active_in_tests: bool = False) -> str:
    """Start background scraping using Streamlit native approach."""
    task_id = str(uuid.uuid4())
    logger.info("start_background_scraping called with task_id: %s", task_id)

    # Initialize session state
    if "task_progress" not in st.session_state:
        st.session_state.task_progress = {}

    # Store task info in session state
    st.session_state.task_progress[task_id] = ProgressInfo(
        progress=0.0,
        message="Starting scraping...",
        timestamp=datetime.now(timezone.utc),
    )
    st.session_state.task_id = task_id

    # Store test behavior preference
    if _is_test_environment():
        st.session_state._test_stay_active = stay_active_in_tests

    # Set scraping trigger flag
    st.session_state.scraping_trigger = True
    st.session_state.scraping_active = True
    st.session_state.scraping_status = "Initializing scraping..."

    logger.info("Scraping trigger set - will be processed manually")
    return task_id


def stop_all_scraping() -> int:
    """Stop all scraping operations with proper thread cleanup."""
    stopped_count = 0
    if st.session_state.get("scraping_active", False):
        st.session_state.scraping_active = False
        st.session_state.scraping_status = "Scraping stopped"

        # Clean up thread reference if exists
        if hasattr(st.session_state, "scraping_thread"):
            thread = st.session_state.scraping_thread
            if thread and thread.is_alive():
                # Thread will terminate when it checks scraping_active
                thread.join(timeout=5.0)  # Wait up to 5 seconds
            delattr(st.session_state, "scraping_thread")
        stopped_count = 1
    return stopped_count


def get_scraping_progress() -> dict[str, ProgressInfo]:
    """Get current scraping progress."""
    return st.session_state.get("task_progress", {})


def get_company_progress() -> dict[str, CompanyProgress]:
    """Get current company-level scraping progress."""
    return st.session_state.get("company_progress", {})
