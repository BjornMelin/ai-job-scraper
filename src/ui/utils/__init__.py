"""UI utilities package for the AI Job Scraper Streamlit application.

This package contains consolidated utility modules for Streamlit UI functionality
including background task management, database helpers, and UI formatting utilities.

Consolidated modules:
- ui_helpers.py: Data formatting, validation, and Streamlit context utilities
- database_helpers.py: Database session management and session state helpers
- background_helpers.py: Background task management and throttled refresh utilities
"""

from .background_helpers import (
    BackgroundTaskManager,
    ProgressInfo,
    StreamlitTaskManager,
    TaskInfo,
    get_scraping_progress,
    get_task_manager,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
    throttled_rerun,
)
from .database_helpers import (
    display_feedback_messages,
    init_session_state_keys,
    render_database_health_widget,
    streamlit_db_session,
)
from .ui_helpers import (
    calculate_eta,
    format_duration,
    format_jobs_count,
    format_salary,
    is_streamlit_context,
    safe_int,
    safe_job_count,
)

__all__ = [
    "BackgroundTaskManager",
    "ProgressInfo",
    "StreamlitTaskManager",
    "TaskInfo",
    "calculate_eta",
    "display_feedback_messages",
    "format_duration",
    "format_jobs_count",
    "format_salary",
    "get_scraping_progress",
    "get_task_manager",
    "init_session_state_keys",
    "is_scraping_active",
    "is_streamlit_context",
    "render_database_health_widget",
    "safe_int",
    "safe_job_count",
    "start_background_scraping",
    "stop_all_scraping",
    "streamlit_db_session",
    "throttled_rerun",
]
