"""UI utilities package for the AI Job Scraper Streamlit application.

This package contains utility modules for Streamlit UI functionality including
background task management, data formatting, validation, and other helper functions.
"""

from .background_tasks import (
    BackgroundTaskManager,
    ProgressInfo,
    StreamlitTaskManager,
    TaskInfo,
    get_scraping_progress,
    get_task_manager,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)

__all__ = [
    "BackgroundTaskManager",
    "CompanyProgress",
    "ProgressInfo",
    "StreamlitTaskManager",
    "TaskInfo",
    "get_scraping_progress",
    "get_scraping_results",
    "get_task_manager",
    "is_scraping_active",
    "render_scraping_controls",
    "start_background_scraping",
    "start_scraping",
    "stop_all_scraping",
]
