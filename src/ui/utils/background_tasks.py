"""Background task utilities for scraping and task management.

This module consolidates background task handling, progress tracking, and utility
functions for the application's scraping workflows.
"""

from src.ui.utils.background_helpers import (
    BackgroundTaskManager,
    CompanyProgress,
    JobService,
    ProgressInfo,
    StreamlitTaskManager,
    TaskInfo,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    get_task_manager,
    is_scraping_active,
    scrape_all,
    start_background_scraping,
    start_scraping,
    stop_all_scraping,
)

__all__ = [
    "BackgroundTaskManager",
    "CompanyProgress",
    "JobService",
    "ProgressInfo",
    "StreamlitTaskManager",
    "TaskInfo",
    "get_company_progress",
    "get_scraping_progress",
    "get_scraping_results",
    "get_task_manager",
    "is_scraping_active",
    "scrape_all",
    "start_background_scraping",
    "start_scraping",
    "stop_all_scraping",
]
