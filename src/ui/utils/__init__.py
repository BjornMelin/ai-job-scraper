"""UI utilities package for the AI Job Scraper Streamlit application.

This package contains utility modules for Streamlit UI functionality including
background task management, data formatting, validation, and other helper functions.
"""

from src.ui.utils.background_helpers import (
    CompanyProgress,
    ProgressInfo,
    TaskInfo,
    add_task,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    get_task,
    is_scraping_active,
    remove_task,
    start_background_scraping,
    stop_all_scraping,
)

__all__ = [
    "CompanyProgress",
    "ProgressInfo",
    "TaskInfo",
    "add_task",
    "get_company_progress",
    "get_scraping_progress",
    "get_scraping_results",
    "get_task",
    "is_scraping_active",
    "remove_task",
    "start_background_scraping",
    "stop_all_scraping",
]
