"""UI utilities package for the AI Job Scraper Streamlit application.

This package contains utility modules for Streamlit UI functionality including:
- Background task management
- Data formatting and display utilities
- Input validation with Pydantic integration
- Computed field helpers for job and company data
- Streamlit context detection and utilities

All modules follow KISS principles with focused responsibilities.
"""

from src.ui.utils.background_helpers import (
    CompanyProgress,
    ProgressInfo,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)

# Computed helper functions are now inlined directly in @computed_field properties
# No longer need a separate module for these simple calculations
from src.ui.utils.formatters import (
    calculate_eta,
    calculate_scraping_speed,
    format_company_stats,
    format_date_relative,
    format_duration,
    format_jobs_count,
    format_salary,
    format_salary_range,
    format_timestamp,
    truncate_text,
)
from src.ui.utils.streamlit_utils import (
    SafeInteger,
    SalaryTuple,
    is_streamlit_context,
)
from src.ui.utils.validators import (
    JobCount,
    SafeInt,
    ensure_non_negative_int,
    ensure_non_negative_int_with_default,
)

__all__ = [
    # Background helpers
    "CompanyProgress",
    "JobCount",
    "ProgressInfo",
    "SafeInt",
    "SafeInteger",
    "SalaryTuple",
    # Formatters
    "calculate_eta",
    "calculate_scraping_speed",
    # Validators
    "ensure_non_negative_int",
    "ensure_non_negative_int_with_default",
    "format_company_stats",
    "format_date_relative",
    "format_duration",
    "format_jobs_count",
    "format_salary",
    "format_salary_range",
    "format_timestamp",
    "get_company_progress",
    "get_scraping_progress",
    "get_scraping_results",
    "is_scraping_active",
    # Streamlit utilities
    "is_streamlit_context",
    "start_background_scraping",
    "stop_all_scraping",
    "truncate_text",
]
