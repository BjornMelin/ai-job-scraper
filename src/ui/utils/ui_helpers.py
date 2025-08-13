"""Comprehensive UI utilities for formatting, context detection, and validation.

This module consolidates utilities for:
- Data formatting
- Streamlit context detection
- Safe integer and job count validation

Provides a library-first approach to UI-related utility functions.
"""

import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import validate_call

if TYPE_CHECKING:
    from src.models import JobSQL

# Type aliases
type SalaryTuple = tuple[int | None, int | None]

logger = logging.getLogger(__name__)


# Formatters from formatters.py
def calculate_scraping_speed(
    jobs_found: int,
    start_time: datetime | None,
    end_time: datetime | None = None,
) -> float:
    """Calculate scraping speed in jobs per minute."""
    try:
        if not isinstance(jobs_found, int) or jobs_found < 0:
            return 0.0

        if start_time is None:
            return 0.0

        effective_end_time = end_time or datetime.now(UTC)
        duration = effective_end_time - start_time
        duration_minutes = duration.total_seconds() / 60.0

        if duration_minutes <= 0:
            return 0.0

        speed = jobs_found / duration_minutes
        return round(speed, 1)

    except Exception:
        logger.exception("Error calculating scraping speed")
        return 0.0


def calculate_eta(
    total_companies: int,
    completed_companies: int,
    time_elapsed: int,
) -> str:
    """Calculate estimated time of arrival for completing all companies."""
    try:
        # Validate inputs
        if not (
            isinstance(total_companies, int)
            and isinstance(completed_companies, int)
            and isinstance(time_elapsed, int)
        ):
            return "Unknown"

        if total_companies <= 0 or completed_companies < 0 or time_elapsed < 0:
            return "Unknown"

        # Check if done
        if completed_companies >= total_companies:
            return "Done"

        # Check if no progress
        if completed_companies == 0 or time_elapsed == 0:
            return "Calculating..."

        # Calculate ETA
        remaining_companies = total_companies - completed_companies
        time_per_company = time_elapsed / completed_companies
        remaining_time = remaining_companies * time_per_company

        return format_duration(int(remaining_time))

    except Exception:
        logger.exception("Error calculating ETA")
        return "Unknown"


def format_duration(seconds: int | float) -> str:
    """Format duration in seconds to human-readable string."""
    try:
        if not isinstance(seconds, int | float) or seconds < 0:
            return "0s"

        seconds = int(seconds)  # Truncate to integer

        if seconds == 0:
            return "0s"

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        # Build result based on largest unit
        result = ""
        if hours > 0:
            result = f"{hours}h"
            if minutes > 0:
                result += f" {minutes}m"
        elif minutes > 0:
            result = f"{minutes}m"
            if remaining_seconds > 0:
                result += f" {remaining_seconds}s"
        else:
            result = f"{remaining_seconds}s"

    except Exception:
        logger.exception("Error formatting duration")
        return "0s"
    else:
        return result


def format_timestamp(dt: datetime | None, format_str: str = "%H:%M:%S") -> str:
    """Format datetime to string or return N/A for None."""
    try:
        if dt is None:
            return "N/A"

        if not isinstance(dt, datetime):
            return "N/A"

        return dt.strftime(format_str)

    except Exception:
        logger.exception("Error formatting timestamp")
        return "N/A"


def format_jobs_count(count: int, singular: str = "job", plural: str = "jobs") -> str:
    """Format job count with proper singular/plural form."""
    try:
        # Handle invalid input gracefully
        if count is None:
            count = 0
        elif not isinstance(count, int | float):
            try:
                count = int(count)
            except (ValueError, TypeError):
                count = 0
        else:
            count = int(count)

        result = f"1 {singular}" if count == 1 else f"{count} {plural}"

    except Exception:
        logger.exception("Error formatting jobs count")
        return "0 jobs"
    else:
        return result


def format_salary(amount: int | float | None) -> str:
    """Format salary amount with k/M suffixes."""
    try:
        if amount is None or not isinstance(amount, int | float) or amount < 0:
            return "$0"

        amount = int(amount)  # Convert to integer

        if amount == 0:
            return "$0"

        # Format based on amount range
        if amount < 1000:
            result = f"${amount}"
        elif amount < 1000000:
            result = f"${amount // 1000}k"
        else:
            millions = amount / 1000000
            result = f"${millions:.1f}M"

    except Exception:
        logger.exception("Error formatting salary")
        return "$0"
    else:
        return result


# Streamlit context from streamlit_context.py
def is_streamlit_context() -> bool:
    """Check if we're running in a proper Streamlit context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        return False


# Validation utilities from validation_utils.py


@validate_call
def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert any value to a non-negative integer."""
    try:
        if value is None:
            return 0
        if isinstance(
            value, bool
        ):  # Check bool before int since bool is subclass of int
            return int(value)
        if isinstance(value, int | float):
            import math

            return max(
                0,
                int(value)
                if isinstance(value, int)
                else (
                    int(value)
                    if isinstance(value, float) and math.isfinite(value)
                    else 0
                ),
            )
        if isinstance(value, str):
            value = value.strip()
            if value:
                try:
                    return max(0, int(float(value)))
                except (ValueError, TypeError):
                    # Extract first number from string
                    import re

                    match = re.search(r"-?\d+(?:\.\d+)?", value)
                    return (
                        max(0, int(float(match.group()))) if match else max(0, default)
                    )
            else:
                # Empty string should use default
                return max(0, default)
    except (ValueError, TypeError, AttributeError):
        logger.warning("Failed to convert %s to safe integer, using default", value)
        return max(0, default)

    # Fallback for unhandled types (use default)
    return max(0, default)


@validate_call
def safe_job_count(value: Any, company_name: str = "unknown") -> int:
    """Safely convert job count values with context-aware logging."""
    try:
        result = safe_int(value)
    except Exception as e:
        logger.warning(
            "Failed to convert job count for %s: %s (%s)",
            company_name,
            value,
            e,
        )
        return 0

    if value != result and value is not None:
        logger.info("Converted job count for %s: %s -> %s", company_name, value, result)
    return result


# Job formatting helpers (replacement for computed fields)


def format_salary_range(salary: SalaryTuple | None) -> str:
    """Format salary range for display.

    Replacement for JobSQL.salary_range_display computed field.

    Args:
        salary: Salary tuple (min, max) or None

    Returns:
        Formatted salary range string
    """
    if not salary or salary == (None, None):
        return "Not specified"

    min_sal, max_sal = salary
    if min_sal and max_sal:
        if min_sal == max_sal:
            return f"${min_sal:,}"
        return f"${min_sal:,} - ${max_sal:,}"
    if min_sal:
        return f"${min_sal:,}+"
    if max_sal:
        return f"Up to ${max_sal:,}"
    return "Not specified"


def calculate_days_since_posted(posted_date: datetime | None) -> int | None:
    """Calculate days since job was posted.

    Replacement for JobSQL.days_since_posted computed field.

    Args:
        posted_date: Job posting date

    Returns:
        Days since posted or None if no date
    """
    if not posted_date:
        return None

    try:
        # Calculate difference using standard datetime
        now_utc = datetime.now(UTC)
        # Ensure posted_date is timezone-aware
        if not posted_date.tzinfo:
            posted_date = posted_date.replace(tzinfo=UTC)
        return (now_utc - posted_date).days
    except Exception:
        logger.exception("Error calculating days since posted")
        return None


def is_job_recently_posted(
    posted_date: datetime | None,
    days_threshold: int = 7,
) -> bool:
    """Check if job was posted within the threshold days.

    Replacement for JobSQL.is_recently_posted computed field.

    Args:
        posted_date: Job posting date
        days_threshold: Number of days to consider as recent (default: 7)

    Returns:
        True if job was posted within threshold days
    """
    days = calculate_days_since_posted(posted_date)
    return days is not None and days <= days_threshold


# Company statistics helpers (replacement for computed fields)


def calculate_total_jobs_count(jobs: list["JobSQL"] | None) -> int:
    """Calculate total job count for company.

    Replacement for CompanySQL.total_jobs_count computed field.

    Args:
        jobs: List of job objects

    Returns:
        Total number of jobs
    """
    if not jobs or not isinstance(jobs, list):
        return 0
    return len(jobs)


def calculate_active_jobs_count(jobs: list["JobSQL"] | None) -> int:
    """Calculate active (non-archived) job count for company.

    Replacement for CompanySQL.active_jobs_count computed field.

    Args:
        jobs: List of job objects

    Returns:
        Number of active (non-archived) jobs
    """
    if not jobs or not isinstance(jobs, list):
        return 0
    try:
        return len([j for j in jobs if not j.archived])
    except (AttributeError, TypeError):
        return 0


def find_last_job_posted(jobs: list["JobSQL"] | None) -> datetime | None:
    """Find the most recent job posting date.

    Replacement for CompanySQL.last_job_posted computed field.

    Args:
        jobs: List of job objects

    Returns:
        Most recent posting date or None
    """
    if not jobs or not isinstance(jobs, list):
        return None
    try:
        return max((j.posted_date for j in jobs if j.posted_date), default=None)
    except (AttributeError, TypeError):
        return None


def format_success_rate_percentage(success_rate: float) -> float:
    """Format success rate as percentage.

    Replacement for CompanySQL.success_rate_percentage computed field.

    Args:
        success_rate: Success rate as decimal (0.0-1.0)

    Returns:
        Success rate as percentage (0-100), rounded to 1 decimal place
    """
    try:
        if not isinstance(success_rate, int | float):
            return 0.0
        return round(success_rate * 100, 1)
    except (TypeError, ValueError):
        return 0.0


# Additional formatter functions for test compatibility


def format_company_stats(stats: dict[str, "Any"]) -> dict[str, "Any"]:
    """Format company statistics for display.

    Args:
        stats: Dictionary containing company statistics

    Returns:
        Formatted statistics dictionary
    """
    try:
        if not isinstance(stats, dict):
            return {}

        formatted = {}
        for key, value in stats.items():
            if key in ["total_jobs", "active_companies"] and isinstance(
                value,
                int | float,
            ):
                formatted[key] = int(value)
            elif key == "success_rate" and isinstance(value, int | float):
                formatted[key] = round(float(value), 2)
            else:
                formatted[key] = value

        return formatted  # noqa: TRY300 - Early return pattern preferred here
    except Exception:
        logger.exception("Error formatting company stats")
        return {}


def format_date_relative(date: datetime | None) -> str:
    """Format date as relative time string using humanize.

    Args:
        date: Date to format

    Returns:
        Relative time string (e.g., "2 hours ago", "Just now")
    """
    try:
        if date is None:
            return "Unknown"

        if not isinstance(date, datetime):
            return "Unknown"

        # Ensure timezone awareness for humanize
        if not date.tzinfo:
            date = date.replace(tzinfo=UTC)

        import humanize

        return humanize.naturaltime(date)

    except Exception:
        logger.exception("Error formatting relative date")
        return "Unknown"


def truncate_text(text: str | None, max_length: int) -> str:
    """Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    try:
        if not text:
            return ""

        if not isinstance(text, str):
            text = str(text)

        if len(text) <= max_length:
            return text

        # Truncate and add ellipsis, ensuring total length doesn't exceed max_length
        if max_length <= 3:
            return "..."[:max_length]
        return text[: max_length - 3] + "..."

    except Exception:
        logger.exception("Error truncating text")
        return ""


# Exposed type aliases
JobCount = int
SafeInteger = int
