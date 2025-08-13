"""Data formatting utilities for human-readable display.

This module provides comprehensive formatting functions for:
- Time and duration formatting (ETA, duration, timestamps, relative dates)
- Numeric formatting (progress percentages, job counts, salary amounts)
- Statistical formatting (success rates, company statistics)
- Text formatting (truncation, display formatting)

All functions are pure formatting utilities with no side effects.
Uses library-first approach with humanize for enhanced readability.
"""

import logging

from datetime import UTC, datetime
from typing import Any

import humanize

# Type aliases
type SalaryTuple = tuple[int | None, int | None]

logger = logging.getLogger(__name__)

# =============================================================================
# TIME & DURATION FORMATTERS
# =============================================================================


def calculate_scraping_speed(
    jobs_found: int,
    start_time: datetime | None,
    end_time: datetime | None = None,
) -> float:
    """Calculate scraping speed in jobs per minute.

    Args:
        jobs_found: Number of jobs scraped
        start_time: When scraping started
        end_time: When scraping ended (defaults to now)

    Returns:
        Jobs per minute as float, 0.0 if invalid inputs
    """
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
    """Calculate estimated time of arrival for completing all companies.

    Args:
        total_companies: Total number of companies to process
        completed_companies: Number of companies already processed
        time_elapsed: Time elapsed in seconds

    Returns:
        Human-readable ETA string
    """
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
    """Format duration in seconds to human-readable string.

    Uses humanize library for consistent, readable output.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string (e.g., "2h 30m", "45s")
    """
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
    """Format datetime to string or return N/A for None.

    Args:
        dt: Datetime object to format
        format_str: Format string (default: "%H:%M:%S")

    Returns:
        Formatted timestamp string or "N/A"
    """
    try:
        if dt is None:
            return "N/A"

        if not isinstance(dt, datetime):
            return "N/A"

        return dt.strftime(format_str)

    except Exception:
        logger.exception("Error formatting timestamp")
        return "N/A"


def format_date_relative(date: datetime | None) -> str:
    """Format date as relative time string using humanize library.

    Args:
        date: Date to format

    Returns:
        Relative time string (e.g., "2 hours ago", "just now")
    """
    try:
        if date is None:
            return "Unknown"

        if not isinstance(date, datetime):
            return "Unknown"

        # Ensure timezone awareness for humanize
        if not date.tzinfo:
            date = date.replace(tzinfo=UTC)

        # Use humanize library for consistent relative time formatting
        return humanize.naturaltime(date)

    except Exception:
        logger.exception("Error formatting relative date")
        return "Unknown"


# =============================================================================
# NUMERIC FORMATTERS
# =============================================================================


def calculate_progress_percentage(completed: int, total: int) -> float:
    """Calculate progress percentage with proper rounding.

    Args:
        completed: Number of completed items
        total: Total number of items

    Returns:
        Progress percentage (0.0-100.0)
    """
    try:
        if not all(isinstance(x, int | float) for x in [completed, total]):
            return 0.0

        if total <= 0 or completed < 0:
            return 0.0

        if completed >= total:
            return 100.0

        percentage = (completed / total) * 100
        return round(percentage, 1)

    except Exception:
        logger.exception("Error calculating progress percentage")
        return 0.0


def format_jobs_count(count: int, singular: str = "job", plural: str = "jobs") -> str:
    """Format job count with proper singular/plural form.

    Args:
        count: Number of items
        singular: Singular form word
        plural: Plural form word

    Returns:
        Formatted count string with proper pluralization
    """
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
    except Exception:
        logger.exception("Error formatting jobs count")
        return "0 jobs"
    else:
        return f"1 {singular}" if count == 1 else f"{count} {plural}"


def format_salary(amount: int | float | None) -> str:
    """Format salary amount with k/M suffixes using humanize library.

    Args:
        amount: Salary amount in dollars

    Returns:
        Formatted salary string (e.g., "$120k", "$1.2M")
    """
    try:
        if amount is None or not isinstance(amount, int | float) or amount < 0:
            return "$0"

        amount = int(amount)  # Convert to integer

        if amount == 0:
            return "$0"

        # Use humanize intword for better formatting
        if amount >= 1000000:
            # For millions, format manually for consistent styling
            millions = amount / 1000000
            return f"${millions:.1f}M"
        if amount >= 1000:
            # For thousands, format manually for consistent styling
            thousands = amount // 1000
            return f"${thousands}k"
    except Exception:
        logger.exception("Error formatting salary")
        return "$0"
    else:
        return f"${amount}"


def format_salary_range(salary: SalaryTuple | None) -> str:
    """Format salary range for display with proper comma separation.

    Args:
        salary: Salary tuple (min, max) or None

    Returns:
        Formatted salary range string
    """
    try:
        if not salary or salary == (None, None):
            return "Not specified"

        min_sal, max_sal = salary

        if min_sal and max_sal:
            if min_sal == max_sal:
                # Use humanize intcomma for consistent number formatting
                return f"${humanize.intcomma(min_sal)}"
            return f"${humanize.intcomma(min_sal)} - ${humanize.intcomma(max_sal)}"

        if min_sal:
            return f"${humanize.intcomma(min_sal)}+"

        if max_sal:
            return f"Up to ${humanize.intcomma(max_sal)}"
    except Exception:
        logger.exception("Error formatting salary range")
        return "Not specified"
    else:
        return "Not specified"


def format_success_rate_percentage(success_rate: float) -> float:
    """Format success rate as percentage.

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


# =============================================================================
# STATISTICAL FORMATTERS
# =============================================================================


def format_company_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Format company statistics for display.

    Args:
        stats: Dictionary containing company statistics

    Returns:
        Formatted statistics dictionary with proper types
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
    except Exception:
        logger.exception("Error formatting company stats")
        return {}
    else:
        return formatted


# =============================================================================
# TEXT FORMATTERS
# =============================================================================


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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SalaryTuple",
    "calculate_eta",
    "calculate_progress_percentage",
    "calculate_scraping_speed",
    "format_company_stats",
    "format_date_relative",
    "format_duration",
    "format_jobs_count",
    "format_salary",
    "format_salary_range",
    "format_success_rate_percentage",
    "format_timestamp",
    "truncate_text",
]
