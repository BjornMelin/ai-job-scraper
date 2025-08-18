"""Data formatting utilities for UI display.

This module provides formatting functions for various data types including:
- Duration and time formatting
- Salary and numeric formatting
- Job counts and statistics
- Date and timestamp formatting

All formatters handle edge cases gracefully and provide consistent output.
"""

import logging

from datetime import UTC, datetime
from typing import Any

import humanize

# Type aliases
type SalaryTuple = tuple[int | None, int | None]

logger = logging.getLogger(__name__)


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

    return result


def format_salary(amount: int | float | None) -> str:
    """Format salary amount using humanize library with k/M suffixes."""
    try:
        if amount is None or not isinstance(amount, int | float) or amount < 0:
            return "$0"

        amount = int(amount)  # Convert to integer

        if amount == 0:
            return "$0"

        # Use humanize for formatting logic but maintain k/M suffix style

        if amount < 1000:
            return f"${amount}"
        if amount < 1000000:
            # Use humanize for accuracy but convert to k suffix
            # humanize.intcomma would give us "125,000" but we want "125k"
            thousands = amount // 1000
            return f"${thousands}k"
        # Use humanize.intword precision for millions
        # Convert "2.8 million" style to "2.8M" style
        intword_result = humanize.intword(amount)
        if "million" in intword_result:
            # Extract the number part and add M suffix
            millions_str = intword_result.replace(" million", "")
            return f"${millions_str}M"
    except Exception:
        # Fallback to original logic for edge cases
        millions = amount / 1000000
        return f"${millions:.1f}M"
        logger.exception("Error formatting salary")
        return "$0"


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
