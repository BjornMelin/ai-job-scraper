"""Utility functions for formatting data and calculating metrics in the UI.

This module provides formatting utilities for the AI Job Scraper Streamlit UI,
including time calculations, progress metrics, and human-readable formatting
functions for dashboard displays.

Key features:
- Scraping speed calculations (jobs per minute)
- ETA estimation based on completion rates
- Human-readable time formatting
- Safe handling of edge cases and invalid data

Example usage:
    # Calculate scraping speed
    speed = calculate_scraping_speed(jobs_found=45, start_time=start, end_time=end)

    # Format ETA for display
    eta = calculate_eta(total_companies=10, completed_companies=3, time_elapsed=300)

    # Format duration for display
    duration_str = format_duration(seconds=125)  # "2m 5s"
"""

import logging

from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def calculate_scraping_speed(
    jobs_found: int,
    start_time: datetime | None,
    end_time: datetime | None = None,
) -> float:
    """Calculate scraping speed in jobs per minute.

    Args:
        jobs_found: Number of jobs found during scraping.
        start_time: When scraping started for this company.
        end_time: When scraping ended. If None, uses current time.

    Returns:
        float: Jobs per minute, or 0.0 if calculation isn't possible.

    Example:
        >>> start = datetime(2024, 1, 1, 10, 0, 0)
        >>> end = datetime(2024, 1, 1, 10, 2, 0)  # 2 minutes later
        >>> calculate_scraping_speed(30, start, end)
        15.0
    """
    try:
        # Validate inputs
        if not isinstance(jobs_found, int) or jobs_found < 0:
            return 0.0

        if start_time is None:
            return 0.0

        # Use current time if end_time not provided
        effective_end_time = end_time or datetime.now(timezone.utc)

        # Calculate duration in minutes
        duration = effective_end_time - start_time
        duration_minutes = duration.total_seconds() / 60.0

        # Avoid division by zero
        if duration_minutes <= 0:
            return 0.0

        # Calculate jobs per minute
        speed = jobs_found / duration_minutes

        # Round to 1 decimal place for display
        return round(speed, 1)

    except Exception:
        logger.exception("Error calculating scraping speed")
        return 0.0


def calculate_eta(
    total_companies: int,
    completed_companies: int,
    time_elapsed: float,
) -> str:
    """Calculate estimated time of arrival (ETA) based on completion rate.

    Args:
        total_companies: Total number of companies to scrape.
        completed_companies: Number of companies already completed.
        time_elapsed: Time elapsed since start in seconds.

    Returns:
        str: Formatted ETA string (e.g., "2m 30s", "1h 15m", "Done")

    Example:
        >>> calculate_eta(10, 3, 300)  # 3 of 10 done in 5 minutes
        "7m 0s"
    """
    result = "Unknown"

    try:
        # Validate inputs
        valid_total = isinstance(total_companies, int) and total_companies > 0
        valid_completed = (
            isinstance(completed_companies, int) and completed_companies >= 0
        )
        valid_time = isinstance(time_elapsed, int | float) and time_elapsed >= 0

        if not (valid_total and valid_completed and valid_time):
            result = "Unknown"
        elif completed_companies >= total_companies:
            result = "Done"
        elif completed_companies == 0 or time_elapsed == 0:
            result = "Calculating..."
        else:
            # Calculate completion rate (companies per second)
            completion_rate = completed_companies / time_elapsed
            # Calculate remaining companies and estimated time
            remaining_companies = total_companies - completed_companies
            estimated_seconds = remaining_companies / completion_rate
            # Format as human-readable duration
            result = format_duration(estimated_seconds)
    except Exception:
        logger.exception("Error calculating ETA")
        result = "Unknown"

    return result


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted duration (e.g., "2m 30s", "1h 15m", "45s")

    Example:
        >>> format_duration(125)
        "2m 5s"
        >>> format_duration(3665)
        "1h 1m"
    """
    try:
        if not isinstance(seconds, int | float) or seconds < 0:
            return "0s"

        # Convert to integer seconds
        total_seconds = int(seconds)

        # Calculate hours, minutes, seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        # Format based on magnitude
        if hours > 0:
            formatted = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        elif minutes > 0:
            formatted = f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
        else:
            formatted = f"{secs}s"
    except Exception:
        logger.exception("Error formatting duration")
        return "0s"

    return formatted


def format_timestamp(dt: datetime | None, format_str: str = "%H:%M:%S") -> str:
    """Format datetime to string with safe handling of None values.

    Args:
        dt: Datetime object to format, or None.
        format_str: Strftime format string.

    Returns:
        str: Formatted timestamp or "N/A" if dt is None.

    Example:
        >>> dt = datetime(2024, 1, 1, 15, 30, 45)
        >>> format_timestamp(dt)
        "15:30:45"
    """
    try:
        return "N/A" if dt is None else dt.strftime(format_str)

    except Exception:
        logger.exception("Error formatting timestamp")
        return "N/A"


def calculate_progress_percentage(
    completed_items: int,
    total_items: int,
) -> float:
    """Calculate progress percentage with safe division.

    Args:
        completed_items: Number of completed items.
        total_items: Total number of items.

    Returns:
        float: Progress percentage (0.0 to 100.0).

    Example:
        >>> calculate_progress_percentage(3, 10)
        30.0
    """
    try:
        if not isinstance(total_items, int) or total_items <= 0:
            return 0.0

        if not isinstance(completed_items, int) or completed_items < 0:
            return 0.0

        # Clamp to maximum of 100%
        percentage = min(100.0, (completed_items / total_items) * 100.0)

        return round(percentage, 1)

    except Exception:
        logger.exception("Error calculating progress percentage")
        return 0.0


def format_jobs_count(count: int, singular: str = "job", plural: str = "jobs") -> str:
    """Format job count with proper pluralization.

    Args:
        count: Number of jobs.
        singular: Singular form of the noun.
        plural: Plural form of the noun.

    Returns:
        str: Formatted count with proper pluralization.

    Example:
        >>> format_jobs_count(1)
        "1 job"
        >>> format_jobs_count(5)
        "5 jobs"
    """
    result = f"0 {plural}"

    try:
        if not isinstance(count, int):
            count = 0
        result = f"{count} {singular}" if count == 1 else f"{count} {plural}"
    except Exception:
        logger.exception("Error formatting jobs count")

    return result


def format_salary(amount: int) -> str:
    """Format salary amount with appropriate units (k/M).

    Args:
        amount: Salary amount in dollars.

    Returns:
        str: Formatted salary string (e.g., "$75k", "$1.2M").

    Example:
        >>> format_salary(75000)
        "$75k"
        >>> format_salary(1200000)
        "$1.2M"
    """
    try:
        if not isinstance(amount, int) or amount < 0:
            return "$0"

        if amount >= 1_000_000:
            return f"${amount / 1_000_000:.1f}M"
        if amount >= 1_000:
            return f"${amount // 1_000}k"
        return f"${amount}"

    except Exception:
        logger.exception("Error formatting salary")
        return "$0"
