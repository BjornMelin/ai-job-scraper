"""UI utilities consolidating formatters, validation, and Streamlit context helpers.

This consolidated module combines formatting utilities, type-safe validation functions,
and Streamlit context detection for optimal organization and reduced file count.

Consolidates:
- formatters.py: Data formatting and time calculations (now using humanize library)
- validation_utils.py: Type-safe validation with Pydantic
- streamlit_context.py: Streamlit runtime detection
"""

import logging

from datetime import datetime, timedelta, timezone
from typing import Any

import humanize

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ===============================================================================
# STREAMLIT CONTEXT DETECTION (from streamlit_context.py)
# ===============================================================================


def is_streamlit_context() -> bool:
    """Check if we're running in a proper Streamlit context.

    This function determines whether the current execution context is within
    a running Streamlit application. This is crucial for preventing page
    functions from executing when modules are imported during testing or
    other non-Streamlit scenarios.

    Returns:
        bool: True if in Streamlit runtime context, False otherwise.

    Examples:
        >>> # In a Streamlit page file:
        >>> if is_streamlit_context():
        ...     render_page()  # Only execute when actually running in Streamlit
    """
    try:
        # Check if Streamlit's script run context exists
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        # If Streamlit is not available or the context doesn't exist
        return False


# ===============================================================================
# VALIDATION UTILITIES (from validation_utils.py)
# ===============================================================================


class SafeIntValidator(BaseModel):
    """Pydantic model for safe integer validation."""

    value: int = Field(ge=0, description="Non-negative integer value")

    @field_validator("value", mode="before")
    @classmethod
    def convert_to_safe_int(cls, v: Any) -> int:
        """Convert various input types to safe non-negative integers.

        Args:
            v: Input value of any type

        Returns:
            Non-negative integer

        Raises:
            ValueError: If value cannot be safely converted to non-negative integer
        """
        if v is None:
            return 0

        # Handle string inputs
        if isinstance(v, str):
            # Remove whitespace and handle empty strings
            v = v.strip()
            if not v:
                return 0

            # Try to convert string to number
            try:
                # Handle float strings by converting to float first
                v = float(v) if "." in v else int(v)
            except ValueError as e:
                error_msg = f"Cannot convert string '{v}' to integer: {e}"
                raise ValueError(error_msg) from e

        # Handle float inputs - round to nearest integer
        if isinstance(v, float):
            if not -1e15 <= v <= 1e15:  # Prevent overflow
                error_msg = f"Float value {v} is too large to convert to integer"
                raise ValueError(error_msg)
            v = round(v)

        # Handle boolean inputs
        if isinstance(v, bool):
            v = int(v)

        # Final integer conversion and validation
        try:
            result = int(v)
        except (ValueError, TypeError) as e:
            error_msg = f"Cannot convert {type(v).__name__} value {v} to integer: {e}"
            raise ValueError(error_msg) from e

        # Ensure non-negative
        if result < 0:
            logger.warning("Negative value %d converted to 0 for safety", result)
            result = 0

        return result


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert any value to a non-negative integer.

    This function provides robust type conversion with comprehensive error handling,
    following library-first principles using Pydantic validation.

    Args:
        value: Input value of any type
        default: Default value to return if conversion fails (default: 0)

    Returns:
        Non-negative integer value

    Examples:
        >>> safe_int("123")
        123
        >>> safe_int("45.7")
        46
        >>> safe_int(None)
        0
        >>> safe_int("invalid", default=10)
        10
        >>> safe_int(-5)
        0
    """
    try:
        validator = SafeIntValidator(value=value)
    except ValidationError as e:
        logger.warning("Failed to convert %s to safe integer: %s", value, e)
        return max(0, default)  # Ensure default is also non-negative
    except Exception:
        logger.exception("Unexpected error converting %s to safe integer", value)
        return max(0, default)

    return validator.value


def safe_job_count(value: Any, company_name: str = "unknown") -> int:
    """Safely convert job count values with context-aware logging.

    Specialized function for converting job counts with additional context
    for better error tracking and debugging.

    Args:
        value: Job count value to convert
        company_name: Company name for context in error messages

    Returns:
        Non-negative integer representing job count
    """
    try:
        result = safe_int(value)
    except Exception as e:
        logger.warning(
            "Failed to convert job count for %s: %s (%s)", company_name, value, e
        )
        return 0

    if value != result and value is not None:
        logger.info("Converted job count for %s: %s -> %s", company_name, value, result)
    return result


# Type aliases for better code documentation
JobCount = int
SafeInteger = int


# ===============================================================================
# FORMATTING UTILITIES (from formatters.py)
# ===============================================================================


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
    """Calculate estimated time of arrival (ETA) using humanize library.

    Args:
        total_companies: Total number of companies to scrape.
        completed_companies: Number of companies already completed.
        time_elapsed: Time elapsed since start in seconds.

    Returns:
        str: Formatted ETA string (e.g., "2 minutes", "1 hour", "Done")

    Example:
        >>> calculate_eta(10, 3, 300)  # 3 of 10 done in 5 minutes
        "7 minutes"
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
            # Use humanize library for natural duration formatting
            result = format_duration(estimated_seconds)
    except Exception:
        logger.exception("Error calculating ETA")
        result = "Unknown"

    return result


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string using humanize library.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted duration (e.g., "2 minutes", "1 hour", "45 seconds")

    Example:
        >>> format_duration(125)
        "2 minutes"
        >>> format_duration(3665)
        "1 hour"
    """
    try:
        if not isinstance(seconds, int | float) or seconds < 0:
            return "0 seconds"

        # Use humanize library for natural duration formatting
        delta = timedelta(seconds=seconds)
        return humanize.naturaldelta(delta)
    except Exception:
        logger.exception("Error formatting duration")
        return "0 seconds"


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
    """Format salary amount using humanize library for clean display.

    Args:
        amount: Salary amount in dollars.

    Returns:
        str: Formatted salary string (e.g., "$75.0 thousand", "$1.2 million").

    Example:
        >>> format_salary(75000)
        "$75.0 thousand"
        >>> format_salary(1200000)
        "$1.2 million"
    """
    try:
        if not isinstance(amount, int) or amount < 0:
            return "$0"

        # Use humanize library for natural number formatting
        formatted_amount = humanize.intword(amount, format="%.1f")
    except Exception:
        logger.exception("Error formatting salary")
        return "$0"
    else:
        return f"${formatted_amount}"
