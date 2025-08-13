"""Computed field helper functions for JobSQL and CompanySQL models.

This module provides pure helper functions that replace SQLModel computed fields
with explicit function calls. All functions are stateless and side-effect-free,
making them easier to test and reason about compared to computed properties.

These functions were extracted from ui_helpers.py to create a focused module
for computed field logic, improving code organization and maintainability.

Design Principles:
- Pure functions with no side effects
- Clear error handling with graceful fallbacks
- Modern Python type hints and patterns
- Library-first approach where applicable
- Comprehensive validation and edge case handling

Function Categories:
- Job computed fields: salary, posting dates, company relationships
- Company computed fields: job counts, statistics, aggregations
"""

import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .formatters import SalaryTuple

if TYPE_CHECKING:
    from src.models import CompanySQL, JobSQL

logger = logging.getLogger(__name__)


# =============================================================================
# JOB COMPUTED FIELD HELPERS
# =============================================================================


def get_salary_min(salary: SalaryTuple | None) -> int | None:
    """Extract minimum salary value from salary tuple.

    Replacement for JobSQL computed field that provides safer access
    to salary data with proper null handling.

    Args:
        salary: Salary tuple (min, max) or None from salary parsing

    Returns:
        Minimum salary value or None if salary data unavailable

    Examples:
        >>> get_salary_min((50000, 80000))
        50000
        >>> get_salary_min(None)
        None
    """
    return salary[0] if salary else None


def get_salary_max(salary: SalaryTuple | None) -> int | None:
    """Extract maximum salary value from salary tuple.

    Replacement for JobSQL computed field that provides safer access
    to salary data with proper null handling.

    Args:
        salary: Salary tuple (min, max) or None from salary parsing

    Returns:
        Maximum salary value or None if salary data unavailable

    Examples:
        >>> get_salary_max((50000, 80000))
        80000
        >>> get_salary_max(None)
        None
    """
    return salary[1] if salary else None


def calculate_days_since_posted(posted_date: datetime | None) -> int | None:
    """Calculate days since job was posted.

    Replacement for JobSQL.days_since_posted computed field.
    Handles timezone-aware calculations with proper error handling.

    Args:
        posted_date: Job posting date, may be timezone-naive or aware

    Returns:
        Days since posted (non-negative integer) or None if date unavailable

    Examples:
        >>> from datetime import datetime, UTC
        >>> posted = datetime(2023, 1, 1, tzinfo=UTC)
        >>> days = calculate_days_since_posted(posted)
        >>> isinstance(days, int) and days >= 0
        True
    """
    if not posted_date:
        return None

    try:
        # Get current time in UTC for consistent calculations
        now_utc = datetime.now(UTC)

        # Ensure posted_date is timezone-aware for accurate comparison
        if not posted_date.tzinfo:
            posted_date = posted_date.replace(tzinfo=UTC)

        # Calculate difference and ensure non-negative result
        days_diff = (now_utc - posted_date).days
        return max(0, days_diff)  # Prevent negative values for future dates

    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Error calculating days since posted: %s", e)
        return None
    except Exception:
        logger.exception("Unexpected error calculating days since posted")
        return None


def is_job_recently_posted(
    posted_date: datetime | None,
    days_threshold: int = 7,
) -> bool:
    """Check if job was posted within the threshold days.

    Replacement for JobSQL.is_recently_posted computed field.
    Provides flexible threshold configuration for different use cases.

    Args:
        posted_date: Job posting date
        days_threshold: Number of days to consider as recent (default: 7)

    Returns:
        True if job was posted within threshold days, False otherwise

    Examples:
        >>> from datetime import datetime, UTC, timedelta
        >>> recent_date = datetime.now(UTC) - timedelta(days=3)
        >>> is_job_recently_posted(recent_date, days_threshold=7)
        True
        >>> old_date = datetime.now(UTC) - timedelta(days=10)
        >>> is_job_recently_posted(old_date, days_threshold=7)
        False
    """
    if days_threshold < 0:
        logger.warning("Invalid days_threshold: %d, using default", days_threshold)
        days_threshold = 7

    days = calculate_days_since_posted(posted_date)
    return days is not None and days <= days_threshold


def get_job_company_name(company_relation: "CompanySQL | None") -> str:
    """Get company name from relationship.

    Replacement for JobSQL.company computed field that provides
    safe access to company names with proper fallback handling.

    Args:
        company_relation: Company relationship object from JobSQL.company

    Returns:
        Company name or 'Unknown' if company data unavailable

    Examples:
        >>> company = CompanySQL(name="Example Corp")
        >>> get_job_company_name(company)
        'Example Corp'
        >>> get_job_company_name(None)
        'Unknown'
    """
    if not company_relation:
        return "Unknown"

    try:
        # Safely access company name with fallback
        name = getattr(company_relation, "name", None)
        return name if name and isinstance(name, str) and name.strip() else "Unknown"

    except (AttributeError, TypeError):
        logger.warning("Error accessing company name from relation")
        return "Unknown"


# =============================================================================
# COMPANY COMPUTED FIELD HELPERS
# =============================================================================


def calculate_total_jobs_count(jobs: list["JobSQL"] | None) -> int:
    """Calculate total job count for company.

    Replacement for CompanySQL.total_jobs_count computed field.
    Provides safe counting with comprehensive input validation.

    Args:
        jobs: List of job objects from company.jobs relationship

    Returns:
        Total number of jobs (non-negative integer)

    Examples:
        >>> jobs = [JobSQL(), JobSQL(), JobSQL()]
        >>> calculate_total_jobs_count(jobs)
        3
        >>> calculate_total_jobs_count(None)
        0
        >>> calculate_total_jobs_count([])
        0
    """
    if not jobs:
        return 0

    try:
        # Validate input is actually a list-like object
        if not hasattr(jobs, "__len__"):
            logger.warning("Invalid jobs input type: %s", type(jobs))
            return 0

        return len(jobs)

    except (TypeError, AttributeError) as e:
        logger.warning("Error calculating total jobs count: %s", e)
        return 0


def calculate_active_jobs_count(jobs: list["JobSQL"] | None) -> int:
    """Calculate active (non-archived) job count for company.

    Replacement for CompanySQL.active_jobs_count computed field.
    Filters out archived jobs with comprehensive error handling.

    Args:
        jobs: List of job objects from company.jobs relationship

    Returns:
        Number of active (non-archived) jobs (non-negative integer)

    Examples:
        >>> jobs = [
        ...     JobSQL(archived=False),
        ...     JobSQL(archived=True),
        ...     JobSQL(archived=False),
        ... ]
        >>> calculate_active_jobs_count(jobs)
        2
    """
    if not jobs:
        return 0

    try:
        # Validate input and filter active jobs
        if not hasattr(jobs, "__iter__"):
            logger.warning("Invalid jobs input type: %s", type(jobs))
            return 0

        active_count = 0
        for job in jobs:
            try:
                # Default to non-archived if attribute missing or invalid
                archived = getattr(job, "archived", False)
                if not archived:
                    active_count += 1
            except (AttributeError, TypeError):
                # Treat jobs with invalid archived status as active
                active_count += 1
    except (TypeError, AttributeError) as e:
        logger.warning("Error calculating active jobs count: %s", e)
        return 0
    else:
        return active_count


def find_last_job_posted(jobs: list["JobSQL"] | None) -> datetime | None:
    """Find the most recent job posting date.

    Replacement for CompanySQL.last_job_posted computed field.
    Handles missing or invalid dates gracefully.

    Args:
        jobs: List of job objects from company.jobs relationship

    Returns:
        Most recent posting date or None if no valid dates found

    Examples:
        >>> from datetime import datetime, UTC
        >>> jobs = [
        ...     JobSQL(posted_date=datetime(2023, 1, 1, tzinfo=UTC)),
        ...     JobSQL(posted_date=datetime(2023, 6, 1, tzinfo=UTC)),
        ...     JobSQL(posted_date=datetime(2023, 3, 1, tzinfo=UTC)),
        ... ]
        >>> result = find_last_job_posted(jobs)
        >>> result.month == 6  # Most recent is June
        True
    """
    if not jobs:
        return None

    try:
        # Validate input and collect valid posting dates
        if not hasattr(jobs, "__iter__"):
            logger.warning("Invalid jobs input type: %s", type(jobs))
            return None

        valid_dates = []
        for job in jobs:
            try:
                posted_date = getattr(job, "posted_date", None)
                if posted_date and isinstance(posted_date, datetime):
                    valid_dates.append(posted_date)
            except (AttributeError, TypeError):
                logger.debug("Skipping job with invalid posted_date attribute")
                continue

        # Return the maximum date or None if no valid dates
        return max(valid_dates) if valid_dates else None

    except (TypeError, ValueError) as e:
        logger.warning("Error finding last job posted date: %s", e)
        return None
    except Exception:
        logger.exception("Unexpected error finding last job posted date")
        return None


# =============================================================================
# UTILITY FUNCTIONS FOR COMPUTED FIELDS
# =============================================================================


def validate_job_list(jobs: list["JobSQL"] | None) -> list["JobSQL"]:
    """Validate and normalize job list input.

    Helper function to ensure consistent job list validation
    across all company computed field functions.

    Args:
        jobs: Raw job list input that may be None or invalid

    Returns:
        Valid job list (empty list if input invalid)

    Examples:
        >>> jobs = [JobSQL(), JobSQL()]
        >>> validated = validate_job_list(jobs)
        >>> len(validated) == 2
        True
        >>> validate_job_list(None)
        []
    """
    if not jobs:
        return []

    try:
        # Ensure we have an iterable list-like object
        if hasattr(jobs, "__iter__") and hasattr(jobs, "__len__"):
            return list(jobs)
        logger.warning("Invalid job list type: %s", type(jobs))
    except (TypeError, AttributeError):
        logger.warning("Error validating job list")
        return []
    else:
        return []


def calculate_job_statistics(
    jobs: list["JobSQL"] | None,
) -> dict[str, int | datetime | None]:
    """Calculate comprehensive job statistics for a company.

    Convenience function that calculates multiple job-related metrics
    in a single pass for efficiency.

    Args:
        jobs: List of job objects from company.jobs relationship

    Returns:
        Dictionary containing job statistics:
        - total_count: Total number of jobs
        - active_count: Number of active jobs
        - archived_count: Number of archived jobs
        - last_posted: Most recent posting date

    Examples:
        >>> jobs = [JobSQL(archived=False), JobSQL(archived=True)]
        >>> stats = calculate_job_statistics(jobs)
        >>> stats["total_count"]
        2
        >>> stats["active_count"]
        1
    """
    validated_jobs = validate_job_list(jobs)

    if not validated_jobs:
        return {
            "total_count": 0,
            "active_count": 0,
            "archived_count": 0,
            "last_posted": None,
        }

    total_count = len(validated_jobs)
    active_count = 0
    archived_count = 0
    valid_dates = []

    for job in validated_jobs:
        try:
            # Count archived vs active jobs
            archived = getattr(job, "archived", False)
            if archived:
                archived_count += 1
            else:
                active_count += 1

            # Collect valid posting dates
            posted_date = getattr(job, "posted_date", None)
            if posted_date and isinstance(posted_date, datetime):
                valid_dates.append(posted_date)

        except (AttributeError, TypeError):
            # Default to active for jobs with missing/invalid data
            active_count += 1

    return {
        "total_count": total_count,
        "active_count": active_count,
        "archived_count": archived_count,
        "last_posted": max(valid_dates) if valid_dates else None,
    }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "SalaryTuple",
    "calculate_active_jobs_count",
    "calculate_days_since_posted",
    "calculate_job_statistics",
    "calculate_total_jobs_count",
    "find_last_job_posted",
    "get_job_company_name",
    "get_salary_max",
    "get_salary_min",
    "is_job_recently_posted",
    "validate_job_list",
]
