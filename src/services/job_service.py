"""Job service for managing job data operations.

This module provides the JobService class with static methods for querying
and updating job records. It handles database operations for job filtering,
status updates, favorite toggling, and notes management.

Simple caching using Streamlit's native @st.cache_data decorator.
"""

import logging

from datetime import UTC, datetime
from typing import Any

# Import streamlit for caching decorators
try:
    import streamlit as st
except ImportError:
    # Create dummy decorator for non-Streamlit environments
    class _DummyStreamlit:
        """Dummy Streamlit class for non-Streamlit environments."""

        @staticmethod
        def cache_data(**_kwargs):
            """Dummy cache decorator that passes through the function unchanged."""

            def decorator(func):
                """Inner decorator function."""
                return func

            return decorator

    st = _DummyStreamlit()

from sqlalchemy import func, or_
from sqlalchemy.orm import joinedload
from sqlmodel import select

from src.constants import SALARY_DEFAULT_MIN, SALARY_UNBOUNDED_THRESHOLD
from src.database import db_session
from src.models import CompanySQL, JobSQL
from src.schemas import Job

logger = logging.getLogger(__name__)

# Type aliases for better readability
type FilterDict = dict[str, Any]
type JobCountStats = dict[str, int]
type JobUpdateBatch = list[dict[str, Any]]


class JobService:
    """Service class for job data operations.

    Provides static methods for querying, filtering, and updating job records
    in the database. This service acts as an abstraction layer between the UI
    and the database models.
    """

    @staticmethod
    def _to_dto(job_sql: JobSQL) -> Job:
        """Convert a single SQLModel object to its Pydantic DTO.

        Helper method for consistent DTO conversion that eliminates
        DetachedInstanceError by creating clean Pydantic objects without
        database session dependencies.

        Args:
            job_sql: SQLModel JobSQL object to convert with all fields populated.

        Returns:
            Job DTO object with all fields copied from the SQLModel instance.

        Raises:
            ValidationError: If SQLModel data doesn't match DTO schema.
        """
        # Extract company name from the relationship
        company_name = "Unknown"
        if hasattr(job_sql, "company_relation") and job_sql.company_relation:
            company_name = job_sql.company_relation.name

        # Create a dictionary with the job data and the resolved company name
        job_data = job_sql.model_dump()
        job_data["company"] = company_name

        # Remove the relationship field that's not part of the DTO
        job_data.pop("company_relation", None)

        return Job.model_validate(job_data)

    @classmethod
    def _to_dtos(cls, jobs_sql: list[JobSQL]) -> list[Job]:
        """Convert a list of SQLModel objects to Pydantic DTOs efficiently.

        Batch conversion helper that processes multiple SQLModel objects using
        the single-object conversion method for consistency.

        Args:
            jobs_sql: List of SQLModel JobSQL objects to convert.

        Returns:
            List of Job DTO objects in the same order as input.

        Raises:
            ValidationError: If any SQLModel data doesn't match DTO schema.
        """
        return [cls._to_dto(js) for js in jobs_sql]

    @staticmethod
    def _apply_filters_to_query(query, filters: FilterDict):
        """Apply filter criteria to a SQLModel query.

        Extracted method to share filtering logic between paginated and
        non-paginated queries.

        Args:
            query: Base SQLModel select query
            filters: Filter criteria dictionary

        Returns:
            Filtered query object
        """
        # Note: Don't early return for empty filters as we still need to apply
        # default filters like archived

        # Apply text search filter
        if text_search := filters.get("text_search", "").strip():
            query = query.filter(
                or_(
                    JobSQL.title.ilike(f"%{text_search}%"),
                    JobSQL.description.ilike(f"%{text_search}%"),
                )
            )

        # Apply company filter using JOIN for better performance
        if (
            company_filter := filters.get("company", [])
        ) and "All" not in company_filter:
            query = query.join(CompanySQL).filter(CompanySQL.name.in_(company_filter))

        # Apply application status filter
        if (
            status_filter := filters.get("application_status", [])
        ) and "All" not in status_filter:
            query = query.filter(JobSQL.application_status.in_(status_filter))

        # Apply date filters
        if date_from := filters.get("date_from"):
            date_from = JobService._parse_date(date_from)
            if date_from:
                query = query.filter(JobSQL.posted_date >= date_from)

        if date_to := filters.get("date_to"):
            date_to = JobService._parse_date(date_to)
            if date_to:
                query = query.filter(JobSQL.posted_date <= date_to)

        # Apply favorites filter
        if filters.get("favorites_only", False):
            query = query.filter(JobSQL.favorite.is_(True))

        # Apply salary range filters with high-value support
        salary_min = filters.get("salary_min")
        if salary_min is not None and salary_min > SALARY_DEFAULT_MIN:
            query = query.filter(func.json_extract(JobSQL.salary, "$[1]") >= salary_min)

        salary_max = filters.get("salary_max")
        if salary_max is not None and salary_max < SALARY_UNBOUNDED_THRESHOLD:
            query = query.filter(func.json_extract(JobSQL.salary, "$[0]") <= salary_max)

        # Filter out archived jobs by default
        if not filters.get("include_archived", False):
            query = query.filter(JobSQL.archived.is_(False))

        # Order by posted date (newest first) by default
        return query.order_by(JobSQL.posted_date.desc().nullslast())

    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_filtered_jobs(filters: FilterDict) -> list[Job]:
        """Get jobs filtered by the provided criteria.

        Uses simple Streamlit caching for improved performance.

        Args:
            filters: Dictionary containing filter criteria:
                - text_search: String to search in title and description
                - company: List of company names or "All"
                - application_status: List of status values or "All"
                - date_from: Start date for filtering
                - date_to: End date for filtering
                - favorites_only: Boolean to show only favorites
                - salary_min: Minimum salary filter (int or None)
                - salary_max: Maximum salary filter (int or None)

        Returns:
            List of Job DTO objects matching the filter criteria.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Start with base query, eagerly loading company relationship
                base_query = select(JobSQL).options(joinedload(JobSQL.company_relation))

                # Apply filters using shared method
                query = JobService._apply_filters_to_query(base_query, filters)

                jobs_sql = session.exec(query).all()

                # Convert SQLModel objects to Pydantic DTOs using helper
                jobs = JobService._to_dtos(jobs_sql)

                logger.info("Retrieved %d jobs with filters: %s", len(jobs), filters)
                return jobs

        except Exception:
            logger.exception("Failed to get filtered jobs")
            raise

    @staticmethod
    def update_job_status(job_id: int, status: str) -> bool:
        """Update the application status of a job.

        Args:
            job_id: Database ID of the job to update.
            status: New application status value.

        Returns:
            True if update was successful, False otherwise.

        Raises:
            Exception: If database update fails.
        """
        try:
            with db_session() as session:
                job = session.exec(select(JobSQL).filter_by(id=job_id)).first()
                if not job:
                    logger.warning("Job with ID %s not found", job_id)
                    return False

                old_status = job.application_status
                job.application_status = status

                # Set application date only if status changed to "Applied"
                # Preserve historical application data - never clear once set
                if (
                    status == "Applied"
                    and old_status != "Applied"
                    and job.application_date is None
                ):
                    job.application_date = datetime.now(UTC)

                logger.info(
                    "Updated job %s status from '%s' to '%s'",
                    job_id,
                    old_status,
                    status,
                )
                return True

        except Exception:
            logger.exception("Failed to update job status for job %s", job_id)
            raise

    @staticmethod
    def toggle_favorite(job_id: int) -> bool:
        """Toggle the favorite status of a job.

        Args:
            job_id: Database ID of the job to toggle.

        Returns:
            New favorite status (True/False) if successful, False if job not found.

        Raises:
            Exception: If database update fails.
        """
        try:
            with db_session() as session:
                job = session.exec(select(JobSQL).filter_by(id=job_id)).first()
                if not job:
                    logger.warning("Job with ID %s not found", job_id)
                    return False

                job.favorite = not job.favorite

                logger.info("Toggled favorite for job %s to %s", job_id, job.favorite)
                return job.favorite

        except Exception:
            logger.exception("Failed to toggle favorite for job %s", job_id)
            raise

    @staticmethod
    def update_notes(job_id: int, notes: str) -> bool:
        """Update the notes for a job.

        Args:
            job_id: Database ID of the job to update.
            notes: New notes content.

        Returns:
            True if update was successful, False otherwise.

        Raises:
            Exception: If database update fails.
        """
        try:
            with db_session() as session:
                job = session.exec(select(JobSQL).filter_by(id=job_id)).first()
                if not job:
                    logger.warning("Job with ID %s not found", job_id)
                    return False

                job.notes = notes

                logger.info("Updated notes for job %s", job_id)
                return True

        except Exception:
            logger.exception("Failed to update notes for job %s", job_id)
            raise

    @staticmethod
    def get_job_by_id(job_id: int) -> Job | None:
        """Get a single job by its ID.

        Args:
            job_id: Database ID of the job to retrieve.

        Returns:
            Job DTO object if found, None otherwise.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Use joinedload for single job lookup with company data
                job_sql = session.exec(
                    select(JobSQL)
                    .options(joinedload(JobSQL.company_relation))
                    .filter_by(id=job_id)
                ).first()

                if job_sql:
                    # Convert to DTO using helper method
                    job = JobService._to_dto(job_sql)

                    logger.info("Retrieved job %s: %s", job_id, job.title)
                    return job
                logger.warning("Job with ID %s not found", job_id)
                return None

        except Exception:
            logger.exception("Failed to get job %s", job_id)
            raise

    @staticmethod
    @st.cache_data(ttl=120)  # Cache for 2 minutes
    def get_job_counts_by_status() -> JobCountStats:
        """Get count of jobs grouped by application status.

        Uses simple Streamlit caching for improved performance.

        Returns:
            Dictionary mapping status names to counts.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                results = session.exec(
                    select(JobSQL.application_status, func.count(JobSQL.id))
                    .filter(JobSQL.archived.is_(False))
                    .group_by(JobSQL.application_status)
                ).all()

                counts = dict(results)
                logger.info("Job counts by status: %s", counts)
                return counts

        except Exception:
            logger.exception("Failed to get job counts")
            raise

    @staticmethod
    def archive_job(job_id: int) -> bool:
        """Archive a job (soft delete).

        Args:
            job_id: Database ID of the job to archive.

        Returns:
            True if archiving was successful, False otherwise.

        Raises:
            Exception: If database update fails.
        """
        try:
            with db_session() as session:
                job = session.exec(select(JobSQL).filter_by(id=job_id)).first()
                if not job:
                    logger.warning("Job with ID %s not found", job_id)
                    return False

                job.archived = True

                logger.info("Archived job %s: %s", job_id, job.title)
                return True

        except Exception:
            logger.exception("Failed to archive job %s", job_id)
            raise

    @staticmethod
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def get_active_companies() -> list[str]:
        """Get list of active company names for scraping.

        Returns:
            List of active company names.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Query for active companies, ordered by name for consistency
                query = (
                    select(CompanySQL.name)
                    .filter(CompanySQL.active.is_(True))
                    .order_by(CompanySQL.name)
                )

                company_names = session.exec(query).all()

                logger.info("Retrieved %d active companies", len(company_names))
                return list(company_names)

        except Exception:
            logger.exception("Failed to get active companies")
            raise

    @staticmethod
    def _parse_date(date_input: str | datetime | None) -> datetime | None:
        """Parse date input into datetime object.

        Supports common formats encountered when scraping job sites:
        - ISO format (2024-12-31)
        - US format (12/31/2024)
        - EU format (31/12/2024)
        - Human readable (December 31, 2024)

        Args:
            date_input: Date as string, datetime object, or None.

        Returns:
            Parsed datetime object or None if input is None/invalid.
        """
        if isinstance(date_input, str):
            date_input = date_input.strip()
            if not date_input:
                return None

            # Try ISO format first (most common for APIs)
            try:
                dt = datetime.fromisoformat(date_input)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
            except ValueError:  # noqa: S110
                # Expected: Continue to alternative date parsing if ISO format fails
                pass
            else:
                return dt

            # Try common formats found in job site scraping
            date_formats = [
                "%Y-%m-%d",  # 2024-12-31 (ISO date)
                "%m/%d/%Y",  # 12/31/2024 (US format)
                "%d/%m/%Y",  # 31/12/2024 (EU format)
                "%B %d, %Y",  # December 31, 2024
                "%d %B %Y",  # 31 December 2024
            ]

            for date_format in date_formats:
                try:
                    return datetime.strptime(date_input, date_format).replace(
                        tzinfo=UTC
                    )

                except ValueError:  # noqa: S112
                    # Expected: Try next date format if this one fails
                    continue

            # If all formats fail, log warning
            logger.warning("Could not parse date: %s", date_input)
        elif date_input is not None and not isinstance(date_input, datetime):
            logger.warning("Unsupported date type: %s", type(date_input))

        return None

    @staticmethod
    def bulk_update_jobs(job_updates: JobUpdateBatch) -> bool:
        """Bulk update job records with favorite, status, and notes changes.

        Args:
            job_updates: List of dicts with keys: id, favorite, application_status,
                notes

        Returns:
            True if updates were successful.

        Raises:
            Exception: If database update fails.
        """
        if not job_updates:
            return True

        try:
            with db_session() as session:
                # Bulk load all jobs to update in a single query to avoid N+1
                job_ids = [update["id"] for update in job_updates]
                jobs_to_update = session.exec(
                    select(JobSQL).where(JobSQL.id.in_(job_ids))
                ).all()

                # Create a lookup dict for efficient updates
                jobs_by_id = {job.id: job for job in jobs_to_update}

                for update in job_updates:
                    job = jobs_by_id.get(update["id"])
                    if job:
                        job.favorite = update.get("favorite", job.favorite)
                        job.application_status = update.get(
                            "application_status", job.application_status
                        )
                        job.notes = update.get("notes", job.notes)

                        # Set application date if status changed to "Applied"
                        if (
                            update.get("application_status") == "Applied"
                            and job.application_status == "Applied"
                            and not job.application_date
                        ):
                            job.application_date = datetime.now(UTC)

                logger.info("Bulk updated %d jobs", len(job_updates))
                return True

        except Exception:
            logger.exception("Failed to bulk update jobs")
            raise

    @staticmethod
    def get_jobs_with_company_names_direct_join(
        filters: FilterDict,
    ) -> list[dict[str, "Any"]]:
        """Alternative implementation using direct SQL JOIN as suggested by Sourcery.

        This method demonstrates the SQL join approach for fetching company names
        directly in the query, as suggested in the PR feedback.

        Args:
            filters: Dictionary containing filter criteria.

        Returns:
            List of dictionaries with job data and company names.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Use explicit JOIN to get company names directly

                query = select(JobSQL, CompanySQL.name.label("company_name")).join(
                    CompanySQL, JobSQL.company_id == CompanySQL.id
                )

                # Apply the same filters as in get_filtered_jobs
                if text_search := filters.get("text_search", "").strip():
                    query = query.filter(
                        or_(
                            JobSQL.title.ilike(f"%{text_search}%"),
                            JobSQL.description.ilike(f"%{text_search}%"),
                        )
                    )

                if (
                    company_filter := filters.get("company", [])
                ) and "All" not in company_filter:
                    query = query.filter(CompanySQL.name.in_(company_filter))

                if (
                    status_filter := filters.get("application_status", [])
                ) and "All" not in status_filter:
                    query = query.filter(JobSQL.application_status.in_(status_filter))

                if date_from := filters.get("date_from"):
                    date_from = JobService._parse_date(date_from)
                    if date_from:
                        query = query.filter(JobSQL.posted_date >= date_from)

                if date_to := filters.get("date_to"):
                    date_to = JobService._parse_date(date_to)
                    if date_to:
                        query = query.filter(JobSQL.posted_date <= date_to)

                if filters.get("favorites_only", False):
                    query = query.filter(JobSQL.favorite.is_(True))

                if not filters.get("include_archived", False):
                    query = query.filter(JobSQL.archived.is_(False))

                query = query.order_by(JobSQL.posted_date.desc().nullslast())

                results = session.exec(query).all()

                # Convert results to dictionary format
                jobs_data = []
                for job_sql, company_name in results:
                    job_dict = job_sql.model_dump()
                    job_dict["company"] = company_name
                    jobs_data.append(job_dict)

                logger.info(
                    "Retrieved %d jobs with direct JOIN approach", len(jobs_data)
                )
                return jobs_data

        except Exception:
            logger.exception("Failed to get jobs with direct JOIN")
            raise

    @staticmethod
    def invalidate_job_cache(job_id: int | None = None) -> bool:  # noqa: ARG004
        """Clear Streamlit cache for job-related data.

        Args:
            job_id: Ignored - Streamlit cache is cleared globally

        Returns:
            True if cache invalidation was successful
        """
        try:
            # Clear relevant Streamlit caches
            JobService.get_filtered_jobs.clear()
            JobService.get_job_counts_by_status.clear()
            JobService.get_active_companies.clear()

            logger.info("Cleared Streamlit job caches")
        except Exception:
            logger.exception("Failed to clear Streamlit cache")
            return False
        else:
            return True
