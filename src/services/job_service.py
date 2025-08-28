"""Job service for managing job data operations.

This module provides the JobService class with static methods for querying
and updating job records. It handles database operations for job filtering,
status updates, favorite toggling, and notes management.

Streamlit caching using Streamlit's native @st.cache_data decorator.
"""

import logging

from datetime import UTC, datetime, timedelta

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

            def decorator(wrapped_func):
                """Inner decorator function."""
                return wrapped_func

            return decorator

    st = _DummyStreamlit()

from sqlalchemy import func
from sqlmodel import select

from src.constants import SALARY_DEFAULT_MIN, SALARY_UNBOUNDED_THRESHOLD
from src.database import db_session
from src.models import CompanySQL, JobSQL
from src.models.job_models import JobPosting, JobScrapeRequest, JobScrapeResult, JobSite
from src.schemas import Job
from src.scraping.job_scraper import job_scraper

logger = logging.getLogger(__name__)

# Type aliases for better readability
type FilterDict = dict[str, object]
type JobCountStats = dict[str, int]
type JobUpdateBatch = list[dict[str, object]]


class JobService:
    """Service class for job data operations.

    Provides static methods for querying, filtering, and updating job records
    in the database. This service acts as an abstraction layer between the UI
    and the database models. Now includes JobSpy integration for scraping.
    """

    def __init__(self):
        """Initialize JobService with JobSpy scraper."""
        self.scraper = job_scraper

    @staticmethod
    def _to_dto(job_sql: JobSQL) -> Job:
        """Convert a single SQLModel object to its Pydantic DTO.

        Helper method for consistent DTO conversion that eliminates
        DetachedInstanceError by creating clean Pydantic objects without
        database session dependencies.

        Note: This method falls back to "Unknown" for company name.
        Use _to_dto_with_company when company name is available.

        Args:
            job_sql: SQLModel JobSQL object to convert with all fields populated.

        Returns:
            Job DTO object with all fields copied from the SQLModel instance.

        Raises:
            ValidationError: If SQLModel data doesn't match DTO schema.
        """
        # Fallback for cases where company name is not available
        company_name = "Unknown"

        # Create a dictionary with the job data and the resolved company name
        job_data = job_sql.model_dump()
        job_data["company"] = company_name

        return Job.model_validate(job_data)

    @staticmethod
    def _to_dto_with_company(job_sql: JobSQL, company_name: str) -> Job:
        """Convert a single SQLModel object to its Pydantic DTO with company name.

        Helper method for consistent DTO conversion that eliminates
        DetachedInstanceError by creating clean Pydantic objects without
        database session dependencies.

        Args:
            job_sql: SQLModel JobSQL object to convert with all fields populated.
            company_name: The resolved company name from the database join.

        Returns:
            Job DTO object with all fields copied from the SQLModel instance.

        Raises:
            ValidationError: If SQLModel data doesn't match DTO schema.
        """
        # Create a dictionary with the job data and the provided company name
        job_data = job_sql.model_dump()
        job_data["company"] = company_name

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

        # Note: Text search filtering is handled by search_service.py using SQLite FTS5
        # This service focuses on database filtering without text search capabilities

        # Apply company filter - assumes CompanySQL is already joined
        if (
            company_filter := filters.get("company", [])
        ) and "All" not in company_filter:
            query = query.filter(CompanySQL.name.in_(company_filter))

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
    def get_filtered_jobs(filters: FilterDict | None = None) -> list[Job]:
        """Get jobs filtered by the provided criteria.

        Uses Streamlit-based caching for improved performance.

        Args:
            filters: Dictionary containing filter criteria:
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
                # Handle empty filters
                if filters is None:
                    filters = {}

                # Join with CompanySQL to get company names
                base_query = select(JobSQL, CompanySQL.name.label("company_name")).join(
                    CompanySQL, JobSQL.company_id == CompanySQL.id
                )

                # Apply filters using shared method
                query = JobService._apply_filters_to_query(base_query, filters)

                results = session.exec(query).all()

                # Convert SQLModel objects to Pydantic DTOs with company names
                jobs = []
                for job_sql, company_name in results:
                    jobs.append(JobService._to_dto_with_company(job_sql, company_name))

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
                # Join with CompanySQL to get company name
                result = session.exec(
                    select(JobSQL, CompanySQL.name.label("company_name"))
                    .join(CompanySQL, JobSQL.company_id == CompanySQL.id)
                    .filter(JobSQL.id == job_id),
                ).first()

                if result:
                    job_sql, company_name = result
                    # Convert to DTO using helper method with company name
                    job = JobService._to_dto_with_company(job_sql, company_name)

                    logger.info("Retrieved job %s: %s", job_id, job.title)
                    return job
                logger.warning("Job with ID %s not found", job_id)
                return None

        except Exception:
            logger.exception("Failed to get job %s", job_id)
            raise

    @staticmethod
    def get_recently_updated_jobs(
        job_ids: list[int], since: datetime, limit: int = 10
    ) -> list[Job]:
        """Get jobs that have been recently updated since a given timestamp.

        This method is used by fragment-based components to detect real-time changes
        and display update notifications without full page refreshes.

        Args:
            job_ids: List of job IDs to check for updates.
            since: Only return jobs updated after this datetime.
            limit: Maximum number of updated jobs to return.

        Returns:
            List of Job DTO objects that have been updated since the given time.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Query for jobs updated since the given timestamp
                query = (
                    select(JobSQL, CompanySQL.name.label("company_name"))
                    .join(CompanySQL, JobSQL.company_id == CompanySQL.id)
                    .filter(JobSQL.id.in_(job_ids), JobSQL.updated_at > since)
                    .order_by(JobSQL.updated_at.desc())
                    .limit(limit)
                )

                results = session.exec(query).all()

                # Convert to DTOs
                updated_jobs = []
                for job_sql, company_name in results:
                    updated_jobs.append(
                        JobService._to_dto_with_company(job_sql, company_name)
                    )

                logger.debug(
                    "Found %d recently updated jobs from %d candidates since %s",
                    len(updated_jobs),
                    len(job_ids),
                    since.isoformat(),
                )

                return updated_jobs

        except Exception:
            logger.exception("Failed to get recently updated jobs")
            raise

    @staticmethod
    @st.cache_data(ttl=120)  # Cache for 2 minutes
    def get_job_counts_by_status() -> JobCountStats:
        """Get count of jobs grouped by application status.

        Uses Streamlit-based caching for improved performance.

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
                    .group_by(JobSQL.application_status),
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
                        tzinfo=UTC,
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
                    select(JobSQL).where(JobSQL.id.in_(job_ids)),
                ).all()

                # Create a lookup dict for efficient updates
                jobs_by_id = {job.id: job for job in jobs_to_update}

                for update in job_updates:
                    job = jobs_by_id.get(update["id"])
                    if job:
                        job.favorite = update.get("favorite", job.favorite)
                        job.application_status = update.get(
                            "application_status",
                            job.application_status,
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

    async def search_and_save_jobs(
        self,
        search_term: str,
        location: str | None = None,
        sites: list[str] | None = None,
        results_wanted: int = 100,
        save_to_db: bool = True,
    ) -> JobScrapeResult:
        """Search for jobs using JobSpy and optionally save to database.

        Args:
            search_term: Job search term (e.g., "software engineer").
            location: Location to search (e.g., "San Francisco").
            sites: List of job sites to search (e.g., ["linkedin", "indeed"]).
            results_wanted: Number of results desired.
            save_to_db: Whether to save jobs to database.

        Returns:
            JobScrapeResult with scraped jobs and metadata.
        """
        try:
            # Convert string sites to JobSite enums
            site_enums = []
            if sites:
                for site in sites:
                    try:
                        site_enum = JobSite.normalize(site)
                        if site_enum:
                            site_enums.append(site_enum)
                    except ValueError:
                        logger.warning("Unknown job site: %s", site)
                        continue

            if not site_enums:
                site_enums = [JobSite.LINKEDIN]  # Default to LinkedIn

            # Create scrape request
            request = JobScrapeRequest(
                site_name=site_enums,
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                linkedin_fetch_description=True,
            )

            # Execute scraping
            result = await self.scraper.scrape_jobs_async(request)

            if save_to_db and result.jobs:
                await self._save_jobs_to_database(result.jobs)
                logger.info("Saved %d jobs to database", len(result.jobs))
            return result  # noqa: TRY300

        except Exception:
            logger.exception("Failed to search and save jobs")
            raise

    async def _save_jobs_to_database(self, jobs: list[JobPosting]) -> None:
        """Save scraped jobs to database with deduplication.

        Args:
            jobs: List of JobPosting objects to save.
        """
        try:
            with db_session() as session:
                for job_posting in jobs:
                    # Check for existing job by link (primary deduplication)
                    existing_job = session.exec(
                        select(JobSQL).filter_by(
                            link=job_posting.job_url or job_posting.job_url_direct
                        )
                    ).first()

                    if existing_job:
                        # Update last_seen timestamp
                        existing_job.last_seen = datetime.now(UTC)
                        continue

                    # Get or create company
                    company_id = await self._get_or_create_company(
                        session, job_posting.company
                    )

                    if not company_id:
                        logger.warning(
                            "Could not create company for: %s", job_posting.company
                        )
                        continue

                    # Create new job record
                    job_data = {
                        "company_id": company_id,
                        "title": job_posting.title,
                        "description": job_posting.description or "",
                        "link": job_posting.job_url or job_posting.job_url_direct or "",
                        "location": job_posting.location or "",
                        "posted_date": job_posting.date_posted,
                        "salary": self._convert_salary(job_posting),
                        "last_seen": datetime.now(UTC),
                    }

                    # Create JobSQL instance with validation
                    job_sql = JobSQL.create_validated(**job_data)
                    session.add(job_sql)

                # Commit all changes
                session.commit()
                logger.info("Successfully saved batch of jobs to database")

        except Exception:
            logger.exception("Failed to save jobs to database")
            # Don't re-raise - let the calling method handle this gracefully

    async def _get_or_create_company(self, session, company_name: str) -> int | None:
        """Get existing company or create new one.

        Args:
            session: Database session.
            company_name: Name of the company.

        Returns:
            Company ID if successful, None otherwise.
        """
        try:
            # Try to find existing company
            existing_company = session.exec(
                select(CompanySQL).filter_by(name=company_name)
            ).first()

            if existing_company:
                return existing_company.id

            # Create new company
            new_company = CompanySQL(
                name=company_name,
                url="",  # We don't have company URL from JobSpy
                active=True,
            )
            session.add(new_company)
            session.flush()  # Get the ID without committing
            return new_company.id  # noqa: TRY300

        except Exception:
            logger.exception("Failed to get or create company: %s", company_name)
            return None

    def _convert_salary(self, job_posting: JobPosting) -> tuple[int | None, int | None]:
        """Convert JobPosting salary to our format.

        Args:
            job_posting: JobPosting with salary information.

        Returns:
            Tuple of (min_salary, max_salary).
        """
        min_salary = None
        max_salary = None

        if job_posting.min_amount:
            min_salary = int(job_posting.min_amount)
        if job_posting.max_amount:
            max_salary = int(job_posting.max_amount)

        return (min_salary, max_salary)

    @staticmethod
    def get_recent_jobs(days: int = 7, limit: int = 100) -> list[Job]:
        """Get recently posted jobs.

        Args:
            days: Number of days back to look.
            limit: Maximum number of jobs to return.

        Returns:
            List of recently posted Job DTOs.
        """
        try:
            with db_session() as session:
                cutoff_date = datetime.now(UTC) - timedelta(days=days)

                query = (
                    select(JobSQL, CompanySQL.name.label("company_name"))
                    .join(CompanySQL, JobSQL.company_id == CompanySQL.id)
                    .filter(JobSQL.posted_date >= cutoff_date)
                    .filter(JobSQL.archived.is_(False))
                    .order_by(JobSQL.posted_date.desc())
                    .limit(limit)
                )

                results = session.exec(query).all()

                jobs = []
                for job_sql, company_name in results:
                    jobs.append(JobService._to_dto_with_company(job_sql, company_name))

                logger.info(
                    "Retrieved %d recent jobs from last %d days", len(jobs), days
                )
                return jobs

        except Exception:
            logger.exception("Failed to get recent jobs")
            return []

    async def refresh_company_jobs(self, company_name: str) -> JobScrapeResult:
        """Refresh jobs for a specific company.

        Args:
            company_name: Name of company to refresh jobs for.

        Returns:
            JobScrapeResult with refreshed jobs.
        """
        try:
            # Search for company-specific jobs
            result = await self.search_and_save_jobs(
                search_term=f"company:{company_name}",
                results_wanted=50,
                save_to_db=True,
            )

            logger.info(
                "Refreshed %d jobs for company: %s", len(result.jobs), company_name
            )
            return result  # noqa: TRY300

        except Exception:
            logger.exception("Failed to refresh company jobs for: %s", company_name)
            raise

    @staticmethod
    def get_jobs_with_company_names_direct_join(
        filters: FilterDict,
    ) -> list[dict[str, object]]:
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
                    CompanySQL,
                    JobSQL.company_id == CompanySQL.id,
                )

                # Apply the same filters as in get_filtered_jobs
                # Note: Text search filtering is handled by search_service.py

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
                    "Retrieved %d jobs with direct JOIN approach",
                    len(jobs_data),
                )
                return jobs_data

        except Exception:
            logger.exception("Failed to get jobs with direct JOIN")
            raise

    @staticmethod
    def invalidate_job_cache(job_id: int | None = None) -> bool:  # noqa: ARG004  # pylint: disable=unused-argument
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

        return True


# Global service instance
job_service = JobService()
