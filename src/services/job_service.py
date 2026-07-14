"""Job service for managing job data operations.

This module provides the JobService class with static methods for querying
and updating job records. It handles database operations for job filtering,
status updates, favorite toggling, and notes management.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import exc as sqlalchemy
from sqlalchemy import func
from sqlmodel import Session, select

from src.constants import SALARY_DEFAULT_MIN, SALARY_UNBOUNDED_THRESHOLD
from src.database import db_session
from src.database_models import CompanySQL, JobSQL
from src.models.job_models import (
    ApplicationStage,
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
)
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
        return JobService._to_dto_with_company(job_sql, "Unknown")

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
        job_data = {
            field: getattr(job_sql, field)
            for field in JobSQL.model_fields
            if field != "salary"
        }
        job_data["salary"] = tuple(job_sql.salary or (None, None))
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

        # Text search is owned by search_service.py; this method owns shared facets.

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
        return query.order_by(
            JobSQL.posted_date.desc().nullslast(),
            JobSQL.id.desc(),
        )

    @staticmethod
    def get_filtered_jobs(
        filters: FilterDict | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Job]:
        """Get jobs filtered by the provided criteria.

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
            if limit is not None and limit < 1:
                raise ValueError("limit must be positive")
            if offset < 0:
                raise ValueError("offset must be nonnegative")

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
                if offset:
                    query = query.offset(offset)
                if limit is not None:
                    query = query.limit(limit)

                results = session.exec(query).all()

                # Convert SQLModel objects to Pydantic DTOs with company names
                jobs = []
                for job_sql, company_name in results:
                    jobs.append(JobService._to_dto_with_company(job_sql, company_name))

                logger.info("Retrieved %d jobs with filters: %s", len(jobs), filters)
                return jobs

        except (sqlalchemy.DatabaseError, sqlalchemy.SQLAlchemyError) as e:
            logger.exception("Database error getting filtered jobs: %s", e)
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Data processing error getting filtered jobs: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error getting filtered jobs: %s", e)
            raise

    @staticmethod
    def count_filtered_jobs(filters: FilterDict | None = None) -> int:
        """Count jobs matching the canonical filter contract."""
        with db_session() as session:
            query = select(func.count(JobSQL.id)).join(
                CompanySQL,
                JobSQL.company_id == CompanySQL.id,
            )
            query = JobService._apply_filters_to_query(query, filters or {}).order_by(
                None
            )
            return session.exec(query).one()

    @staticmethod
    def update_job_status(job_id: int, status: ApplicationStage) -> bool:
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
                    status == ApplicationStage.APPLIED
                    and old_status != ApplicationStage.APPLIED
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

        except (sqlalchemy.DatabaseError, sqlalchemy.NoResultFound) as e:
            logger.exception(
                "Database error updating job status for job %s: %s", job_id, e
            )
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "Data validation error updating job status for job %s: %s",
                job_id,
                e,
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error updating job status for job %s: %s",
                job_id,
                e,
            )
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

        except (sqlalchemy.DatabaseError, sqlalchemy.NoResultFound) as e:
            logger.exception(
                "Database error toggling favorite for job %s: %s", job_id, e
            )
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "Data validation error toggling favorite for job %s: %s",
                job_id,
                e,
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error toggling favorite for job %s: %s",
                job_id,
                e,
            )
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

        except (sqlalchemy.DatabaseError, sqlalchemy.NoResultFound) as e:
            logger.exception("Database error updating notes for job %s: %s", job_id, e)
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "Data validation error updating notes for job %s: %s",
                job_id,
                e,
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error updating notes for job %s: %s", job_id, e
            )
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

        except (sqlalchemy.DatabaseError, sqlalchemy.SQLAlchemyError) as e:
            logger.exception("Database error getting job %s: %s", job_id, e)
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Data processing error getting job %s: %s", job_id, e)
            raise
        except Exception as e:
            logger.exception("Unexpected error getting job %s: %s", job_id, e)
            raise

    @staticmethod
    def get_job_counts_by_status() -> JobCountStats:
        """Get count of jobs grouped by application status.

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

        except (sqlalchemy.DatabaseError, sqlalchemy.SQLAlchemyError) as e:
            logger.exception("Database error getting job counts: %s", e)
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Data processing error getting job counts: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error getting job counts: %s", e)
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

        except (sqlalchemy.DatabaseError, sqlalchemy.NoResultFound) as e:
            logger.exception("Database error archiving job %s: %s", job_id, e)
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Data validation error archiving job %s: %s", job_id, e)
            raise
        except Exception as e:
            logger.exception("Unexpected error archiving job %s: %s", job_id, e)
            raise

    @staticmethod
    def _parse_date(date_input: str | date | datetime | None) -> datetime | None:
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
        if isinstance(date_input, datetime):
            return date_input if date_input.tzinfo else date_input.replace(tzinfo=UTC)
        if isinstance(date_input, date):
            return datetime.combine(date_input, time.min, tzinfo=UTC)
        if isinstance(date_input, str):
            date_input = date_input.strip()
            if not date_input:
                return None

            # Try ISO format first (most common for APIs)
            try:
                dt = datetime.fromisoformat(date_input)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
            except ValueError:
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

                except ValueError:
                    # Expected: Try next date format if this one fails
                    continue

            # If all formats fail, log warning
            logger.warning("Could not parse date: %s", date_input)
        elif date_input is not None:
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
                            update.get("application_status") == ApplicationStage.APPLIED
                            and job.application_status == ApplicationStage.APPLIED
                            and not job.application_date
                        ):
                            job.application_date = datetime.now(UTC)

                logger.info("Bulk updated %d jobs", len(job_updates))
                return True

        except (sqlalchemy.DatabaseError, sqlalchemy.IntegrityError) as e:
            logger.exception("Database error bulk updating jobs: %s", e)
            raise
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.warning("Data validation error bulk updating jobs: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error bulk updating jobs: %s", e)
            raise

    async def search_and_save_jobs(
        self,
        search_term: str,
        location: str | None = None,
        sites: Sequence[str | JobSite] | None = None,
        is_remote: bool = False,
        job_type: JobType | None = None,
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
            site_enums: list[JobSite] = []
            if sites:
                for site in sites:
                    site_enum = (
                        site if isinstance(site, JobSite) else JobSite.normalize(site)
                    )
                    if site_enum:
                        site_enums.append(site_enum)
                    else:
                        logger.warning("Unknown job site: %s", site)

            if not site_enums:
                site_enums = [JobSite.LINKEDIN]  # Default to LinkedIn

            # Create scrape request
            request = JobScrapeRequest(
                site_name=site_enums,
                search_term=search_term,
                location=location,
                is_remote=is_remote,
                job_type=job_type,
                results_wanted=results_wanted,
                linkedin_fetch_description=True,
            )

            # Execute scraping
            result = await self.scraper.scrape_jobs_async(request)

            persistence = {"inserted": 0, "updated": 0, "skipped": 0}
            if save_to_db:
                persistence = self._save_jobs_to_database(result.jobs)
                logger.info("Job persistence complete: %s", persistence)
            result.metadata["persistence"] = persistence
            return result

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Data processing error searching and saving jobs: %s", e)
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.exception("Network error searching and saving jobs: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error searching and saving jobs: %s", e)
            raise

    def _save_jobs_to_database(
        self,
        jobs: list[JobPosting],
        *,
        session: Session | None = None,
    ) -> dict[str, int]:
        """Persist jobs alone or within a caller-owned transaction."""
        if session is not None:
            return self._persist_jobs(session, jobs)
        with db_session() as owned_session:
            return self._persist_jobs(owned_session, jobs)

    def _persist_jobs(
        self,
        session: Session,
        jobs: list[JobPosting],
    ) -> dict[str, int]:
        """Merge one provider result without erasing richer stored fields."""
        stats = {"inserted": 0, "updated": 0, "skipped": 0}
        for posting in jobs:
            link = posting.job_url_direct or posting.job_url
            if not link or not posting.company:
                stats["skipped"] += 1
                continue

            company_id = self._get_or_create_company(
                session,
                posting.company,
                posting.company_url_direct or posting.company_url,
            )
            now = datetime.now(UTC)
            existing = session.exec(
                select(JobSQL).where(JobSQL.link == link),
            ).first()
            incoming_salary = self._convert_salary(posting)
            if existing is None:
                candidate = JobSQL.create_validated(
                    company_id=company_id,
                    title=posting.title,
                    description=posting.description or "",
                    link=link,
                    location=posting.location or "",
                    posted_date=self._parse_date(posting.date_posted),
                    salary=incoming_salary,
                    last_seen=now,
                )
                session.add(candidate)
                stats["inserted"] += 1
                continue

            current_salary = tuple(existing.salary or (None, None))
            merged_salary = tuple(
                incoming if incoming is not None else current
                for incoming, current in zip(
                    incoming_salary, current_salary, strict=True
                )
            )
            candidate = JobSQL.create_validated(
                company_id=company_id,
                title=posting.title,
                description=(posting.description or "").strip() or existing.description,
                link=link,
                location=(posting.location or "").strip() or existing.location,
                posted_date=self._parse_date(posting.date_posted)
                or self._parse_date(existing.posted_date),
                salary=merged_salary,
                last_seen=now,
            )
            changed = False
            for field in (
                "company_id",
                "title",
                "description",
                "location",
                "posted_date",
                "salary",
                "content_hash",
            ):
                incoming = getattr(candidate, field)
                current = getattr(existing, field)
                if field == "salary":
                    current = tuple(current or (None, None))
                elif field == "posted_date":
                    current = self._parse_date(current)
                    incoming = self._parse_date(incoming)
                if current != incoming:
                    setattr(existing, field, incoming)
                    changed = True
            existing.last_seen = now
            stats["updated" if changed else "skipped"] += 1
        return stats

    @staticmethod
    def _get_or_create_company(
        session,
        company_name: str,
        company_url: str | None,
    ) -> int:
        """Resolve the company facet owned by a persisted job."""
        company = session.exec(
            select(CompanySQL).where(CompanySQL.name == company_name),
        ).first()
        if company is None:
            company = CompanySQL(name=company_name, url=company_url)
            session.add(company)
            session.flush()
        elif not company.url and company_url:
            company.url = company_url
        if company.id is None:
            raise RuntimeError("Company ID was not assigned")
        return company.id

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

        except (sqlalchemy.DatabaseError, sqlalchemy.SQLAlchemyError) as e:
            logger.exception("Database error getting recent jobs: %s", e)
            return []
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Data processing error getting recent jobs: %s", e)
            return []
        except Exception as e:
            logger.exception("Unexpected error getting recent jobs: %s", e)
            return []


# Global service instance
job_service = JobService()
