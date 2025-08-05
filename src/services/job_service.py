"""Job service for managing job data operations.

This module provides the JobService class with static methods for querying
and updating job records. It handles database operations for job filtering,
status updates, favorite toggling, and notes management.
"""

import logging

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import joinedload
from sqlmodel import Session, select
from src.database import get_session
from src.models import CompanySQL, JobSQL

logger = logging.getLogger(__name__)


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with proper error handling.

    Provides automatic commit on success, rollback on error, and session cleanup.
    Follows SQLModel 2025 best practices for session management.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class JobService:
    """Service class for job data operations.

    Provides static methods for querying, filtering, and updating job records
    in the database. This service acts as an abstraction layer between the UI
    and the database models.
    """

    @staticmethod
    def get_filtered_jobs(filters: dict[str, Any]) -> list[JobSQL]:
        """Get jobs filtered by the provided criteria.

        Args:
            filters: Dictionary containing filter criteria:
                - text_search: String to search in title and description
                - company: List of company names or "All"
                - application_status: List of status values or "All"
                - date_from: Start date for filtering
                - date_to: End date for filtering
                - favorites_only: Boolean to show only favorites

        Returns:
            List of JobSQL objects matching the filter criteria.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Start with base query, eagerly loading company relationship
                query = select(JobSQL).options(joinedload(JobSQL.company_relation))

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
                    query = query.join(CompanySQL).filter(
                        CompanySQL.name.in_(company_filter)
                    )

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

                # Filter out archived jobs by default
                if not filters.get("include_archived", False):
                    query = query.filter(JobSQL.archived.is_(False))

                # Order by posted date (newest first) by default
                query = query.order_by(JobSQL.posted_date.desc().nullslast())

                jobs = session.exec(query).all()

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
                    job.application_date = datetime.now(datetime.UTC)

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
    def get_job_by_id(job_id: int) -> JobSQL | None:
        """Get a single job by its ID.

        Args:
            job_id: Database ID of the job to retrieve.

        Returns:
            JobSQL object if found, None otherwise.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                job = session.exec(
                    select(JobSQL)
                    .options(joinedload(JobSQL.company_relation))
                    .filter_by(id=job_id)
                ).first()

                if job:
                    logger.info("Retrieved job %s: %s", job_id, job.title)
                else:
                    logger.warning("Job with ID %s not found", job_id)

                return job

        except Exception:
            logger.exception("Failed to get job %s", job_id)
            raise

    @staticmethod
    def get_job_counts_by_status() -> dict[str, int]:
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
                    .group_by(JobSQL.application_status)
                ).all()

                counts = dict(results)
                logger.info("Job counts by status: %s", counts)
                return counts

        except Exception:
            logger.exception("Failed to get job counts")
            raise

    @staticmethod
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
                return datetime.fromisoformat(date_input)
            except ValueError:
                pass

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
                        tzinfo=datetime.UTC
                    )

                except ValueError:
                    continue

            # If all formats fail, log warning
            logger.warning("Could not parse date: %s", date_input)
        elif date_input is not None and not isinstance(date_input, datetime):
            logger.warning("Unsupported date type: %s", type(date_input))

        return None
