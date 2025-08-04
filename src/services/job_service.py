"""Job service for managing job data operations.

This module provides the JobService class with static methods for querying
and updating job records. It handles database operations for job filtering,
status updates, favorite toggling, and notes management.
"""

import logging

from datetime import datetime
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from src.database import get_session
from src.models import CompanySQL, JobSQL

logger = logging.getLogger(__name__)


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
        session = get_session()

        try:
            # Start with base query, eagerly loading company relationship
            query = session.query(JobSQL).options(joinedload(JobSQL.company_relation))

            # Apply text search filter
            text_search = filters.get("text_search", "").strip()
            if text_search:
                search_pattern = f"%{text_search}%"
                query = query.filter(
                    or_(
                        JobSQL.title.ilike(search_pattern),
                        JobSQL.description.ilike(search_pattern),
                    )
                )

            # Apply company filter
            company_filter = filters.get("company", [])
            if company_filter and "All" not in company_filter:
                # Get company IDs for the selected companies
                company_ids = (
                    session.query(CompanySQL.id)
                    .filter(CompanySQL.name.in_(company_filter))
                    .subquery()
                )
                query = query.filter(JobSQL.company_id.in_(company_ids))

            # Apply application status filter
            status_filter = filters.get("application_status", [])
            if status_filter and "All" not in status_filter:
                query = query.filter(JobSQL.application_status.in_(status_filter))

            # Apply date filters
            date_from = filters.get("date_from")
            if date_from:
                if isinstance(date_from, str):
                    date_from = datetime.fromisoformat(date_from)
                query = query.filter(JobSQL.posted_date >= date_from)

            date_to = filters.get("date_to")
            if date_to:
                if isinstance(date_to, str):
                    date_to = datetime.fromisoformat(date_to)
                query = query.filter(JobSQL.posted_date <= date_to)

            # Apply favorites filter
            if filters.get("favorites_only", False):
                query = query.filter(JobSQL.favorite.is_(True))

            # Filter out archived jobs by default
            if not filters.get("include_archived", False):
                query = query.filter(JobSQL.archived.is_(False))

            # Order by posted date (newest first) by default
            query = query.order_by(JobSQL.posted_date.desc().nullslast())

            jobs = query.all()

            logger.info(f"Retrieved {len(jobs)} jobs with filters: {filters}")
            return jobs

        except Exception as e:
            logger.error(f"Failed to get filtered jobs: {e}")
            raise
        finally:
            session.close()

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
        session = get_session()

        try:
            job = session.query(JobSQL).filter_by(id=job_id).first()
            if not job:
                logger.warning(f"Job with ID {job_id} not found")
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
                job.application_date = datetime.now()

            session.commit()

            logger.info(
                f"Updated job {job_id} status from '{old_status}' to '{status}'"
            )
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update job status for job {job_id}: {e}")
            raise
        finally:
            session.close()

    @staticmethod
    def toggle_favorite(job_id: int) -> bool:
        """Toggle the favorite status of a job.

        Args:
            job_id: Database ID of the job to toggle.

        Returns:
            New favorite status (True/False) if successful, None if failed.

        Raises:
            Exception: If database update fails.
        """
        session = get_session()

        try:
            job = session.query(JobSQL).filter_by(id=job_id).first()
            if not job:
                logger.warning(f"Job with ID {job_id} not found")
                return False

            job.favorite = not job.favorite
            session.commit()

            logger.info(f"Toggled favorite for job {job_id} to {job.favorite}")
            return job.favorite

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to toggle favorite for job {job_id}: {e}")
            raise
        finally:
            session.close()

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
        session = get_session()

        try:
            job = session.query(JobSQL).filter_by(id=job_id).first()
            if not job:
                logger.warning(f"Job with ID {job_id} not found")
                return False

            job.notes = notes
            session.commit()

            logger.info(f"Updated notes for job {job_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update notes for job {job_id}: {e}")
            raise
        finally:
            session.close()

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
        session = get_session()

        try:
            job = (
                session.query(JobSQL)
                .options(joinedload(JobSQL.company_relation))
                .filter_by(id=job_id)
                .first()
            )

            if job:
                logger.info(f"Retrieved job {job_id}: {job.title}")
            else:
                logger.warning(f"Job with ID {job_id} not found")

            return job

        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise
        finally:
            session.close()

    @staticmethod
    def get_job_counts_by_status() -> dict[str, int]:
        """Get count of jobs grouped by application status.

        Returns:
            Dictionary mapping status names to counts.

        Raises:
            Exception: If database query fails.
        """
        session = get_session()

        try:
            from sqlalchemy import func

            results = (
                session.query(JobSQL.application_status, func.count(JobSQL.id))
                .filter(JobSQL.archived.is_(False))
                .group_by(JobSQL.application_status)
                .all()
            )

            counts = dict(results)
            logger.info(f"Job counts by status: {counts}")
            return counts

        except Exception as e:
            logger.error(f"Failed to get job counts: {e}")
            raise
        finally:
            session.close()
