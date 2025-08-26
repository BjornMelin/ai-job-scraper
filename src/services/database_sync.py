"""Smart database synchronization service for AI Job Scraper.

This module implements the SmartSyncEngine, a robust service that
synchronizes scraped job data with the database while preserving user data
and preventing data loss. It uses content hashing for change detection and
implements smart archiving rules.
"""

import hashlib
import logging

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, not_
from sqlmodel import Session, select

from src.database import SessionLocal
from src.models import JobSQL

logger = logging.getLogger(__name__)


class SmartSyncEngine:
    """Database synchronization engine for job data.

    This engine provides safe synchronization of scraped job data
    with the database, implementing the following features:

    - Content-based change detection using MD5 hashes
    - Preservation of user-editable data during updates
    - Smart archiving of stale jobs with user data
    - Permanent deletion of jobs without user interaction
    - Error handling and logging
    - Transactional safety with rollback on errors

    The engine follows the database sync requirements DB-SYNC-01 through DB-SYNC-04
    from the project requirements document.
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize the SmartSyncEngine.

        Args:
            session: Optional database session. If not provided, creates new sessions
                    as needed using SessionLocal().
        """
        self._session = session
        self._session_owned = session is None

    def _get_session(self) -> Session:
        """Get or create a database session.

        Returns:
            Session: Database session for operations.
        """
        return self._session if self._session else SessionLocal()

    def _close_session_if_owned(self, session: Session) -> None:
        """Close session if it was created by this engine.

        Args:
            session: Database session to potentially close.
        """
        if self._session_owned and session != self._session:
            session.close()

    def sync_jobs(self, jobs: list[JobSQL]) -> dict[str, int]:
        """Synchronize jobs with the database.

        This method performs the core synchronization logic:
        1. Identifies jobs to insert (new jobs not in database)
        2. Identifies jobs to update (existing jobs with content changes)
        3. Identifies jobs to archive (stale jobs with user data)
        4. Identifies jobs to delete (stale jobs without user data)

        All operations are performed within a single transaction for consistency.

        Args:
            jobs: List of JobSQL objects from scrapers to synchronize.

        Returns:
            dict[str, int]: Statistics about the sync operation containing:
                - 'inserted': Number of new jobs added
                - 'updated': Number of existing jobs updated
                - 'archived': Number of stale jobs archived
                - 'deleted': Number of stale jobs permanently deleted
                - 'skipped': Number of jobs skipped (no changes needed)

        Raises:
            Exception: If database operations fail, the transaction is rolled back
                     and the original exception is re-raised.
        """
        session = self._get_session()
        stats = {"inserted": 0, "updated": 0, "archived": 0, "deleted": 0, "skipped": 0}

        try:
            logger.info("Starting sync of %d jobs", len(jobs))

            # Step 1: Bulk load existing jobs to avoid N+1 query pattern
            current_links = {job.link for job in jobs if job.link}
            if current_links:
                existing_jobs_query = session.exec(
                    select(JobSQL).where(JobSQL.link.in_(current_links)),
                )
                existing_jobs_map = {job.link: job for job in existing_jobs_query}
                logger.debug("Bulk loaded %d existing jobs", len(existing_jobs_map))
            else:
                existing_jobs_map = {}

            # Step 2: Process incoming jobs (insert/update) using bulk-loaded data
            # Track processed links within this batch to prevent duplicates
            processed_links = set()

            for job in jobs:
                if not job.link:
                    logger.warning("Skipping job without link: %s", job.title)
                    continue

                # Check for duplicates within the current batch
                if job.link in processed_links:
                    logger.warning("Skipping duplicate link in batch: %s", job.link)
                    stats["skipped"] += 1
                    continue

                operation = self._sync_single_job_optimized(
                    session,
                    job,
                    existing_jobs_map,
                )
                stats[operation] += 1
                processed_links.add(job.link)

            # Step 3: Handle stale jobs (archive/delete)
            stale_stats = self._handle_stale_jobs(session, current_links)
            stats["archived"] += stale_stats["archived"]
            stats["deleted"] += stale_stats["deleted"]

            # Step 4: Commit all changes
            session.commit()

            logger.info(
                "Sync completed successfully. "
                "Inserted: %d, "
                "Updated: %d, "
                "Archived: %d, "
                "Deleted: %d, "
                "Skipped: %d",
                stats["inserted"],
                stats["updated"],
                stats["archived"],
                stats["deleted"],
                stats["skipped"],
            )
        except Exception:
            logger.exception("Sync failed, rolling back transaction")
            session.rollback()
            raise
        else:
            return stats
        finally:
            self._close_session_if_owned(session)

    def _sync_single_job(self, session: Session, job: JobSQL) -> str:
        """Synchronize a single job with the database.

        Args:
            session: Database session for operations.
            job: JobSQL object to synchronize.

        Returns:
            str: Operation performed ('inserted', 'updated', or 'skipped').
        """
        if existing := session.exec(
            select(JobSQL).where(JobSQL.link == job.link),
        ).first():
            return self._update_existing_job(existing, job)
        return self._insert_new_job(session, job)

    def _sync_single_job_optimized(
        self,
        session: Session,
        job: JobSQL,
        existing_jobs_map: dict[str, JobSQL],
    ) -> str:
        """Synchronize a single job with the database using pre-loaded existing jobs.

        This optimized version uses a pre-loaded map of existing jobs to avoid
        individual database queries for each job, eliminating the N+1 query pattern.

        Args:
            session: Database session for operations.
            job: JobSQL object to synchronize.
            existing_jobs_map: Pre-loaded map of {link: JobSQL} for existing jobs.

        Returns:
            str: Operation performed ('inserted', 'updated', or 'skipped').
        """
        existing = existing_jobs_map.get(job.link)

        if existing:
            return self._update_existing_job(existing, job)
        return self._insert_new_job(session, job)

    def _insert_new_job(self, session: Session, job: JobSQL) -> str:
        """Insert a new job into the database.

        Args:
            session: Database session for operations.
            job: New JobSQL object to insert.

        Returns:
            str: Always returns 'inserted'.
        """
        # Ensure required fields are set
        job.last_seen = datetime.now(UTC)
        if not job.application_status:
            job.application_status = "New"
        if not job.content_hash:
            job.content_hash = self._generate_content_hash(job)

        session.add(job)
        logger.debug("Inserting new job: %s at %s", job.title, job.link)
        return "inserted"

    def _update_existing_job(self, existing: JobSQL, new_job: JobSQL) -> str:
        """Update an existing job while preserving user data.

        This method implements the core user data preservation logic per
        requirement DB-SYNC-03. It only updates scraped fields while keeping
        all user-editable fields intact.

        Args:
            existing: Existing JobSQL object in database.
            new_job: New JobSQL object from scraper.

        Returns:
            str: Operation performed ('updated' or 'skipped').
        """
        new_content_hash = self._generate_content_hash(new_job)

        # Check if content has actually changed
        if existing.content_hash == new_content_hash:
            # Content unchanged, just update last_seen and skip
            existing.last_seen = datetime.now(UTC)
            # Unarchive if it was archived (job is back!)
            if existing.archived:
                existing.archived = False
                logger.info("Unarchiving job that returned: %s", existing.title)
                return "updated"
            logger.debug("Job content unchanged, skipping: %s", existing.title)
            return "skipped"

        # Content changed, update scraped fields while preserving user data
        self._update_scraped_fields(existing, new_job, new_content_hash)
        logger.debug("Updating job with content changes: %s", existing.title)
        return "updated"

    def _update_scraped_fields(
        self,
        existing: JobSQL,
        new_job: JobSQL,
        new_content_hash: str,
    ) -> None:
        """Update only scraped fields, preserving user-editable fields.

        This method carefully updates only the fields that come from scraping
        while preserving all user-editable fields per DB-SYNC-03.

        Args:
            existing: Existing JobSQL object to update.
            new_job: New JobSQL object with updated data.
            new_content_hash: Pre-computed content hash for the new job.
        """
        # Update scraped fields
        existing.title = new_job.title
        existing.company_id = new_job.company_id
        existing.description = new_job.description
        existing.location = new_job.location
        existing.posted_date = new_job.posted_date
        existing.salary = new_job.salary
        existing.content_hash = new_content_hash
        existing.last_seen = datetime.now(UTC)

        # Unarchive if it was archived (job is back!)
        if existing.archived:
            existing.archived = False
            logger.info("Unarchiving job that returned: %s", existing.title)

        # PRESERVE user-editable fields (do not modify):
        # - existing.favorite
        # - existing.notes
        # - existing.application_status
        # - existing.application_date

    def _handle_stale_jobs(
        self,
        session: Session,
        current_links: set[str],
    ) -> dict[str, int]:
        """Handle jobs that are no longer present in current scrape.

        This method implements the smart archiving logic per DB-SYNC-04:
        - Jobs with user data (favorites, notes, app status != "New") are archived
        - Jobs without user data are permanently deleted

        Args:
            session: Database session for operations.
            current_links: Set of job links from current scrape.

        Returns:
            dict[str, int]: Statistics with 'archived' and 'deleted' counts.
        """
        stats = {"archived": 0, "deleted": 0}

        # Find all non-archived jobs not in current scrape
        if current_links:
            # Normal case: exclude jobs with links in current_links
            stale_jobs = session.exec(
                select(JobSQL).where(
                    not_(JobSQL.archived),
                    not_(JobSQL.link.in_(current_links)),
                ),
            ).all()
        else:
            # Edge case: when current_links is empty, all non-archived jobs are stale
            stale_jobs = session.exec(select(JobSQL).where(not_(JobSQL.archived))).all()

        for job in stale_jobs:
            if self._has_user_data(job):
                # Archive jobs with user interaction
                job.archived = True
                stats["archived"] += 1
                logger.debug("Archiving job with user data: %s", job.title)
            else:
                # Delete jobs without user interaction
                session.delete(job)
                stats["deleted"] += 1
                logger.debug("Deleting job without user data: %s", job.title)

        return stats

    def _has_user_data(self, job: JobSQL) -> bool:
        """Check if a job has user-entered data that should be preserved.

        Args:
            job: JobSQL object to check.

        Returns:
            bool: True if job has user data, False otherwise.
        """
        return (
            job.favorite
            or (job.notes or "").strip() != ""
            or job.application_status != "New"
        )

    def _generate_content_hash(self, job: JobSQL) -> str:
        """Generate MD5 hash of job content for change detection.

        The hash includes all relevant scraped fields to detect meaningful changes
        in job content per DB-SYNC-02. This ensures updates are triggered when
        any significant job detail changes.

        Args:
            job: JobSQL object to hash.

        Returns:
            str: MD5 hash of job content.
        """
        # Always use company_id for consistent hashing regardless of
        # relationship loading
        company_identifier = str(job.company_id) if job.company_id else "unknown"

        # Include all relevant scraped fields for change detection
        content_parts = [
            job.title or "",
            job.description or "",
            job.location or "",
            company_identifier,
        ]

        # Handle salary field (tuple or list format)
        if hasattr(job, "salary") and job.salary:
            if isinstance(job.salary, tuple | list) and len(job.salary) >= 2:
                salary_str = f"{job.salary[0] or ''}-{job.salary[1] or ''}"
            else:
                salary_str = str(job.salary)
            content_parts.append(salary_str)

        # Handle posted_date if available (normalize timezone for consistent hashing)
        if job.posted_date:
            # Convert to naive datetime to ensure consistent hash regardless of timezone
            naive_date = (
                job.posted_date.replace(tzinfo=None)
                if job.posted_date.tzinfo
                else job.posted_date
            )
            content_parts.append(naive_date.isoformat())

        content = "".join(content_parts)
        # MD5 is safe for non-cryptographic content fingerprinting/change detection
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get_sync_statistics(self) -> dict[str, int]:
        """Get current database statistics for monitoring.

        Returns:
            dict[str, int]: Database statistics including:
                - 'total_jobs': Total number of jobs (including archived)
                - 'active_jobs': Number of non-archived jobs
                - 'archived_jobs': Number of archived jobs
                - 'favorited_jobs': Number of favorited jobs
                - 'applied_jobs': Number of jobs with applications submitted
        """
        session = self._get_session()
        try:
            # Get basic counts using efficient count queries
            total_jobs = session.exec(select(func.count(JobSQL.id))).scalar()
            active_jobs = session.exec(
                select(func.count(JobSQL.id)).where(not_(JobSQL.archived)),
            ).scalar()
            archived_jobs = session.exec(
                select(func.count(JobSQL.id)).where(JobSQL.archived),
            ).scalar()
            favorited_jobs = session.exec(
                select(func.count(JobSQL.id)).where(JobSQL.favorite),
            ).scalar()

            # Count applied jobs (status != "New")
            applied_jobs = session.exec(
                select(func.count(JobSQL.id)).where(JobSQL.application_status != "New"),
            ).scalar()

            return {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "archived_jobs": archived_jobs,
                "favorited_jobs": favorited_jobs,
                "applied_jobs": applied_jobs,
            }
        finally:
            self._close_session_if_owned(session)

    def cleanup_old_jobs(self, days_threshold: int = 90) -> int:
        """Clean up very old jobs that have been archived for a long time.

        This method provides a way to eventually clean up jobs that have been
        archived for an extended period, helping manage database size.

        Args:
            days_threshold: Number of days after which archived jobs without
                          recent user interaction can be deleted.

        Returns:
            int: Number of jobs deleted.

        Note:
            This method should be used carefully and typically run as a
            scheduled maintenance task, not during regular sync operations.
        """
        session = self._get_session()
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=days_threshold)

            # Find archived jobs that haven't been seen in a long time
            # and don't have recent application activity
            old_jobs = session.exec(
                select(JobSQL).where(
                    JobSQL.archived,
                    JobSQL.last_seen < cutoff_date,
                    (JobSQL.application_date.is_(None))
                    | (JobSQL.application_date < cutoff_date),
                ),
            ).all()

            count = 0
            for job in old_jobs:
                session.delete(job)
                count += 1

            session.commit()
            logger.info("Cleaned up %d old archived jobs", count)
        except Exception:
            logger.exception("Cleanup failed")
            session.rollback()
            raise
        else:
            return count
        finally:
            self._close_session_if_owned(session)
