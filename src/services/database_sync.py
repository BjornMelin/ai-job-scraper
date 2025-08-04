"""Smart database synchronization service for AI Job Scraper.

This module implements the SmartSyncEngine, a robust service that intelligently
synchronizes scraped job data with the database while preserving user data
and preventing data loss. It uses content hashing for change detection and
implements smart archiving rules.
"""

import hashlib
import logging

from datetime import datetime, timedelta

from sqlmodel import Session, select

from ..database import SessionLocal
from ..models import JobSQL

logger = logging.getLogger(__name__)


class SmartSyncEngine:
    """Intelligent database synchronization engine for job data.

    This engine provides safe, intelligent synchronization of scraped job data
    with the database, implementing the following features:

    - Content-based change detection using MD5 hashes
    - Preservation of user-editable data during updates
    - Smart archiving of stale jobs with user data
    - Permanent deletion of jobs without user interaction
    - Comprehensive error handling and logging
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
        if self._session:
            return self._session
        return SessionLocal()

    def _close_session_if_owned(self, session: Session) -> None:
        """Close session if it was created by this engine.

        Args:
            session: Database session to potentially close.
        """
        if self._session_owned and session != self._session:
            session.close()

    def sync_jobs(self, jobs: list[JobSQL]) -> dict[str, int]:
        """Synchronize jobs with the database intelligently.

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
            logger.info(f"Starting sync of {len(jobs)} jobs")

            # Step 1: Process incoming jobs (insert/update)
            current_links = {job.link for job in jobs if job.link}
            for job in jobs:
                if not job.link:
                    logger.warning(f"Skipping job without link: {job.title}")
                    continue

                operation = self._sync_single_job(session, job)
                stats[operation] += 1

            # Step 2: Handle stale jobs (archive/delete)
            stale_stats = self._handle_stale_jobs(session, current_links)
            stats["archived"] += stale_stats["archived"]
            stats["deleted"] += stale_stats["deleted"]

            # Step 3: Commit all changes
            session.commit()

            logger.info(
                f"Sync completed successfully. "
                f"Inserted: {stats['inserted']}, "
                f"Updated: {stats['updated']}, "
                f"Archived: {stats['archived']}, "
                f"Deleted: {stats['deleted']}, "
                f"Skipped: {stats['skipped']}"
            )

            return stats

        except Exception as e:
            logger.error(f"Sync failed, rolling back transaction: {e}")
            session.rollback()
            raise
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
        existing = session.exec(select(JobSQL).where(JobSQL.link == job.link)).first()

        if existing:
            return self._update_existing_job(session, existing, job)
        else:
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
        job.last_seen = datetime.now()
        if not job.application_status:
            job.application_status = "New"
        if not job.content_hash:
            job.content_hash = self._generate_content_hash(job)

        session.add(job)
        logger.debug(f"Inserting new job: {job.title} at {job.link}")
        return "inserted"

    def _update_existing_job(
        self, session: Session, existing: JobSQL, new_job: JobSQL
    ) -> str:
        """Update an existing job while preserving user data.

        This method implements the core user data preservation logic per
        requirement DB-SYNC-03. It only updates scraped fields while keeping
        all user-editable fields intact.

        Args:
            session: Database session for operations.
            existing: Existing JobSQL object in database.
            new_job: New JobSQL object from scraper.

        Returns:
            str: Operation performed ('updated' or 'skipped').
        """
        new_content_hash = self._generate_content_hash(new_job)

        # Check if content has actually changed
        if existing.content_hash == new_content_hash:
            # Content unchanged, just update last_seen and skip
            existing.last_seen = datetime.now()
            # Unarchive if it was archived (job is back!)
            if existing.archived:
                existing.archived = False
                logger.info(f"Unarchiving job that returned: {existing.title}")
                return "updated"
            return "skipped"

        # Content changed, update scraped fields while preserving user data
        self._update_scraped_fields(existing, new_job, new_content_hash)
        logger.debug(f"Updating job with content changes: {existing.title}")
        return "updated"

    def _update_scraped_fields(
        self, existing: JobSQL, new_job: JobSQL, new_content_hash: str
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
        existing.last_seen = datetime.now()

        # Unarchive if it was archived (job is back!)
        if existing.archived:
            existing.archived = False
            logger.info(f"Unarchiving job that returned: {existing.title}")

        # PRESERVE user-editable fields (do not modify):
        # - existing.favorite
        # - existing.notes
        # - existing.application_status
        # - existing.application_date

    def _handle_stale_jobs(
        self, session: Session, current_links: set[str]
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
        stale_jobs = session.exec(
            select(JobSQL).where(
                JobSQL.archived == False,  # noqa: E712 (SQLModel requires == False)
                ~JobSQL.link.in_(current_links),
            )
        ).all()

        for job in stale_jobs:
            if self._has_user_data(job):
                # Archive jobs with user interaction
                job.archived = True
                stats["archived"] += 1
                logger.debug(f"Archiving job with user data: {job.title}")
            else:
                # Delete jobs without user interaction
                session.delete(job)
                stats["deleted"] += 1
                logger.debug(f"Deleting job without user data: {job.title}")

        return stats

    def _has_user_data(self, job: JobSQL) -> bool:
        """Check if a job has user-entered data that should be preserved.

        Args:
            job: JobSQL object to check.

        Returns:
            bool: True if job has user data, False otherwise.
        """
        return (
            job.favorite or job.notes.strip() != "" or job.application_status != "New"
        )

    def _generate_content_hash(self, job: JobSQL) -> str:
        """Generate MD5 hash of job content for change detection.

        The hash includes title, description, and company information to detect
        meaningful changes in job content per DB-SYNC-02.

        Args:
            job: JobSQL object to hash.

        Returns:
            str: MD5 hash of job content.
        """
        # Use company_id if available, otherwise fallback to extracting company name
        company_identifier = str(job.company_id) if job.company_id else "unknown"
        content = f"{job.title}{job.description}{company_identifier}"
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
            # Get basic counts
            total_jobs = len(session.exec(select(JobSQL)).all())
            active_jobs = len(
                session.exec(select(JobSQL).where(not JobSQL.archived)).all()
            )  # noqa: E712
            archived_jobs = len(
                session.exec(select(JobSQL).where(JobSQL.archived)).all()
            )  # noqa: E712
            favorited_jobs = len(
                session.exec(select(JobSQL).where(JobSQL.favorite)).all()
            )  # noqa: E712

            # Count applied jobs (status != "New")
            applied_jobs = len(
                session.exec(
                    select(JobSQL).where(JobSQL.application_status != "New")
                ).all()
            )

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
            cutoff_date = datetime.now() - timedelta(days=days_threshold)

            # Find archived jobs that haven't been seen in a long time
            # and don't have recent application activity
            old_jobs = session.exec(
                select(JobSQL).where(
                    JobSQL.archived == True,  # noqa: E712
                    JobSQL.last_seen < cutoff_date,
                    (JobSQL.application_date == None)  # noqa: E711
                    | (JobSQL.application_date < cutoff_date),
                )
            ).all()

            count = 0
            for job in old_jobs:
                session.delete(job)
                count += 1

            session.commit()
            logger.info(f"Cleaned up {count} old archived jobs")
            return count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            session.rollback()
            raise
        finally:
            self._close_session_if_owned(session)
