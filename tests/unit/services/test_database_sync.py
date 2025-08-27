"""Comprehensive tests for SmartSyncEngine database synchronization service.

This test suite validates the SmartSyncEngine for real-world synchronization scenarios,
focusing on content-based change detection, user data preservation, and smart archiving.
Tests cover sync operations, stale job handling, content hashing, and error conditions.
"""

import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select
from tests.factories import CompanyFactory, JobFactory

from src.models import JobSQL
from src.services.database_sync import SmartSyncEngine

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)


@pytest.fixture
def test_engine():
    """Create a test-specific SQLite engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def sample_companies(test_session):
    """Create sample companies for testing."""
    CompanyFactory._meta.sqlalchemy_session = test_session
    return CompanyFactory.create_batch(2)


@pytest.fixture
def sync_engine(test_session):
    """Create SmartSyncEngine with test session."""
    return SmartSyncEngine(session=test_session)


@pytest.fixture
def sync_engine_no_session():
    """Create SmartSyncEngine without session (will create its own)."""
    with patch("src.services.database_sync.SessionLocal") as mock_session_factory:
        mock_session = Mock()
        mock_session_factory.return_value = mock_session
        yield SmartSyncEngine(), mock_session


class TestSmartSyncEngineInitialization:
    """Test SmartSyncEngine initialization and session management."""

    def test_init_with_session(self, test_session):
        """Test initialization with provided session."""
        engine = SmartSyncEngine(session=test_session)

        assert engine._session is test_session
        assert engine._session_owned is False

    def test_init_without_session(self):
        """Test initialization without session (creates own)."""
        engine = SmartSyncEngine()

        assert engine._session is None
        assert engine._session_owned is True

    def test_get_session_with_provided_session(self, test_session):
        """Test _get_session with provided session."""
        engine = SmartSyncEngine(session=test_session)

        session = engine._get_session()

        assert session is test_session

    def test_get_session_without_provided_session(self, sync_engine_no_session):
        """Test _get_session without provided session."""
        engine, mock_session = sync_engine_no_session

        with patch("src.services.database_sync.SessionLocal") as mock_session_local:
            mock_session_local.return_value = mock_session
            session = engine._get_session()

            assert session is mock_session


class TestSmartSyncEngineContentHashing:
    """Test content hashing functionality."""

    def test_generate_content_hash_basic(self, sync_engine, sample_companies):
        """Test basic content hash generation."""
        job = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job opportunity",
            link="https://company.com/job/1",
            location="San Francisco, CA",
        )

        hash_value = sync_engine._generate_content_hash(job)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # MD5 hash length
        assert hash_value.isalnum()

    def test_generate_content_hash_consistency(self, sync_engine, sample_companies):
        """Test that same content generates same hash."""
        job1 = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job opportunity",
            link="https://company.com/job/1",
            location="San Francisco, CA",
        )

        job2 = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job opportunity",
            # Different link, but content should hash same
            link="https://company.com/job/2",
            location="San Francisco, CA",
        )

        hash1 = sync_engine._generate_content_hash(job1)
        hash2 = sync_engine._generate_content_hash(job2)

        # Should be same because link isn't part of content hash
        assert hash1 == hash2

    def test_generate_content_hash_different_content(
        self, sync_engine, sample_companies
    ):
        """Test that different content generates different hash."""
        job1 = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job opportunity",
            location="San Francisco, CA",
        )

        job2 = JobSQL(
            company_id=sample_companies[0].id,
            title="Senior Software Engineer",  # Different title
            description="Great job opportunity",
            location="San Francisco, CA",
        )

        hash1 = sync_engine._generate_content_hash(job1)
        hash2 = sync_engine._generate_content_hash(job2)

        assert hash1 != hash2

    def test_generate_content_hash_with_salary(self, sync_engine, sample_companies):
        """Test content hash includes salary information."""
        job_with_salary = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job",
            location="SF",
            salary=(100000, 150000),
        )

        job_without_salary = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job",
            location="SF",
        )

        hash_with = sync_engine._generate_content_hash(job_with_salary)
        hash_without = sync_engine._generate_content_hash(job_without_salary)

        assert hash_with != hash_without

    def test_generate_content_hash_with_posted_date(
        self, sync_engine, sample_companies
    ):
        """Test content hash includes posted date."""
        date1 = datetime(2024, 1, 15, tzinfo=UTC)
        date2 = datetime(2024, 1, 16, tzinfo=UTC)

        job1 = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job",
            location="SF",
            posted_date=date1,
        )

        job2 = JobSQL(
            company_id=sample_companies[0].id,
            title="Software Engineer",
            description="Great job",
            location="SF",
            posted_date=date2,
        )

        hash1 = sync_engine._generate_content_hash(job1)
        hash2 = sync_engine._generate_content_hash(job2)

        assert hash1 != hash2


class TestSmartSyncEngineJobInserts:
    """Test job insertion functionality."""

    def test_insert_new_job_basic(self, sync_engine, test_session, sample_companies):
        """Test inserting a new job."""
        job = JobSQL(
            company_id=sample_companies[0].id,
            title="New Engineer",
            description="Exciting opportunity",
            link="https://company.com/new-job",
            location="Remote",
        )

        result = sync_engine._insert_new_job(test_session, job)

        assert result == "inserted"
        assert job.application_status == "New"
        assert job.last_seen is not None
        assert job.content_hash is not None
        assert len(job.content_hash) == 32

    def test_insert_new_job_preserves_status(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that existing application status is preserved."""
        job = JobSQL(
            company_id=sample_companies[0].id,
            title="New Engineer",
            description="Exciting opportunity",
            link="https://company.com/new-job",
            location="Remote",
            application_status="Applied",  # Pre-set status
        )

        result = sync_engine._insert_new_job(test_session, job)

        assert result == "inserted"
        assert job.application_status == "Applied"  # Should preserve existing

    def test_insert_new_job_sets_content_hash(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that content hash is generated if not provided."""
        job = JobSQL(
            company_id=sample_companies[0].id,
            title="New Engineer",
            description="Exciting opportunity",
            link="https://company.com/new-job",
            location="Remote",
        )

        sync_engine._insert_new_job(test_session, job)

        assert job.content_hash is not None
        assert len(job.content_hash) == 32


class TestSmartSyncEngineJobUpdates:
    """Test job update functionality and user data preservation."""

    def test_update_existing_job_no_changes(
        self, sync_engine, test_session, sample_companies
    ):
        """Test updating job with no content changes (skip)."""
        # Create existing job
        existing_job = JobFactory.create(
            company_id=sample_companies[0].id,
            title="Existing Engineer",
            content_hash="original_hash",
            session=test_session,
        )

        # Create new job with same content (will generate same hash)
        new_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Existing Engineer",  # Same content
            description=existing_job.description,
            location=existing_job.location,
        )
        new_job.content_hash = "original_hash"  # Same hash

        result = sync_engine._update_existing_job(existing_job, new_job)

        assert result == "skipped"
        assert existing_job.last_seen is not None  # Should update last_seen

    def test_update_existing_job_with_changes(
        self, sync_engine, test_session, sample_companies
    ):
        """Test updating job with content changes."""
        # Create existing job
        existing_job = JobFactory.create(
            company_id=sample_companies[0].id,
            title="Old Title",
            content_hash="old_hash",
            favorite=True,  # User data to preserve
            notes="My notes",  # User data to preserve
            session=test_session,
        )

        # Create new job with different content
        new_job = JobSQL(
            company_id=sample_companies[0].id,
            title="New Title",  # Changed
            description="New description",  # Changed
            location=existing_job.location,
        )

        result = sync_engine._update_existing_job(existing_job, new_job)

        assert result == "updated"
        assert existing_job.title == "New Title"  # Should update scraped field
        assert (
            existing_job.description == "New description"
        )  # Should update scraped field
        assert existing_job.favorite is True  # Should preserve user data
        assert existing_job.notes == "My notes"  # Should preserve user data

    def test_update_existing_job_preserves_user_data(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that user-editable fields are preserved during update."""
        base_date = datetime.now(UTC)

        # Create existing job with user data
        existing_job = JobFactory.create(
            company_id=sample_companies[0].id,
            favorite=True,
            notes="Important job notes",
            application_status="Applied",
            application_date=base_date - timedelta(days=5),
            session=test_session,
        )

        # Update with new scraped data
        new_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Updated Title",
            description="Updated description",
            location="New Location",
        )

        sync_engine._update_existing_job(existing_job, new_job)

        # Scraped fields should be updated
        assert existing_job.title == "Updated Title"
        assert existing_job.description == "Updated description"
        assert existing_job.location == "New Location"

        # User fields should be preserved
        assert existing_job.favorite is True
        assert existing_job.notes == "Important job notes"
        assert existing_job.application_status == "Applied"
        assert existing_job.application_date == base_date - timedelta(days=5)

    def test_update_existing_job_unarchives_returned_job(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that archived jobs are unarchived when they return."""
        # Create archived job
        existing_job = JobFactory.create(
            company_id=sample_companies[0].id,
            archived=True,
            content_hash="old_hash",
            session=test_session,
        )

        # Job returns with same content
        new_job = JobSQL(
            company_id=sample_companies[0].id,
            title=existing_job.title,
            description=existing_job.description,
            location=existing_job.location,
        )
        new_job.content_hash = "old_hash"  # Same content

        result = sync_engine._update_existing_job(existing_job, new_job)

        assert (
            result == "updated"
        )  # Even with same content, it's an update due to unarchiving
        assert existing_job.archived is False  # Should be unarchived


class TestSmartSyncEngineUserDataDetection:
    """Test user data detection logic."""

    def test_has_user_data_favorite_job(self, sync_engine):
        """Test detection of favorited jobs."""
        job = JobSQL(favorite=True, application_status="New")

        assert sync_engine._has_user_data(job) is True

    def test_has_user_data_with_notes(self, sync_engine):
        """Test detection of jobs with notes."""
        job = JobSQL(favorite=False, notes="Some notes", application_status="New")

        assert sync_engine._has_user_data(job) is True

    def test_has_user_data_applied_status(self, sync_engine):
        """Test detection of jobs with non-New application status."""
        job = JobSQL(favorite=False, notes="", application_status="Applied")

        assert sync_engine._has_user_data(job) is True

    def test_has_user_data_no_user_data(self, sync_engine):
        """Test detection of jobs without user data."""
        job = JobSQL(favorite=False, notes="", application_status="New")

        assert sync_engine._has_user_data(job) is False

    def test_has_user_data_whitespace_notes(self, sync_engine):
        """Test that whitespace-only notes don't count as user data."""
        job = JobSQL(favorite=False, notes="   \n\t   ", application_status="New")

        assert sync_engine._has_user_data(job) is False


class TestSmartSyncEngineStaleJobHandling:
    """Test stale job handling and smart archiving."""

    def test_handle_stale_jobs_archive_with_user_data(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that stale jobs with user data are archived."""
        # Create job with user data that won't be in current scrape
        stale_job = JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://stale.com/job/1",
            favorite=True,  # Has user data
            archived=False,
            session=test_session,
        )

        # Current scrape doesn't include this job
        current_links = {"https://company.com/job/2"}

        stats = sync_engine._handle_stale_jobs(test_session, current_links)

        assert stats["archived"] == 1
        assert stats["deleted"] == 0

        # Verify job was archived, not deleted
        test_session.refresh(stale_job)
        assert stale_job.archived is True

    def test_handle_stale_jobs_delete_without_user_data(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that stale jobs without user data are deleted."""
        # Create job without user data
        stale_job = JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://stale.com/job/1",
            favorite=False,
            notes="",
            application_status="New",
            archived=False,
            session=test_session,
        )
        job_id = stale_job.id

        # Current scrape doesn't include this job
        current_links = {"https://company.com/job/2"}

        stats = sync_engine._handle_stale_jobs(test_session, current_links)

        assert stats["archived"] == 0
        assert stats["deleted"] == 1

        # Verify job was deleted
        deleted_job = test_session.exec(select(JobSQL).filter_by(id=job_id)).first()
        assert deleted_job is None

    def test_handle_stale_jobs_ignores_already_archived(
        self, sync_engine, test_session, sample_companies
    ):
        """Test that already archived jobs are ignored."""
        # Create already archived job
        JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://archived.com/job/1",
            archived=True,
            session=test_session,
        )

        # Current scrape doesn't include this job
        current_links = {"https://company.com/job/2"}

        stats = sync_engine._handle_stale_jobs(test_session, current_links)

        assert stats["archived"] == 0
        assert stats["deleted"] == 0

    def test_handle_stale_jobs_empty_current_links(
        self, sync_engine, test_session, sample_companies
    ):
        """Test handling when current_links is empty (all jobs are stale)."""
        # Create job that will be stale
        JobFactory.create(
            company_id=sample_companies[0].id,
            favorite=False,
            notes="",
            application_status="New",
            archived=False,
            session=test_session,
        )

        # Empty current scrape - all jobs are stale
        current_links = set()

        stats = sync_engine._handle_stale_jobs(test_session, current_links)

        assert stats["deleted"] >= 1  # At least our test job was deleted

    def test_handle_stale_jobs_mixed_user_data(
        self, sync_engine, test_session, sample_companies
    ):
        """Test handling mix of jobs with and without user data."""
        # Job with user data (should be archived)
        with_user_data = JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://stale1.com/job",
            favorite=True,
            archived=False,
            session=test_session,
        )

        # Job without user data (should be deleted)
        without_user_data = JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://stale2.com/job",
            favorite=False,
            notes="",
            application_status="New",
            archived=False,
            session=test_session,
        )

        # Current scrape doesn't include either job
        current_links = {"https://company.com/job/3"}

        stats = sync_engine._handle_stale_jobs(test_session, current_links)

        assert stats["archived"] == 1
        assert stats["deleted"] == 1

        # Verify the results
        test_session.refresh(with_user_data)
        assert with_user_data.archived is True

        deleted_job = test_session.exec(
            select(JobSQL).filter_by(id=without_user_data.id)
        ).first()
        assert deleted_job is None


class TestSmartSyncEngineFullSync:
    """Test complete sync operations."""

    def test_sync_jobs_insert_new_jobs(self, sync_engine, sample_companies):
        """Test syncing with all new jobs."""
        new_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="New Job 1",
                description="Description 1",
                link="https://company.com/job/1",
                location="SF",
            ),
            JobSQL(
                company_id=sample_companies[1].id,
                title="New Job 2",
                description="Description 2",
                link="https://company.com/job/2",
                location="NYC",
            ),
        ]

        stats = sync_engine.sync_jobs(new_jobs)

        assert stats["inserted"] == 2
        assert stats["updated"] == 0
        assert stats["archived"] == 0
        assert stats["deleted"] == 0
        assert stats["skipped"] == 0

    def test_sync_jobs_update_existing_jobs(
        self, sync_engine, test_session, sample_companies
    ):
        """Test syncing with updated existing jobs."""
        # Create existing job
        existing_job = JobFactory.create(
            company_id=sample_companies[0].id,
            title="Old Title",
            link="https://company.com/job/1",
            session=test_session,
        )

        # Create updated version
        updated_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="New Title",  # Updated
                description=existing_job.description,
                link="https://company.com/job/1",  # Same link
                location=existing_job.location,
            ),
        ]

        stats = sync_engine.sync_jobs(updated_jobs)

        assert stats["inserted"] == 0
        assert stats["updated"] == 1
        assert stats["archived"] == 0
        assert stats["deleted"] == 0
        assert stats["skipped"] == 0

    def test_sync_jobs_skip_unchanged_jobs(
        self, sync_engine, test_session, sample_companies
    ):
        """Test syncing with unchanged existing jobs."""
        # Create existing job
        existing_job = JobFactory.create(
            company_id=sample_companies[0].id,
            title="Same Title",
            link="https://company.com/job/1",
            session=test_session,
        )

        # Create identical version
        unchanged_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Same Title",
                description=existing_job.description,
                link="https://company.com/job/1",
                location=existing_job.location,
            ),
        ]
        # Set same content hash to simulate no changes
        unchanged_jobs[0].content_hash = existing_job.content_hash

        stats = sync_engine.sync_jobs(unchanged_jobs)

        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["archived"] == 0
        assert stats["deleted"] == 0
        assert stats["skipped"] == 1

    def test_sync_jobs_handles_duplicates_in_batch(self, sync_engine, sample_companies):
        """Test that duplicate links within batch are handled."""
        duplicate_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Job 1",
                description="Description 1",
                link="https://company.com/job/1",  # Duplicate link
                location="SF",
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Job 2",
                description="Description 2",
                link="https://company.com/job/1",  # Duplicate link
                location="NYC",
            ),
        ]

        stats = sync_engine.sync_jobs(duplicate_jobs)

        assert stats["inserted"] == 1  # Only one inserted
        assert stats["skipped"] == 1  # One skipped due to duplicate

    def test_sync_jobs_handles_jobs_without_links(self, sync_engine, sample_companies):
        """Test that jobs without links are skipped with warning."""
        jobs_with_missing_links = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Good Job",
                description="Has link",
                link="https://company.com/job/1",
                location="SF",
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Bad Job",
                description="Missing link",
                link=None,  # No link
                location="NYC",
            ),
        ]

        stats = sync_engine.sync_jobs(jobs_with_missing_links)

        assert stats["inserted"] == 1  # Only the good job
        assert stats["skipped"] == 1  # Bad job skipped

    def test_sync_jobs_comprehensive_workflow(
        self, sync_engine, test_session, sample_companies
    ):
        """Test comprehensive sync with all operation types."""
        datetime.now(UTC)

        # Set up existing jobs
        # Job 1: Will be updated
        existing_job_1 = JobFactory.create(
            company_id=sample_companies[0].id,
            title="Old Title 1",
            link="https://company.com/job/1",
            session=test_session,
        )

        # Job 2: Will be skipped (no changes)
        existing_job_2 = JobFactory.create(
            company_id=sample_companies[0].id,
            title="Unchanged Title",
            link="https://company.com/job/2",
            session=test_session,
        )

        # Job 3: Will be archived (has user data, not in current scrape)
        stale_job_with_user_data = JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://company.com/stale/1",
            favorite=True,
            session=test_session,
        )

        # Job 4: Will be deleted (no user data, not in current scrape)
        stale_job_without_user_data = JobFactory.create(
            company_id=sample_companies[0].id,
            link="https://company.com/stale/2",
            favorite=False,
            notes="",
            application_status="New",
            session=test_session,
        )

        # Prepare sync batch
        sync_jobs = [
            # Updated job 1
            JobSQL(
                company_id=sample_companies[0].id,
                title="New Title 1",  # Changed
                description=existing_job_1.description,
                link="https://company.com/job/1",
                location=existing_job_1.location,
            ),
            # Unchanged job 2
            JobSQL(
                company_id=sample_companies[0].id,
                title="Unchanged Title",  # Same
                description=existing_job_2.description,
                link="https://company.com/job/2",
                location=existing_job_2.location,
            ),
            # New job 3
            JobSQL(
                company_id=sample_companies[1].id,
                title="Brand New Job",
                description="New opportunity",
                link="https://company.com/job/3",
                location="Remote",
            ),
        ]

        # Set same content hash for unchanged job to simulate no changes
        sync_jobs[1].content_hash = existing_job_2.content_hash

        # Execute sync
        stats = sync_engine.sync_jobs(sync_jobs)

        # Verify statistics
        assert stats["inserted"] == 1  # New job 3
        assert stats["updated"] == 1  # Updated job 1
        assert stats["skipped"] == 1  # Unchanged job 2
        assert stats["archived"] == 1  # Stale job with user data
        assert stats["deleted"] == 1  # Stale job without user data

        # Verify database state
        test_session.refresh(existing_job_1)
        assert existing_job_1.title == "New Title 1"

        test_session.refresh(stale_job_with_user_data)
        assert stale_job_with_user_data.archived is True

        deleted_job = test_session.exec(
            select(JobSQL).filter_by(id=stale_job_without_user_data.id)
        ).first()
        assert deleted_job is None

    def test_sync_jobs_transaction_rollback_on_error(
        self, test_session, sample_companies
    ):
        """Test that sync rolls back on error."""
        engine = SmartSyncEngine(session=test_session)

        # Create a job that will cause issues
        problematic_jobs = [
            JobSQL(
                company_id=None,  # This might cause constraint issues
                title="Problem Job",
                description="Will cause error",
                link="https://company.com/bad",
                location="Error City",
            ),
        ]

        # Mock a database error during processing
        with patch.object(
            engine,
            "_sync_single_job_optimized",
            side_effect=Exception("Database error"),
        ):
            with pytest.raises(Exception, match="Database error"):
                engine.sync_jobs(problematic_jobs)


class TestSmartSyncEngineStatistics:
    """Test statistics and monitoring functionality."""

    def test_get_sync_statistics_basic(
        self, sync_engine, test_session, sample_companies
    ):
        """Test basic sync statistics retrieval."""
        # Create sample data
        JobFactory.create_batch(
            3,
            company_id=sample_companies[0].id,
            archived=False,
            session=test_session,
        )
        JobFactory.create(
            company_id=sample_companies[0].id,
            archived=True,
            session=test_session,
        )
        JobFactory.create(
            company_id=sample_companies[0].id,
            favorite=True,
            archived=False,
            session=test_session,
        )
        JobFactory.create(
            company_id=sample_companies[0].id,
            application_status="Applied",
            archived=False,
            session=test_session,
        )

        stats = sync_engine.get_sync_statistics()

        assert stats["total_jobs"] == 6
        assert stats["active_jobs"] == 5  # Non-archived
        assert stats["archived_jobs"] == 1
        assert stats["favorited_jobs"] == 1
        assert stats["applied_jobs"] == 1  # Non-"New" status

    def test_get_sync_statistics_empty_database(self, sync_engine, test_session):
        """Test statistics with empty database."""
        stats = sync_engine.get_sync_statistics()

        assert stats["total_jobs"] == 0
        assert stats["active_jobs"] == 0
        assert stats["archived_jobs"] == 0
        assert stats["favorited_jobs"] == 0
        assert stats["applied_jobs"] == 0


class TestSmartSyncEngineCleanup:
    """Test cleanup and maintenance functionality."""

    def test_cleanup_old_jobs_basic(self, sync_engine, test_session, sample_companies):
        """Test cleanup of old archived jobs."""
        base_date = datetime.now(UTC)

        # Create old archived job without recent user activity
        old_job = JobFactory.create(
            company_id=sample_companies[0].id,
            archived=True,
            last_seen=base_date - timedelta(days=100),  # Very old
            application_date=None,  # No application activity
            session=test_session,
        )

        # Create old archived job with recent application activity
        old_job_with_activity = JobFactory.create(
            company_id=sample_companies[0].id,
            archived=True,
            last_seen=base_date - timedelta(days=100),  # Very old
            application_date=base_date - timedelta(days=10),  # Recent activity
            session=test_session,
        )

        # Create recent archived job
        recent_job = JobFactory.create(
            company_id=sample_companies[0].id,
            archived=True,
            last_seen=base_date - timedelta(days=30),  # Recent
            session=test_session,
        )

        # Cleanup jobs older than 90 days
        deleted_count = sync_engine.cleanup_old_jobs(days_threshold=90)

        assert deleted_count == 1  # Only the truly old job without activity

        # Verify the right job was deleted
        remaining_jobs = test_session.exec(select(JobSQL)).all()
        job_ids = [job.id for job in remaining_jobs]

        assert old_job.id not in job_ids  # Should be deleted
        assert old_job_with_activity.id in job_ids  # Should remain (recent activity)
        assert recent_job.id in job_ids  # Should remain (recent)

    def test_cleanup_old_jobs_no_old_jobs(
        self, sync_engine, test_session, sample_companies
    ):
        """Test cleanup when no old jobs exist."""
        # Create only recent jobs
        JobFactory.create_batch(
            3,
            company_id=sample_companies[0].id,
            archived=True,
            last_seen=datetime.now(UTC) - timedelta(days=30),
            session=test_session,
        )

        deleted_count = sync_engine.cleanup_old_jobs(days_threshold=90)

        assert deleted_count == 0

    def test_cleanup_old_jobs_transaction_error(self, test_session, sample_companies):
        """Test cleanup handles transaction errors properly."""
        engine = SmartSyncEngine(session=test_session)

        # Mock a database error during cleanup
        with patch.object(
            test_session, "commit", side_effect=Exception("Commit failed")
        ):
            with pytest.raises(Exception, match="Commit failed"):
                engine.cleanup_old_jobs()


class TestSmartSyncEngineErrorHandling:
    """Test error handling and edge cases."""

    def test_sync_jobs_empty_list(self, sync_engine):
        """Test syncing empty job list."""
        stats = sync_engine.sync_jobs([])

        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["archived"] == 0
        assert stats["deleted"] == 0
        assert stats["skipped"] == 0

    def test_sync_engine_without_session_closes_properly(self):
        """Test that sync engine properly closes sessions it creates."""
        # This is more of an integration test to ensure proper resource cleanup
        with patch("src.services.database_sync.SessionLocal") as mock_session_local:
            mock_session = Mock()
            mock_session_local.return_value = mock_session

            engine = SmartSyncEngine()  # No session provided

            # Execute operation that uses session
            engine.sync_jobs([])

            # Verify session was closed
            mock_session.close.assert_called_once()

    def test_sync_engine_with_provided_session_no_close(self, test_session):
        """Test that provided sessions are not closed by sync engine."""
        original_close = test_session.close
        close_called = False

        def mock_close():
            nonlocal close_called
            close_called = True
            original_close()

        test_session.close = mock_close

        engine = SmartSyncEngine(session=test_session)
        engine.sync_jobs([])

        # Should not call close on provided session
        assert not close_called


class TestSmartSyncEngineIntegration:
    """Integration tests for SmartSyncEngine workflows."""

    def test_realistic_scraping_workflow(
        self, sync_engine, test_session, sample_companies
    ):
        """Test realistic job scraping and synchronization workflow."""
        base_date = datetime.now(UTC)

        # Initial scrape: insert new jobs
        initial_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Software Engineer",
                description="Python development",
                link="https://company.com/job/1",
                location="SF",
                posted_date=base_date - timedelta(days=1),
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Data Scientist",
                description="ML and analytics",
                link="https://company.com/job/2",
                location="NYC",
                posted_date=base_date - timedelta(days=2),
            ),
        ]

        # First sync
        stats1 = sync_engine.sync_jobs(initial_jobs)
        assert stats1["inserted"] == 2

        # User interacts with jobs
        jobs = test_session.exec(select(JobSQL)).all()
        jobs[0].favorite = True
        jobs[0].notes = "Interesting position"
        jobs[1].application_status = "Applied"
        jobs[1].application_date = base_date
        test_session.commit()

        # Second scrape: one job updated, one unchanged, one removed, one new
        second_scrape = [
            # Job 1: Updated (title changed)
            JobSQL(
                company_id=sample_companies[0].id,
                title="Senior Software Engineer",  # Changed
                description="Python development",
                link="https://company.com/job/1",
                location="SF",
                posted_date=base_date - timedelta(days=1),
            ),
            # Job 2: Unchanged
            JobSQL(
                company_id=sample_companies[0].id,
                title="Data Scientist",
                description="ML and analytics",
                link="https://company.com/job/2",
                location="NYC",
                posted_date=base_date - timedelta(days=2),
            ),
            # Job 3: New job
            JobSQL(
                company_id=sample_companies[1].id,
                title="DevOps Engineer",
                description="Infrastructure automation",
                link="https://company2.com/job/1",
                location="Remote",
                posted_date=base_date,
            ),
        ]

        # Second sync
        stats2 = sync_engine.sync_jobs(second_scrape)
        assert stats2["inserted"] == 1  # New DevOps job
        assert stats2["updated"] == 1  # Updated Software Engineer
        assert stats2["skipped"] == 1  # Unchanged Data Scientist
        assert stats2["archived"] == 0  # No jobs were stale
        assert stats2["deleted"] == 0

        # Verify user data preservation
        updated_jobs = test_session.exec(select(JobSQL)).all()
        software_eng = next(
            job for job in updated_jobs if "Software Engineer" in job.title
        )
        data_scientist = next(
            job for job in updated_jobs if "Data Scientist" in job.title
        )

        # Software Engineer: updated title, preserved user data
        assert software_eng.title == "Senior Software Engineer"
        assert software_eng.favorite is True
        assert software_eng.notes == "Interesting position"

        # Data Scientist: unchanged, preserved user data
        assert data_scientist.application_status == "Applied"
        assert data_scientist.application_date == base_date

    def test_multi_scrape_archiving_workflow(
        self, sync_engine, test_session, sample_companies
    ):
        """Test workflow with jobs disappearing and reappearing."""
        # Create initial job
        initial_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Temp Position",
            description="Temporary role",
            link="https://company.com/temp/1",
            location="SF",
        )

        # First sync: insert job
        stats1 = sync_engine.sync_jobs([initial_job])
        assert stats1["inserted"] == 1

        # User marks as favorite
        job = test_session.exec(select(JobSQL)).first()
        job.favorite = True
        test_session.commit()

        # Second sync: job disappears (archived due to user data)
        stats2 = sync_engine.sync_jobs([])  # Empty scrape
        assert stats2["archived"] == 1

        # Third sync: job reappears (unarchived)
        reappeared_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Temp Position",
            description="Temporary role",
            link="https://company.com/temp/1",
            location="SF",
        )

        stats3 = sync_engine.sync_jobs([reappeared_job])
        assert stats3["updated"] == 1  # Unarchived and updated

        # Verify job is unarchived and user data preserved
        final_job = test_session.exec(select(JobSQL)).first()
        assert final_job.archived is False
        assert final_job.favorite is True
