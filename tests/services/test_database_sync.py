"""Comprehensive tests for SmartSyncEngine database sync service.

This test suite validates the SmartSyncEngine class for robust database synchronization
with job data persistence. Tests cover all sync operations, change detection, archiving,
error handling, and transaction integrity scenarios.
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import hashlib
import logging

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select
from src.models import CompanySQL, JobSQL
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
    companies = [
        CompanySQL(
            name="TechCorp",
            url="https://techcorp.com/careers",
            active=True,
        ),
        CompanySQL(
            name="InnovateLabs",
            url="https://innovatelabs.com/jobs",
            active=True,
        ),
        CompanySQL(
            name="DataCorp",
            url="https://datacorp.com/careers",
            active=False,
        ),
    ]

    for company in companies:
        test_session.add(company)
    test_session.commit()

    # Refresh to get IDs
    for company in companies:
        test_session.refresh(company)

    return companies


@pytest.fixture
def sample_jobs(test_session, sample_companies):
    """Create sample jobs for comprehensive sync testing."""
    base_date = datetime.now(timezone.utc)

    jobs = [
        # Active job that will be updated
        JobSQL(
            company_id=sample_companies[0].id,
            title="Senior Python Developer",
            description="Python development role",
            link="https://techcorp.com/jobs/python-dev-001",
            location="San Francisco, CA",
            posted_date=base_date - timedelta(days=1),
            salary=(120000, 160000),
            content_hash="original_hash_001",
            application_status="New",
            last_seen=base_date - timedelta(hours=24),
        ),
        # Existing job with user data
        JobSQL(
            company_id=sample_companies[0].id,
            title="Machine Learning Engineer",
            description="ML engineering position",
            link="https://techcorp.com/jobs/ml-eng-002",
            location="New York, NY",
            posted_date=base_date - timedelta(days=5),
            salary=(140000, 180000),
            content_hash="original_hash_002",
            application_status="Applied",
            application_date=base_date - timedelta(days=2),
            favorite=True,
            notes="Great company culture",
            last_seen=base_date - timedelta(hours=48),
        ),
        # Stale job without user data (will be deleted)
        JobSQL(
            company_id=sample_companies[1].id,
            title="Frontend Developer",
            description="React development role",
            link="https://innovatelabs.com/jobs/frontend-003",
            location="Austin, TX",
            posted_date=base_date - timedelta(days=10),
            salary=(90000, 120000),
            content_hash="original_hash_003",
            application_status="New",
            last_seen=base_date - timedelta(days=7),
        ),
        # Stale job with user data (will be archived)
        JobSQL(
            company_id=sample_companies[1].id,
            title="DevOps Engineer",
            description="Cloud infrastructure role",
            link="https://innovatelabs.com/jobs/devops-004",
            location="Remote",
            posted_date=base_date - timedelta(days=15),
            salary=(110000, 150000),
            content_hash="original_hash_004",
            application_status="Interested",
            favorite=False,
            notes="Remote work available",
            last_seen=base_date - timedelta(days=10),
        ),
        # Already archived job
        JobSQL(
            company_id=sample_companies[0].id,
            title="Archived Position",
            description="This job is archived",
            link="https://techcorp.com/jobs/archived-005",
            location="Somewhere",
            posted_date=base_date - timedelta(days=30),
            salary=(80000, 100000),
            content_hash="original_hash_005",
            application_status="New",
            archived=True,
            last_seen=base_date - timedelta(days=20),
        ),
    ]

    for job in jobs:
        test_session.add(job)
    test_session.commit()

    # Refresh to get IDs and relationships
    for job in jobs:
        test_session.refresh(job)

    return jobs


class TestSmartSyncEngineInitialization:
    """Test SmartSyncEngine initialization and session management."""

    def test_init_with_session(self, test_session):
        """Test initialization with provided session."""
        engine = SmartSyncEngine(session=test_session)
        assert engine._session is test_session
        assert not engine._session_owned

    def test_init_without_session(self):
        """Test initialization without provided session."""
        engine = SmartSyncEngine()
        assert engine._session is None
        assert engine._session_owned

    def test_get_session_with_provided_session(self, test_session):
        """Test _get_session when session is provided."""
        engine = SmartSyncEngine(session=test_session)
        assert engine._get_session() is test_session

    @patch("src.services.database_sync.SessionLocal")
    def test_get_session_without_provided_session(self, mock_session_local):
        """Test _get_session creates new session when none provided."""
        mock_session = Mock()
        mock_session_local.return_value = mock_session

        engine = SmartSyncEngine()
        result = engine._get_session()

        assert result is mock_session
        mock_session_local.assert_called_once()

    def test_close_session_if_owned_with_owned_session(self, test_session):
        """Test session is closed when owned by engine."""
        engine = SmartSyncEngine()
        mock_session = Mock()

        engine._close_session_if_owned(mock_session)

        mock_session.close.assert_called_once()

    def test_close_session_if_owned_with_provided_session(self, test_session):
        """Test session is not closed when provided to engine."""
        engine = SmartSyncEngine(session=test_session)
        mock_session = Mock()

        engine._close_session_if_owned(mock_session)

        mock_session.close.assert_not_called()


class TestSyncJobs:
    """Test the main sync_jobs method with various scenarios."""

    def test_sync_jobs_empty_list(self, test_session, sample_jobs):
        """Test syncing empty job list."""
        engine = SmartSyncEngine(session=test_session)

        # Calculate expected values BEFORE sync (since sync modifies the jobs)
        expected_archived = len(
            [j for j in sample_jobs if not j.archived and engine._has_user_data(j)]
        )
        expected_deleted = len(
            [j for j in sample_jobs if not j.archived and not engine._has_user_data(j)]
        )

        result = engine.sync_jobs([])

        expected_stats = {
            "inserted": 0,
            "updated": 0,
            "archived": expected_archived,
            "deleted": expected_deleted,
            "skipped": 0,
        }
        assert result == expected_stats

    def test_sync_jobs_new_jobs_only(self, test_session, sample_companies):
        """Test syncing only new jobs."""
        engine = SmartSyncEngine(session=test_session)

        new_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="New Software Engineer",
                description="New development role",
                link="https://techcorp.com/jobs/new-001",
                location="Seattle, WA",
                salary=(130000, 170000),
            ),
            JobSQL(
                company_id=sample_companies[1].id,
                title="New Data Scientist",
                description="Data science position",
                link="https://innovatelabs.com/jobs/new-002",
                location="Portland, OR",
                salary=(125000, 165000),
            ),
        ]

        result = engine.sync_jobs(new_jobs)

        assert result["inserted"] == 2
        assert result["updated"] == 0
        assert result["archived"] == 0
        assert result["deleted"] == 0
        assert result["skipped"] == 0

        # Verify jobs were inserted with correct default values
        inserted_jobs = test_session.exec(
            select(JobSQL).where(
                JobSQL.title.in_(["New Software Engineer", "New Data Scientist"])
            )
        ).all()

        assert len(inserted_jobs) == 2
        for job in inserted_jobs:
            assert job.application_status == "New"
            assert job.last_seen is not None
            assert job.content_hash is not None
            assert not job.archived

    def test_sync_jobs_existing_no_changes(self, test_session, sample_jobs):
        """Test syncing existing jobs with no content changes."""
        engine = SmartSyncEngine(session=test_session)

        # Create jobs with same content as existing
        existing_job = sample_jobs[0]  # Senior Python Developer

        # Update existing job's content hash to what sync engine would generate
        existing_job.content_hash = engine._generate_content_hash(existing_job)
        test_session.commit()

        unchanged_job = JobSQL(
            company_id=existing_job.company_id,
            title=existing_job.title,
            description=existing_job.description,
            link=existing_job.link,
            location=existing_job.location,
            salary=existing_job.salary,
            posted_date=existing_job.posted_date,
        )

        # Ensure salary is properly copied as the same format (list, not tuple)
        if existing_job.salary:
            unchanged_job.salary = existing_job.salary
        # Generate content hash using the sync engine to ensure consistency
        unchanged_job.content_hash = engine._generate_content_hash(unchanged_job)

        # Calculate expected stale jobs BEFORE sync (all except the one being synced)
        stale_jobs = [
            j for j in sample_jobs if not j.archived and j.link != existing_job.link
        ]
        expected_archived = len([j for j in stale_jobs if engine._has_user_data(j)])
        expected_deleted = len([j for j in stale_jobs if not engine._has_user_data(j)])

        # Capture original last_seen before sync
        original_last_seen = existing_job.last_seen

        result = engine.sync_jobs([unchanged_job])

        assert result["inserted"] == 0
        assert result["updated"] == 0
        assert result["archived"] == expected_archived
        assert result["deleted"] == expected_deleted
        assert result["skipped"] == 1

        # Verify last_seen was updated
        updated_job = test_session.exec(
            select(JobSQL).where(JobSQL.id == existing_job.id)
        ).first()
        assert updated_job.last_seen > original_last_seen

    def test_sync_jobs_existing_with_changes(self, test_session, sample_jobs):
        """Test syncing existing jobs with content changes."""
        engine = SmartSyncEngine(session=test_session)

        existing_job = sample_jobs[0]  # Senior Python Developer

        # Store original last_seen for comparison
        original_last_seen = existing_job.last_seen

        updated_job = JobSQL(
            company_id=existing_job.company_id,
            title="Senior Python Developer (Updated)",  # Changed title
            description=existing_job.description,
            link=existing_job.link,
            location=existing_job.location,
            salary=existing_job.salary,
            posted_date=existing_job.posted_date,
        )

        # Calculate expected stale jobs BEFORE sync (all except the one being synced)
        stale_jobs = [
            j for j in sample_jobs if not j.archived and j.link != existing_job.link
        ]
        expected_archived = len([j for j in stale_jobs if engine._has_user_data(j)])
        expected_deleted = len([j for j in stale_jobs if not engine._has_user_data(j)])

        result = engine.sync_jobs([updated_job])

        assert result["inserted"] == 0
        assert result["updated"] == 1
        assert result["archived"] == expected_archived
        assert result["deleted"] == expected_deleted
        assert result["skipped"] == 0

        # Verify job was updated
        refreshed_job = test_session.exec(
            select(JobSQL).where(JobSQL.id == existing_job.id)
        ).first()
        assert refreshed_job.title == "Senior Python Developer (Updated)"
        assert refreshed_job.last_seen > original_last_seen

    def test_sync_jobs_mixed_scenarios(
        self, test_session, sample_jobs, sample_companies
    ):
        """Test syncing with mixed insert/update/archive/delete scenarios."""
        engine = SmartSyncEngine(session=test_session)

        existing_job_1 = sample_jobs[0]  # Will be updated
        existing_job_2 = sample_jobs[1]  # Will be skipped (no changes)
        # sample_jobs[2] will be deleted (no user data)
        # sample_jobs[3] will be archived (has user data)

        # Update existing jobs' content hashes to what sync engine would generate
        existing_job_2.content_hash = engine._generate_content_hash(existing_job_2)
        test_session.commit()

        sync_jobs = [
            # Update existing job
            JobSQL(
                company_id=existing_job_1.company_id,
                title="Senior Python Developer (Updated)",
                description=existing_job_1.description,
                link=existing_job_1.link,
                location=existing_job_1.location,
                salary=existing_job_1.salary,
                posted_date=existing_job_1.posted_date,
            ),
            # Keep existing job unchanged
            JobSQL(
                company_id=existing_job_2.company_id,
                title=existing_job_2.title,
                description=existing_job_2.description,
                link=existing_job_2.link,
                location=existing_job_2.location,
                salary=existing_job_2.salary,  # Will be corrected below
                posted_date=existing_job_2.posted_date,
            ),
            # Insert new job
            JobSQL(
                company_id=sample_companies[0].id,
                title="New Backend Engineer",
                description="Backend development role",
                link="https://techcorp.com/jobs/backend-new",
                location="Denver, CO",
                salary=(115000, 155000),
            ),
        ]

        # Ensure the unchanged job has the same salary format for hash consistency
        if existing_job_2.salary:
            sync_jobs[1].salary = existing_job_2.salary

        # Calculate expected stale jobs BEFORE sync (jobs not in current sync)
        current_links = {job.link for job in sync_jobs}
        stale_jobs = [
            j for j in sample_jobs if not j.archived and j.link not in current_links
        ]
        expected_archived = len([j for j in stale_jobs if engine._has_user_data(j)])
        expected_deleted = len([j for j in stale_jobs if not engine._has_user_data(j)])

        result = engine.sync_jobs(sync_jobs)

        assert result["inserted"] == 1  # New job
        assert result["updated"] == 1  # Updated job
        assert result["archived"] == expected_archived  # Job(s) with user data
        assert result["deleted"] == expected_deleted  # Job(s) without user data
        assert result["skipped"] == 1  # Unchanged job

    def test_sync_jobs_unarchive_returning_job(self, test_session, sample_jobs):
        """Test that archived jobs are unarchived when they return."""
        engine = SmartSyncEngine(session=test_session)

        # Archive a job first
        job_to_archive = sample_jobs[0]
        job_to_archive.archived = True
        test_session.commit()

        # Now sync it back
        returning_job = JobSQL(
            company_id=job_to_archive.company_id,
            title=job_to_archive.title,
            description=job_to_archive.description,
            link=job_to_archive.link,
            location=job_to_archive.location,
            salary=job_to_archive.salary,
            posted_date=job_to_archive.posted_date,
        )

        result = engine.sync_jobs([returning_job])

        assert result["updated"] == 1  # Job was unarchived and updated

        # Verify job is no longer archived
        updated_job = test_session.exec(
            select(JobSQL).where(JobSQL.id == job_to_archive.id)
        ).first()
        assert not updated_job.archived

    def test_sync_jobs_skip_jobs_without_links(self, test_session, sample_companies):
        """Test that jobs without links are skipped."""
        engine = SmartSyncEngine(session=test_session)

        jobs_with_invalid_links = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Job Without Link",
                description="This job has no link",
                link="",  # Empty link
                location="Nowhere",
                salary=(100000, 130000),
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Job With Valid Link",
                description="This job has a valid link",
                link="https://example.com/valid-job",
                location="Somewhere",
                salary=(100000, 130000),
            ),
        ]

        result = engine.sync_jobs(jobs_with_invalid_links)

        # Only one job should be processed
        assert result["inserted"] == 1

        # Verify only the valid job was inserted
        inserted_jobs = test_session.exec(
            select(JobSQL).where(
                JobSQL.title.in_(["Job Without Link", "Job With Valid Link"])
            )
        ).all()

        assert len(inserted_jobs) == 1
        assert inserted_jobs[0].title == "Job With Valid Link"

    def test_sync_jobs_transaction_rollback_on_error(
        self, test_session, sample_companies
    ):
        """Test that transaction is rolled back on error."""
        engine = SmartSyncEngine(session=test_session)

        # Create a job that will cause an error during processing
        problem_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Problem Job",
            description="This will cause issues",
            link="https://example.com/problem",
            location="Error City",
            salary=(100000, 130000),
        )

        # Mock the _sync_single_job_optimized to raise an exception
        with (
            patch.object(
                engine,
                "_sync_single_job_optimized",
                side_effect=Exception("Database error"),
            ),
            pytest.raises(Exception, match="Database error"),
        ):
            engine.sync_jobs([problem_job])

        # Verify no jobs were inserted due to rollback
        inserted_jobs = test_session.exec(
            select(JobSQL).where(JobSQL.title == "Problem Job")
        ).all()
        assert len(inserted_jobs) == 0


class TestJobSyncOperations:
    """Test individual job sync operations."""

    def test_insert_new_job(self, test_session, sample_companies):
        """Test inserting a new job."""
        engine = SmartSyncEngine(session=test_session)

        new_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Test New Job",
            description="New job description",
            link="https://example.com/new-job",
            location="Test City",
            salary=(100000, 130000),
        )

        result = engine._insert_new_job(test_session, new_job)

        assert result == "inserted"
        assert new_job.application_status == "New"
        assert new_job.last_seen is not None
        assert new_job.content_hash is not None

        # Verify job is in session
        assert new_job in test_session

    def test_insert_new_job_with_existing_status(self, test_session, sample_companies):
        """Test inserting a job with pre-existing application status."""
        engine = SmartSyncEngine(session=test_session)

        new_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Test Job with Status",
            description="Job with pre-set status",
            link="https://example.com/job-with-status",
            location="Test City",
            salary=(100000, 130000),
            application_status="Interested",  # Pre-set status
        )

        result = engine._insert_new_job(test_session, new_job)

        assert result == "inserted"
        assert (
            new_job.application_status == "Interested"
        )  # Should preserve existing status
        assert new_job.last_seen is not None

    def test_update_existing_job_no_content_change(self, test_session, sample_jobs):
        """Test updating existing job with no content changes."""
        engine = SmartSyncEngine(session=test_session)

        existing_job = sample_jobs[0]
        original_last_seen = existing_job.last_seen

        # Generate the correct content hash for the existing job first
        correct_hash = engine._generate_content_hash(existing_job)
        existing_job.content_hash = correct_hash
        test_session.commit()

        # Create job with same content (including company relationship)
        new_job = JobSQL(
            company_id=existing_job.company_id,
            title=existing_job.title,
            description=existing_job.description,
            location=existing_job.location,
            salary=existing_job.salary,
            posted_date=existing_job.posted_date,
        )
        # Set the company relationship to match the existing job
        new_job.company_relation = existing_job.company_relation

        result = engine._update_existing_job(existing_job, new_job)

        assert result == "skipped"
        # Convert original_last_seen to timezone-aware for comparison if needed
        if (
            original_last_seen.tzinfo is None
            and existing_job.last_seen.tzinfo is not None
        ):
            original_last_seen = original_last_seen.replace(tzinfo=timezone.utc)
        assert existing_job.last_seen > original_last_seen

    def test_update_existing_job_with_content_changes(self, test_session, sample_jobs):
        """Test updating existing job with content changes."""
        engine = SmartSyncEngine(session=test_session)

        existing_job = sample_jobs[1]  # ML Engineer with user data
        original_favorite = existing_job.favorite
        original_notes = existing_job.notes
        original_app_status = existing_job.application_status
        original_app_date = existing_job.application_date

        # Create job with different content
        new_job = JobSQL(
            company_id=existing_job.company_id,
            title="Updated ML Engineer Title",  # Changed title
            description=existing_job.description,
            location=existing_job.location,
            salary=existing_job.salary,
        )

        result = engine._update_existing_job(existing_job, new_job)

        assert result == "updated"

        # Verify scraped fields were updated
        assert existing_job.title == "Updated ML Engineer Title"

        # Verify user data was preserved
        assert existing_job.favorite == original_favorite
        assert existing_job.notes == original_notes
        assert existing_job.application_status == original_app_status
        assert existing_job.application_date == original_app_date

    def test_update_scraped_fields_preserves_user_data(self, test_session, sample_jobs):
        """Test that _update_scraped_fields preserves user-editable fields."""
        engine = SmartSyncEngine(session=test_session)

        existing_job = sample_jobs[1]  # Job with user data
        original_user_data = {
            "favorite": existing_job.favorite,
            "notes": existing_job.notes,
            "application_status": existing_job.application_status,
            "application_date": existing_job.application_date,
        }

        new_job = JobSQL(
            company_id=existing_job.company_id,
            title="Completely New Title",
            description="Completely new description",
            location="New Location",
            salary=(200000, 250000),  # New salary
        )

        new_hash = "new_content_hash"

        engine._update_scraped_fields(existing_job, new_job, new_hash)

        # Verify scraped fields were updated
        assert existing_job.title == "Completely New Title"
        assert existing_job.description == "Completely new description"
        assert existing_job.location == "New Location"
        assert existing_job.salary == (200000, 250000)
        assert existing_job.content_hash == "new_content_hash"

        # Verify user data was preserved
        assert existing_job.favorite == original_user_data["favorite"]
        assert existing_job.notes == original_user_data["notes"]
        assert (
            existing_job.application_status == original_user_data["application_status"]
        )
        assert existing_job.application_date == original_user_data["application_date"]


class TestStaleJobHandling:
    """Test handling of stale jobs (archive/delete logic)."""

    def test_handle_stale_jobs_with_mixed_user_data(self, test_session, sample_jobs):
        """Test stale job handling with mixed user data scenarios."""
        engine = SmartSyncEngine(session=test_session)

        # Current links don't include the stale jobs
        current_links = {
            sample_jobs[0].link,
            sample_jobs[1].link,
        }  # Keep first two jobs

        result = engine._handle_stale_jobs(test_session, current_links)

        # Should archive job with user data and delete job without
        assert result["archived"] == 1  # DevOps job has notes
        assert result["deleted"] == 1  # Frontend job has no user data

        # Verify specific jobs were handled correctly
        devops_job = test_session.exec(
            select(JobSQL).where(JobSQL.title == "DevOps Engineer")
        ).first()
        assert devops_job.archived

        frontend_jobs = test_session.exec(
            select(JobSQL).where(JobSQL.title == "Frontend Developer")
        ).all()
        assert len(frontend_jobs) == 0  # Should be deleted

    def test_handle_stale_jobs_empty_current_links(self, test_session, sample_jobs):
        """Test stale job handling when no current jobs provided."""
        engine = SmartSyncEngine(session=test_session)

        result = engine._handle_stale_jobs(test_session, set())

        # All non-archived jobs should be processed
        expected_archived = len(
            [j for j in sample_jobs if not j.archived and engine._has_user_data(j)]
        )
        expected_deleted = len(
            [j for j in sample_jobs if not j.archived and not engine._has_user_data(j)]
        )

        assert result["archived"] == expected_archived
        assert result["deleted"] == expected_deleted

    def test_handle_stale_jobs_ignores_already_archived(
        self, test_session, sample_jobs
    ):
        """Test that already archived jobs are ignored."""
        engine = SmartSyncEngine(session=test_session)

        # Archive one job manually
        sample_jobs[0].archived = True
        test_session.commit()

        current_links = set()  # No current jobs

        initial_archived_count = len([j for j in sample_jobs if j.archived])

        result = engine._handle_stale_jobs(test_session, current_links)

        # Verify the pre-archived job wasn't counted
        final_archived_count = test_session.exec(
            select(JobSQL).where(JobSQL.archived)
        ).count()

        assert final_archived_count == initial_archived_count + result["archived"]

    def test_has_user_data_various_scenarios(self, test_session, sample_companies):
        """Test _has_user_data method with various job states."""
        engine = SmartSyncEngine(session=test_session)

        test_cases = [
            # Job with favorite flag
            (
                JobSQL(
                    company_id=sample_companies[0].id,
                    title="Favorite Job",
                    description="Test",
                    link="https://example.com/favorite",
                    location="Test",
                    favorite=True,
                    application_status="New",
                ),
                True,
            ),
            # Job with notes
            (
                JobSQL(
                    company_id=sample_companies[0].id,
                    title="Job with Notes",
                    description="Test",
                    link="https://example.com/notes",
                    location="Test",
                    notes="Some notes here",
                    application_status="New",
                ),
                True,
            ),
            # Job with application status other than "New"
            (
                JobSQL(
                    company_id=sample_companies[0].id,
                    title="Applied Job",
                    description="Test",
                    link="https://example.com/applied",
                    location="Test",
                    application_status="Applied",
                ),
                True,
            ),
            # Job with whitespace-only notes (should be False)
            (
                JobSQL(
                    company_id=sample_companies[0].id,
                    title="Whitespace Notes Job",
                    description="Test",
                    link="https://example.com/whitespace",
                    location="Test",
                    notes="   \n\t  ",
                    application_status="New",
                ),
                False,
            ),
            # Job with no user data
            (
                JobSQL(
                    company_id=sample_companies[0].id,
                    title="No User Data Job",
                    description="Test",
                    link="https://example.com/no-user-data",
                    location="Test",
                    application_status="New",
                ),
                False,
            ),
        ]

        for job, expected_has_user_data in test_cases:
            assert engine._has_user_data(job) == expected_has_user_data


class TestContentHashGeneration:
    """Test content hash generation for change detection."""

    def test_generate_content_hash_basic(self, test_session, sample_companies):
        """Test basic content hash generation."""
        engine = SmartSyncEngine(session=test_session)

        job = JobSQL(
            company_id=sample_companies[0].id,
            title="Test Job",
            description="Test description",
            location="Test Location",
            link="https://example.com/test",
            salary=(100000, 130000),
        )

        hash_result = engine._generate_content_hash(job)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 hash length

    def test_generate_content_hash_consistency(self, test_session, sample_companies):
        """Test that identical jobs produce identical hashes."""
        engine = SmartSyncEngine(session=test_session)

        job1 = JobSQL(
            company_id=sample_companies[0].id,
            title="Test Job",
            description="Test description",
            location="Test Location",
            link="https://example.com/test1",
            salary=(100000, 130000),
        )

        job2 = JobSQL(
            company_id=sample_companies[0].id,
            title="Test Job",
            description="Test description",
            location="Test Location",
            link="https://example.com/test2",  # Different link shouldn't affect hash
            salary=(100000, 130000),
        )

        hash1 = engine._generate_content_hash(job1)
        hash2 = engine._generate_content_hash(job2)

        assert hash1 == hash2

    def test_generate_content_hash_change_detection(
        self, test_session, sample_companies
    ):
        """Test that content changes produce different hashes."""
        engine = SmartSyncEngine(session=test_session)

        base_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Test Job",
            description="Test description",
            location="Test Location",
            link="https://example.com/test",
            salary=(100000, 130000),
        )

        # Test different field changes
        changed_jobs = [
            # Title change
            JobSQL(
                company_id=base_job.company_id,
                title="Changed Test Job",
                description=base_job.description,
                location=base_job.location,
                link=base_job.link,
                salary=base_job.salary,
            ),
            # Description change
            JobSQL(
                company_id=base_job.company_id,
                title=base_job.title,
                description="Changed test description",
                location=base_job.location,
                link=base_job.link,
                salary=base_job.salary,
            ),
            # Location change
            JobSQL(
                company_id=base_job.company_id,
                title=base_job.title,
                description=base_job.description,
                location="Changed Location",
                link=base_job.link,
                salary=base_job.salary,
            ),
            # Salary change
            JobSQL(
                company_id=base_job.company_id,
                title=base_job.title,
                description=base_job.description,
                location=base_job.location,
                link=base_job.link,
                salary=(120000, 150000),
            ),
        ]

        base_hash = engine._generate_content_hash(base_job)

        for changed_job in changed_jobs:
            changed_hash = engine._generate_content_hash(changed_job)
            assert changed_hash != base_hash

    def test_generate_content_hash_with_none_values(
        self, test_session, sample_companies
    ):
        """Test content hash generation handles None values gracefully."""
        engine = SmartSyncEngine(session=test_session)

        job = JobSQL(
            company_id=None,  # None company_id
            title=None,  # None title
            description=None,  # None description
            location="Test Location",
            link="https://example.com/test-none",
            salary=(None, None),  # None salary
            posted_date=None,  # None posted_date
        )

        # Should not raise an exception
        hash_result = engine._generate_content_hash(job)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_generate_content_hash_with_company_relationship(
        self, test_session, sample_jobs
    ):
        """Test content hash consistency with company_id regardless of loading."""
        engine = SmartSyncEngine(session=test_session)

        job = sample_jobs[0]  # Has company relationship loaded

        hash_result = engine._generate_content_hash(job)
        assert isinstance(hash_result, str)

        # Manually verify the hash includes company_id (not company name)
        content_parts = [
            job.title or "",
            job.description or "",
            job.location or "",
            str(job.company_id),  # Should use company_id for consistency
        ]
        if hasattr(job, "salary") and job.salary:
            if isinstance(job.salary, tuple | list) and len(job.salary) >= 2:
                salary_str = f"{job.salary[0] or ''}-{job.salary[1] or ''}"
            else:
                salary_str = str(job.salary)
            content_parts.append(salary_str)
        if job.posted_date:
            # Normalize timezone for consistent hashing
            naive_date = (
                job.posted_date.replace(tzinfo=None)
                if job.posted_date.tzinfo
                else job.posted_date
            )
            content_parts.append(naive_date.isoformat())

        expected_hash = hashlib.md5("".join(content_parts).encode("utf-8")).hexdigest()  # noqa: S324
        assert hash_result == expected_hash


class TestSyncStatisticsAndUtilities:
    """Test statistics and utility methods."""

    def test_get_sync_statistics(self, test_session, sample_jobs):
        """Test get_sync_statistics method."""
        engine = SmartSyncEngine(session=test_session)

        stats = engine.get_sync_statistics()

        # Verify expected statistics structure
        expected_keys = {
            "total_jobs",
            "active_jobs",
            "archived_jobs",
            "favorited_jobs",
            "applied_jobs",
        }
        assert set(stats.keys()) == expected_keys

        # Verify counts are reasonable
        assert stats["total_jobs"] >= stats["active_jobs"] + stats["archived_jobs"]
        assert stats["favorited_jobs"] <= stats["total_jobs"]
        assert stats["applied_jobs"] <= stats["total_jobs"]

        # Count manually and verify
        total_jobs = len(sample_jobs)
        active_jobs = len([j for j in sample_jobs if not j.archived])
        archived_jobs = len([j for j in sample_jobs if j.archived])
        favorited_jobs = len([j for j in sample_jobs if j.favorite])
        applied_jobs = len([j for j in sample_jobs if j.application_status != "New"])

        assert stats["total_jobs"] == total_jobs
        assert stats["active_jobs"] == active_jobs
        assert stats["archived_jobs"] == archived_jobs
        assert stats["favorited_jobs"] == favorited_jobs
        assert stats["applied_jobs"] == applied_jobs

    def test_cleanup_old_jobs(self, test_session, sample_companies):
        """Test cleanup_old_jobs method."""
        engine = SmartSyncEngine(session=test_session)

        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Create old archived jobs
        old_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Very Old Job 1",
                description="Old job 1",
                link="https://example.com/old-1",
                location="Old Location",
                salary=(80000, 100000),
                archived=True,
                last_seen=old_date,
                application_date=None,  # No recent application
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Very Old Job 2",
                description="Old job 2",
                link="https://example.com/old-2",
                location="Old Location",
                salary=(85000, 105000),
                archived=True,
                last_seen=old_date,
                application_date=old_date - timedelta(days=10),  # Old application
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Old Job with Recent Application",
                description="Old job with recent app",
                link="https://example.com/old-recent",
                location="Old Location",
                salary=(90000, 110000),
                archived=True,
                last_seen=old_date,
                application_date=datetime.now(timezone.utc)
                - timedelta(days=10),  # Recent app
            ),
        ]

        for job in old_jobs:
            test_session.add(job)
        test_session.commit()

        # Cleanup jobs older than 90 days
        deleted_count = engine.cleanup_old_jobs(days_threshold=90)

        # Should delete 2 jobs (first two), but keep the one with recent application
        assert deleted_count == 2

        remaining_jobs = test_session.exec(
            select(JobSQL).where(
                JobSQL.title.in_(
                    [
                        "Very Old Job 1",
                        "Very Old Job 2",
                        "Old Job with Recent Application",
                    ]
                )
            )
        ).all()

        assert len(remaining_jobs) == 1
        assert remaining_jobs[0].title == "Old Job with Recent Application"

    def test_cleanup_old_jobs_error_handling(self, test_session):
        """Test cleanup_old_jobs handles errors properly."""
        engine = SmartSyncEngine(session=test_session)

        # Mock session.exec to raise an exception
        with (
            patch.object(test_session, "exec", side_effect=Exception("Cleanup error")),
            pytest.raises(Exception, match="Cleanup error"),
        ):
            engine.cleanup_old_jobs()


class TestConcurrencyAndEdgeCases:
    """Test concurrent operations and edge cases."""

    def test_sync_jobs_concurrent_modifications(
        self, test_session, sample_jobs, sample_companies
    ):
        """Test sync behavior with concurrent job modifications."""
        engine = SmartSyncEngine(session=test_session)

        # Simulate concurrent modification by updating a job during sync
        existing_job = sample_jobs[0]

        def modify_job_during_sync():
            # Modify the job in the database during sync
            concurrent_job = test_session.exec(
                select(JobSQL).where(JobSQL.id == existing_job.id)
            ).first()
            concurrent_job.notes = "Modified during sync"
            test_session.commit()

            # Return the original sync operation result
            return "updated"

        with patch.object(
            engine, "_sync_single_job_optimized", side_effect=modify_job_during_sync
        ):
            updated_job = JobSQL(
                company_id=existing_job.company_id,
                title="Updated Title",
                description=existing_job.description,
                link=existing_job.link,
                location=existing_job.location,
                salary=existing_job.salary,
            )

            result = engine.sync_jobs([updated_job])

            # Sync should still complete successfully
            assert result["updated"] == 1

    def test_sync_jobs_with_duplicate_links(self, test_session, sample_companies):
        """Test sync behavior with duplicate job links."""
        engine = SmartSyncEngine(session=test_session)

        duplicate_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Job Version 1",
                description="First version",
                link="https://example.com/duplicate",
                location="Location 1",
                salary=(100000, 130000),
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="Job Version 2",
                description="Second version",
                link="https://example.com/duplicate",  # Same link
                location="Location 2",
                salary=(110000, 140000),
            ),
        ]

        # The second job should update the first due to same link
        result = engine.sync_jobs(duplicate_jobs)

        # Should result in one insert and one update/skip
        assert result["inserted"] + result["updated"] + result["skipped"] == 2

        # Verify only one job exists with this link
        jobs_with_link = test_session.exec(
            select(JobSQL).where(JobSQL.link == "https://example.com/duplicate")
        ).all()

        assert len(jobs_with_link) == 1

    def test_sync_jobs_bulk_optimization(self, test_session, sample_companies):
        """Test that bulk loading optimization works correctly."""
        engine = SmartSyncEngine(session=test_session)

        # Create many jobs to test bulk loading
        many_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title=f"Bulk Job {i}",
                description=f"Bulk job description {i}",
                link=f"https://example.com/bulk-{i}",
                location="Bulk Location",
                salary=(100000 + i * 1000, 130000 + i * 1000),
            )
            for i in range(100)
        ]

        # Sync should handle large batches efficiently
        result = engine.sync_jobs(many_jobs)

        assert result["inserted"] == 100
        assert result["updated"] == 0
        assert result["skipped"] == 0

    def test_sync_jobs_memory_efficiency(self, test_session, sample_companies):
        """Test that sync operations are memory efficient."""
        engine = SmartSyncEngine(session=test_session)

        # Create jobs with large descriptions to test memory usage
        large_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title=f"Large Job {i}",
                description="X" * 10000,  # Large description
                link=f"https://example.com/large-{i}",
                location="Large Location",
                salary=(100000, 130000),
            )
            for i in range(10)
        ]

        # Should handle large jobs without memory issues
        result = engine.sync_jobs(large_jobs)

        assert result["inserted"] == 10

    def test_session_management_edge_cases(self):
        """Test session management in edge cases."""
        # Test with None session initially
        engine = SmartSyncEngine(session=None)
        assert engine._session is None
        assert engine._session_owned

        # Test session closing behavior
        mock_session = Mock()
        engine._close_session_if_owned(mock_session)
        mock_session.close.assert_called_once()

        # Test with provided session
        real_session = Mock()
        engine_with_session = SmartSyncEngine(session=real_session)
        engine_with_session._close_session_if_owned(mock_session)
        # Should not close when session was provided
        mock_session.close.assert_called_once()  # Still just once from before


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_sync_jobs_database_connection_error(self, test_session):
        """Test handling of database connection errors."""
        engine = SmartSyncEngine()

        with (
            patch.object(
                engine, "_get_session", side_effect=Exception("Connection failed")
            ),
            pytest.raises(Exception, match="Connection failed"),
        ):
            engine.sync_jobs([])

    def test_sync_jobs_partial_failure_recovery(self, test_session, sample_companies):
        """Test recovery from partial failures during sync."""
        engine = SmartSyncEngine(session=test_session)

        jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Good Job",
                description="This will work",
                link="https://example.com/good",
                location="Good Location",
                salary=(100000, 130000),
            ),
            JobSQL(
                company_id=None,  # This might cause issues
                title="Problem Job",
                description="This might cause issues",
                link="https://example.com/problem",
                location="Problem Location",
                salary=(100000, 130000),
            ),
        ]

        # Mock to make second job fail
        call_count = 0

        def failing_insert(_session, _job):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second job fails
                raise RuntimeError("Insert failed")
            return "inserted"

        with (
            patch.object(engine, "_insert_new_job", side_effect=failing_insert),
            pytest.raises(Exception, match="Insert failed"),
        ):
            engine.sync_jobs(jobs)

        # Verify rollback - no jobs should be inserted
        inserted_jobs = test_session.exec(
            select(JobSQL).where(JobSQL.title.in_(["Good Job", "Problem Job"]))
        ).all()

        assert len(inserted_jobs) == 0

    def test_content_hash_generation_with_invalid_data(self, test_session):
        """Test content hash generation with invalid or edge case data."""
        engine = SmartSyncEngine(session=test_session)

        # Test with various problematic data
        edge_case_jobs = [
            # Job with unicode characters
            JobSQL(
                title="Job with émojis 🚀",
                description="Ünicödé description with spëcial chars",
                location="São Paulo, BR",
                link="https://example.com/unicode",
                salary=(100000, 130000),
            ),
            # Job with very long strings
            JobSQL(
                title="x" * 1000,
                description="y" * 10000,
                location="z" * 500,
                link="https://example.com/long",
                salary=(100000, 130000),
            ),
        ]

        for job in edge_case_jobs:
            # Should not raise exceptions
            hash_result = engine._generate_content_hash(job)
            assert isinstance(hash_result, str)
            assert len(hash_result) == 32
