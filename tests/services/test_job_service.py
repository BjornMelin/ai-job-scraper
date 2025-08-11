"""Comprehensive tests for JobService class.

This test suite validates JobService methods for real-world usage scenarios,
focusing on business functionality and Pydantic DTO conversion accuracy.
Tests cover filtering, state mutations, edge cases, and error conditions.
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import logging

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select
from src.models import CompanySQL, JobSQL
from src.schemas import Job
from src.services.job_service import JobService

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
            scrape_count=5,
            success_rate=0.8,
        ),
        CompanySQL(
            name="InnovateLabs",
            url="https://innovatelabs.com/jobs",
            active=True,
            scrape_count=3,
            success_rate=1.0,
        ),
        CompanySQL(
            name="DataDriven Inc",
            url="https://datadriven.com/careers",
            active=False,
            scrape_count=2,
            success_rate=0.5,
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
    """Create diverse sample jobs for comprehensive testing."""
    base_date = datetime.now(timezone.utc)

    jobs = [
        # TechCorp jobs
        JobSQL(
            company_id=sample_companies[0].id,
            title="Senior Python Developer",
            description=(
                "Looking for experienced Python developer with Django expertise. "
                "Work on scalable web applications."
            ),
            link="https://techcorp.com/jobs/python-dev-001",
            location="San Francisco, CA",
            posted_date=base_date - timedelta(days=1),
            salary=(120000, 160000),
            favorite=True,
            notes="Great benefits package, remote-friendly",
            content_hash="hash001",
            application_status="Applied",
            application_date=base_date - timedelta(hours=12),
        ),
        JobSQL(
            company_id=sample_companies[0].id,
            title="Machine Learning Engineer",
            description=(
                "Build and deploy ML models at scale. TensorFlow and PyTorch "
                "experience required."
            ),
            link="https://techcorp.com/jobs/ml-eng-002",
            location="New York, NY",
            posted_date=base_date - timedelta(days=3),
            salary=(140000, 180000),
            favorite=False,
            notes="",
            content_hash="hash002",
            application_status="Interested",
        ),
        # InnovateLabs jobs
        JobSQL(
            company_id=sample_companies[1].id,
            title="Full Stack Developer",
            description=(
                "JavaScript, React, and Node.js development. Building next-gen "
                "web applications."
            ),
            link="https://innovatelabs.com/jobs/fullstack-003",
            location="Austin, TX",
            posted_date=base_date - timedelta(days=5),
            salary=(100000, 140000),
            favorite=True,
            notes="Startup environment, equity options",
            content_hash="hash003",
            application_status="New",
        ),
        JobSQL(
            company_id=sample_companies[1].id,
            title="DevOps Engineer",
            description=(
                "AWS, Docker, Kubernetes experience. Manage cloud infrastructure "
                "and CI/CD pipelines."
            ),
            link="https://innovatelabs.com/jobs/devops-004",
            location="Remote",
            posted_date=base_date - timedelta(days=7),
            salary=(110000, 150000),
            favorite=False,
            notes="Remote-first culture",
            content_hash="hash004",
            application_status="Applied",
            application_date=base_date - timedelta(days=2),
        ),
        # DataDriven Inc jobs (inactive company)
        JobSQL(
            company_id=sample_companies[2].id,
            title="Data Scientist",
            description=(
                "Statistical modeling and machine learning for business insights. "
                "Python, R, SQL required."
            ),
            link="https://datadriven.com/jobs/data-sci-005",
            location="Seattle, WA",
            posted_date=base_date - timedelta(days=10),
            salary=(130000, 170000),
            favorite=False,
            notes="",
            content_hash="hash005",
            application_status="Rejected",
        ),
        # Archived job
        JobSQL(
            company_id=sample_companies[0].id,
            title="Archived Position",
            description="This job should not appear in normal queries.",
            link="https://techcorp.com/jobs/archived-006",
            location="Somewhere",
            posted_date=base_date - timedelta(days=30),
            salary=(80000, 100000),
            favorite=False,
            notes="Position was filled",
            content_hash="hash006",
            application_status="New",
            archived=True,
        ),
    ]

    for job in jobs:
        test_session.add(job)
    test_session.commit()

    # Refresh to get IDs and relationships
    for job in jobs:
        test_session.refresh(job)

    return jobs


@pytest.fixture
def mock_db_session(test_session):
    """Mock db_session context manager to use test session."""
    with patch("src.services.job_service.db_session") as mock_session:
        mock_session.return_value.__enter__.return_value = test_session
        mock_session.return_value.__exit__.return_value = None
        yield mock_session


class TestJobServiceFiltering:
    """Test JobService.get_filtered_jobs with various filter combinations."""

    def test_get_all_jobs_no_filters(self, mock_db_session, sample_jobs):
        """Test retrieving all non-archived jobs with empty filters."""
        filters = {}

        jobs = JobService.get_filtered_jobs(filters)

        # Should return all non-archived jobs (5 jobs, excluding archived)
        assert len(jobs) == 5
        assert all(isinstance(job, Job) for job in jobs)
        assert all(not job.archived for job in jobs)

        # Verify DTO conversion - should have company names as strings
        company_names = {job.company for job in jobs}
        expected_companies = {"TechCorp", "InnovateLabs", "DataDriven Inc"}
        assert company_names == expected_companies

        # Jobs should be ordered by posted_date desc
        job_dates = [job.posted_date for job in jobs if job.posted_date]
        assert job_dates == sorted(job_dates, reverse=True)

    def test_text_search_filter(self, mock_db_session, sample_jobs):
        """Test text search in job titles and descriptions."""
        # Search for "Python" - should match Senior Python Developer
        filters = {"text_search": "Python"}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 2  # Python developer + Data Scientist (mentions Python)
        titles = {job.title for job in jobs}
        assert "Senior Python Developer" in titles
        assert "Data Scientist" in titles

        # Search for "React" - should match Full Stack Developer
        filters = {"text_search": "React"}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 1
        assert jobs[0].title == "Full Stack Developer"

        # Case-insensitive search
        filters = {"text_search": "MACHINE LEARNING"}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 2  # ML Engineer + Data Scientist
        titles = {job.title for job in jobs}
        assert "Machine Learning Engineer" in titles
        assert "Data Scientist" in titles

    def test_company_filter(self, mock_db_session, sample_jobs):
        """Test filtering by company names."""
        # Filter by single company
        filters = {"company": ["TechCorp"]}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 2  # 2 TechCorp jobs (excluding archived)
        assert all(job.company == "TechCorp" for job in jobs)

        # Filter by multiple companies
        filters = {"company": ["TechCorp", "InnovateLabs"]}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 4
        company_names = {job.company for job in jobs}
        assert company_names == {"TechCorp", "InnovateLabs"}

        # "All" should return all jobs
        filters = {"company": ["All"]}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 5  # All non-archived jobs

    def test_application_status_filter(self, mock_db_session, sample_jobs):
        """Test filtering by application status."""
        # Filter by Applied status
        filters = {"application_status": ["Applied"]}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 2
        assert all(job.application_status == "Applied" for job in jobs)

        # Filter by multiple statuses
        filters = {"application_status": ["New", "Interested"]}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 2
        statuses = {job.application_status for job in jobs}
        assert statuses == {"New", "Interested"}

        # "All" should return all jobs
        filters = {"application_status": ["All"]}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 5

    def test_date_range_filter(self, mock_db_session, sample_jobs):
        """Test filtering by date ranges."""
        # Test basic date filtering functionality
        base_date = datetime.now(timezone.utc)

        # Test date_from filtering (jobs posted after a certain date)
        # Use a date that's definitely older than our sample data
        old_date = base_date - timedelta(days=30)
        filters = {"date_from": old_date.strftime("%Y-%m-%d")}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return all non-archived jobs since they're all newer
        assert len(jobs) == 5  # All non-archived jobs

        # Test date_to filtering (jobs posted before a certain date)
        # Use a future date to get all jobs
        future_date = base_date + timedelta(days=1)
        filters = {"date_to": future_date.strftime("%Y-%m-%d")}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return all non-archived jobs
        assert len(jobs) == 5  # All non-archived jobs

        # Test date_from with future date (should return no jobs)
        future_date = base_date + timedelta(days=10)
        filters = {"date_from": future_date.strftime("%Y-%m-%d")}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return no jobs since no jobs are posted in the future
        assert len(jobs) == 0

        # Test date_to with very old date (should return no jobs)
        very_old_date = base_date - timedelta(days=365)
        filters = {"date_to": very_old_date.strftime("%Y-%m-%d")}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return no jobs since no jobs are that old
        assert len(jobs) == 0

    def test_favorites_only_filter(self, mock_db_session, sample_jobs):
        """Test filtering for favorite jobs only."""
        filters = {"favorites_only": True}
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 2  # 2 favorite jobs
        assert all(job.favorite for job in jobs)

        # Verify specific favorite jobs
        titles = {job.title for job in jobs}
        expected_titles = {"Senior Python Developer", "Full Stack Developer"}
        assert titles == expected_titles

    def test_include_archived_filter(self, mock_db_session, sample_jobs):
        """Test including archived jobs in results."""
        # Default behavior excludes archived
        filters = {}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 5

        # Explicitly include archived jobs
        filters = {"include_archived": True}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 6  # Now includes the archived job

        archived_jobs = [job for job in jobs if job.archived]
        assert len(archived_jobs) == 1
        assert archived_jobs[0].title == "Archived Position"

    def test_combined_filters(self, mock_db_session, sample_jobs):
        """Test complex filter combinations."""
        # Search for Python jobs at TechCorp that are favorites
        filters = {
            "text_search": "Python",
            "company": ["TechCorp"],
            "favorites_only": True,
        }
        jobs = JobService.get_filtered_jobs(filters)

        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Senior Python Developer"
        assert job.company == "TechCorp"
        assert job.favorite
        assert "Python" in job.description

    def test_empty_results(self, mock_db_session, sample_jobs):
        """Test filters that return no results."""
        # Search for non-existent technology
        filters = {"text_search": "COBOL"}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0

        # Filter by non-existent company
        filters = {"company": ["NonExistentCorp"]}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0

        # Filter by non-existent status
        filters = {"application_status": ["NonExistentStatus"]}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0


class TestJobServiceStateMutations:
    """Test JobService methods that modify job state."""

    def test_update_job_status_success(self, mock_db_session, sample_jobs):
        """Test successful job status updates."""
        job = sample_jobs[0]  # Senior Python Developer

        # Update to different status
        result = JobService.update_job_status(job.id, "Interviewing")

        assert result is True

        # Verify the job was updated in database
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.application_status == "Interviewing"

    def test_update_job_status_to_applied_sets_date(self, mock_db_session, sample_jobs):
        """Test that updating status to Applied sets application_date."""
        job = sample_jobs[2]  # Full Stack Developer (status: New)
        assert job.application_date is None

        # Update to Applied status
        result = JobService.update_job_status(job.id, "Applied")

        assert result is True

        # Verify application_date was set
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.application_status == "Applied"
        assert updated_job.application_date is not None
        assert isinstance(updated_job.application_date, datetime)

    def test_update_job_status_preserves_existing_application_date(
        self, mock_db_session, sample_jobs
    ):
        """Test that existing application_date is preserved when re-applying."""
        job = sample_jobs[0]  # Already has application_date
        original_app_date = job.application_date

        # Change status away from Applied and back
        JobService.update_job_status(job.id, "Interviewing")
        JobService.update_job_status(job.id, "Applied")

        # Verify original application_date is preserved
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.application_date == original_app_date

    def test_update_job_status_invalid_id(self, mock_db_session, sample_jobs):
        """Test updating status for non-existent job ID."""
        result = JobService.update_job_status(99999, "Applied")

        assert result is False

    def test_toggle_favorite_success(self, mock_db_session, sample_jobs):
        """Test successful favorite toggling."""
        job = sample_jobs[1]  # ML Engineer (not favorite)
        original_favorite = job.favorite

        # Toggle favorite status
        result = JobService.toggle_favorite(job.id)

        assert result == (not original_favorite)

        # Verify the job was updated in database
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.favorite != original_favorite

    def test_toggle_favorite_multiple_times(self, mock_db_session, sample_jobs):
        """Test toggling favorite multiple times."""
        job = sample_jobs[0]  # Senior Python Developer (favorite)
        original_favorite = job.favorite

        # Toggle twice should return to original state
        result1 = JobService.toggle_favorite(job.id)
        result2 = JobService.toggle_favorite(job.id)

        assert result1 != original_favorite
        assert result2 == original_favorite

        # Verify final state matches original
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.favorite == original_favorite

    def test_toggle_favorite_invalid_id(self, mock_db_session, sample_jobs):
        """Test toggling favorite for non-existent job ID."""
        result = JobService.toggle_favorite(99999)

        assert result is False

    def test_update_notes_success(self, mock_db_session, sample_jobs):
        """Test successful notes update."""
        job = sample_jobs[1]  # ML Engineer (empty notes)
        new_notes = "Interesting role, need to research the tech stack more"

        result = JobService.update_notes(job.id, new_notes)

        assert result is True

        # Verify the job was updated in database
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.notes == new_notes

    def test_update_notes_empty_string(self, mock_db_session, sample_jobs):
        """Test updating notes to empty string."""
        job = sample_jobs[0]  # Has existing notes

        result = JobService.update_notes(job.id, "")

        assert result is True

        # Verify notes were cleared
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.notes == ""

    def test_update_notes_invalid_id(self, mock_db_session, sample_jobs):
        """Test updating notes for non-existent job ID."""
        result = JobService.update_notes(99999, "Some notes")

        assert result is False

    def test_archive_job_success(self, mock_db_session, sample_jobs):
        """Test successful job archiving."""
        job = sample_jobs[1]  # ML Engineer (not archived)

        result = JobService.archive_job(job.id)

        assert result is True

        # Verify the job was archived in database
        updated_job = mock_db_session.return_value.__enter__.return_value.exec(
            select(JobSQL).filter_by(id=job.id)
        ).first()
        assert updated_job.archived is True

    def test_archive_job_invalid_id(self, mock_db_session, sample_jobs):
        """Test archiving non-existent job ID."""
        result = JobService.archive_job(99999)

        assert result is False


class TestJobServiceQueries:
    """Test JobService query methods."""

    def test_get_job_by_id_success(self, mock_db_session, sample_jobs):
        """Test retrieving job by valid ID."""
        job_id = sample_jobs[0].id

        job = JobService.get_job_by_id(job_id)

        assert job is not None
        assert isinstance(job, Job)
        assert job.id == job_id
        assert job.title == "Senior Python Developer"
        assert job.company == "TechCorp"

        # Verify DTO conversion
        assert isinstance(job.company, str)
        assert job.salary == (120000, 160000)

    def test_get_job_by_id_not_found(self, mock_db_session, sample_jobs):
        """Test retrieving job by invalid ID."""
        job = JobService.get_job_by_id(99999)

        assert job is None

    def test_get_job_counts_by_status(self, mock_db_session, sample_jobs):
        """Test job count aggregation by status."""
        counts = JobService.get_job_counts_by_status()

        # Expected counts from non-archived jobs
        expected_counts = {
            "Applied": 2,
            "Interested": 1,
            "New": 1,
            "Rejected": 1,
        }

        assert counts == expected_counts

    def test_get_active_companies(self, mock_db_session, sample_companies):
        """Test retrieving active company names."""
        companies = JobService.get_active_companies()

        # Should return only active companies, sorted by name
        expected_companies = ["InnovateLabs", "TechCorp"]
        assert companies == expected_companies


class TestJobServiceDTOConversion:
    """Test proper DTO conversion and relationship handling."""

    def test_jobs_returned_as_pydantic_dtos(self, mock_db_session, sample_jobs):
        """Test that get_filtered_jobs returns Pydantic Job DTOs."""
        jobs = JobService.get_filtered_jobs({})

        for job in jobs:
            # Verify it's a Pydantic Job, not SQLModel JobSQL
            assert isinstance(job, Job)
            assert not isinstance(job, JobSQL)

            # Verify all expected fields are present
            assert hasattr(job, "id")
            assert hasattr(job, "company")
            assert hasattr(job, "title")
            assert hasattr(job, "description")

            # Verify company is a string, not relationship object
            assert isinstance(job.company, str)
            assert job.company in ["TechCorp", "InnovateLabs", "DataDriven Inc"]

    def test_get_job_by_id_returns_dto(self, mock_db_session, sample_jobs):
        """Test that get_job_by_id returns Pydantic DTO."""
        job_id = sample_jobs[0].id
        job = JobService.get_job_by_id(job_id)

        assert isinstance(job, Job)
        assert not isinstance(job, JobSQL)
        assert isinstance(job.company, str)
        assert job.company == "TechCorp"

    def test_job_with_no_company_relationship(self, mock_db_session, test_session):
        """Test handling of job with null company relationship."""
        # Create job without company_id
        job = JobSQL(
            company_id=None,
            title="Freelance Job",
            description="Independent contractor position",
            link="https://example.com/freelance",
            location="Remote",
            content_hash="hash_freelance",
        )
        test_session.add(job)
        test_session.commit()
        test_session.refresh(job)

        # Should handle gracefully and return "Unknown" for company
        result_job = JobService.get_job_by_id(job.id)

        assert result_job is not None
        assert result_job.company == "Unknown"


class TestJobServiceErrorHandling:
    """Test error handling and edge cases."""

    def test_database_error_during_filtering(self, sample_jobs):
        """Test handling of database errors during filtering."""
        with patch("src.services.job_service.db_session") as mock_session:
            # Simulate database error
            mock_session.side_effect = Exception("Database connection failed")

            with pytest.raises(Exception, match="Database connection failed"):
                JobService.get_filtered_jobs({})

    def test_database_error_during_status_update(self, sample_jobs):
        """Test handling of database errors during status updates."""
        with patch("src.services.job_service.db_session") as mock_session:
            # Simulate database error
            mock_session.side_effect = Exception("Database write failed")

            with pytest.raises(Exception, match="Database write failed"):
                JobService.update_job_status(1, "Applied")

    def test_invalid_date_formats(self, mock_db_session, sample_jobs):
        """Test handling of invalid date formats in filters."""
        # Invalid date format should be ignored gracefully
        filters = {
            "date_from": "invalid-date-format",
            "date_to": "also-invalid",
        }

        # Should not raise exception
        jobs = JobService.get_filtered_jobs(filters)

        # Should return all jobs since date filters are ignored
        assert len(jobs) == 5

    def test_empty_and_none_filters(self, mock_db_session, sample_jobs):
        """Test handling of empty and None filter values."""
        filters = {
            "text_search": "",  # Empty string
            "company": [],  # Empty list
            "application_status": None,  # None value
            "date_from": None,
            "date_to": "",
            "favorites_only": False,
            "salary_min": None,
            "salary_max": None,
        }

        # Should handle gracefully and return all jobs
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 5

    def test_salary_min_filter(self, mock_db_session, sample_jobs):
        """Test filtering by minimum salary."""
        # Filter for jobs where max salary >= 170,000 (jobs that can pay at least 170k)
        filters = {"salary_min": 170000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should include: ML Engineer (140k-180k), Data Scientist (130k-170k)
        # Should exclude: Full Stack (100k-140k), DevOps (110k-150k), Python (120k-160k)
        assert len(jobs) == 2
        titles = {job.title for job in jobs}
        assert "Machine Learning Engineer" in titles
        assert "Data Scientist" in titles

    def test_salary_min_filter_boundary(self, mock_db_session, sample_jobs):
        """Test filtering by minimum salary where job max salary equals filter."""
        # Filter where filter minimum exactly matches job's max salary
        filters = {"salary_min": 170000}  # Data Scientist has max_salary=170000
        jobs = JobService.get_filtered_jobs(filters)

        # Should include jobs where max_salary == 170,000
        job_titles = {job.title for job in jobs}
        assert "Data Scientist" in job_titles

        # Test with partial overlap
        filters = {
            "salary_min": 140000
        }  # Should catch Full Stack (100k-140k) at boundary
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert (
            "Full Stack Developer" in job_titles
        )  # min_salary < 140k but max_salary >= 140k
        assert "Machine Learning Engineer" in job_titles  # Also qualifies
        assert "Data Scientist" in job_titles  # Also qualifies

    def test_salary_max_filter(self, mock_db_session, sample_jobs):
        """Test filtering by maximum salary."""
        # Filter for jobs where min salary <= 110,000 (jobs I can get with 110k exp)
        filters = {"salary_max": 110000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should include: Full Stack (100k-140k), DevOps (110k-150k)
        # Should exclude: Python (120k-160k), ML Engineer (140k-180k), Data Scientist
        assert len(jobs) == 2
        titles = {job.title for job in jobs}
        assert "Full Stack Developer" in titles
        assert "DevOps Engineer" in titles

    def test_salary_max_filter_boundary(self, mock_db_session, sample_jobs):
        """Test filtering by maximum salary where job min salary equals filter."""
        # Filter where filter maximum exactly matches job's min salary
        filters = {"salary_max": 110000}  # DevOps has min_salary=110000
        jobs = JobService.get_filtered_jobs(filters)

        # Should include jobs where min_salary == 110,000
        job_titles = {job.title for job in jobs}
        assert "DevOps Engineer" in job_titles
        assert "Full Stack Developer" in job_titles  # min_salary=100k <= 110k

    def test_salary_range_filter(self, mock_db_session, sample_jobs):
        """Test filtering by both minimum and maximum salary."""
        # Filter for jobs where I can earn 140-160k (max >= 140k AND min <= 160k)
        filters = {"salary_min": 140000, "salary_max": 160000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should include:
        # - Full Stack (100k-140k): max 140k >= 140k ✓, min 100k <= 160k ✓
        # - DevOps (110k-150k): max 150k >= 140k ✓, min 110k <= 160k ✓
        # - Python (120k-160k): max 160k >= 140k ✓, min 120k <= 160k ✓
        # - ML Engineer (140k-180k): max 180k >= 140k ✓, min 140k <= 160k ✓
        # - Data Scientist (130k-170k): max 170k >= 140k ✓, min 130k <= 160k ✓
        # All should match!
        assert len(jobs) == 5

    def test_salary_filter_with_none_values(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test salary filtering handles None salary values gracefully."""
        # Create a job with None salary
        job_with_none_salary = JobSQL(
            company_id=sample_companies[0].id,
            title="Internship Position",
            description="Unpaid internship opportunity",
            link="https://example.com/internship",
            location="Remote",
            salary=(None, None),
            content_hash="hash_internship",
        )
        test_session.add(job_with_none_salary)
        test_session.commit()
        test_session.refresh(job_with_none_salary)

        # Filter with salary_min should exclude the None salary job
        filters = {"salary_min": 50000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should not include the internship with None salary
        job_titles = {job.title for job in jobs}
        assert "Internship Position" not in job_titles

    def test_salary_filter_with_partial_none_values(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test salary filtering handles partially None salary values correctly."""
        # Create jobs with various None salary scenarios
        job_with_none_min = JobSQL(
            company_id=sample_companies[0].id,
            title="Min Salary None",
            description="Job with min_salary=None",
            link="https://example.com/min_none",
            location="Remote",
            salary=(None, 120000),
            content_hash="hash_min_none",
        )
        job_with_none_max = JobSQL(
            company_id=sample_companies[0].id,
            title="Max Salary None",
            description="Job with max_salary=None",
            link="https://example.com/max_none",
            location="Remote",
            salary=(90000, None),
            content_hash="hash_max_none",
        )

        test_session.add(job_with_none_min)
        test_session.add(job_with_none_max)
        test_session.commit()
        test_session.refresh(job_with_none_min)
        test_session.refresh(job_with_none_max)

        # Test job with None min_salary: should match if max_salary >= salary_min
        filters = {"salary_min": 100000}  # max_salary (120k) >= 100k
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Min Salary None" in job_titles

        filters = {"salary_min": 130000}  # max_salary (120k) < 130k
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Min Salary None" not in job_titles

        # Test job with None max_salary: should match if min_salary <= salary_max
        filters = {"salary_max": 100000}  # min_salary (90k) <= 100k
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Max Salary None" in job_titles

        filters = {"salary_max": 80000}  # min_salary (90k) > 80k
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Max Salary None" not in job_titles

    def test_salary_filter_edge_cases(self, mock_db_session, sample_jobs):
        """Test salary filtering edge cases."""
        # Very high minimum salary - should return no jobs
        filters = {"salary_min": 500000}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0

        # Very low maximum salary - should return no jobs
        filters = {"salary_max": 10000}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0

        # Zero values should be handled gracefully
        filters = {"salary_min": 0, "salary_max": 0}
        jobs = JobService.get_filtered_jobs(filters)
        # With salary_max = 0, no jobs should match (no job has min salary <= 0)
        assert len(jobs) == 0

    def test_high_value_salary_unbounded_filtering(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test that salary_max=750000 acts as unbounded for high-value positions."""
        # Create high-value jobs above 750k
        high_value_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="VP of Engineering",
                description="Executive leadership position",
                link="https://techcorp.com/jobs/vp-eng",
                location="San Francisco, CA",
                posted_date=datetime.now(timezone.utc) - timedelta(days=1),
                salary=(800000, 1200000),
                content_hash="hash_vp",
                application_status="New",
            ),
            JobSQL(
                company_id=sample_companies[1].id,
                title="Principal Staff Engineer",
                description="Senior technical leadership role",
                link="https://innovatelabs.com/jobs/principal-staff",
                location="Seattle, WA",
                posted_date=datetime.now(timezone.utc) - timedelta(days=2),
                salary=(750000, 950000),
                content_hash="hash_principal",
                application_status="New",
            ),
        ]

        for job in high_value_jobs:
            test_session.add(job)
        test_session.commit()

        # Test 1: salary_max < 750000 should apply upper bound filtering
        filters = {"salary_max": 600000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should exclude all high-value jobs (none have min salary <= 600k)
        job_titles = {job.title for job in jobs}
        assert "VP of Engineering" not in job_titles
        assert "Principal Staff Engineer" not in job_titles

        # Test 2: salary_max = 750000 should include ALL jobs (unbounded)
        filters = {"salary_max": 750000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should include all jobs, including high-value ones
        job_titles = {job.title for job in jobs}
        assert "VP of Engineering" in job_titles
        assert "Principal Staff Engineer" in job_titles
        assert len(jobs) >= 2  # At minimum the 2 high-value jobs we created

        # Test 3: salary_min with unbounded max
        filters = {"salary_min": 500000, "salary_max": 750000}
        jobs = JobService.get_filtered_jobs(filters)

        # Should include high-value jobs where max salary >= 500k
        job_titles = {job.title for job in jobs}
        assert "VP of Engineering" in job_titles
        assert "Principal Staff Engineer" in job_titles
        assert len(jobs) == 2  # Only the high-value jobs meet this criteria

    def test_salary_zero_minimum_handling(self, mock_db_session, sample_jobs):
        """Test that salary_min=0 is properly ignored."""
        # Test that salary_min=0 doesn't filter anything
        filters = {"salary_min": 0}
        jobs_with_zero = JobService.get_filtered_jobs(filters)

        # Should be same as no salary filter
        filters_no_salary = {}
        jobs_without_filter = JobService.get_filtered_jobs(filters_no_salary)

        assert len(jobs_with_zero) == len(jobs_without_filter)

        # Test that salary_min=1 does apply filtering
        filters = {"salary_min": 1}
        jobs_with_one = JobService.get_filtered_jobs(filters)

        # All sample jobs have salaries > 1, so should be same result
        assert len(jobs_with_one) == len(jobs_without_filter)

    def test_salary_combined_with_other_filters(self, mock_db_session, sample_jobs):
        """Test salary filtering combined with other filter types."""
        # Combine salary filter with company filter
        filters = {
            "salary_min": 100000,
            "company": ["TechCorp"],
        }
        jobs = JobService.get_filtered_jobs(filters)

        # Should include both TechCorp jobs (both have salaries >= 100k)
        assert len(jobs) == 2
        assert all(job.company == "TechCorp" for job in jobs)
        titles = {job.title for job in jobs}
        assert titles == {"Senior Python Developer", "Machine Learning Engineer"}

        # Combine salary with text search and status
        filters = {
            "salary_max": 150000,
            "text_search": "Developer",
            "application_status": ["New"],
        }
        jobs = JobService.get_filtered_jobs(filters)

        # Should include Full Stack Developer (New status, max 140k, "Developer")
        assert len(jobs) == 1
        assert jobs[0].title == "Full Stack Developer"
        assert jobs[0].application_status == "New"

    def test_salary_combined_with_archived_jobs(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test salary filtering combined with archived jobs."""
        # Create an archived job with high salary
        archived_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Archived High Salary Position",
            description="Archived executive role",
            link="https://example.com/archived",
            location="San Francisco, CA",
            salary=(200000, 300000),
            content_hash="hash_archived",
            archived=True,
        )
        test_session.add(archived_job)
        test_session.commit()

        # Test salary filter normally excludes archived
        filters = {"salary_min": 200000}
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Archived High Salary Position" not in job_titles

        # Test salary filter with archived included
        filters = {"salary_min": 200000, "include_archived": True}
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Archived High Salary Position" in job_titles

    def test_salary_combined_with_date_filters(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test salary filtering combined with date range filters."""
        from datetime import datetime, timedelta, timezone

        # Create a job with recent date and good salary
        recent_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Recent High Paying Job",
            description="Recently posted executive role",
            link="https://example.com/recent",
            location="San Francisco, CA",
            posted_date=datetime.now(timezone.utc) - timedelta(days=5),
            salary=(180000, 220000),
            content_hash="hash_recent",
        )

        # Create an old job with good salary
        old_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Old High Paying Job",
            description="Old executive role",
            link="https://example.com/old",
            location="San Francisco, CA",
            posted_date=datetime.now(timezone.utc) - timedelta(days=60),
            salary=(180000, 220000),
            content_hash="hash_old",
        )

        test_session.add(recent_job)
        test_session.add(old_job)
        test_session.commit()

        # Test salary + recent date filter
        filters = {
            "salary_min": 180000,
            "date_from": datetime.now(timezone.utc) - timedelta(days=30),
        }
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Recent High Paying Job" in job_titles
        assert "Old High Paying Job" not in job_titles

    def test_salary_combined_with_favorites(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test salary filtering combined with favorites filter."""
        # Create a favorite job with good salary
        favorite_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Favorite High Paying Job",
            description="Favorite executive role",
            link="https://example.com/favorite",
            location="San Francisco, CA",
            salary=(180000, 220000),
            content_hash="hash_favorite",
            favorite=True,
        )
        test_session.add(favorite_job)
        test_session.commit()

        # Test salary + favorites filter
        filters = {"salary_min": 180000, "favorites_only": True}
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Favorite High Paying Job" in job_titles
        assert len(jobs) == 1  # Only the favorite should match


class TestJobServiceHighValueSalaries:
    """Test suite specifically for high-value salary functionality."""

    def test_format_salary_display_helper(self):
        """Test salary formatting helper for UI display."""
        from src.ui.utils.formatters import format_salary

        # Test various amounts using the shared utility
        assert format_salary(0) == "$0"
        assert format_salary(500) == "$500"
        assert format_salary(75000) == "$75k"
        assert format_salary(150000) == "$150k"
        assert format_salary(750000) == "$750k"
        assert format_salary(1000000) == "$1.0M"
        assert format_salary(1500000) == "$1.5M"
        assert format_salary(2300000) == "$2.3M"

    def test_unbounded_range_scenarios(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test various unbounded range scenarios for comprehensive coverage."""
        # Create jobs at the boundary
        boundary_jobs = [
            JobSQL(
                company_id=sample_companies[0].id,
                title="Director Level (750k exact)",
                description="Director position exactly at boundary",
                link="https://techcorp.com/jobs/director-750",
                location="San Francisco, CA",
                posted_date=datetime.now(timezone.utc) - timedelta(days=1),
                salary=(500000, 750000),
                content_hash="hash_dir750",
                application_status="New",
            ),
            JobSQL(
                company_id=sample_companies[0].id,
                title="SVP Level (above boundary)",
                description="Senior VP position above boundary",
                link="https://techcorp.com/jobs/svp-high",
                location="San Francisco, CA",
                posted_date=datetime.now(timezone.utc) - timedelta(days=1),
                salary=(750001, 1500000),
                content_hash="hash_svp",
                application_status="New",
            ),
        ]

        for job in boundary_jobs:
            test_session.add(job)
        test_session.commit()

        # Test boundary conditions
        filters = {"salary_max": 749999}  # Just below boundary
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        # Director job (500k-750k) should be INCLUDED because min salary 500k <= 749999
        # SVP job (750001-1500000) should be EXCLUDED because min salary 750001 > 749999
        assert "Director Level (750k exact)" in job_titles
        assert "SVP Level (above boundary)" not in job_titles

        filters = {"salary_max": 750000}  # Exactly at boundary (unbounded)
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Director Level (750k exact)" in job_titles
        assert "SVP Level (above boundary)" in job_titles

    def test_salary_max_just_above_unbounded_threshold(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test salary_max set just above the unbounded threshold."""
        from src.constants import SALARY_UNBOUNDED_THRESHOLD

        # Create a job with salary above the threshold
        high_value_job = JobSQL(
            company_id=sample_companies[0].id,
            title="Ultra High Value Position",
            description="Position with salary above threshold + 1",
            link="https://example.com/ultra-high",
            location="San Francisco, CA",
            salary=(SALARY_UNBOUNDED_THRESHOLD + 1, 2000000),
            content_hash="hash_ultra_high",
        )
        test_session.add(high_value_job)
        test_session.commit()

        # Test salary_max just above threshold (should still be unbounded)
        filters = {"salary_max": SALARY_UNBOUNDED_THRESHOLD + 1}
        jobs = JobService.get_filtered_jobs(filters)
        job_titles = {job.title for job in jobs}
        assert "Ultra High Value Position" in job_titles


class TestJobServiceDateParsing:
    """Test JobService._parse_date method with various formats."""

    def test_parse_iso_date_formats(self):
        """Test parsing ISO date formats."""
        # ISO date string
        result = JobService._parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

        # ISO datetime string
        result = JobService._parse_date("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_common_date_formats(self):
        """Test parsing common date formats."""
        # US format
        result = JobService._parse_date("01/15/2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

        # EU format
        result = JobService._parse_date("15/01/2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

        # Human readable format
        result = JobService._parse_date("January 15, 2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_invalid_dates(self):
        """Test handling of invalid date formats."""
        # Invalid format
        result = JobService._parse_date("not-a-date")
        assert result is None

        # Empty string
        result = JobService._parse_date("")
        assert result is None

        # None input
        result = JobService._parse_date(None)
        assert result is None

        # Unsupported type
        result = JobService._parse_date(12345)
        assert result is None


class TestJobServiceIntegration:
    """Integration tests combining multiple JobService operations."""

    def test_job_lifecycle_workflow(self, mock_db_session, sample_jobs):
        """Test complete job lifecycle: view -> favorite -> apply -> archive."""
        job_id = sample_jobs[1].id  # ML Engineer

        # 1. Get job details
        job = JobService.get_job_by_id(job_id)
        assert job is not None
        assert job.application_status == "Interested"
        assert not job.favorite

        # 2. Toggle favorite
        result = JobService.toggle_favorite(job_id)
        assert result is True

        # 3. Add notes
        result = JobService.update_notes(
            job_id, "Looks promising, need to prepare for ML interview"
        )
        assert result is True

        # 4. Apply for the job
        result = JobService.update_job_status(job_id, "Applied")
        assert result is True

        # 5. Verify final state
        updated_job = JobService.get_job_by_id(job_id)
        assert updated_job.favorite is True
        assert updated_job.application_status == "Applied"
        assert updated_job.application_date is not None
        assert updated_job.notes == "Looks promising, need to prepare for ML interview"

        # 6. Archive the job
        result = JobService.archive_job(job_id)
        assert result is True

        # 7. Verify job no longer appears in default filtering
        jobs = JobService.get_filtered_jobs({})
        job_ids = [j.id for j in jobs]
        assert job_id not in job_ids

        # 8. But appears when including archived
        jobs = JobService.get_filtered_jobs({"include_archived": True})
        archived_job = next((j for j in jobs if j.id == job_id), None)
        assert archived_job is not None
        assert archived_job.archived is True

    def test_search_and_filter_workflow(self, mock_db_session, sample_jobs):
        """Test realistic search and filter workflow."""
        # User searches for "Python" jobs
        jobs = JobService.get_filtered_jobs({"text_search": "Python"})
        assert len(jobs) == 2

        # User narrows down to TechCorp only
        jobs = JobService.get_filtered_jobs(
            {"text_search": "Python", "company": ["TechCorp"]}
        )
        assert len(jobs) == 1
        assert jobs[0].title == "Senior Python Developer"

        # User filters for favorites only
        jobs = JobService.get_filtered_jobs(
            {"text_search": "Python", "company": ["TechCorp"], "favorites_only": True}
        )
        assert len(jobs) == 1  # The Python job is already favorite

        # User checks application status distribution
        counts = JobService.get_job_counts_by_status()
        assert counts["Applied"] >= 1

    def test_high_value_job_workflow(
        self, mock_db_session, test_session, sample_companies
    ):
        """Test workflow with high-value positions."""
        # Create a high-value position
        high_value_job = JobSQL(
            company_id=sample_companies[0].id,
            title="CTO Position",
            description="Chief Technology Officer role with equity",
            link="https://techcorp.com/jobs/cto",
            location="San Francisco, CA",
            posted_date=datetime.now(timezone.utc) - timedelta(days=1),
            salary=(900000, 1500000),
            content_hash="hash_cto",
            application_status="New",
        )
        test_session.add(high_value_job)
        test_session.commit()
        test_session.refresh(high_value_job)

        # User searches with high salary filter (unbounded)
        jobs = JobService.get_filtered_jobs(
            {
                "salary_min": 800000,
                "salary_max": 750000,  # This should be treated as unbounded
            }
        )

        # Should find the CTO position
        assert len(jobs) == 1
        assert jobs[0].title == "CTO Position"

        # User manages this high-value job through normal workflow
        job_id = jobs[0].id

        # Mark as favorite and add notes
        JobService.toggle_favorite(job_id)
        JobService.update_notes(
            job_id, "Incredible opportunity, prepare executive pitch"
        )

        # Verify changes
        updated_job = JobService.get_job_by_id(job_id)
        assert updated_job.favorite is True
        assert "executive pitch" in updated_job.notes.lower()


class TestJobServiceDatabaseOptimizedFiltering:
    """Test database-optimized filtering for UI tab functionality."""

    def test_get_filtered_jobs_favorites_only_database_query(
        self, mock_db_session, sample_jobs
    ):
        """Test favorites_only filter uses database query optimization."""
        # Test basic favorites filtering
        filters = {"favorites_only": True}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return only favorite jobs using database query
        assert len(jobs) == 2
        assert all(job.favorite for job in jobs)

        # Verify specific favorite jobs are returned
        favorite_titles = {job.title for job in jobs}
        expected_titles = {"Senior Python Developer", "Full Stack Developer"}
        assert favorite_titles == expected_titles

    def test_get_filtered_jobs_application_status_applied_database_query(
        self, mock_db_session, sample_jobs
    ):
        """Test application_status filtering for Applied jobs uses database query."""
        # Test filtering for Applied status
        filters = {"application_status": ["Applied"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return only applied jobs using database query
        assert len(jobs) == 2
        assert all(job.application_status == "Applied" for job in jobs)

        # Verify specific applied jobs
        applied_titles = {job.title for job in jobs}
        expected_titles = {"Senior Python Developer", "DevOps Engineer"}
        assert applied_titles == expected_titles

    def test_get_filtered_jobs_application_status_multiple_statuses(
        self, mock_db_session, sample_jobs
    ):
        """Test application_status filtering with multiple status values."""
        # Test filtering for multiple statuses
        filters = {"application_status": ["Applied", "Interested"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return jobs with either status
        assert len(jobs) == 3
        statuses = {job.application_status for job in jobs}
        assert statuses == {"Applied", "Interested"}

        # Verify we get the expected jobs
        titles = {job.title for job in jobs}
        expected_titles = {
            "Senior Python Developer",  # Applied
            "Machine Learning Engineer",  # Interested
            "DevOps Engineer",  # Applied
        }
        assert titles == expected_titles

    def test_get_filtered_jobs_application_status_single_value(
        self, mock_db_session, sample_jobs
    ):
        """Test application_status filtering with single status value."""
        # Test each status individually
        test_cases = [
            ("New", {"Full Stack Developer"}),
            ("Interested", {"Machine Learning Engineer"}),
            ("Applied", {"Senior Python Developer", "DevOps Engineer"}),
            ("Rejected", {"Data Scientist"}),
        ]

        for status, expected_titles in test_cases:
            filters = {"application_status": [status]}
            jobs = JobService.get_filtered_jobs(filters)

            titles = {job.title for job in jobs}
            assert titles == expected_titles
            assert all(job.application_status == status for job in jobs)

    def test_get_filtered_jobs_combined_favorites_and_status(
        self, mock_db_session, sample_jobs
    ):
        """Test combining favorites_only with application_status filtering."""
        # Test favorite jobs that are also applied
        filters = {"favorites_only": True, "application_status": ["Applied"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return only favorite AND applied jobs
        assert len(jobs) == 1  # Senior Python Developer
        job = jobs[0]
        assert job.title == "Senior Python Developer"
        assert job.favorite is True
        assert job.application_status == "Applied"

        # Test favorite jobs with multiple statuses
        filters = {"favorites_only": True, "application_status": ["Applied", "New"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return favorite jobs with either Applied or New status
        assert len(jobs) == 2
        titles = {job.title for job in jobs}
        expected_titles = {"Senior Python Developer", "Full Stack Developer"}
        assert titles == expected_titles
        assert all(job.favorite for job in jobs)
        assert all(job.application_status in ["Applied", "New"] for job in jobs)

    def test_get_filtered_jobs_text_search_with_status_filter(
        self, mock_db_session, sample_jobs
    ):
        """Test combining text search with application status filtering."""
        # Search for "Developer" with Applied status
        filters = {"text_search": "Developer", "application_status": ["Applied"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should find Python Developer (Applied) but not Full Stack Developer (New)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Senior Python Developer"
        assert job.application_status == "Applied"
        assert "Developer" in job.title

        # Search for "Engineer" with multiple statuses
        filters = {
            "text_search": "Engineer",
            "application_status": ["Applied", "Interested"],
        }
        jobs = JobService.get_filtered_jobs(filters)

        # Should find ML Engineer (Interested) and DevOps Engineer (Applied)
        assert len(jobs) == 2
        titles = {job.title for job in jobs}
        expected_titles = {"Machine Learning Engineer", "DevOps Engineer"}
        assert titles == expected_titles

    def test_get_filtered_jobs_company_with_status_filter(
        self, mock_db_session, sample_jobs
    ):
        """Test combining company filter with application status filtering."""
        # Filter TechCorp jobs with Applied status
        filters = {"company": ["TechCorp"], "application_status": ["Applied"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should find only TechCorp Applied jobs
        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Senior Python Developer"
        assert job.company == "TechCorp"
        assert job.application_status == "Applied"

        # Filter InnovateLabs jobs with any status except Rejected
        filters = {
            "company": ["InnovateLabs"],
            "application_status": ["New", "Interested", "Applied"],
        }
        jobs = JobService.get_filtered_jobs(filters)

        # Should find both InnovateLabs jobs (they're New and Applied)
        assert len(jobs) == 2
        assert all(job.company == "InnovateLabs" for job in jobs)
        titles = {job.title for job in jobs}
        expected_titles = {"Full Stack Developer", "DevOps Engineer"}
        assert titles == expected_titles

    def test_get_filtered_jobs_date_range_with_status_filter(
        self, mock_db_session, sample_jobs
    ):
        """Test combining date filters with application status filtering."""
        from datetime import datetime, timedelta, timezone

        base_date = datetime.now(timezone.utc)

        # Filter recent jobs (last 4 days) with Applied status
        recent_date = base_date - timedelta(days=4)
        filters = {
            "date_from": recent_date.strftime("%Y-%m-%d"),
            "application_status": ["Applied"],
        }
        jobs = JobService.get_filtered_jobs(filters)

        # Should find recent Applied jobs
        applied_jobs = [job for job in jobs if job.application_status == "Applied"]
        assert applied_jobs  # At least Python Developer should match

        # All returned jobs should be recent and Applied
        assert all(job.application_status == "Applied" for job in jobs)

    def test_get_filtered_jobs_status_empty_list_returns_all(
        self, mock_db_session, sample_jobs
    ):
        """Test that empty application_status list returns all jobs."""
        filters = {"application_status": []}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return all non-archived jobs
        assert len(jobs) == 5

        # Should include all different statuses
        statuses = {job.application_status for job in jobs}
        expected_statuses = {"New", "Interested", "Applied", "Rejected"}
        assert statuses == expected_statuses

    def test_get_filtered_jobs_status_all_returns_all(
        self, mock_db_session, sample_jobs
    ):
        """Test that application_status=['All'] returns all jobs."""
        filters = {"application_status": ["All"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return all non-archived jobs
        assert len(jobs) == 5

        # Should include all different statuses
        statuses = {job.application_status for job in jobs}
        expected_statuses = {"New", "Interested", "Applied", "Rejected"}
        assert statuses == expected_statuses

    def test_get_filtered_jobs_nonexistent_status_returns_empty(
        self, mock_db_session, sample_jobs
    ):
        """Test filtering by non-existent application status."""
        filters = {"application_status": ["NonExistentStatus"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return no jobs
        assert len(jobs) == 0

    def test_get_filtered_jobs_mixed_valid_invalid_status(
        self, mock_db_session, sample_jobs
    ):
        """Test filtering with mix of valid and invalid status values."""
        filters = {"application_status": ["Applied", "NonExistentStatus", "New"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Should return jobs matching valid statuses only
        assert len(jobs) == 3  # Applied (2) + New (1) = 3
        statuses = {job.application_status for job in jobs}
        assert statuses == {"Applied", "New"}

    def test_get_filtered_jobs_case_sensitive_status_filter(
        self, mock_db_session, sample_jobs
    ):
        """Test that application status filtering is case sensitive."""
        # Test with incorrect case
        filters = {"application_status": ["applied"]}  # lowercase
        jobs = JobService.get_filtered_jobs(filters)

        # Should return no jobs (case sensitive)
        assert len(jobs) == 0

        # Test with correct case
        filters = {"application_status": ["Applied"]}  # correct case
        jobs = JobService.get_filtered_jobs(filters)

        # Should return Applied jobs
        assert len(jobs) == 2
        assert all(job.application_status == "Applied" for job in jobs)

    def test_get_filtered_jobs_performance_database_query_optimization(
        self, mock_db_session, sample_jobs
    ):
        """Test that status filtering uses database queries, not Python filtering."""
        from unittest.mock import MagicMock

        # Mock session.exec to verify query contains WHERE clause for status
        original_exec = mock_db_session.return_value.__enter__.return_value.exec
        query_strings = []

        def capture_exec(statement, *args, **kwargs):
            query_strings.append(str(statement))
            return original_exec(statement, *args, **kwargs)

        mock_db_session.return_value.__enter__.return_value.exec = MagicMock(
            side_effect=capture_exec
        )

        # Filter by specific status
        filters = {"application_status": ["Applied"]}
        jobs = JobService.get_filtered_jobs(filters)

        # Verify that the query contains application_status filtering
        assert query_strings
        query_string = " ".join(query_strings)

        # The query should include a WHERE clause for application_status
        assert "application_status" in query_string.lower()

        # Should still return the correct results
        assert len(jobs) == 2
        assert all(job.application_status == "Applied" for job in jobs)

    def test_favorites_and_applied_tab_optimization_simulation(
        self, mock_db_session, sample_jobs
    ):
        """Test simulating the UI tab filtering optimization from jobs.py."""
        # Simulate _get_favorites_jobs() helper function
        favorites_filters = {
            "text_search": "",  # Empty search from session state
            "company": [],  # No company filter
            "application_status": [],  # No status filter for favorites tab
            "date_from": None,
            "date_to": None,
            "favorites_only": True,  # Database-level filtering for favorites
            "include_archived": False,
        }

        favorites_jobs = JobService.get_filtered_jobs(favorites_filters)

        # Should return only favorite jobs via database query
        assert len(favorites_jobs) == 2
        assert all(job.favorite for job in favorites_jobs)
        favorite_titles = {job.title for job in favorites_jobs}
        assert favorite_titles == {"Senior Python Developer", "Full Stack Developer"}

        # Simulate _get_applied_jobs() helper function
        applied_filters = {
            "text_search": "",  # Empty search from session state
            "company": [],  # No company filter
            "application_status": ["Applied"],  # Database-level filtering for applied
            "date_from": None,
            "date_to": None,
            "favorites_only": False,
            "include_archived": False,
        }

        applied_jobs = JobService.get_filtered_jobs(applied_filters)

        # Should return only applied jobs via database query
        assert len(applied_jobs) == 2
        assert all(job.application_status == "Applied" for job in applied_jobs)
        applied_titles = {job.title for job in applied_jobs}
        assert applied_titles == {"Senior Python Developer", "DevOps Engineer"}

        # Test that they can be combined for intersection
        favorites_and_applied_filters = {
            "text_search": "",
            "company": [],
            "application_status": ["Applied"],
            "date_from": None,
            "date_to": None,
            "favorites_only": True,  # Both filters active
            "include_archived": False,
        }

        intersection_jobs = JobService.get_filtered_jobs(favorites_and_applied_filters)

        # Should return jobs that are both favorite AND applied
        assert len(intersection_jobs) == 1
        job = intersection_jobs[0]
        assert job.title == "Senior Python Developer"
        assert job.favorite is True
        assert job.application_status == "Applied"
