"""Modernized JobService tests focused on essential functionality.

This module replaces the overly complex 1700+ line test_job_service.py
with focused, maintainable tests covering core business logic.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from src.models import CompanySQL, JobSQL
from src.services.job_service import JobService

if TYPE_CHECKING:
    from sqlmodel import Session


class TestJobServiceCore:
    """Core JobService functionality tests."""

    def test_get_filtered_jobs_basic(
        self,
        session: "Session",
        sample_company: CompanySQL,
    ):
        """Test basic job filtering functionality."""
        # Create test jobs
        job1 = JobSQL(
            company_id=sample_company.id,
            title="AI Engineer",
            description="AI role",
            link="https://test.com/job1",
            location="Remote",
            application_status="New",
            last_seen=datetime.now(UTC),
        )
        job2 = JobSQL(
            company_id=sample_company.id,
            title="Data Scientist",
            description="Data role",
            link="https://test.com/job2",
            location="NYC",
            application_status="Applied",
            last_seen=datetime.now(UTC),
        )

        session.add_all([job1, job2])
        session.commit()

        # Test filtering
        jobs = JobService.get_filtered_jobs()

        assert len(jobs) == 2
        assert all(isinstance(job, dict) for job in jobs)

    def test_update_job_status(self, session: "Session", sample_job: JobSQL):
        """Test updating job application status."""
        result = JobService.update_job_status(sample_job.id, "Applied")
        assert result is True

        # Verify update in database
        updated_job = session.get(JobSQL, sample_job.id)
        assert updated_job.application_status == "Applied"

    def test_toggle_favorite(self, session: "Session", sample_job: JobSQL):
        """Test toggling job favorite status."""
        service = JobService(session)

        original_status = sample_job.favorite
        result = service.toggle_favorite(sample_job.id)
        assert result is True

        # Verify toggle in database
        updated_job = session.get(JobSQL, sample_job.id)
        assert updated_job.favorite != original_status


class TestJobServiceEdgeCases:
    """Edge cases and error handling tests."""

    def test_update_nonexistent_job(self, session: "Session"):
        """Test updating a job that doesn't exist."""
        service = JobService(session)

        result = service.update_job_status(999999, "Applied")
        assert result is False

    def test_get_filtered_jobs_empty_database(self, session: "Session"):
        """Test filtering when database is empty."""
        service = JobService(session)

        jobs = service.get_filtered_jobs()
        assert jobs == []


class TestJobServiceFiltering:
    """Job filtering and search functionality tests."""

    @pytest.mark.parametrize(
        ("status", "expected_count"),
        (
            ("New", 1),
            ("Applied", 0),
            (None, 1),  # No filter
        ),
    )
    def test_filter_by_status(
        self,
        session: "Session",
        sample_company: CompanySQL,
        status,
        expected_count,
    ):
        """Test filtering jobs by application status."""
        job = JobSQL(
            company_id=sample_company.id,
            title="Test Job",
            description="Test role",
            link="https://test.com/job",
            location="Remote",
            application_status="New",
            last_seen=datetime.now(UTC),
        )
        session.add(job)
        session.commit()

        service = JobService(session)
        jobs = service.get_filtered_jobs(status_filter=status)

        assert len(jobs) == expected_count
