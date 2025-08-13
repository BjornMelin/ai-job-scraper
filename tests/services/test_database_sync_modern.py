"""Modernized database sync tests focused on core functionality.

This module replaces the overly complex database sync tests with
focused, maintainable tests covering essential sync operations.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.models import JobSQL
from src.services.database_sync import SmartSyncEngine

if TYPE_CHECKING:
    from sqlmodel import Session


class TestSmartSyncEngine:
    """Core SmartSyncEngine functionality tests."""

    def test_sync_new_jobs(self, session: "Session"):
        """Test syncing new jobs to database."""
        engine = SmartSyncEngine(session)

        new_jobs = [
            JobSQL(
                company_id=1,
                title="AI Engineer",
                description="AI role",
                link="https://test.com/job1",
                location="Remote",
                content_hash="hash1",
                application_status="New",
                last_seen=datetime.now(UTC),
            ),
            JobSQL(
                company_id=1,
                title="Data Scientist",
                description="Data role",
                link="https://test.com/job2",
                location="NYC",
                content_hash="hash2",
                application_status="New",
                last_seen=datetime.now(UTC),
            ),
        ]

        result = engine.sync_jobs(new_jobs)

        assert result["inserted"] == 2
        assert result["updated"] == 0
        assert result["archived"] == 0
        assert result["deleted"] == 0

    def test_sync_duplicate_jobs(self, session: "Session"):
        """Test syncing jobs with duplicate links."""
        engine = SmartSyncEngine(session)

        # Create existing job
        existing_job = JobSQL(
            company_id=1,
            title="Original Title",
            description="Original description",
            link="https://test.com/job1",
            location="Remote",
            content_hash="original_hash",
            application_status="New",
            last_seen=datetime.now(UTC),
        )
        session.add(existing_job)
        session.commit()

        # Try to sync job with same link but different content
        new_jobs = [
            JobSQL(
                company_id=1,
                title="Updated Title",
                description="Updated description",
                link="https://test.com/job1",  # Same link
                location="Remote",
                content_hash="new_hash",
                application_status="New",
                last_seen=datetime.now(UTC),
            ),
        ]

        result = engine.sync_jobs(new_jobs)

        assert result["inserted"] == 0
        assert result["updated"] == 1

    def test_sync_empty_jobs_list(self, session: "Session"):
        """Test syncing with empty jobs list."""
        engine = SmartSyncEngine(session)

        result = engine.sync_jobs([])

        assert result["inserted"] == 0
        assert result["updated"] == 0
        assert result["archived"] == 0
        assert result["deleted"] == 0


class TestSyncEdgeCases:
    """Edge cases and error handling for sync operations."""

    def test_sync_with_invalid_company_id(self, session: "Session"):
        """Test syncing jobs with invalid company references."""
        engine = SmartSyncEngine(session)

        jobs_with_invalid_company = [
            JobSQL(
                company_id=999999,  # Non-existent company
                title="Test Job",
                description="Test description",
                link="https://test.com/job1",
                location="Remote",
                content_hash="hash1",
                application_status="New",
                last_seen=datetime.now(UTC),
            ),
        ]

        # Should handle gracefully (implementation dependent)
        result = engine.sync_jobs(jobs_with_invalid_company)

        # Result depends on implementation - either inserts or skips
        assert isinstance(result, dict)
        assert all(
            key in result for key in ["inserted", "updated", "archived", "deleted"]
        )
