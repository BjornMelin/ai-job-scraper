"""Comprehensive tests for scraper.py module.

This module provides extensive test coverage for the main scraping orchestration,
including workflow integration, error handling, data normalization, and CLI interface.
Tests focus on real-world scenarios and edge cases to ensure robust operation.
"""

import hashlib

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from sqlmodel import Session
from src.models import CompanySQL, JobSQL
from src.scraper import (
    _normalize_board_jobs,
    get_or_create_company,
    scrape_all,
)
from src.scraper import (
    app as scraper_cli,
)


class TestGetOrCreateCompany:
    """Test suite for company management functions."""

    def test_get_or_create_company_new(self, session: Session) -> None:
        """Test creating a new company when it doesn't exist."""
        company_name = "New Tech Corp"

        company_id = get_or_create_company(session, company_name)

        # Verify company was created
        assert isinstance(company_id, int)
        company = session.get(CompanySQL, company_id)
        assert company is not None
        assert company.name == company_name
        assert company.active is True
        assert company.url == ""

    def test_get_or_create_company_existing(self, session: Session) -> None:
        """Test retrieving existing company without creating duplicate."""
        company_name = "Existing Corp"

        # Create existing company
        existing_company = CompanySQL(
            name=company_name, url="https://existing.com/careers", active=True
        )
        session.add(existing_company)
        session.commit()
        session.refresh(existing_company)

        # Should return existing company ID
        company_id = get_or_create_company(session, company_name)

        assert company_id == existing_company.id
        # Verify no duplicate was created
        companies = (
            session.query(CompanySQL).filter(CompanySQL.name == company_name).all()
        )
        assert len(companies) == 1

    @patch("src.scraper.logger")
    def test_get_or_create_company_database_error(
        self, mock_logger: Mock, session: Session
    ) -> None:
        """Test handling database errors during company creation."""
        company_name = "Error Corp"

        # Mock session to raise exception on commit
        with (
            patch.object(session, "commit", side_effect=Exception("Database error")),
            pytest.raises(Exception, match="Database error"),
        ):
            get_or_create_company(session, company_name)


class TestNormalizeBoardJobs:
    """Test suite for job board data normalization."""

    @patch("src.scraper.CompanyService")
    @patch("src.scraper.SessionLocal")
    def test_normalize_board_jobs_success(
        self, mock_session_local: Mock, mock_company_service: Mock
    ) -> None:
        """Test successful normalization of job board data."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock company service bulk operation
        mock_company_service.bulk_get_or_create_companies.return_value = {
            "Tech Corp": 1,
            "Data Co": 2,
        }

        raw_jobs = [
            {
                "title": "AI Engineer",
                "company": "Tech Corp",
                "description": "AI role description",
                "job_url": "https://tech.com/jobs/1",
                "location": "San Francisco, CA",
                "date_posted": datetime.now(timezone.utc),
                "min_amount": 100000,
                "max_amount": 150000,
            },
            {
                "title": "ML Engineer",
                "company": "Data Co",
                "description": "ML role description",
                "job_url": "https://data.com/jobs/2",
                "location": "Remote",
                "date_posted": None,
                "min_amount": None,
                "max_amount": 200000,
            },
        ]

        result = _normalize_board_jobs(raw_jobs)

        assert len(result) == 2

        # Verify first job
        job1 = result[0]
        assert job1.title == "AI Engineer"
        assert job1.company_id == 1
        assert job1.location == "San Francisco, CA"
        assert job1.salary == "$100000-$150000"
        assert job1.application_status == "New"
        assert job1.content_hash is not None

        # Verify second job
        job2 = result[1]
        assert job2.title == "ML Engineer"
        assert job2.company_id == 2
        assert job2.salary == "$200000"

        # Verify session was properly closed
        mock_session.close.assert_called_once()

    @patch("src.scraper.SessionLocal")
    def test_normalize_board_jobs_empty_input(self, mock_session_local: Mock) -> None:
        """Test handling empty job list."""
        result = _normalize_board_jobs([])

        assert result == []
        # Session should not be created for empty input
        mock_session_local.assert_not_called()

    @patch("src.scraper.CompanyService")
    @patch("src.scraper.SessionLocal")
    @patch("src.scraper.logger")
    def test_normalize_board_jobs_malformed_data(
        self, mock_logger: Mock, mock_session_local: Mock, mock_company_service: Mock
    ) -> None:
        """Test handling malformed job data."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_company_service.bulk_get_or_create_companies.return_value = {
            "Good Corp": 1,
        }

        raw_jobs = [
            {
                "title": "Valid Job",
                "company": "Good Corp",
                "description": "Valid description",
                "job_url": "https://good.com/job1",
                "location": "Location",
                "min_amount": 100000,
                "max_amount": 150000,
            },
            {
                # Missing required fields
                "title": "",
                "company": "",
                "job_url": "https://bad.com/job2",
            },
            {
                # Job that will cause normalization exception
                "title": "Error Job",
                "company": "Bad Corp",  # Company not in mapping
                "job_url": "https://bad.com/job3",
            },
        ]

        result = _normalize_board_jobs(raw_jobs)

        # Should successfully process valid job and skip invalid ones
        assert len(result) == 1
        assert result[0].title == "Valid Job"

        # Verify error logging for malformed jobs
        mock_logger.warning.assert_called()
        # Note: exception logging depends on specific normalization failures
        # which may not occur with this test setup

    @patch("src.scraper.CompanyService")
    @patch("src.scraper.SessionLocal")
    def test_normalize_board_jobs_salary_parsing(
        self, mock_session_local: Mock, mock_company_service: Mock
    ) -> None:
        """Test various salary format parsing scenarios."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_company_service.bulk_get_or_create_companies.return_value = {"Corp": 1}

        test_cases = [
            (100000, 150000, "$100000-$150000"),
            (100000, None, "$100000+"),
            (None, 150000, "$150000"),
            (None, None, ""),
        ]

        for min_amt, max_amt, expected_salary in test_cases:
            raw_jobs = [
                {
                    "title": "Test Job",
                    "company": "Corp",
                    "description": "Description",
                    "job_url": f"https://corp.com/job-{min_amt}-{max_amt}",
                    "min_amount": min_amt,
                    "max_amount": max_amt,
                }
            ]

            result = _normalize_board_jobs(raw_jobs)

            assert len(result) == 1
            assert result[0].salary == expected_salary

    @patch("src.scraper.CompanyService")
    @patch("src.scraper.SessionLocal")
    @patch("src.scraper.logger")
    def test_normalize_board_jobs_database_error(
        self, mock_logger: Mock, mock_session_local: Mock, mock_company_service: Mock
    ) -> None:
        """Test handling database errors during normalization."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock database error during company bulk operation
        mock_company_service.bulk_get_or_create_companies.side_effect = Exception(
            "DB Error"
        )

        raw_jobs = [
            {
                "title": "Test Job",
                "company": "Corp",
                "description": "Description",
                "job_url": "https://corp.com/job1",
            }
        ]

        with pytest.raises(Exception, match="DB Error"):
            _normalize_board_jobs(raw_jobs)

        # Verify session rollback was called
        mock_session.rollback.assert_called_once()

    @patch("src.scraper.CompanyService")
    @patch("src.scraper.SessionLocal")
    def test_normalize_board_jobs_content_hash_generation(
        self, mock_session_local: Mock, mock_company_service: Mock
    ) -> None:
        """Test content hash generation for change detection."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_company_service.bulk_get_or_create_companies.return_value = {"Corp": 1}

        raw_job = {
            "title": "Test Job",
            "company": "Corp",
            "description": "Test description",
            "job_url": "https://corp.com/job1",
        }

        result = _normalize_board_jobs([raw_job])

        assert len(result) == 1
        job = result[0]

        # Verify content hash is generated properly
        expected_content = "Test JobTest descriptionCorp"
        expected_hash = hashlib.sha256(expected_content.encode()).hexdigest()
        assert job.content_hash == expected_hash


class TestScrapeAll:
    """Test suite for the main scraping orchestration function."""

    @patch("src.scraper.SmartSyncEngine")
    @patch("src.scraper.scrape_job_boards")
    @patch("src.scraper.scrape_company_pages")
    @patch("src.scraper.random_delay")
    def test_scrape_all_success_mixed_sources(
        self,
        mock_delay: Mock,
        mock_company_scraper: Mock,
        mock_board_scraper: Mock,
        mock_sync_engine: Mock,
    ) -> None:
        """Test successful scraping with data from both sources."""
        # Mock company scraper results
        company_job = JobSQL(
            company_id=1,
            title="AI Engineer",
            description="Company AI role",
            link="https://company.com/job1",
            location="SF",
            content_hash="company_hash",
            application_status="New",
            last_seen=datetime.now(timezone.utc),
        )
        mock_company_scraper.return_value = [company_job]

        # Mock job board scraper results
        mock_board_scraper.return_value = [
            {
                "title": "ML Engineer",
                "company": "Board Corp",
                "description": "Board ML role",
                "job_url": "https://board.com/job1",
                "location": "Remote",
                "min_amount": 120000,
                "max_amount": 160000,
            }
        ]

        # Mock sync engine
        mock_sync_instance = Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 2,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        # Mock _normalize_board_jobs to return JobSQL objects
        with patch("src.scraper._normalize_board_jobs") as mock_normalize:
            board_job = JobSQL(
                company_id=2,
                title="ML Engineer",
                description="Board ML role",
                link="https://board.com/job1",
                location="Remote",
                content_hash="board_hash",
                application_status="New",
                last_seen=datetime.now(timezone.utc),
            )
            mock_normalize.return_value = [board_job]

            result = scrape_all(max_jobs_per_company=50)

        # Verify result
        assert result == {
            "inserted": 2,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        # Verify all scrapers were called
        mock_company_scraper.assert_called_once_with(50)
        mock_board_scraper.assert_called_once()

        # Verify sync was called with both jobs (both are AI-related)
        mock_sync_instance.sync_jobs.assert_called_once()
        sync_jobs = mock_sync_instance.sync_jobs.call_args[0][0]
        assert len(sync_jobs) == 2

    @patch("src.scraper.SmartSyncEngine")
    @patch("src.scraper.scrape_job_boards")
    @patch("src.scraper.scrape_company_pages")
    @patch("src.scraper.random_delay")
    @patch("src.scraper.logger")
    def test_scrape_all_company_scraper_fails(
        self,
        mock_logger: Mock,
        mock_delay: Mock,
        mock_company_scraper: Mock,
        mock_board_scraper: Mock,
        mock_sync_engine: Mock,
    ) -> None:
        """Test handling company scraper failure."""
        # Mock company scraper to raise exception
        mock_company_scraper.side_effect = Exception("Company scraper failed")

        # Mock successful job board scraper
        mock_board_scraper.return_value = [
            {
                "title": "AI Engineer",
                "company": "Board Corp",
                "description": "AI role",
                "job_url": "https://board.com/job1",
            }
        ]

        # Mock sync engine
        mock_sync_instance = Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 1,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        with patch("src.scraper._normalize_board_jobs") as mock_normalize:
            mock_normalize.return_value = [Mock(spec=JobSQL, title="AI Engineer")]

            result = scrape_all()

        # Should continue with job board results despite company scraper failure
        assert "inserted" in result
        mock_logger.exception.assert_called_with("Company scraping failed")

    @patch("src.scraper.SmartSyncEngine")
    @patch("src.scraper.scrape_job_boards")
    @patch("src.scraper.scrape_company_pages")
    @patch("src.scraper.logger")
    def test_scrape_all_both_scrapers_fail(
        self,
        mock_logger: Mock,
        mock_company_scraper: Mock,
        mock_board_scraper: Mock,
        mock_sync_engine: Mock,
    ) -> None:
        """Test safety guard when both scrapers return empty results."""
        mock_company_scraper.return_value = []
        mock_board_scraper.return_value = []

        result = scrape_all()

        # Should return empty stats and skip sync
        expected = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }
        assert result == expected

        # Verify safety warning was logged
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args_list[0]
        assert (
            "Both company pages and job boards scrapers returned empty results"
            in str(warning_call)
        )

        # Sync engine should not be called
        mock_sync_engine.assert_not_called()

    @patch("src.scraper.SmartSyncEngine")
    @patch("src.scraper.scrape_job_boards")
    @patch("src.scraper.scrape_company_pages")
    @patch("src.scraper.logger")
    def test_scrape_all_low_job_count_warning(
        self,
        mock_logger: Mock,
        mock_company_scraper: Mock,
        mock_board_scraper: Mock,
        mock_sync_engine: Mock,
    ) -> None:
        """Test warning for suspiciously low job counts."""
        # Return small number of jobs (below threshold)
        mock_company_scraper.return_value = []
        mock_board_scraper.return_value = [
            {
                "title": "AI Engineer",
                "company": "Corp",
                "description": "Role",
                "job_url": "https://corp.com/job1",
            }
        ]

        mock_sync_instance = Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 1,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        with patch("src.scraper._normalize_board_jobs") as mock_normalize:
            mock_normalize.return_value = [Mock(spec=JobSQL, title="AI Engineer")]

            scrape_all()

        # Verify warning was logged for low job count
        mock_logger.warning.assert_called()
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("suspiciously low" in call for call in warning_calls)

    @patch("src.scraper.SmartSyncEngine")
    @patch("src.scraper.scrape_job_boards")
    @patch("src.scraper.scrape_company_pages")
    def test_scrape_all_no_relevant_jobs_after_filtering(
        self,
        mock_company_scraper: Mock,
        mock_board_scraper: Mock,
        mock_sync_engine: Mock,
    ) -> None:
        """Test handling when no jobs remain after AI/ML filtering."""
        mock_company_scraper.return_value = []

        # Return non-AI jobs that will be filtered out
        mock_board_scraper.return_value = [
            {
                "title": "Sales Manager",  # Non-AI job
                "company": "Corp",
                "description": "Sales role",
                "job_url": "https://corp.com/sales1",
            }
        ]

        with patch("src.scraper._normalize_board_jobs") as mock_normalize:
            mock_normalize.return_value = [Mock(spec=JobSQL, title="Sales Manager")]

            result = scrape_all()

        # Should skip sync due to no relevant jobs
        expected = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }
        assert result == expected
        mock_sync_engine.assert_not_called()

    def test_scrape_all_max_jobs_validation(self) -> None:
        """Test validation of max_jobs_per_company parameter."""
        # Test invalid type
        with pytest.raises(ValueError, match="max_jobs_per_company must be an integer"):
            scrape_all(max_jobs_per_company="invalid")  # type: ignore

        # Test invalid value
        with pytest.raises(ValueError, match="max_jobs_per_company must be at least 1"):
            scrape_all(max_jobs_per_company=0)

        with pytest.raises(ValueError, match="max_jobs_per_company must be at least 1"):
            scrape_all(max_jobs_per_company=-1)

    @patch("src.scraper.SmartSyncEngine")
    @patch("src.scraper.scrape_job_boards")
    @patch("src.scraper.scrape_company_pages")
    def test_scrape_all_deduplication_logic(
        self,
        mock_company_scraper: Mock,
        mock_board_scraper: Mock,
        mock_sync_engine: Mock,
    ) -> None:
        """Test job deduplication by link."""
        # Create jobs with duplicate links
        duplicate_job1 = JobSQL(
            company_id=1,
            title="AI Engineer V1",
            description="First version",
            link="https://same.com/job1",
            location="Location1",
            content_hash="hash1",
            application_status="New",
            last_seen=datetime.now(timezone.utc),
        )

        duplicate_job2 = JobSQL(
            company_id=2,
            title="AI Engineer V2",
            description="Second version",
            link="https://same.com/job1",  # Same link
            location="Location2",
            content_hash="hash2",
            application_status="New",
            last_seen=datetime.now(timezone.utc),
        )

        mock_company_scraper.return_value = [duplicate_job1]
        mock_board_scraper.return_value = []

        with patch("src.scraper._normalize_board_jobs") as mock_normalize:
            mock_normalize.return_value = [duplicate_job2]

            mock_sync_instance = Mock()
            mock_sync_engine.return_value = mock_sync_instance
            mock_sync_instance.sync_jobs.return_value = {
                "inserted": 1,
                "updated": 0,
                "archived": 0,
                "deleted": 0,
                "skipped": 0,
            }

            scrape_all()

        # Verify only one job was passed to sync (latest one)
        mock_sync_instance.sync_jobs.assert_called_once()
        sync_jobs = mock_sync_instance.sync_jobs.call_args[0][0]
        assert len(sync_jobs) == 1
        # Should keep the last occurrence (from normalize_board_jobs)
        assert sync_jobs[0].title == "AI Engineer V2"


class TestScraperCLI:
    """Test suite for CLI interface."""

    @patch("src.scraper.scrape_all")
    def test_cli_scrape_command_default(self, mock_scrape_all: Mock) -> None:
        """Test CLI scrape command with default parameters."""
        mock_scrape_all.return_value = {
            "inserted": 5,
            "updated": 2,
            "archived": 1,
            "deleted": 0,
            "skipped": 3,
        }

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(scraper_cli, [])

        assert result.exit_code == 0
        assert "Scraping completed successfully!" in result.stdout
        assert "✅ Inserted: 5 new jobs" in result.stdout
        assert "🔄 Updated: 2 existing jobs" in result.stdout
        assert "📋 Archived: 1 stale jobs with user data" in result.stdout
        assert "🗑️  Deleted: 0 stale jobs without user data" in result.stdout
        assert "⏭️  Skipped: 3 jobs (no changes)" in result.stdout

        # Verify scrape_all was called with default value
        mock_scrape_all.assert_called_once_with(50)

    @patch("src.scraper.scrape_all")
    def test_cli_scrape_command_custom_max_jobs(self, mock_scrape_all: Mock) -> None:
        """Test CLI scrape command with custom max jobs parameter."""
        mock_scrape_all.return_value = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(scraper_cli, ["--max-jobs", "25"])

        assert result.exit_code == 0
        mock_scrape_all.assert_called_once_with(25)

    @patch("src.scraper.scrape_all")
    def test_cli_scrape_command_error_handling(self, mock_scrape_all: Mock) -> None:
        """Test CLI error handling when scrape_all fails."""
        mock_scrape_all.side_effect = Exception("Scraping failed")

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(scraper_cli, [])

        # With the exception, the CLI should fail but typer catches it
        # and may return various exit codes - let's just check it's not 0
        assert result.exit_code != 0
        # Exception should propagate through CLI
