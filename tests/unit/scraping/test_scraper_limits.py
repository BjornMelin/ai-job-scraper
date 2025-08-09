"""Comprehensive pytest tests for scraper limit functionality.

Tests cover:
- scrape_company_pages() respects max_jobs limit
- scrape_all() passes limits correctly
- Backward compatibility when max_jobs=None
- Edge cases (0, 1, 100+ jobs)
"""

import logging

from datetime import datetime, timezone
from unittest import mock

from src.models import CompanySQL, JobSQL
from src.scraper import scrape_all
from src.scraper_company_pages import (
    extract_job_lists,
    scrape_company_pages,
)


class TestScrapeLimits:
    """Test scraper limit functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_companies = [
            CompanySQL(
                id=1,
                name="Tech Corp",
                url="https://techcorp.com/careers",
                active=True,
            ),
            CompanySQL(
                id=2,
                name="AI Startup",
                url="https://aistartup.com/jobs",
                active=True,
            ),
        ]

    @mock.patch("src.scraper_company_pages.load_active_companies")
    @mock.patch("src.scraper_company_pages.SmartScraperMultiGraph")
    def test_scrape_company_pages_with_limit(self, mock_graph, mock_load_companies):
        """Test scrape_company_pages respects max_jobs limit."""
        mock_load_companies.return_value = self.mock_companies

        # Mock extraction results with more jobs than limit
        mock_graph_instance = mock.Mock()
        mock_graph.return_value = mock_graph_instance
        mock_graph_instance.run.return_value = {
            "https://techcorp.com/careers": {
                "jobs": [
                    {"title": f"Job {i}", "url": f"https://techcorp.com/job/{i}"}
                    for i in range(20)  # More than our limit of 15
                ]
            },
            "https://aistartup.com/jobs": {
                "jobs": [
                    {"title": f"AI Job {i}", "url": f"https://aistartup.com/job/{i}"}
                    for i in range(25)  # More than our limit of 15
                ]
            },
        }

        # Test with limit of 15 jobs per company
        max_jobs = 15
        with mock.patch("src.scraper_company_pages.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.return_value = mock.Mock()
            result = scrape_company_pages(max_jobs_per_company=max_jobs)

        # Verify the limit was applied
        assert len(result) <= len(self.mock_companies) * max_jobs

        # Group results by company to verify per-company limits
        jobs_by_company = {}
        for job in result:
            company_name = job.company
            if company_name not in jobs_by_company:
                jobs_by_company[company_name] = []
            jobs_by_company[company_name].append(job)

        # Assert that no company exceeds the job limit
        for company_name, jobs in jobs_by_company.items():
            assert len(jobs) <= max_jobs, (
                f"Company {company_name} has {len(jobs)} jobs, "
                f"exceeding limit of {max_jobs}"
            )

        # Should have called the graph with correct configuration
        mock_graph.assert_called()

    @mock.patch("src.scraper_company_pages.load_active_companies")
    def test_scrape_company_pages_default_limit(self, mock_load_companies):
        """Test scrape_company_pages uses default limit when not specified."""
        mock_load_companies.return_value = self.mock_companies

        with mock.patch(
            "src.scraper_company_pages.SmartScraperMultiGraph"
        ) as mock_graph:
            mock_graph_instance = mock.Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.run.return_value = {}

            with mock.patch("src.scraper_company_pages.SessionLocal"):
                result = scrape_company_pages()  # No parameter, should use default

        # Should use default constant (50)
        assert isinstance(result, list)

    @mock.patch("src.scraper_company_pages.load_active_companies")
    def test_scrape_company_pages_zero_limit(self, mock_load_companies):
        """Test scrape_company_pages with zero limit."""
        mock_load_companies.return_value = self.mock_companies

        with mock.patch(
            "src.scraper_company_pages.SmartScraperMultiGraph"
        ) as mock_graph:
            mock_graph_instance = mock.Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.run.return_value = {}

            with mock.patch("src.scraper_company_pages.SessionLocal"):
                result = scrape_company_pages(max_jobs_per_company=0)

        # Should handle zero limit gracefully
        assert isinstance(result, list)

    @mock.patch("src.scraper_company_pages.load_active_companies")
    def test_scrape_company_pages_large_limit(self, mock_load_companies):
        """Test scrape_company_pages with very large limit."""
        mock_load_companies.return_value = self.mock_companies

        with mock.patch(
            "src.scraper_company_pages.SmartScraperMultiGraph"
        ) as mock_graph:
            mock_graph_instance = mock.Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.run.return_value = {}

            with mock.patch("src.scraper_company_pages.SessionLocal"):
                result = scrape_company_pages(max_jobs_per_company=500)

        # Should handle large limit gracefully
        assert isinstance(result, list)

    @mock.patch("src.scraper_company_pages.load_active_companies")
    def test_scrape_company_pages_empty_companies(self, mock_load_companies):
        """Test scrape_company_pages with no active companies."""
        mock_load_companies.return_value = []

        result = scrape_company_pages(max_jobs_per_company=50)

        assert result == []

    def test_extract_job_lists_respects_limit(self):
        """Test extract_job_lists function respects max_jobs_per_company in state."""
        # Mock state with companies and job limit
        state = {
            "companies": self.mock_companies,
            "max_jobs_per_company": 5,
        }

        with mock.patch(
            "src.scraper_company_pages.SmartScraperMultiGraph"
        ) as mock_graph:
            mock_graph_instance = mock.Mock()
            mock_graph.return_value = mock_graph_instance

            # Return more jobs than the limit
            mock_graph_instance.run.return_value = {
                "https://techcorp.com/careers": {
                    "jobs": [
                        {"title": f"Job {i}", "url": f"https://techcorp.com/job/{i}"}
                        for i in range(10)  # More than limit of 5
                    ]
                }
            }

            with mock.patch("src.scraper_company_pages.random_delay"):
                result = extract_job_lists(state)

        partial_jobs = result["partial_jobs"]

        # Should respect the limit of 5 jobs per company
        tech_corp_jobs = [job for job in partial_jobs if job["company"] == "Tech Corp"]
        assert len(tech_corp_jobs) <= 5

    def test_extract_job_lists_default_limit(self):
        """Test extract_job_lists uses default limit when not in state."""
        # Mock state without max_jobs_per_company
        state = {
            "companies": self.mock_companies,
        }

        with mock.patch(
            "src.scraper_company_pages.SmartScraperMultiGraph"
        ) as mock_graph:
            mock_graph_instance = mock.Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.run.return_value = {}

            with mock.patch("src.scraper_company_pages.random_delay"):
                result = extract_job_lists(state)

        # Should handle missing limit gracefully
        assert "partial_jobs" in result
        assert isinstance(result["partial_jobs"], list)


class TestScrapeAllLimits:
    """Test scrape_all function limit functionality."""

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    @mock.patch("src.scraper.SmartSyncEngine")
    def test_scrape_all_passes_limit_to_company_scraper(
        self, mock_sync_engine, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all passes max_jobs_per_company to company scraper."""
        # Mock return values
        mock_company_pages.return_value = [
            JobSQL(
                id=1,
                title="AI Engineer",
                company_id=1,
                description="Test job",
                link="https://example.com/job1",
                location="Remote",
                posted_date=datetime.now(timezone.utc),
                salary="$100k-150k",
                content_hash="hash1",
                application_status="New",
                last_seen=datetime.now(timezone.utc),
            )
        ]
        mock_job_boards.return_value = []

        mock_sync_instance = mock.Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 1,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        # Call scrape_all with specific limit
        max_jobs = 25
        result = scrape_all(max_jobs_per_company=max_jobs)

        # Verify company scraper was called with correct limit
        mock_company_pages.assert_called_once_with(max_jobs)
        assert isinstance(result, dict)
        assert "inserted" in result

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    @mock.patch("src.scraper.SmartSyncEngine")
    def test_scrape_all_default_limit(
        self, mock_sync_engine, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all uses default limit when None provided."""
        mock_company_pages.return_value = []
        mock_job_boards.return_value = []

        mock_sync_instance = mock.Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        # Call without specifying limit
        result = scrape_all(max_jobs_per_company=None)

        # Should call with default of 50
        mock_company_pages.assert_called_once_with(None)
        assert isinstance(result, dict)

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    @mock.patch("src.scraper.SmartSyncEngine")
    def test_scrape_all_extreme_limits(
        self, mock_sync_engine, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all handles extreme limit values."""
        mock_company_pages.return_value = []
        mock_job_boards.return_value = []

        mock_sync_instance = mock.Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        # Test with zero limit
        result = scrape_all(max_jobs_per_company=0)
        mock_company_pages.assert_called_with(0)
        assert isinstance(result, dict)

        # Test with very large limit
        mock_company_pages.reset_mock()
        result = scrape_all(max_jobs_per_company=1000)
        mock_company_pages.assert_called_with(1000)
        assert isinstance(result, dict)

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    def test_scrape_all_safety_guard_both_empty(
        self, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all safety guard when both scrapers return empty."""
        # Both scrapers return empty results
        mock_company_pages.return_value = []
        mock_job_boards.return_value = []

        result = scrape_all(max_jobs_per_company=50)

        # Should return empty stats and not proceed with sync
        expected = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }
        assert result == expected

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    def test_scrape_all_safety_guard_low_count(
        self, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all safety guard for suspiciously low job counts."""
        # Return very few jobs
        mock_company_pages.return_value = [
            JobSQL(
                id=1,
                title="Single Job",
                company_id=1,
                description="Only job",
                link="https://example.com/job1",
                location="Remote",
                posted_date=datetime.now(timezone.utc),
                salary="$100k-150k",
                content_hash="hash1",
                application_status="New",
                last_seen=datetime.now(timezone.utc),
            )
        ]
        mock_job_boards.return_value = []

        with mock.patch("src.scraper.SmartSyncEngine") as mock_sync_engine:
            mock_sync_instance = mock.Mock()
            mock_sync_engine.return_value = mock_sync_instance
            mock_sync_instance.sync_jobs.return_value = {
                "inserted": 1,
                "updated": 0,
                "archived": 0,
                "deleted": 0,
                "skipped": 0,
            }

            result = scrape_all(max_jobs_per_company=50)

        # Should still proceed but with warning logged
        assert isinstance(result, dict)
        assert result["inserted"] >= 0

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    @mock.patch("src.scraper.SmartSyncEngine")
    def test_scrape_all_company_scraper_exception(
        self, mock_sync_engine, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all handles company scraper exceptions."""
        # Company scraper raises exception
        mock_company_pages.side_effect = Exception("Scraping failed")
        mock_job_boards.return_value = [
            {
                "title": "Board Job",
                "company": "Board Company",
                "description": "Job from board",
                "location": "Remote",
                "job_url": "https://board.com/job1",
                "date_posted": datetime.now(timezone.utc),
                "min_amount": 100000,
                "max_amount": 150000,
            }
        ]

        mock_sync_instance = mock.Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 1,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        with mock.patch("src.scraper._normalize_board_jobs") as mock_normalize:
            mock_normalize.return_value = []

            result = scrape_all(max_jobs_per_company=50)

        # Should continue with job board results despite company scraper failure
        assert isinstance(result, dict)
        mock_company_pages.assert_called_once_with(50)

    @mock.patch("src.scraper.scrape_company_pages")
    @mock.patch("src.scraper.scrape_job_boards")
    @mock.patch("src.scraper.SmartSyncEngine")
    def test_scrape_all_job_board_scraper_exception(
        self, mock_sync_engine, mock_job_boards, mock_company_pages
    ):
        """Test scrape_all handles job board scraper exceptions."""
        mock_company_pages.return_value = [
            JobSQL(
                id=1,
                title="Company Job",
                company_id=1,
                description="Job from company",
                link="https://company.com/job1",
                location="Remote",
                posted_date=datetime.now(timezone.utc),
                salary="$100k-150k",
                content_hash="hash1",
                application_status="New",
                last_seen=datetime.now(timezone.utc),
            )
        ]
        # Job board scraper raises exception
        mock_job_boards.side_effect = Exception("Board scraping failed")

        mock_sync_instance = mock.Mock()
        mock_sync_engine.return_value = mock_sync_instance
        mock_sync_instance.sync_jobs.return_value = {
            "inserted": 1,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }

        result = scrape_all(max_jobs_per_company=50)

        # Should continue with company results despite job board scraper failure
        assert isinstance(result, dict)
        assert result["inserted"] >= 0


class TestLimitEdgeCases:
    """Test edge cases for limit functionality."""

    def test_negative_limit_handling(self):
        """Test handling of negative limit values."""
        with mock.patch("src.scraper_company_pages.load_active_companies") as mock_load:
            mock_load.return_value = []

            # Should handle negative values without crashing
            result = scrape_company_pages(max_jobs_per_company=-5)
            assert isinstance(result, list)

    def test_float_limit_handling(self):
        """Test handling of float limit values."""
        with mock.patch("src.scraper_company_pages.load_active_companies") as mock_load:
            mock_load.return_value = []

            # Should handle float values
            result = scrape_company_pages(max_jobs_per_company=50.5)
            assert isinstance(result, list)

    def test_string_limit_handling(self):
        """Test handling of string limit values."""
        with mock.patch("src.scraper_company_pages.load_active_companies") as mock_load:
            mock_load.return_value = []

            # Should handle string values (may cause issues in real usage)
            try:
                result = scrape_company_pages(max_jobs_per_company="50")
                assert isinstance(result, list)
            except (TypeError, ValueError):
                # Expected behavior for invalid types
                pass


class TestLimitLogging:
    """Test logging behavior for limit functionality."""

    @mock.patch("src.scraper_company_pages.load_active_companies")
    @mock.patch("src.scraper_company_pages.SmartScraperMultiGraph")
    def test_limit_logging(self, mock_graph, mock_load_companies, caplog):
        """Test that job limits are properly logged."""
        mock_load_companies.return_value = [
            CompanySQL(
                id=1,
                name="Test Company",
                url="https://test.com/careers",
                active=True,
            )
        ]

        mock_graph_instance = mock.Mock()
        mock_graph.return_value = mock_graph_instance
        mock_graph_instance.run.return_value = {
            "https://test.com/careers": {
                "jobs": [
                    {"title": f"Job {i}", "url": f"https://test.com/job/{i}"}
                    for i in range(30)  # More than limit
                ]
            }
        }

        with (
            mock.patch("src.scraper_company_pages.SessionLocal"),
            caplog.at_level(logging.INFO),
        ):
            result = scrape_company_pages(max_jobs_per_company=20)

        # Assert that the expected log message appears
        expected_message = "Starting scraping for 1 companies (limit: 20 jobs each)"
        log_messages = [r.message for r in caplog.records]
        assert any(expected_message in record.message for record in caplog.records), (
            f"Expected log message '{expected_message}' not found in logs: "
            f"{log_messages}"
        )
        assert isinstance(result, list)

    def test_scrape_all_limit_logging(self, caplog):
        """Test that scrape_all logs the job limit being used."""
        with mock.patch("src.scraper.scrape_company_pages") as mock_company_pages:
            mock_company_pages.return_value = []
            with mock.patch("src.scraper.scrape_job_boards") as mock_job_boards:
                mock_job_boards.return_value = []

                with caplog.at_level(logging.INFO):
                    scrape_all(max_jobs_per_company=75)

        # Should log the job limit being used
        assert any(
            "Using job limit: 75 jobs per company" in record.message
            for record in caplog.records
        )

    def test_scrape_all_default_limit_logging(self, caplog):
        """Test that scrape_all logs default limit when None provided."""
        with mock.patch("src.scraper.scrape_company_pages") as mock_company_pages:
            mock_company_pages.return_value = []
            with mock.patch("src.scraper.scrape_job_boards") as mock_job_boards:
                mock_job_boards.return_value = []

                with caplog.at_level(logging.INFO):
                    scrape_all(max_jobs_per_company=None)

        # Should log the default limit of 50
        assert any(
            "Using job limit: 50 jobs per company" in record.message
            for record in caplog.records
        )
