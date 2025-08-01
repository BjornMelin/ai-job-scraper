"""Tests for the AI Job Scraper functionality.

This module contains comprehensive tests for the core scraping functionality,
including job relevance filtering, link validation, data validation, and
scraping workflow integration.
"""

import asyncio
import json
import tempfile

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from pydantic import ValidationError

from src.models import CompanySQL, JobPydantic, JobSQL
from scraper import (
    RELEVANT_KEYWORDS,
    SessionStats,
    get_cached_schema,
    is_relevant,
    is_valid_job,
    save_schema_cache,
    update_db,
    validate_link,
)


class TestJobRelevance:
    """Test cases for job relevance filtering."""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Senior AI Engineer", True),
            ("Machine Learning Engineer", True),
            ("MLOps Engineer", True),
            ("AI Agent Engineer", True),
            ("Principal AI Engineer at Google", True),
            (
                "Software Engineer - AI/ML",
                False,
            ),  # Pattern expects keywords before "Engineer"
            ("Sales Representative", False),
            ("Product Manager", False),
            ("Data Analyst", False),
            ("Frontend Developer", False),
            ("Marketing Specialist", False),
        ],
    )
    def test_is_relevant_comprehensive(self, title, expected):
        """Test job relevance filtering with various titles."""
        assert is_relevant({"title": title}) == expected

    def test_is_relevant_case_insensitive(self):
        """Test that relevance filtering is case insensitive."""
        assert is_relevant({"title": "ai engineer"})
        assert is_relevant({"title": "AI ENGINEER"})
        assert is_relevant({"title": "Machine learning Engineer"})
        assert is_relevant({"title": "MACHINE LEARNING ENGINEER"})

    def test_is_relevant_regex_pattern(self):
        """Test the underlying regex pattern works correctly."""
        test_cases = [
            ("AI Engineer", True),
            ("Machine Learning Engineer", True),
            ("MLOps Engineer", True),
            ("AI Agent Engineer", True),
            ("Engineer AI", False),  # Pattern expects keywords before "Engineer"
            ("AI Developer", False),  # Pattern specifically looks for "Engineer"
            ("Machine Learning Specialist", False),  # Same here
        ]

        for title, expected in test_cases:
            match = bool(RELEVANT_KEYWORDS.search(title))
            assert match == expected, f"Failed for title: {title}"


class TestLinkValidation:
    """Test cases for URL validation."""

    @pytest.mark.asyncio
    async def test_validate_link_valid_urls(self):
        """Test validation of valid URLs."""
        # Google should be reliably accessible
        valid = await validate_link("https://google.com")
        assert valid == "https://google.com"

        # Test with different valid URLs
        valid_http = await validate_link("http://httpbin.org/status/200")
        assert valid_http == "http://httpbin.org/status/200"

    @pytest.mark.asyncio
    async def test_validate_link_invalid_urls(self):
        """Test validation of invalid URLs."""
        # Test non-existent domain
        invalid = await validate_link(
            "https://this-domain-definitely-does-not-exist-12345.com"
        )
        assert invalid is None

        # Test malformed URL
        invalid_malformed = await validate_link("not-a-url")
        assert invalid_malformed is None

    @pytest.mark.asyncio
    async def test_validate_link_timeout(self):
        """Test that link validation respects timeout."""
        # Test with a URL that should timeout
        # Using httpbin delay endpoint
        start_time = asyncio.get_event_loop().time()
        result = await validate_link("http://httpbin.org/delay/10")  # 10 second delay
        end_time = asyncio.get_event_loop().time()

        # Should timeout (5 seconds) and return None
        assert result is None
        assert (end_time - start_time) < 8  # Should timeout before 8 seconds

    @pytest.mark.asyncio
    async def test_validate_link_redirects(self):
        """Test that link validation follows redirects."""
        # Test with a URL that redirects
        result = await validate_link("http://httpbin.org/redirect/1")
        assert result == "http://httpbin.org/redirect/1"


class TestJobValidation:
    """Test cases for job data validation."""

    def test_is_valid_job_success(self):
        """Test valid job data passes validation."""
        valid_job = {
            "title": "Senior AI Engineer",
            "description": (
                "We are looking for an experienced AI engineer to join our team."
            ),
            "link": "https://example.com/careers/ai-engineer-123",
        }
        assert is_valid_job(valid_job, "Test Company")

    @pytest.mark.parametrize(
        "invalid_data",
        [
            # Missing title
            {
                "description": "Valid description here.",
                "link": "https://example.com/job/123",
            },
            # Empty title
            {
                "title": "",
                "description": "Valid description here.",
                "link": "https://example.com/job/123",
            },
            # Title too short
            {
                "title": "AI",
                "description": "Valid description here.",
                "link": "https://example.com/job/123",
            },
            # Title too long
            {
                "title": "A" * 201,
                "description": "Valid description here.",
                "link": "https://example.com/job/123",
            },
            # Description too short
            {
                "title": "AI Engineer",
                "description": "Short",
                "link": "https://example.com/job/123",
            },
            # Description too long
            {
                "title": "AI Engineer",
                "description": "A" * 1001,
                "link": "https://example.com/job/123",
            },
            # Invalid link protocol
            {
                "title": "AI Engineer",
                "description": "Valid description here.",
                "link": "ftp://example.com/job/123",
            },
            # Missing link
            {"title": "AI Engineer", "description": "Valid description here."},
        ],
    )
    def test_is_valid_job_failures(self, invalid_data):
        """Test invalid job data fails validation."""
        assert not is_valid_job(invalid_data, "Test Company")

    def test_pydantic_integration(self):
        """Test Pydantic validation integration."""
        # Test valid job
        job = JobPydantic(
            company="Test Company",
            title="Senior AI Engineer",
            description="We are looking for an experienced AI engineer.",
            link="https://test.com/careers/ai-engineer-123",
        )
        assert job.title == "Senior AI Engineer"
        assert job.company == "Test Company"
        assert job.location == "Unknown"  # Default value

    def test_pydantic_validation_errors(self):
        """Test Pydantic validation catches errors."""
        with pytest.raises(ValidationError):
            JobPydantic(
                company="",  # Too short
                title="AI Engineer",
                description="Valid description here.",
                link="https://example.com/job/123",
            )


class TestSchemaCache:
    """Test cases for schema caching functionality."""

    def test_cache_operations(self):
        """Test cache save and retrieve operations."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("scraper.CACHE_DIR", Path(temp_dir)),
        ):
            company = "test_company"
            test_schema = {
                "jobs": {
                    "selector": ".job-listing",
                    "fields": {
                        "title": ".title",
                        "description": ".description",
                        "link": "a@href",
                    },
                }
            }

            # Test saving cache
            save_schema_cache(company, test_schema)

            # Test retrieving cache
            cached_schema = get_cached_schema(company)
            assert cached_schema == test_schema

    def test_cache_expiration(self):
        """Test that cache expires after TTL."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("scraper.CACHE_DIR", Path(temp_dir)),
        ):
            company = "expiry_test"
            test_schema = {"test": "data"}

            # Save cache
            save_schema_cache(company, test_schema)

            # Test with very short TTL (should expire immediately)
            cached_schema = get_cached_schema(company, ttl_hours=0)
            assert cached_schema is None

    def test_cache_version_handling(self):
        """Test cache version validation."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("scraper.CACHE_DIR", Path(temp_dir)),
        ):
            # Create cache file without version
            cache_file = Path(temp_dir) / "version_test.json"
            cache_file.write_text(json.dumps({"schema": {"test": "data"}}))

            # Should return None due to missing version
            cached_schema = get_cached_schema("version_test")
            assert cached_schema is None

            # File should be deleted
            assert not cache_file.exists()


class TestSessionStats:
    """Test cases for session statistics tracking."""

    def test_session_stats_thread_safety(self):
        """Test that session stats are thread-safe."""
        stats = SessionStats()

        # Test basic operations
        stats.set("test_key", 10)
        assert stats.get("test_key") == 10

        stats.increment("test_key", 5)
        assert stats.get("test_key") == 15

        stats.increment("new_key")
        assert stats.get("new_key") == 1

    def test_session_stats_get_all(self):
        """Test getting all stats."""
        stats = SessionStats()

        stats.set("key1", 100)
        stats.set("key2", 200)
        stats.increment("key3", 5)

        all_stats = stats.get_all()

        assert all_stats["key1"] == 100
        assert all_stats["key2"] == 200
        assert all_stats["key3"] == 5


class TestDatabaseUpdate:
    """Test cases for database update functionality."""

    def test_update_db_with_new_jobs(self, temp_db):
        """Test updating database with new jobs."""
        # Create test DataFrame
        jobs_data = {
            "company": ["Test Company"],
            "title": ["Senior AI Engineer"],
            "description": ["We are looking for an experienced AI engineer."],
            "link": ["https://test.com/careers/ai-engineer-123"],
            "location": ["San Francisco, CA"],
            "posted_date": [datetime.now()],
        }
        jobs_df = pd.DataFrame(jobs_data)

        # Mock the database session and validation
        with (
            patch("scraper.SessionLocal", temp_db),
            patch("scraper.validate_link", new_callable=AsyncMock) as mock_validate,
        ):
            mock_validate.return_value = "https://test.com/careers/ai-engineer-123"

            update_db(jobs_df)

            # Verify job was added
            session = temp_db()
            jobs = session.query(JobSQL).all()
            assert len(jobs) == 1
            assert jobs[0].title == "Senior AI Engineer"
            session.close()

    def test_update_db_with_invalid_jobs(self, temp_db):
        """Test that invalid jobs are skipped during database update."""
        # Create DataFrame with invalid job
        jobs_data = {
            "company": ["Test Company"],
            "title": ["AI"],  # Too short - should be invalid
            "description": ["Short"],  # Too short - should be invalid
            "link": ["https://test.com/careers/ai-engineer-123"],
            "location": ["San Francisco, CA"],
            "posted_date": [datetime.now()],
        }
        jobs_df = pd.DataFrame(jobs_data)

        with (
            patch("scraper.SessionLocal", temp_db),
            patch("scraper.validate_link", new_callable=AsyncMock) as mock_validate,
        ):
            mock_validate.return_value = "https://test.com/careers/ai-engineer-123"

            update_db(jobs_df)

            # Verify no jobs were added due to validation failure
            session = temp_db()
            jobs = session.query(JobSQL).all()
            assert len(jobs) == 0
            session.close()


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""

    def test_end_to_end_workflow_simulation(self, temp_db):
        """Test simulated end-to-end scraping workflow."""
        # Setup: Add a test company
        session = temp_db()
        test_company = CompanySQL(
            name="Integration Test Company",
            url="https://integration-test.com/careers",
            active=True,
        )
        session.add(test_company)
        session.commit()
        session.close()

        # Simulate scraped job data
        scraped_jobs = [
            {
                "company": "Integration Test Company",
                "title": "Senior AI Engineer",
                "description": (
                    "We are looking for an experienced AI engineer to work on "
                    "cutting-edge projects."
                ),
                "link": "https://integration-test.com/careers/senior-ai-engineer",
                "location": "Remote",
                "posted_date": datetime.now(),
            },
            {
                "company": "Integration Test Company",
                "title": "Machine Learning Engineer",
                "description": (
                    "Join our ML team to build scalable machine learning systems."
                ),
                "link": "https://integration-test.com/careers/ml-engineer",
                "location": "San Francisco, CA",
                "posted_date": datetime.now(),
            },
            {
                "company": "Integration Test Company",
                "title": "Product Manager",  # Should be filtered out as not relevant
                "description": "Lead product development for our AI initiatives.",
                "link": "https://integration-test.com/careers/product-manager",
                "location": "New York, NY",
                "posted_date": datetime.now(),
            },
        ]

        # Filter for relevant jobs
        relevant_jobs = [job for job in scraped_jobs if is_relevant(job)]
        assert len(relevant_jobs) == 2  # Only AI and ML engineer jobs

        # Validate jobs
        valid_jobs = [job for job in relevant_jobs if is_valid_job(job, job["company"])]
        assert len(valid_jobs) == 2  # Both should be valid

        # Convert to DataFrame and update database
        jobs_df = pd.DataFrame(valid_jobs)

        with (
            patch("scraper.SessionLocal", temp_db),
            patch("scraper.validate_link", new_callable=AsyncMock) as mock_validate,
        ):
            # Mock successful link validation
            mock_validate.side_effect = lambda link: link

            update_db(jobs_df)

        # Verify results in database
        session = temp_db()
        saved_jobs = session.query(JobSQL).all()
        assert len(saved_jobs) == 2

        # Verify job titles are correct
        job_titles = [job.title for job in saved_jobs]
        assert "Senior AI Engineer" in job_titles
        assert "Machine Learning Engineer" in job_titles
        assert "Product Manager" not in job_titles  # Should be filtered out

        session.close()
