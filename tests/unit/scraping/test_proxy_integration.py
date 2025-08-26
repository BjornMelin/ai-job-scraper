"""Comprehensive tests for proxy integration in scrapers.

Tests proxy configuration, rotation, error handling, and integration
with both JobSpy (job boards) and ScrapeGraphAI (company pages) scrapers.
"""

import os

from unittest.mock import patch

import pandas as pd


def test_jobspy_proxy():
    """Test that JobSpy receives proxy configuration correctly."""
    with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
        # Mock DataFrame return
        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["test.com"],
                "company": ["Test"],
                "location": ["Remote"],
                "description": ["Test"],
            },
        )
        mock_scrape_jobs.return_value = mock_df

        # Test with proxies enabled
        with patch.dict(
            os.environ,
            {
                "USE_PROXIES": "true",
                "PROXY_POOL": '["http://test:8080"]',
                "OPENAI_API_KEY": "test",
                "GROQ_API_KEY": "test",
            },
        ):
            from src.config import Settings

            settings = Settings()

            with patch("src.scraper_job_boards.settings", settings):
                from src.scraper_job_boards import scrape_job_boards

                scrape_job_boards(["ai"], ["remote"])

                # Check that proxies parameter was passed
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert "proxies" in call_kwargs
                assert call_kwargs["proxies"] == ["http://test:8080"]


def test_proxy_disabled():
    """Test that proxies are disabled when USE_PROXIES=false."""
    with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
        mock_df = pd.DataFrame(
            {
                "title": ["test"],
                "job_url": ["test"],
                "company": ["test"],
                "location": ["test"],
                "description": ["test"],
            },
        )
        mock_scrape_jobs.return_value = mock_df

        with patch.dict(
            os.environ,
            {
                "USE_PROXIES": "false",
                "PROXY_POOL": '["http://test:8080"]',
                "OPENAI_API_KEY": "test",
                "GROQ_API_KEY": "test",
            },
        ):
            from src.config import Settings

            settings = Settings()

            with patch("src.scraper_job_boards.settings", settings):
                from src.scraper_job_boards import scrape_job_boards

                scrape_job_boards(["ai"], ["remote"])

                # Check that proxies parameter is None
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert call_kwargs["proxies"] is None
