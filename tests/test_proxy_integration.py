"""Comprehensive tests for proxy integration in scrapers.

Tests proxy configuration, rotation, error handling, and integration
with both JobSpy (job boards) and ScrapeGraphAI (company pages) scrapers.
"""

import os

from unittest.mock import MagicMock, patch

import pandas as pd


def test_jobspy_proxy():
    """Test that JobSpy receives proxy configuration correctly."""
    print("Testing JobSpy proxy integration...")

    with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
        # Mock DataFrame return
        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["test.com"],
                "company": ["Test"],
                "location": ["Remote"],
                "description": ["Test"],
            }
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
                print("✓ JobSpy proxy configuration working")


def test_scrapegraphai_proxy():
    """Test that ScrapeGraphAI receives proxy configuration correctly."""
    print("Testing ScrapeGraphAI proxy integration...")

    with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
        mock_instance = MagicMock()
        mock_graph.return_value = mock_instance
        mock_instance.run.return_value = {"https://test.com": {"jobs": []}}

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
            from src.models import CompanySQL

            settings = Settings()

            with (
                patch("src.scraper_company_pages.settings", settings),
                patch(
                    "src.scraper_company_pages.get_proxy",
                    return_value="http://test:8080",
                ),
            ):
                from src.scraper_company_pages import extract_job_lists

                    company = CompanySQL(
                        id=1, name="Test", url="https://test.com", active=True
                    )
                    extract_job_lists({"companies": [company]})

                    # Check that loader_kwargs contains proxy config
                    call_args, call_kwargs = mock_graph.call_args
                    config = (
                        call_args[2]
                        if len(call_args) > 2
                        else call_kwargs.get("config")
                    )
                    assert "loader_kwargs" in config
                    assert "proxy" in config["loader_kwargs"]
                    assert (
                        config["loader_kwargs"]["proxy"]["server"] == "http://test:8080"
                    )
                    print("✓ ScrapeGraphAI proxy configuration working")


def test_proxy_disabled():
    """Test that proxies are disabled when USE_PROXIES=false."""
    print("Testing disabled proxy configuration...")

    with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
        mock_df = pd.DataFrame(
            {
                "title": ["test"],
                "job_url": ["test"],
                "company": ["test"],
                "location": ["test"],
                "description": ["test"],
            }
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
                print("✓ Proxy correctly disabled")


if __name__ == "__main__":
    test_jobspy_proxy()
    test_scrapegraphai_proxy()
    test_proxy_disabled()
    print("\n🎉 All proxy integration tests passed!")
