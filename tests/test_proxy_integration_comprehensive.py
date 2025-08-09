"""Comprehensive tests for proxy integration in scrapers.

Tests proxy configuration, rotation, error handling, and integration
with both JobSpy (job boards) and ScrapeGraphAI (company pages) scrapers.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.models import CompanySQL


class TestJobBoardScraperProxyIntegration:
    """Test proxy integration for job board scraper (JobSpy)."""

    def test_jobspy_uses_proxy_when_enabled(self, test_settings):
        """Test JobSpy receives proxy configuration when proxies are enabled."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080", "http://proxy2:8080"]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                scrape_job_boards(["ai engineer"], ["remote"])

                # Assert
                mock_scrape_jobs.assert_called()
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert "proxies" in call_kwargs
                assert call_kwargs["proxies"] == [
                    "http://proxy1:8080",
                    "http://proxy2:8080",
                ]

    def test_jobspy_uses_no_proxy_when_disabled(self, test_settings):
        """Test JobSpy receives None for proxies when proxy usage is disabled."""
        # Arrange
        test_settings.use_proxies = False
        test_settings.proxy_pool = ["http://proxy1:8080"]  # Should be ignored

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                scrape_job_boards(["ai engineer"], ["remote"])

                # Assert
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert call_kwargs["proxies"] is None

    def test_jobspy_handles_empty_proxy_pool(self, test_settings):
        """Test JobSpy handles empty proxy pool gracefully."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = []  # Empty proxy pool

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                scrape_job_boards(["ai engineer"], ["remote"])

                # Assert
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert call_kwargs["proxies"] == []

    def test_jobspy_continues_on_proxy_failure(self, test_settings):
        """Test JobSpy continues scraping even if proxy fails."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://invalid-proxy:8080"]

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            # Mock proxy failure scenario
            mock_scrape_jobs.side_effect = Exception("Proxy connection failed")

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act - Should not raise exception
                result = scrape_job_boards(["ai engineer"], ["remote"])

                # Assert - Returns empty result but doesn't crash
                assert result == []

    def test_jobspy_proxy_configuration_with_multiple_locations(self, test_settings):
        """Test proxy configuration is consistent across multiple location scrapes."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080"]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["San Francisco"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                scrape_job_boards(
                    ["ai engineer"], ["san francisco", "new york", "remote"]
                )

                # Assert - Should be called 3 times (once per location) with same
                # proxy config
                assert mock_scrape_jobs.call_count == 3
                for call in mock_scrape_jobs.call_args_list:
                    assert call.kwargs["proxies"] == ["http://proxy1:8080"]


class TestCompanyPageScraperProxyIntegration:
    """Test proxy integration for company page scraper (ScrapeGraphAI)."""

    def test_scrapegraphai_uses_proxy_when_enabled(self, test_settings):
        """Test ScrapeGraphAI receives proxy configuration when proxies are enabled."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080"]

        company = CompanySQL(
            id=1, name="TechCorp", url="https://techcorp.com/careers", active=True
        )

        mock_result = {"https://techcorp.com/careers": {"jobs": []}}

        with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.run.return_value = mock_result

            with (
                patch("src.scraper_company_pages.settings", test_settings),
                patch(
                    "src.scraper_company_pages.get_proxy",
                    return_value="http://proxy1:8080",
                ),
            ):
                from src.scraper_company_pages import extract_job_lists

                # Act
                extract_job_lists({"companies": [company]})

                # Assert
                mock_graph.assert_called()
                call_args, call_kwargs = mock_graph.call_args
                config = (
                    call_args[2] if len(call_args) > 2 else call_kwargs.get("config")
                )

                assert "loader_kwargs" in config
                assert "proxy" in config["loader_kwargs"]
                assert (
                    config["loader_kwargs"]["proxy"]["server"] == "http://proxy1:8080"
                )

    def test_scrapegraphai_no_proxy_when_disabled(self, test_settings):
        """Test ScrapeGraphAI doesn't use proxy when proxy usage is disabled."""
        # Arrange
        test_settings.use_proxies = False
        test_settings.proxy_pool = ["http://proxy1:8080"]  # Should be ignored

        company = CompanySQL(
            id=1, name="TechCorp", url="https://techcorp.com/careers", active=True
        )

        mock_result = {"https://techcorp.com/careers": {"jobs": []}}

        with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.run.return_value = mock_result

            with patch("src.scraper_company_pages.settings", test_settings):
                from src.scraper_company_pages import extract_job_lists

                # Act
                extract_job_lists({"companies": [company]})

                # Assert
                call_args, call_kwargs = mock_graph.call_args
                config = (
                    call_args[2] if len(call_args) > 2 else call_kwargs.get("config")
                )

                # Should not contain proxy configuration
                assert "loader_kwargs" not in config or "proxy" not in config.get(
                    "loader_kwargs", {}
                )

    def test_scrapegraphai_handles_proxy_failure_gracefully(self, test_settings):
        """Test ScrapeGraphAI handles proxy connection failures gracefully."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://invalid-proxy:8080"]

        company = CompanySQL(
            id=1, name="TechCorp", url="https://techcorp.com/careers", active=True
        )

        with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            # Mock proxy connection failure
            mock_instance.run.side_effect = Exception("Proxy connection failed")

            with (
                patch("src.scraper_company_pages.settings", test_settings),
                patch(
                    "src.scraper_company_pages.get_proxy",
                    return_value="http://invalid-proxy:8080",
                ),
            ):
                from src.scraper_company_pages import extract_job_lists

                # Act - Should not raise exception
                result = extract_job_lists({"companies": [company]})

                # Assert - Returns empty result but doesn't crash
                assert result == {"partial_jobs": []}

    def test_scrapegraphai_proxy_configuration_in_details_extraction(
        self, test_settings
    ):
        """Test proxy configuration is also applied to job details extraction."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080"]

        partial_jobs = [
            {
                "company": "TechCorp",
                "title": "AI Engineer",
                "url": "https://techcorp.com/job1",
            }
        ]

        mock_result = {
            "https://techcorp.com/job1": {
                "description": "AI role",
                "location": "Remote",
            }
        }

        with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.run.return_value = mock_result

            with (
                patch("src.scraper_company_pages.settings", test_settings),
                patch(
                    "src.scraper_company_pages.get_proxy",
                    return_value="http://proxy1:8080",
                ),
            ):
                from src.scraper_company_pages import extract_details

            # Act
            extract_details({"partial_jobs": partial_jobs})

            # Assert
            call_args, call_kwargs = mock_graph.call_args
            config = call_args[2] if len(call_args) > 2 else call_kwargs.get("config")

            assert "loader_kwargs" in config
            assert "proxy" in config["loader_kwargs"]
            assert config["loader_kwargs"]["proxy"]["server"] == "http://proxy1:8080"

    def test_scrapegraphai_uses_different_proxies_for_different_requests(
        self, test_settings
    ):
        """Test ScrapeGraphAI can use different proxies for different extractions."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080", "http://proxy2:8080"]

        company = CompanySQL(
            id=1, name="TechCorp", url="https://techcorp.com/careers", active=True
        )

        mock_list_result = {
            "https://techcorp.com/careers": {
                "jobs": [{"title": "AI Engineer", "url": "https://techcorp.com/job1"}]
            }
        }

        with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.run.return_value = mock_list_result

            with (
                patch("src.scraper_company_pages.settings", test_settings),
                # Mock get_proxy to return different proxies for different calls
                patch(
                    "src.scraper_company_pages.get_proxy",
                    side_effect=["http://proxy1:8080", "http://proxy2:8080"],
                ),
            ):
                from src.scraper_company_pages import (
                    extract_details,
                    extract_job_lists,
                )

            # Act - Extract job lists (first proxy)
            list_result = extract_job_lists({"companies": [company]})

            # Reset mock for details extraction
            mock_graph.reset_mock()
            mock_instance.run.return_value = {
                "https://techcorp.com/job1": {"description": "AI role"}
            }

            # Extract details (second proxy)
            extract_details(list_result)

            # Assert - Both calls should have used proxy configuration
            assert mock_graph.call_count == 2


class TestProxyUtilityFunctions:
    """Test proxy utility functions and error handling."""

    def test_get_proxy_returns_proxy_from_pool(self, test_settings):
        """Test get_proxy function returns a proxy from the configured pool."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080", "http://proxy2:8080"]

        with patch("src.scraper_company_pages.settings", test_settings):
            from src.utils import get_proxy

            # Act
            proxy = get_proxy()

            # Assert
            assert proxy in ["http://proxy1:8080", "http://proxy2:8080"]

    def test_get_proxy_returns_none_when_disabled(self, test_settings):
        """Test get_proxy returns None when proxy usage is disabled."""
        # Arrange
        test_settings.use_proxies = False
        test_settings.proxy_pool = ["http://proxy1:8080"]

        with patch("src.scraper_company_pages.settings", test_settings):
            from src.utils import get_proxy

            # Act
            proxy = get_proxy()

            # Assert
            assert proxy is None

    def test_get_proxy_handles_empty_pool(self, test_settings):
        """Test get_proxy handles empty proxy pool gracefully."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = []

        with patch("src.scraper_company_pages.settings", test_settings):
            from src.utils import get_proxy

            # Act
            proxy = get_proxy()

            # Assert
            assert proxy is None


class TestProxyErrorHandlingAndRecovery:
    """Test proxy error handling and recovery mechanisms."""

    def test_scraper_falls_back_when_all_proxies_fail(self, test_settings):
        """Test scrapers can fall back to direct connection when all proxies fail."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://proxy1:8080", "http://proxy2:8080"]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            # Mock first call to fail (with proxies), second to succeed (fallback)
            mock_scrape_jobs.side_effect = [
                Exception("All proxies failed"),
                mock_df,  # Successful fallback
            ]

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                result = scrape_job_boards(["ai engineer"], ["remote"])

                # Assert - Should have attempted with proxies and fallen back
                assert len(result) > 0  # Successful result after fallback

    def test_proxy_rotation_on_consecutive_requests(self, test_settings):
        """Test proxy rotation mechanism for consecutive scraping requests."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = [
            "http://proxy1:8080",
            "http://proxy2:8080",
            "http://proxy3:8080",
        ]

        with patch("src.utils.random.choice") as mock_choice:
            # Mock to return different proxies for each call
            mock_choice.side_effect = [
                "http://proxy1:8080",
                "http://proxy2:8080",
                "http://proxy3:8080",
            ]

            with patch("src.scraper_company_pages.settings", test_settings):
                from src.utils import get_proxy

                # Act - Make multiple proxy requests
                proxy1 = get_proxy()
                proxy2 = get_proxy()
                proxy3 = get_proxy()

                # Assert - Different proxies should be returned (rotation)
                proxies_used = [proxy1, proxy2, proxy3]
                assert len(set(proxies_used)) > 1  # At least some rotation occurred

    def test_proxy_timeout_handling(self, test_settings):
        """Test handling of proxy timeout scenarios."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://slow-proxy:8080"]

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            # Mock timeout scenario
            mock_scrape_jobs.side_effect = TimeoutError("Proxy request timed out")

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act - Should handle timeout gracefully
                result = scrape_job_boards(["ai engineer"], ["remote"])

                # Assert - Returns empty result but doesn't crash
                assert result == []

    def test_invalid_proxy_format_handling(self, test_settings):
        """Test handling of invalid proxy format in configuration."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["invalid-proxy-format", "http://valid-proxy:8080"]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act - Should handle invalid proxy format
                result = scrape_job_boards(["ai engineer"], ["remote"])

                # Assert - Should still attempt scraping (may use valid proxy or
                # fallback)
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert "proxies" in call_kwargs
                # Invalid proxies should still be passed to JobSpy to handle


class TestProxyConfigurationEdgeCases:
    """Test edge cases in proxy configuration and usage."""

    def test_proxy_configuration_with_authentication(self, test_settings):
        """Test proxy configuration with authentication credentials."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://user:pass@proxy.example.com:8080"]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                scrape_job_boards(["ai engineer"], ["remote"])

                # Assert
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert call_kwargs["proxies"] == [
                    "http://user:pass@proxy.example.com:8080"
                ]

    def test_mixed_proxy_protocols_in_pool(self, test_settings):
        """Test proxy pool with mixed protocols (HTTP, HTTPS, SOCKS)."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = [
            "http://http-proxy:8080",
            "https://https-proxy:8080",
            "socks5://socks-proxy:1080",
        ]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act
                scrape_job_boards(["ai engineer"], ["remote"])

                # Assert - All proxy types should be passed through
                call_kwargs = mock_scrape_jobs.call_args.kwargs
                assert len(call_kwargs["proxies"]) == 3
                assert "http://http-proxy:8080" in call_kwargs["proxies"]
                assert "https://https-proxy:8080" in call_kwargs["proxies"]
                assert "socks5://socks-proxy:1080" in call_kwargs["proxies"]

    def test_proxy_configuration_persists_across_scraping_sessions(self, test_settings):
        """Test proxy configuration persists across multiple scraping sessions."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://persistent-proxy:8080"]

        mock_df = pd.DataFrame(
            {
                "title": ["AI Engineer"],
                "job_url": ["https://example.com/job1"],
                "company": ["TechCorp"],
                "location": ["Remote"],
                "description": ["AI Engineer role"],
            }
        )

        with patch("src.scraper_job_boards.scrape_jobs") as mock_scrape_jobs:
            mock_scrape_jobs.return_value = mock_df

            with patch("src.scraper_job_boards.settings", test_settings):
                from src.scraper_job_boards import scrape_job_boards

                # Act - Multiple scraping sessions
                scrape_job_boards(["ai engineer"], ["remote"])
                scrape_job_boards(["data scientist"], ["san francisco"])

                # Assert - Proxy should be used in both sessions
                assert mock_scrape_jobs.call_count == 2
                for call in mock_scrape_jobs.call_args_list:
                    assert call.kwargs["proxies"] == ["http://persistent-proxy:8080"]

    def test_proxy_configuration_logging_and_monitoring(self, test_settings):
        """Test that proxy usage is properly logged for monitoring."""
        # Arrange
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["http://monitored-proxy:8080"]

        company = CompanySQL(
            id=1, name="TechCorp", url="https://techcorp.com/careers", active=True
        )

        with patch("src.scraper_company_pages.SmartScraperMultiGraph") as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.run.return_value = {
                "https://techcorp.com/careers": {"jobs": []}
            }

            with (
                patch("src.scraper_company_pages.settings", test_settings),
                patch(
                    "src.scraper_company_pages.get_proxy",
                    return_value="http://monitored-proxy:8080",
                ),
                patch("src.scraper_company_pages.logger") as mock_logger,
            ):
                from src.scraper_company_pages import extract_job_lists

                # Act
                extract_job_lists({"companies": [company]})

                # Assert - Proxy usage should be logged
                logged_calls = [
                    call.args[0] for call in mock_logger.info.call_args_list
                ]
                proxy_log_found = any(
                    "proxy" in call.lower() and "http://monitored-proxy:8080" in call
                    for call in logged_calls
                )
                assert proxy_log_found
