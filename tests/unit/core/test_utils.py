"""Tests for utils module."""

import time

from unittest.mock import patch

import pytest

from src.core_utils import (
    get_proxy,
    random_delay,
    random_user_agent,
)


@pytest.mark.unit
class TestGetProxy:
    """Test proxy selection."""

    @patch("src.core_utils.settings")
    def test_get_proxy_disabled(self, mock_settings):
        """Test that None is returned when proxies are disabled."""
        mock_settings.use_proxies = False
        mock_settings.proxy_pool = ["proxy1", "proxy2"]

        proxy = get_proxy()

        assert proxy is None

    @patch("src.core_utils.settings")
    def test_get_proxy_empty_pool(self, mock_settings):
        """Test that None is returned when proxy pool is empty."""
        mock_settings.use_proxies = True
        mock_settings.proxy_pool = []

        proxy = get_proxy()

        assert proxy is None

    @patch("src.core_utils.settings")
    @patch("src.core_utils.secrets.choice")
    def test_get_proxy_enabled(self, mock_choice, mock_settings):
        """Test that a proxy is returned when enabled and pool is not empty."""
        mock_settings.use_proxies = True
        mock_settings.proxy_pool = ["proxy1", "proxy2", "proxy3"]
        mock_choice.return_value = "proxy2"

        proxy = get_proxy()

        mock_choice.assert_called_once_with(["proxy1", "proxy2", "proxy3"])
        assert proxy == "proxy2"

    @patch("src.core_utils.settings")
    def test_get_proxy_randomness(self, mock_settings):
        """Test that get_proxy returns different values over multiple calls."""
        mock_settings.use_proxies = True
        mock_settings.proxy_pool = ["proxy1", "proxy2", "proxy3", "proxy4", "proxy5"]

        # Call multiple times and collect results
        results = set()
        for _ in range(20):  # 20 calls should give us some variety
            proxy = get_proxy()
            results.add(proxy)

        # Should get at least 2 different proxies (with high probability)
        assert len(results) >= 2

        # All results should be from the proxy pool
        for result in results:
            assert result in mock_settings.proxy_pool

    def test_all_functions_importable(self):
        """Test that all utility functions can be imported."""
        # This test ensures all functions are properly exported
        functions = [
            get_proxy,
            random_user_agent,
            random_delay,
        ]

        for func in functions:
            assert callable(func)

    @patch("src.core_utils.settings")
    def test_utils_respect_settings(self, mock_settings):
        """Test that utility functions respect settings configuration."""
        # Configure mock settings
        mock_settings.use_proxies = False
        mock_settings.proxy_pool = []

        # Test that functions use the settings
        proxy = get_proxy()
        assert proxy is None  # Should be None when disabled


@pytest.mark.unit
class TestRandomUserAgent:
    """Test random user agent generation."""

    def test_random_user_agent_returns_string(self):
        """Test that random_user_agent returns a non-empty string."""
        user_agent = random_user_agent()

        assert isinstance(user_agent, str)
        assert len(user_agent) > 0

    def test_random_user_agent_contains_browser_info(self):
        """Test that user agent strings contain expected browser information."""
        user_agent = random_user_agent()

        # Should contain typical user agent components
        expected_components = [
            "Mozilla",
            "AppleWebKit",
            "Safari",
        ]

        # At least one of these should be present
        assert any(component in user_agent for component in expected_components)

    def test_random_user_agent_variety(self):
        """Test that random_user_agent returns different values."""
        user_agents = set()

        # Collect user agents from multiple calls
        for _ in range(20):
            user_agent = random_user_agent()
            user_agents.add(user_agent)

        # Should get at least 2 different user agents
        assert len(user_agents) >= 2

    def test_random_user_agent_common_browsers(self):
        """Test that user agents represent common browsers."""
        user_agents = [random_user_agent() for _ in range(50)]  # Get a good sample

        # Check that we get user agents for different browsers/platforms
        has_chrome = any("Chrome" in ua for ua in user_agents)
        _has_firefox = any("Firefox" in ua for ua in user_agents)
        has_safari = any("Safari" in ua for ua in user_agents)

        # Should have at least Chrome and Safari (most common)
        assert has_chrome
        assert has_safari

    def test_random_user_agent_no_empty_strings(self):
        """Test that random_user_agent never returns empty strings."""
        for _ in range(10):
            user_agent = random_user_agent()
            assert user_agent
            assert user_agent.strip() == user_agent  # No leading/trailing whitespace


@pytest.mark.unit
class TestRandomDelay:
    """Test random delay functionality."""

    def test_random_delay_default_parameters(self):
        """Test random_delay with default parameters."""
        start_time = time.time()

        random_delay()

        end_time = time.time()
        elapsed = end_time - start_time

        # Should delay between 1.0 and 5.0 seconds (default)
        assert 1.0 <= elapsed <= 5.5  # Adding small buffer for timing precision

    def test_random_delay_custom_parameters(self):
        """Test random_delay with custom parameters."""
        start_time = time.time()

        random_delay(0.1, 0.3)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should delay between 0.1 and 0.3 seconds
        assert 0.1 <= elapsed <= 0.4  # Adding small buffer for timing precision

    def test_random_delay_zero_minimum(self):
        """Test random_delay with zero minimum."""
        start_time = time.time()

        random_delay(0.0, 0.2)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should delay between 0.0 and 0.2 seconds
        assert 0.0 <= elapsed <= 0.3

    def test_random_delay_equal_min_max(self):
        """Test random_delay when min and max are equal."""
        start_time = time.time()

        random_delay(0.2, 0.2)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should delay exactly 0.2 seconds (within precision)
        assert 0.15 <= elapsed <= 0.25

    @patch("src.core_utils.time.sleep")
    @patch("src.core_utils.random.uniform")
    def test_random_delay_calls_correct_functions(self, mock_uniform, mock_sleep):
        """Test that random_delay calls the correct underlying functions."""
        mock_uniform.return_value = 2.5

        random_delay(1.0, 4.0)

        mock_uniform.assert_called_once_with(1.0, 4.0)
        mock_sleep.assert_called_once_with(2.5)

    def test_random_delay_variability(self):
        """Test that random_delay produces variable delays."""
        delays = []

        for _ in range(5):
            start_time = time.time()
            random_delay(0.1, 0.3)
            end_time = time.time()
            delays.append(end_time - start_time)

        # Should have some variability in delays (not all exactly the same)
        unique_delays = len({round(delay, 2) for delay in delays})
        assert unique_delays >= 2  # Should have at least some variation
