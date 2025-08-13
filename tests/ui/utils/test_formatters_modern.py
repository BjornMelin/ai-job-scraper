"""Modernized UI formatters tests focused on essential functionality.

This module replaces the overly complex 1200+ line formatters test
with focused, maintainable tests covering core formatting operations.
"""

from datetime import UTC, datetime

import pytest

from src.ui.utils.formatters import (
    format_company_stats,
    format_date_relative,
    format_salary_range,
    truncate_text,
)


class TestSalaryFormatting:
    """Test salary range formatting."""

    @pytest.mark.parametrize(
        ("salary_input", "expected"),
        (
            ((100000, 150000), "$100,000 - $150,000"),
            ((100000, None), "$100,000+"),
            ((None, 150000), "Up to $150,000"),
            ((None, None), "Not specified"),
            (None, "Not specified"),
        ),
    )
    def test_format_salary_range(self, salary_input, expected):
        """Test salary range formatting with various inputs."""
        result = format_salary_range(salary_input)
        assert result == expected


class TestDateFormatting:
    """Test date formatting utilities."""

    def test_format_recent_date(self):
        """Test formatting recent dates."""
        now = datetime.now(UTC)
        result = format_date_relative(now)
        assert "ago" in result or "Just now" in result

    def test_format_none_date(self):
        """Test formatting None date."""
        result = format_date_relative(None)
        assert result == "Unknown"


class TestTextFormatting:
    """Test text formatting utilities."""

    @pytest.mark.parametrize(
        ("text", "max_length", "expected"),
        (
            ("Short text", 50, "Short text"),
            (
                "This is a very long text that should be truncated",
                20,
                "This is a very lo...",
            ),
            ("", 10, ""),
            (None, 10, ""),
        ),
    )
    def test_truncate_text(self, text, max_length, expected):
        """Test text truncation with various inputs."""
        result = truncate_text(text, max_length)
        assert result == expected


class TestCompanyStatsFormatting:
    """Test company statistics formatting."""

    def test_format_company_stats_basic(self):
        """Test basic company stats formatting."""
        stats = {"total_jobs": 25, "active_companies": 5, "success_rate": 0.85}

        result = format_company_stats(stats)

        assert isinstance(result, dict)
        assert "total_jobs" in result
        assert "active_companies" in result
