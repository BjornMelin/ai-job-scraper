"""Tests for UI formatter utility functions.

Tests the calculation and formatting utilities used throughout the
scraping dashboard for ETA calculation, duration formatting, and
other display formatting functions.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.ui.utils.formatters import (
    calculate_eta,
    calculate_progress_percentage,
    calculate_scraping_speed,
    format_duration,
    format_jobs_count,
    format_timestamp,
)


class TestCalculateScrapingSpeed:
    """Test the scraping speed calculation function."""

    def test_calculates_correct_speed_with_valid_times(self):
        """Test scraping speed calculation with valid start and end times."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 2, tzinfo=timezone.utc)  # 2 minutes
        jobs_found = 30

        # Act
        speed = calculate_scraping_speed(jobs_found, start_time, end_time)

        # Assert - 30 jobs in 2 minutes = 15 jobs/minute
        assert speed == 15.0

    def test_calculates_speed_with_current_time_when_no_end_time(self):
        """Test speed calculation uses current time when end_time is None."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        jobs_found = 20

        with patch("src.ui.utils.formatters.datetime") as mock_datetime:
            # Mock current time to be 4 minutes after start
            mock_datetime.now.return_value = datetime(
                2024, 1, 1, 10, 4, tzinfo=timezone.utc
            )
            mock_datetime.timezone = timezone

            # Act
            speed = calculate_scraping_speed(jobs_found, start_time, None)

            # Assert - 20 jobs in 4 minutes = 5 jobs/minute
            assert speed == 5.0

    def test_returns_zero_for_invalid_inputs(self):
        """Test function returns zero for invalid input values."""
        # Test cases for invalid inputs
        test_cases = [
            (
                -5,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),  # Negative jobs
            (
                "invalid",
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),  # Non-int jobs
            (10, None, datetime.now(timezone.utc)),  # None start_time
            (10, "invalid", datetime.now(timezone.utc)),  # Invalid start_time
        ]

        for jobs_found, start_time, end_time in test_cases:
            # Act
            speed = calculate_scraping_speed(jobs_found, start_time, end_time)

            # Assert
            assert speed == 0.0

    def test_returns_zero_for_zero_duration(self):
        """Test function returns zero when duration is zero or negative."""
        # Arrange - same start and end time
        same_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        jobs_found = 10

        # Act
        speed = calculate_scraping_speed(jobs_found, same_time, same_time)

        # Assert
        assert speed == 0.0

    def test_returns_zero_for_negative_duration(self):
        """Test function returns zero when end time is before start time."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
        end_time = datetime(
            2024, 1, 1, 10, 0, tzinfo=timezone.utc
        )  # Earlier than start
        jobs_found = 10

        # Act
        speed = calculate_scraping_speed(jobs_found, start_time, end_time)

        # Assert
        assert speed == 0.0

    def test_handles_exceptions_gracefully(self):
        """Test function handles unexpected exceptions gracefully."""
        # Arrange - Create a mock datetime that raises an exception
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.formatters.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Unexpected error")

            # Act
            speed = calculate_scraping_speed(10, start_time, None)

            # Assert
            assert speed == 0.0

    def test_rounds_to_one_decimal_place(self):
        """Test function rounds results to one decimal place."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 3, tzinfo=timezone.utc)  # 3 minutes
        jobs_found = 10  # Should give 3.333... jobs/minute

        # Act
        speed = calculate_scraping_speed(jobs_found, start_time, end_time)

        # Assert
        assert speed == 3.3


class TestCalculateETA:
    """Test the ETA calculation function."""

    def test_calculates_correct_eta_with_valid_inputs(self):
        """Test ETA calculation with valid completion data."""
        # Arrange
        total_companies = 10
        completed_companies = 4
        time_elapsed = 120  # 2 minutes for 4 companies

        # Act
        eta = calculate_eta(total_companies, completed_companies, time_elapsed)

        # Assert - 6 remaining companies at 30s each = 180s = 3m 0s
        assert eta == "3m 0s"

    def test_returns_done_when_all_completed(self):
        """Test function returns 'Done' when all companies are completed."""
        # Arrange
        total_companies = 5
        completed_companies = 5
        time_elapsed = 300

        # Act
        eta = calculate_eta(total_companies, completed_companies, time_elapsed)

        # Assert
        assert eta == "Done"

    def test_returns_done_when_completed_exceeds_total(self):
        """Test function returns 'Done' when completed exceeds total."""
        # Arrange - Edge case
        total_companies = 5
        completed_companies = 7  # More than total
        time_elapsed = 300

        # Act
        eta = calculate_eta(total_companies, completed_companies, time_elapsed)

        # Assert
        assert eta == "Done"

    def test_returns_calculating_for_no_progress(self):
        """Test function returns 'Calculating...' when no progress made."""
        test_cases = [
            (10, 0, 120),  # No completed companies
            (10, 5, 0),  # No elapsed time
        ]

        for total, completed, elapsed in test_cases:
            # Act
            eta = calculate_eta(total, completed, elapsed)

            # Assert
            assert eta == "Calculating..."

    def test_returns_unknown_for_invalid_inputs(self):
        """Test function returns 'Unknown' for invalid input values."""
        test_cases = [
            (-5, 2, 120),  # Negative total
            (0, 2, 120),  # Zero total
            (10, -1, 120),  # Negative completed
            (10, 2, -50),  # Negative elapsed time
            ("invalid", 2, 120),  # Invalid total type
            (10, "invalid", 120),  # Invalid completed type
            (10, 2, "invalid"),  # Invalid elapsed type
        ]

        for total, completed, elapsed in test_cases:
            # Act
            eta = calculate_eta(total, completed, elapsed)

            # Assert
            assert eta == "Unknown"

    def test_handles_exceptions_gracefully(self):
        """Test function handles calculation exceptions gracefully."""
        # Arrange - Create conditions that might cause calculation errors
        total_companies = 10
        completed_companies = 2
        time_elapsed = 60

        with pytest.mock.patch(
            "src.ui.utils.formatters.format_duration"
        ) as mock_format:
            mock_format.side_effect = Exception("Format error")

            # Act
            eta = calculate_eta(total_companies, completed_companies, time_elapsed)

            # Assert
            assert eta == "Unknown"

    def test_calculates_eta_for_large_numbers(self):
        """Test ETA calculation works with large company counts."""
        # Arrange
        total_companies = 1000
        completed_companies = 100  # 10% done
        time_elapsed = 3600  # 1 hour for 100 companies

        # Act
        eta = calculate_eta(total_companies, completed_companies, time_elapsed)

        # Assert - 900 remaining at 36s each = 32400s = 9h 0m
        assert "9h" in eta


class TestFormatDuration:
    """Test the duration formatting function."""

    def test_formats_seconds_only(self):
        """Test formatting for durations less than a minute."""
        test_cases = [
            (0, "0s"),
            (1, "1s"),
            (45, "45s"),
            (59, "59s"),
        ]

        for seconds, expected in test_cases:
            # Act
            result = format_duration(seconds)

            # Assert
            assert result == expected

    def test_formats_minutes_and_seconds(self):
        """Test formatting for durations with minutes."""
        test_cases = [
            (60, "1m"),  # Exactly 1 minute
            (65, "1m 5s"),  # 1 minute 5 seconds
            (125, "2m 5s"),  # 2 minutes 5 seconds
            (180, "3m"),  # Exactly 3 minutes
            (3599, "59m 59s"),  # Just under 1 hour
        ]

        for seconds, expected in test_cases:
            # Act
            result = format_duration(seconds)

            # Assert
            assert result == expected

    def test_formats_hours_and_minutes(self):
        """Test formatting for durations with hours."""
        test_cases = [
            (3600, "1h"),  # Exactly 1 hour
            (3660, "1h 1m"),  # 1 hour 1 minute
            (7200, "2h"),  # Exactly 2 hours
            (7380, "2h 3m"),  # 2 hours 3 minutes
            (90061, "25h 1m"),  # Over 24 hours
        ]

        for seconds, expected in test_cases:
            # Act
            result = format_duration(seconds)

            # Assert
            assert result == expected

    def test_formats_hours_only_when_no_minutes(self):
        """Test hours-only formatting when minutes is zero."""
        # Arrange
        seconds = 7200  # Exactly 2 hours

        # Act
        result = format_duration(seconds)

        # Assert
        assert result == "2h"

    def test_handles_invalid_inputs(self):
        """Test function handles invalid inputs gracefully."""
        test_cases = [
            (-5, "0s"),  # Negative seconds
            ("invalid", "0s"),  # String input
            (None, "0s"),  # None input
            (float("inf"), "0s"),  # Infinity
        ]

        for seconds, expected in test_cases:
            # Act
            result = format_duration(seconds)

            # Assert
            assert result == expected

    def test_handles_float_inputs(self):
        """Test function handles float inputs correctly."""
        test_cases = [
            (65.7, "1m 5s"),  # Float gets truncated
            (125.9, "2m 5s"),  # Float gets truncated
            (3600.5, "1h"),  # Float gets truncated
        ]

        for seconds, expected in test_cases:
            # Act
            result = format_duration(seconds)

            # Assert
            assert result == expected

    def test_handles_exceptions_gracefully(self):
        """Test function handles calculation exceptions gracefully."""
        # Arrange - Mock to raise exception during calculation
        with pytest.mock.patch("builtins.int") as mock_int:
            mock_int.side_effect = Exception("Conversion error")

            # Act
            result = format_duration(120)

            # Assert
            assert result == "0s"


class TestFormatTimestamp:
    """Test the timestamp formatting function."""

    def test_formats_datetime_with_default_format(self):
        """Test datetime formatting with default format string."""
        # Arrange
        dt = datetime(2024, 1, 1, 15, 30, 45, tzinfo=timezone.utc)

        # Act
        result = format_timestamp(dt)

        # Assert
        assert result == "15:30:45"

    def test_formats_datetime_with_custom_format(self):
        """Test datetime formatting with custom format string."""
        # Arrange
        dt = datetime(2024, 1, 1, 15, 30, 45, tzinfo=timezone.utc)
        custom_format = "%Y-%m-%d %H:%M"

        # Act
        result = format_timestamp(dt, custom_format)

        # Assert
        assert result == "2024-01-01 15:30"

    def test_returns_na_for_none_input(self):
        """Test function returns 'N/A' for None input."""
        # Act
        result = format_timestamp(None)

        # Assert
        assert result == "N/A"

    def test_handles_invalid_format_string_gracefully(self):
        """Test function handles invalid format strings gracefully."""
        # Arrange
        dt = datetime(2024, 1, 1, 15, 30, 45, tzinfo=timezone.utc)
        invalid_format = "%invalid"

        # Act
        result = format_timestamp(dt, invalid_format)

        # Assert
        assert result == "N/A"

    def test_handles_exceptions_gracefully(self):
        """Test function handles formatting exceptions gracefully."""
        # Arrange - Create a mock datetime that raises an exception
        dt = datetime(2024, 1, 1, 15, 30, 45, tzinfo=timezone.utc)

        with pytest.mock.patch.object(dt, "strftime") as mock_strftime:
            mock_strftime.side_effect = Exception("Format error")

            # Act
            result = format_timestamp(dt)

            # Assert
            assert result == "N/A"


class TestCalculateProgressPercentage:
    """Test the progress percentage calculation function."""

    def test_calculates_correct_percentage(self):
        """Test progress percentage calculation with valid inputs."""
        test_cases = [
            (0, 10, 0.0),  # No progress
            (5, 10, 50.0),  # Half progress
            (10, 10, 100.0),  # Complete
            (3, 7, 42.9),  # Decimal result rounded
        ]

        for completed, total, expected in test_cases:
            # Act
            result = calculate_progress_percentage(completed, total)

            # Assert
            assert result == expected

    def test_handles_edge_cases(self):
        """Test function handles edge cases correctly."""
        test_cases = [
            (0, 0, 0.0),  # Both zero
            (5, 0, 0.0),  # Zero total (division by zero)
            (-1, 10, 0.0),  # Negative completed
            (5, -1, 0.0),  # Negative total
        ]

        for completed, total, expected in test_cases:
            # Act
            result = calculate_progress_percentage(completed, total)

            # Assert
            assert result == expected

    def test_caps_at_100_percent(self):
        """Test function caps percentage at 100% when completed exceeds total."""
        # Arrange
        completed = 15
        total = 10

        # Act
        result = calculate_progress_percentage(completed, total)

        # Assert
        assert result == 100.0

    def test_handles_invalid_types(self):
        """Test function handles invalid input types gracefully."""
        test_cases = [
            ("invalid", 10, 0.0),
            (5, "invalid", 0.0),
            (None, 10, 0.0),
            (5, None, 0.0),
        ]

        for completed, total, expected in test_cases:
            # Act
            result = calculate_progress_percentage(completed, total)

            # Assert
            assert result == expected

    def test_handles_exceptions_gracefully(self):
        """Test function handles calculation exceptions gracefully."""
        # Arrange - Mock to raise exception during calculation
        with pytest.mock.patch("builtins.min") as mock_min:
            mock_min.side_effect = Exception("Calculation error")

            # Act
            result = calculate_progress_percentage(5, 10)

            # Assert
            assert result == 0.0


class TestFormatJobsCount:
    """Test the jobs count formatting function."""

    def test_formats_singular_correctly(self):
        """Test formatting uses singular form for count of 1."""
        # Act
        result = format_jobs_count(1)

        # Assert
        assert result == "1 job"

    def test_formats_plural_correctly(self):
        """Test formatting uses plural form for counts other than 1."""
        test_cases = [
            (0, "0 jobs"),
            (2, "2 jobs"),
            (25, "25 jobs"),
            (100, "100 jobs"),
        ]

        for count, expected in test_cases:
            # Act
            result = format_jobs_count(count)

            # Assert
            assert result == expected

    def test_supports_custom_singular_plural(self):
        """Test formatting supports custom singular and plural forms."""
        # Act
        result = format_jobs_count(1, singular="company", plural="companies")

        # Assert
        assert result == "1 company"

        # Act
        result = format_jobs_count(5, singular="company", plural="companies")

        # Assert
        assert result == "5 companies"

    def test_handles_invalid_count_gracefully(self):
        """Test function handles invalid count values gracefully."""
        test_cases = [
            ("invalid", "0 jobs"),
            (None, "0 jobs"),
            (-5, "-5 jobs"),  # Negative numbers still get formatted
        ]

        for count, expected in test_cases:
            # Act
            result = format_jobs_count(count)

            # Assert
            assert result == expected

    def test_handles_exceptions_gracefully(self):
        """Test function handles formatting exceptions gracefully."""
        # Arrange - Create conditions that might cause formatting errors
        count = 5

        with pytest.mock.patch("builtins.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = Exception("Type check error")

            # Act
            result = format_jobs_count(count)

            # Assert
            assert result == "0 jobs"  # Fallback value


class TestFormatterUtilitiesIntegration:
    """Integration tests for formatter utilities working together."""

    def test_realistic_scraping_dashboard_scenario(self):
        """Test formatters work together in realistic dashboard scenario."""
        # Arrange - Simulate real scraping data
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 15, tzinfo=timezone.utc)  # 15 minutes
        jobs_found = 75

        # Calculate metrics
        speed = calculate_scraping_speed(jobs_found, start_time, end_time)
        duration = format_duration(900)  # 15 minutes
        timestamp = format_timestamp(start_time)
        jobs_text = format_jobs_count(jobs_found)
        progress = calculate_progress_percentage(3, 5)  # 3 of 5 companies done
        eta = calculate_eta(5, 3, 900)  # 3 of 5 companies in 15 minutes

        # Assert all formatters work correctly together
        assert speed == 5.0  # 75 jobs in 15 minutes
        assert duration == "15m"
        assert timestamp == "10:00:00"
        assert jobs_text == "75 jobs"
        assert progress == 60.0  # 3/5 = 60%
        assert eta == "10m 0s"  # 2 remaining at 450s each

    def test_formatters_handle_edge_cases_consistently(self):
        """Test all formatters handle edge cases consistently."""
        # Test with zero/empty values
        speed = calculate_scraping_speed(0, None, None)
        duration = format_duration(0)
        timestamp = format_timestamp(None)
        jobs_text = format_jobs_count(0)
        progress = calculate_progress_percentage(0, 0)
        eta = calculate_eta(0, 0, 0)

        # Assert consistent handling of empty/zero values
        assert speed == 0.0
        assert duration == "0s"
        assert timestamp == "N/A"
        assert jobs_text == "0 jobs"
        assert progress == 0.0
        assert eta == "Unknown"

    def test_formatters_maintain_precision_and_readability(self):
        """Test formatters balance precision with readability."""
        # Arrange - Test values that require rounding/formatting decisions
        speed = calculate_scraping_speed(
            17,
            datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 10, 3, tzinfo=timezone.utc),  # 3 minutes
        )
        duration = format_duration(3725)  # 1h 2m 5s
        progress = calculate_progress_percentage(7, 13)  # ~53.8%

        # Assert appropriate precision for UI display
        assert speed == 5.7  # Rounded to 1 decimal
        assert duration == "1h 2m"  # Drops seconds for readability at hour level
        assert progress == 53.8  # Rounded to 1 decimal

    def test_formatters_handle_boundary_conditions(self):
        """Test formatters handle boundary conditions correctly."""
        # Test maximum/minimum realistic values
        large_speed = calculate_scraping_speed(
            1000,
            datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 10, 1, tzinfo=timezone.utc),  # 1 minute
        )
        long_duration = format_duration(86400)  # 24 hours
        full_progress = calculate_progress_percentage(100, 100)
        long_eta = calculate_eta(100, 10, 3600)  # 10 of 100 in 1 hour

        # Assert boundary conditions are handled appropriately
        assert large_speed == 1000.0  # High speeds are supported
        assert long_duration == "24h"  # Long durations format correctly
        assert full_progress == 100.0  # Full progress handled
        assert "54h" in long_eta  # Long ETAs calculate correctly
