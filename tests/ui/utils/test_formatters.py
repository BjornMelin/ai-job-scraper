"""Tests for UI formatter utility functions.

Tests the calculation and formatting utilities used throughout the
scraping dashboard for ETA calculation, duration formatting, and
other display formatting functions.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.ui.utils.formatters import (
    calculate_eta,
    calculate_progress_percentage,
    calculate_scraping_speed,
    format_duration,
    format_jobs_count,
    format_salary,
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

        # Assert - 6 remaining companies at 30s each = 180s = 3m
        assert eta == "3m"

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
        # Arrange - Use invalid input types to trigger the exception path naturally
        result1 = calculate_eta("invalid", 2, 60)
        result2 = calculate_eta(10, "invalid", 60)
        result3 = calculate_eta(10, 2, "invalid")

        # Assert
        assert result1 == "Unknown"
        assert result2 == "Unknown"
        assert result3 == "Unknown"

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
        # Arrange - Use invalid input types to trigger the exception path naturally
        result1 = format_duration("invalid")
        result2 = format_duration(None)

        # Assert
        assert result1 == "0s"
        assert result2 == "0s"


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
        invalid_format = "%Q"  # Invalid format specifier

        # Act
        result = format_timestamp(dt, invalid_format)

        # Assert - Invalid format just returns the format string itself
        assert result == "%Q"

    def test_handles_exceptions_gracefully(self):
        """Test function handles formatting exceptions gracefully."""
        # Arrange - Test with None and invalid format
        result1 = format_timestamp(None)
        result2 = format_timestamp("invalid")

        # Assert
        assert result1 == "N/A"
        assert result2 == "N/A"


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
        # Arrange - Use invalid input types to trigger the exception path naturally
        result1 = calculate_progress_percentage("invalid", 10)
        result2 = calculate_progress_percentage(5, "invalid")

        # Assert
        assert result1 == 0.0
        assert result2 == 0.0


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
        # Arrange - Use invalid input types to trigger exception path naturally
        result1 = format_jobs_count(None)
        result2 = format_jobs_count("invalid")

        # Assert - Function should still return valid output
        assert result1 == "0 jobs"
        assert result2 == "0 jobs"


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
        assert eta == "10m"  # 2 remaining at 300s each (5m per company)

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
        assert "9h" in long_eta  # Long ETAs calculate correctly


class TestFormatSalary:
    """Test the salary formatting function."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0, "$0"),
            (500, "$500"),
            (999, "$999"),
            (1000, "$1k"),
            (1500, "$1k"),
            (75000, "$75k"),
            (125000, "$125k"),
            (999000, "$999k"),
            (1000000, "$1.0M"),
            (1200000, "$1.2M"),
            (1250000, "$1.2M"),  # 1.25 rounds to 1.2
            (1260000, "$1.3M"),  # Rounds up
            (2500000, "$2.5M"),
            (10000000, "$10.0M"),
        ],
    )
    def test_formats_salary_amounts_correctly(self, amount, expected):
        """Test salary formatting for various amounts."""
        # Act
        result = format_salary(amount)

        # Assert
        assert result == expected

    def test_handles_negative_amounts(self):
        """Test function handles negative salary amounts gracefully."""
        # Act
        result = format_salary(-50000)

        # Assert
        assert result == "$0"

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "invalid",
            None,
            3.14,  # Float input
            "75000",  # String number
            [],  # List
            {},  # Dict
        ],
    )
    def test_handles_invalid_input_types(self, invalid_input):
        """Test function handles invalid input types gracefully."""
        # Act
        result = format_salary(invalid_input)

        # Assert
        assert result == "$0"

    def test_handles_very_large_amounts(self):
        """Test function handles very large salary amounts."""
        # Act
        result = format_salary(999999999)  # Nearly $1B

        # Assert
        assert result == "$1000.0M"

    def test_handles_boundary_values(self):
        """Test function handles boundary values correctly."""
        test_cases = [
            (999, "$999"),  # Just under $1k
            (1000, "$1k"),  # Exactly $1k
            (1001, "$1k"),  # Just over $1k
            (999999, "$999k"),  # Just under $1M
            (1000000, "$1.0M"),  # Exactly $1M
            (1000001, "$1.0M"),  # Just over $1M
        ]

        for amount, expected in test_cases:
            # Act
            result = format_salary(amount)

            # Assert
            assert result == expected, f"Failed for amount {amount}"

    def test_handles_exceptions_gracefully(self):
        """Test function handles calculation exceptions gracefully."""
        # Arrange - Use invalid input types to trigger the exception path naturally
        result1 = format_salary("invalid")
        result2 = format_salary(None)

        # Assert
        assert result1 == "$0"
        assert result2 == "$0"

    def test_realistic_tech_salary_ranges(self):
        """Test function with realistic tech industry salary ranges."""
        # Common tech salaries for different roles
        test_cases = [
            (65000, "$65k"),  # Junior developer
            (95000, "$95k"),  # Mid-level developer
            (140000, "$140k"),  # Senior developer
            (180000, "$180k"),  # Staff engineer
            (250000, "$250k"),  # Principal engineer
            (350000, "$350k"),  # Engineering manager
            (500000, "$500k"),  # Director level
            (1200000, "$1.2M"),  # VP level
        ]

        for amount, expected in test_cases:
            # Act
            result = format_salary(amount)

            # Assert
            assert result == expected, f"Failed for salary {amount}"


class TestFormattersRealWorldScenarios:
    """Test formatters with real-world job scraping scenarios."""

    def test_job_posting_timestamp_formatting(self):
        """Test formatting job posting timestamps for different time zones."""
        # Test cases for different time zones (all converted to UTC for storage)
        test_cases = [
            (datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc), "09:30:00"),
            (datetime(2024, 3, 20, 14, 45, 30, tzinfo=timezone.utc), "14:45:30"),
            (datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc), "23:59:59"),
        ]

        for dt, expected in test_cases:
            # Act
            result = format_timestamp(dt)

            # Assert
            assert result == expected

    def test_job_posting_date_formatting(self):
        """Test formatting job posting dates for display."""
        # Test custom date formats for job postings
        dt = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

        test_cases = [
            ("%Y-%m-%d", "2024-01-15"),  # ISO date
            ("%B %d, %Y", "January 15, 2024"),  # Human readable
            ("%m/%d/%Y", "01/15/2024"),  # US format
            ("%d/%m/%Y", "15/01/2024"),  # European format
        ]

        for format_str, expected in test_cases:
            # Act
            result = format_timestamp(dt, format_str)

            # Assert
            assert result == expected

    def test_salary_range_formatting(self):
        """Test formatting salary ranges commonly found in job postings."""
        # Common salary range scenarios
        test_cases = [
            # Entry level positions
            (45000, "$45k"),
            (55000, "$55k"),
            # Mid-level positions
            (75000, "$75k"),
            (95000, "$95k"),
            # Senior positions
            (125000, "$125k"),
            (155000, "$155k"),
            # Executive positions
            (250000, "$250k"),
            (350000, "$350k"),
            # C-level positions
            (1000000, "$1.0M"),
            (2500000, "$2.5M"),
        ]

        for amount, expected in test_cases:
            # Act
            result = format_salary(amount)

            # Assert
            assert result == expected

    @pytest.mark.parametrize(
        ("jobs_found", "duration_seconds", "expected_speed"),
        [
            (10, 60, 10.0),  # 10 jobs in 1 minute = 10 jobs/min
            (50, 300, 10.0),  # 50 jobs in 5 minutes = 10 jobs/min
            (100, 600, 10.0),  # 100 jobs in 10 minutes = 10 jobs/min
            (25, 150, 10.0),  # 25 jobs in 2.5 minutes = 10 jobs/min
            (1, 30, 2.0),  # 1 job in 30 seconds = 2 jobs/min
            (200, 1200, 10.0),  # 200 jobs in 20 minutes = 10 jobs/min
        ],
    )
    def test_consistent_scraping_speeds(
        self, jobs_found, duration_seconds, expected_speed
    ):
        """Test scraping speed calculations with various realistic scenarios."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        end_time = start_time + timedelta(seconds=duration_seconds)

        # Act
        speed = calculate_scraping_speed(jobs_found, start_time, end_time)

        # Assert
        assert speed == expected_speed

    def test_progress_tracking_during_batch_scraping(self):
        """Test progress tracking during realistic batch scraping scenarios."""
        # Scenario: Scraping 50 companies, various progress states
        total_companies = 50
        test_cases = [
            (0, 0.0, "Just started"),
            (5, 10.0, "Early progress"),
            (12, 24.0, "Quarter done"),
            (25, 50.0, "Half way"),
            (37, 74.0, "Three quarters"),
            (48, 96.0, "Almost done"),
            (50, 100.0, "Complete"),
        ]

        for completed, expected_percentage, description in test_cases:
            # Act
            percentage = calculate_progress_percentage(completed, total_companies)

            # Assert
            assert percentage == expected_percentage, f"Failed at {description}"

    def test_eta_calculation_for_large_scraping_jobs(self):
        """Test ETA calculations for large-scale scraping operations."""
        # Scenario: Large scraping job with 500 companies
        total_companies = 500
        test_cases = [
            (50, 3600, "9h"),  # 50 companies in 1 hour -> 9h remaining
            (100, 7200, "8h"),  # 100 companies in 2 hours -> 8h remaining
            (250, 18000, "5h"),  # 250 companies in 5 hours -> 5h remaining
            (450, 32400, "1h"),  # 450 companies in 9 hours -> 1h remaining
        ]

        for completed, time_elapsed, expected_hour_part in test_cases:
            # Act
            eta = calculate_eta(total_companies, completed, time_elapsed)

            # Assert
            assert expected_hour_part in eta, (
                f"Expected {expected_hour_part} in ETA: {eta}"
            )

    def test_jobs_count_display_variations(self):
        """Test job count displays for various scenarios."""
        # Test different count scenarios with custom labels
        test_cases = [
            (0, "job", "jobs", "0 jobs"),
            (1, "job", "jobs", "1 job"),
            (5, "job", "jobs", "5 jobs"),
            (1000, "job", "jobs", "1000 jobs"),
            (1, "position", "positions", "1 position"),
            (25, "position", "positions", "25 positions"),
            (1, "opening", "openings", "1 opening"),
            (100, "opening", "openings", "100 openings"),
        ]

        for count, singular, plural, expected in test_cases:
            # Act
            result = format_jobs_count(count, singular, plural)

            # Assert
            assert result == expected


class TestFormattersInternationalFormats:
    """Test formatters with international formats and edge cases."""

    def test_timestamp_formatting_with_different_locales(self):
        """Test timestamp formatting that would work across different locales."""
        # Test various timestamp formats that are commonly used internationally
        dt = datetime(2024, 3, 15, 14, 30, 45, tzinfo=timezone.utc)

        test_cases = [
            ("%H:%M:%S", "14:30:45"),  # 24-hour format (international)
            ("%I:%M:%S %p", "02:30:45 PM"),  # 12-hour format (US)
            ("%Y-%m-%d %H:%M:%S", "2024-03-15 14:30:45"),  # ISO format
            ("%d.%m.%Y %H:%M", "15.03.2024 14:30"),  # German format
            ("%d/%m/%Y %H:%M", "15/03/2024 14:30"),  # UK format
            ("%m/%d/%Y %I:%M %p", "03/15/2024 02:30 PM"),  # US format
        ]

        for format_str, expected in test_cases:
            # Act
            result = format_timestamp(dt, format_str)

            # Assert
            assert result == expected, f"Failed for format {format_str}"

    def test_large_salary_formatting_edge_cases(self):
        """Test salary formatting with very large amounts and edge cases."""
        test_cases = [
            # Edge cases around formatting thresholds
            (999, "$999"),  # Just under 1k
            (1000, "$1k"),  # Exactly 1k
            (1001, "$1k"),  # Just over 1k (truncated)
            (1999, "$1k"),  # Truncated to 1k (uses integer division)
            (2000, "$2k"),  # Rounded to 2k
            (999999, "$999k"),  # Just under 1M
            (1000000, "$1.0M"),  # Exactly 1M
            (1100000, "$1.1M"),  # 1.1M
            (1149999, "$1.1M"),  # Rounds down
            (1150000, "$1.1M"),  # Also rounds down (only 1 decimal place)
            # Very large amounts
            (100000000, "$100.0M"),  # $100M
            (999999999, "$1000.0M"),  # Just under $1B
        ]

        for amount, expected in test_cases:
            # Act
            result = format_salary(amount)

            # Assert
            assert result == expected, f"Failed for amount ${amount:,}"

    def test_duration_formatting_extreme_cases(self):
        """Test duration formatting with extreme time values."""
        test_cases = [
            # Very short durations
            (0.1, "0s"),  # Sub-second rounds to 0
            (0.9, "0s"),  # Sub-second rounds to 0
            # Very long durations
            (86400, "24h"),  # 1 day
            (90000, "25h"),  # Over 24 hours
            (172800, "48h"),  # 2 days
            (604800, "168h"),  # 1 week
            (2592000, "720h"),  # 1 month (30 days)
            # Edge cases around formatting boundaries
            (3599, "59m 59s"),  # Just under 1 hour
            (3600, "1h"),  # Exactly 1 hour
            (3601, "1h"),  # Just over 1 hour (doesn't show 0 minutes)
            (3660, "1h 1m"),  # 1 hour 1 minute
            (7199, "1h 59m"),  # Just under 2 hours
            (7200, "2h"),  # Exactly 2 hours
        ]

        for seconds, expected in test_cases:
            # Act
            result = format_duration(seconds)

            # Assert
            assert result == expected, f"Failed for {seconds} seconds"

    @pytest.mark.parametrize(
        ("total", "completed", "elapsed", "expected_patterns"),
        [
            # Fast scraping scenarios
            (100, 50, 300, ["5m"]),  # 50/100 in 5min -> 5min remaining
            (10, 8, 240, ["1m"]),  # 8/10 in 4min -> 1m remaining
            # Slow scraping scenarios
            (50, 5, 3600, ["9h"]),  # 5/50 in 1h -> 9h remaining
            (100, 10, 7200, ["18h"]),  # 10/100 in 2h -> 18h remaining
            # Variable speed scenarios
            (200, 25, 900, ["1h 45m", "1h"]),  # Allow some variance in calculation
        ],
    )
    def test_eta_calculation_realistic_scenarios(
        self, total, completed, elapsed, expected_patterns
    ):
        """Test ETA calculations with realistic scraping scenarios."""
        # Act
        eta = calculate_eta(total, completed, elapsed)

        # Assert - Check if any expected pattern is found
        pattern_found = any(pattern in eta for pattern in expected_patterns)
        assert pattern_found, (
            f"None of {expected_patterns} found in ETA: {eta} "
            f"(total={total}, completed={completed}, elapsed={elapsed}s)"
        )

    def test_progress_percentage_precision_cases(self):
        """Test progress percentage with precision edge cases."""
        test_cases = [
            # Precision edge cases
            (1, 3, 33.3),  # 1/3 = 33.333... -> 33.3%
            (2, 3, 66.7),  # 2/3 = 66.666... -> 66.7%
            (1, 7, 14.3),  # 1/7 = 14.285... -> 14.3%
            (1, 6, 16.7),  # 1/6 = 16.666... -> 16.7%
            (5, 6, 83.3),  # 5/6 = 83.333... -> 83.3%
            # Large numbers
            (333, 1000, 33.3),  # 333/1000 = 33.3%
            (667, 1000, 66.7),  # 667/1000 = 66.7%
            (999, 1000, 99.9),  # 999/1000 = 99.9%
        ]

        for completed, total, expected in test_cases:
            # Act
            result = calculate_progress_percentage(completed, total)

            # Assert
            assert result == expected, (
                f"Failed for {completed}/{total}: expected {expected}%, got {result}%"
            )

    def test_scraping_speed_with_microsecond_precision(self):
        """Test scraping speed calculations with high precision timing."""
        # Test with microsecond-level precision (realistic for fast operations)
        start_time = datetime(2024, 1, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        test_cases = [
            # Very fast scraping (sub-second operations)
            (
                datetime(2024, 1, 1, 10, 0, 0, 500000, tzinfo=timezone.utc),
                1,
                120.0,
            ),  # 0.5s
            (datetime(2024, 1, 1, 10, 0, 1, 0, tzinfo=timezone.utc), 2, 120.0),  # 1s
            (datetime(2024, 1, 1, 10, 0, 5, 0, tzinfo=timezone.utc), 10, 120.0),  # 5s
            # Normal scraping speeds
            (datetime(2024, 1, 1, 10, 0, 30, 0, tzinfo=timezone.utc), 15, 30.0),  # 30s
            (datetime(2024, 1, 1, 10, 1, 0, 0, tzinfo=timezone.utc), 20, 20.0),  # 1min
        ]

        for end_time, jobs, expected_speed in test_cases:
            # Act
            speed = calculate_scraping_speed(jobs, start_time, end_time)

            # Assert
            assert speed == expected_speed, (
                f"Failed for {jobs} jobs in {(end_time - start_time).total_seconds()}s"
            )

    def test_jobs_count_edge_cases_and_localization_ready(self):
        """Test job count formatting edge cases and patterns ready for localization."""
        # Test negative numbers (edge case)
        result = format_jobs_count(-1)
        assert result == "-1 jobs"

        # Test zero with custom labels
        result = format_jobs_count(0, "opportunity", "opportunities")
        assert result == "0 opportunities"

        # Test large numbers
        result = format_jobs_count(1000000)
        assert result == "1000000 jobs"

        # Test with labels that could be internationalized
        international_test_cases = [
            (1, "posizione", "posizioni", "1 posizione"),  # Italian
            (5, "posizione", "posizioni", "5 posizioni"),  # Italian
            (1, "emploi", "emplois", "1 emploi"),  # French
            (3, "emploi", "emplois", "3 emplois"),  # French
            (1, "trabajo", "trabajos", "1 trabajo"),  # Spanish
            (7, "trabajo", "trabajos", "7 trabajos"),  # Spanish
        ]

        for count, singular, plural, expected in international_test_cases:
            # Act
            result = format_jobs_count(count, singular, plural)

            # Assert
            assert result == expected, f"Failed for {count} with {singular}/{plural}"


class TestFormattersStressAndErrorConditions:
    """Test formatters under stress conditions and error scenarios."""

    def test_format_salary_with_system_limits(self):
        """Test salary formatting near system integer limits."""
        import sys

        # Test with very large but valid integers
        large_amount = min(sys.maxsize, 2**31 - 1)  # Stay within reasonable bounds
        result = format_salary(large_amount)
        # Should not crash and should return some formatted value
        assert result.startswith("$")
        assert len(result) > 1

    def test_concurrent_formatter_calls(self):
        """Test formatters handle concurrent calls correctly."""
        import queue
        import threading

        results = queue.Queue()

        def worker():
            # Simulate concurrent formatting calls
            for i in range(10):
                salary_result = format_salary(75000 + i * 1000)
                duration_result = format_duration(3600 + i * 60)
                progress_result = calculate_progress_percentage(i, 10)
                results.put((salary_result, duration_result, progress_result))

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all results are valid
        collected_results = []
        while not results.empty():
            collected_results.append(results.get())

        assert len(collected_results) == 30  # 3 threads x 10 iterations

        # Verify all results are properly formatted
        for salary, duration, progress in collected_results:
            assert salary.startswith("$")
            assert (
                "k" in salary
                or "M" in salary
                or salary
                in [
                    "$75",
                    "$76",
                    "$77",
                    "$78",
                    "$79",
                    "$80",
                    "$81",
                    "$82",
                    "$83",
                    "$84",
                ]
            )
            assert any(unit in duration for unit in ["s", "m", "h"])
            assert 0.0 <= progress <= 100.0

    def test_formatters_handle_various_exceptions(self):
        """Test formatters gracefully handle various exception types."""
        # Test format_salary with invalid inputs
        assert format_salary(None) == "$0"
        assert format_salary("invalid") == "$0"
        assert format_salary(-100) == "$0"

        # Test calculate_scraping_speed with invalid inputs
        assert calculate_scraping_speed(None, None, None) == 0.0
        assert calculate_scraping_speed("invalid", None, None) == 0.0
        assert calculate_scraping_speed(-5, None, None) == 0.0

    def test_memory_usage_with_large_datasets(self):
        """Test formatters don't consume excessive memory with large inputs."""
        # Test with large iteration counts to ensure no memory leaks
        import gc
        import tracemalloc

        tracemalloc.start()

        # Perform many formatting operations
        for i in range(1000):
            format_salary(75000 + i)
            format_duration(3600 + i)
            calculate_progress_percentage(i % 100, 100)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 1MB for this test)
        assert peak < 1024 * 1024, f"Peak memory usage too high: {peak} bytes"

        # Force garbage collection
        gc.collect()

    def test_formatter_performance_benchmarks(self):
        """Basic performance benchmarks for formatters."""
        import time

        # Benchmark format_salary
        start_time = time.perf_counter()
        for i in range(1000):
            format_salary(75000 + i)
        salary_time = time.perf_counter() - start_time

        # Benchmark format_duration
        start_time = time.perf_counter()
        for i in range(1000):
            format_duration(3600 + i)
        duration_time = time.perf_counter() - start_time

        # Benchmark calculate_progress_percentage
        start_time = time.perf_counter()
        for i in range(1000):
            calculate_progress_percentage(i % 100, 100)
        progress_time = time.perf_counter() - start_time

        # Each benchmark should complete in reasonable time (< 0.1s for 1000 calls)
        assert salary_time < 0.1, f"format_salary too slow: {salary_time}s"
        assert duration_time < 0.1, f"format_duration too slow: {duration_time}s"
        assert progress_time < 0.1, (
            f"calculate_progress_percentage too slow: {progress_time}s"
        )
