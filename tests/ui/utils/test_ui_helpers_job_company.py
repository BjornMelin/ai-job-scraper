"""Tests for UI helpers job and company utility functions.

Tests the job and company formatting and calculation functions in ui_helpers.py:
- Salary range formatting and extraction
- Days since posted calculations
- Company statistics calculations
- Job relationship helpers
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from src.ui.utils.ui_helpers import (
    calculate_active_jobs_count,
    calculate_days_since_posted,
    calculate_total_jobs_count,
    find_last_job_posted,
    format_salary_range,
    format_success_rate_percentage,
    get_job_company_name,
    get_salary_max,
    get_salary_min,
    is_job_recently_posted,
)


class TestSalaryHelpers:
    """Test salary-related helper functions."""

    def test_get_salary_min_with_valid_tuple(self):
        """Test get_salary_min extracts minimum value correctly."""
        test_cases = [
            ((50000, 75000), 50000),
            ((100000, 150000), 100000),
            ((0, 50000), 0),
            ((75000, 75000), 75000),  # Same min/max
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = get_salary_min(salary_tuple)

            # Assert
            assert result == expected

    def test_get_salary_min_with_none_values(self):
        """Test get_salary_min handles None values correctly."""
        test_cases = [
            ((None, 75000), None),
            ((50000, None), 50000),
            ((None, None), None),
            (None, None),  # Entire tuple is None
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = get_salary_min(salary_tuple)

            # Assert
            assert result == expected

    def test_get_salary_max_with_valid_tuple(self):
        """Test get_salary_max extracts maximum value correctly."""
        test_cases = [
            ((50000, 75000), 75000),
            ((100000, 150000), 150000),
            ((50000, 0), 0),
            ((75000, 75000), 75000),  # Same min/max
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = get_salary_max(salary_tuple)

            # Assert
            assert result == expected

    def test_get_salary_max_with_none_values(self):
        """Test get_salary_max handles None values correctly."""
        test_cases = [
            ((75000, None), None),
            ((None, 50000), 50000),
            ((None, None), None),
            (None, None),  # Entire tuple is None
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = get_salary_max(salary_tuple)

            # Assert
            assert result == expected

    def test_format_salary_range_with_complete_range(self):
        """Test format_salary_range with both min and max values."""
        test_cases = [
            ((50000, 75000), "$50,000 - $75,000"),
            ((100000, 150000), "$100,000 - $150,000"),
            ((75000, 125000), "$75,000 - $125,000"),
            ((200000, 300000), "$200,000 - $300,000"),
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = format_salary_range(salary_tuple)

            # Assert
            assert result == expected

    def test_format_salary_range_with_same_min_max(self):
        """Test format_salary_range when min and max are the same."""
        test_cases = [
            ((75000, 75000), "$75,000"),
            ((100000, 100000), "$100,000"),
            ((50000, 50000), "$50,000"),
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = format_salary_range(salary_tuple)

            # Assert
            assert result == expected

    def test_format_salary_range_with_partial_values(self):
        """Test format_salary_range with only min or max values."""
        test_cases = [
            ((50000, None), "From $50,000"),
            ((None, 75000), "Up to $75,000"),
            ((100000, None), "From $100,000"),
            ((None, 150000), "Up to $150,000"),
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = format_salary_range(salary_tuple)

            # Assert
            assert result == expected

    def test_format_salary_range_with_none_or_empty(self):
        """Test format_salary_range with None or empty values."""
        test_cases = [
            (None, "Not specified"),
            ((None, None), "Not specified"),
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = format_salary_range(salary_tuple)

            # Assert
            assert result == expected

    def test_format_salary_range_with_zero_values(self):
        """Test format_salary_range handles zero values correctly."""
        test_cases = [
            ((0, 50000), "Up to $50,000"),  # Zero min treated as None
            ((50000, 0), "From $50,000"),  # Zero max treated as None
            ((0, 0), "Not specified"),  # Both zero treated as None
        ]

        for salary_tuple, expected in test_cases:
            # Act
            result = format_salary_range(salary_tuple)

            # Assert
            assert result == expected

    @pytest.mark.parametrize(
        ("salary_tuple", "expected"),
        [
            # Realistic tech salary ranges
            ((65000, 85000), "$65,000 - $85,000"),  # Junior developer
            ((90000, 120000), "$90,000 - $120,000"),  # Mid-level developer
            ((130000, 170000), "$130,000 - $170,000"),  # Senior developer
            ((180000, 250000), "$180,000 - $250,000"),  # Staff engineer
            ((250000, 400000), "$250,000 - $400,000"),  # Principal engineer
            # Contract/hourly equivalents (annualized)
            ((104000, 156000), "$104,000 - $156,000"),  # $50-75/hour
            ((130000, 208000), "$130,000 - $208,000"),  # $62.5-100/hour
        ],
    )
    def test_format_salary_range_realistic_values(self, salary_tuple, expected):
        """Test format_salary_range with realistic salary values."""
        # Act
        result = format_salary_range(salary_tuple)

        # Assert
        assert result == expected


class TestDaysCalculations:
    """Test date and time calculation functions."""

    def test_calculate_days_since_posted_with_recent_date(self):
        """Test calculate_days_since_posted with recent posting dates."""
        # Arrange - Use fixed time for consistent testing
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone = timezone

            test_cases = [
                (
                    datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
                    0,
                ),  # Same day, earlier
                (datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc), 1),  # 1 day ago
                (datetime(2024, 1, 13, 10, 0, 0, tzinfo=timezone.utc), 2),  # 2 days ago
                (datetime(2024, 1, 8, 10, 0, 0, tzinfo=timezone.utc), 7),  # 1 week ago
                (
                    datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    14,
                ),  # 2 weeks ago
            ]

            for posted_date, expected_days in test_cases:
                # Act
                result = calculate_days_since_posted(posted_date)

                # Assert
                assert result == expected_days

    def test_calculate_days_since_posted_with_naive_datetime(self):
        """Test calculate_days_since_posted handles naive datetime correctly."""
        # Arrange - Use fixed time for consistent testing
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone = timezone

            # Naive datetime (no timezone info)
            naive_posted_date = datetime(  # noqa: DTZ001
                2024, 1, 14, 10, 0, 0
            )  # No tzinfo

            # Act
            result = calculate_days_since_posted(naive_posted_date)

            # Assert - Should be 1 day (naive datetime gets UTC timezone)
            assert result == 1

    def test_calculate_days_since_posted_with_none(self):
        """Test calculate_days_since_posted handles None input."""
        # Act
        result = calculate_days_since_posted(None)

        # Assert
        assert result is None

    def test_calculate_days_since_posted_handles_exceptions(self):
        """Test calculate_days_since_posted handles calculation exceptions."""
        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            # Arrange - Mock datetime.now to raise exception
            mock_datetime.now.side_effect = Exception("Time calculation error")

            posted_date = datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc)

            # Act
            result = calculate_days_since_posted(posted_date)

            # Assert
            assert result is None

    def test_is_job_recently_posted_with_recent_jobs(self):
        """Test is_job_recently_posted identifies recent jobs correctly."""
        # Arrange - Use fixed time for consistent testing
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone = timezone

            test_cases = [
                (datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc), True),  # Today
                (
                    datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
                    True,
                ),  # 1 day ago
                (
                    datetime(2024, 1, 9, 10, 0, 0, tzinfo=timezone.utc),
                    True,
                ),  # 6 days ago
                (
                    datetime(2024, 1, 8, 10, 0, 0, tzinfo=timezone.utc),
                    True,
                ),  # 7 days ago (threshold)
                (
                    datetime(2024, 1, 7, 10, 0, 0, tzinfo=timezone.utc),
                    False,
                ),  # 8 days ago
                (
                    datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    False,
                ),  # 14 days ago
            ]

            for posted_date, expected in test_cases:
                # Act
                result = is_job_recently_posted(posted_date)

                # Assert
                assert result == expected

    def test_is_job_recently_posted_with_custom_threshold(self):
        """Test is_job_recently_posted with custom threshold."""
        # Arrange - Use fixed time for consistent testing
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone = timezone

            test_cases = [
                (
                    datetime(2024, 1, 12, 10, 0, 0, tzinfo=timezone.utc),
                    3,
                    True,
                ),  # 3 days ago, threshold=3
                (
                    datetime(2024, 1, 11, 10, 0, 0, tzinfo=timezone.utc),
                    3,
                    False,
                ),  # 4 days ago, threshold=3
                (
                    datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    14,
                    True,
                ),  # 14 days ago, threshold=14
                (
                    datetime(2023, 12, 31, 10, 0, 0, tzinfo=timezone.utc),
                    14,
                    False,
                ),  # 15 days ago, threshold=14
            ]

            for posted_date, threshold, expected in test_cases:
                # Act
                result = is_job_recently_posted(posted_date, threshold)

                # Assert
                assert result == expected

    def test_is_job_recently_posted_with_none_date(self):
        """Test is_job_recently_posted handles None date."""
        # Act
        result = is_job_recently_posted(None)

        # Assert
        assert result is False

    def test_is_job_recently_posted_handles_calculation_errors(self):
        """Test is_job_recently_posted handles calculation errors gracefully."""
        with patch("src.ui.utils.ui_helpers.calculate_days_since_posted") as mock_calc:
            # Arrange - Mock to return None (calculation error)
            mock_calc.return_value = None

            posted_date = datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc)

            # Act
            result = is_job_recently_posted(posted_date)

            # Assert
            assert result is False


class TestJobCompanyHelpers:
    """Test job and company relationship helper functions."""

    def test_get_job_company_name_with_valid_company(self):
        """Test get_job_company_name extracts company name correctly."""
        # Arrange
        mock_company = Mock()
        mock_company.name = "Test Company Inc."

        # Act
        result = get_job_company_name(mock_company)

        # Assert
        assert result == "Test Company Inc."

    def test_get_job_company_name_with_none_company(self):
        """Test get_job_company_name handles None company."""
        # Act
        result = get_job_company_name(None)

        # Assert
        assert result == "Unknown"

    def test_get_job_company_name_with_various_company_names(self):
        """Test get_job_company_name with various company name formats."""
        test_cases = [
            "Google",
            "Microsoft Corporation",
            "Acme Corp.",
            "StartUp123",
            "Company with Spaces & Symbols!",
            "",  # Empty name
            "   ",  # Whitespace only
        ]

        for company_name in test_cases:
            # Arrange
            mock_company = Mock()
            mock_company.name = company_name

            # Act
            result = get_job_company_name(mock_company)

            # Assert
            assert result == company_name


class TestCompanyStatistics:
    """Test company statistics calculation functions."""

    def test_calculate_total_jobs_count_with_jobs_list(self):
        """Test calculate_total_jobs_count counts all jobs correctly."""
        test_cases = [
            ([], 0),  # Empty list
            ([Mock(), Mock(), Mock()], 3),  # 3 jobs
            ([Mock() for _ in range(10)], 10),  # 10 jobs
            ([Mock() for _ in range(100)], 100),  # Large number
        ]

        for jobs_list, expected_count in test_cases:
            # Act
            result = calculate_total_jobs_count(jobs_list)

            # Assert
            assert result == expected_count

    def test_calculate_total_jobs_count_with_none(self):
        """Test calculate_total_jobs_count handles None input."""
        # Act
        result = calculate_total_jobs_count(None)

        # Assert
        assert result == 0

    def test_calculate_active_jobs_count_with_mixed_jobs(self):
        """Test calculate_active_jobs_count counts only non-archived jobs."""
        # Arrange
        active_job1 = Mock()
        active_job1.archived = False

        active_job2 = Mock()
        active_job2.archived = False

        archived_job1 = Mock()
        archived_job1.archived = True

        archived_job2 = Mock()
        archived_job2.archived = True

        test_cases = [
            ([], 0),  # Empty list
            ([active_job1], 1),  # One active job
            ([archived_job1], 0),  # One archived job
            ([active_job1, active_job2], 2),  # Two active jobs
            ([archived_job1, archived_job2], 0),  # Two archived jobs
            ([active_job1, archived_job1, active_job2, archived_job2], 2),  # Mixed
        ]

        for jobs_list, expected_count in test_cases:
            # Act
            result = calculate_active_jobs_count(jobs_list)

            # Assert
            assert result == expected_count

    def test_calculate_active_jobs_count_with_none(self):
        """Test calculate_active_jobs_count handles None input."""
        # Act
        result = calculate_active_jobs_count(None)

        # Assert
        assert result == 0

    def test_find_last_job_posted_with_various_dates(self):
        """Test find_last_job_posted finds most recent posting date."""
        # Arrange
        date1 = datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
        date2 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)  # Most recent
        date3 = datetime(2024, 1, 5, 10, 0, 0, tzinfo=timezone.utc)

        job1 = Mock()
        job1.posted_date = date1

        job2 = Mock()
        job2.posted_date = date2

        job3 = Mock()
        job3.posted_date = date3

        # Act
        result = find_last_job_posted([job1, job2, job3])

        # Assert
        assert result == date2

    def test_find_last_job_posted_with_none_dates(self):
        """Test find_last_job_posted handles jobs with None posted_date."""
        # Arrange
        date1 = datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc)

        job_with_date = Mock()
        job_with_date.posted_date = date1

        job_without_date = Mock()
        job_without_date.posted_date = None

        # Act
        result = find_last_job_posted([job_with_date, job_without_date])

        # Assert
        assert result == date1

    def test_find_last_job_posted_with_all_none_dates(self):
        """Test find_last_job_posted when all jobs have None posted_date."""
        # Arrange
        job1 = Mock()
        job1.posted_date = None

        job2 = Mock()
        job2.posted_date = None

        # Act
        result = find_last_job_posted([job1, job2])

        # Assert
        assert result is None

    def test_find_last_job_posted_with_empty_list(self):
        """Test find_last_job_posted handles empty jobs list."""
        # Act
        result = find_last_job_posted([])

        # Assert
        assert result is None

    def test_find_last_job_posted_with_none_input(self):
        """Test find_last_job_posted handles None input."""
        # Act
        result = find_last_job_posted(None)

        # Assert
        assert result is None

    def test_format_success_rate_percentage_with_various_rates(self):
        """Test format_success_rate_percentage converts decimals to percentages."""
        test_cases = [
            (0.0, 0.0),  # 0%
            (0.1, 10.0),  # 10%
            (0.25, 25.0),  # 25%
            (0.5, 50.0),  # 50%
            (0.75, 75.0),  # 75%
            (0.9, 90.0),  # 90%
            (1.0, 100.0),  # 100%
        ]

        for decimal_rate, expected_percentage in test_cases:
            # Act
            result = format_success_rate_percentage(decimal_rate)

            # Assert
            assert result == expected_percentage

    def test_format_success_rate_percentage_with_precision(self):
        """Test format_success_rate_percentage rounds to 1 decimal place."""
        test_cases = [
            (0.123, 12.3),  # 12.3%
            (0.456, 45.6),  # 45.6%
            (0.789, 78.9),  # 78.9%
            (0.9999, 100.0),  # Rounds to 100.0%
            (0.1234, 12.3),  # Rounds down
            (0.1235, 12.3),  # Rounds down (Python's banker's rounding)
        ]

        for decimal_rate, expected_percentage in test_cases:
            # Act
            result = format_success_rate_percentage(decimal_rate)

            # Assert
            assert result == expected_percentage

    def test_format_success_rate_percentage_edge_cases(self):
        """Test format_success_rate_percentage handles edge cases."""
        test_cases = [
            (0.0001, 0.0),  # Very small rate
            (0.9999, 100.0),  # Very close to 100%
            (1.5, 150.0),  # Above 100% (theoretical)
            (-0.1, -10.0),  # Negative rate (theoretical)
        ]

        for decimal_rate, expected_percentage in test_cases:
            # Act
            result = format_success_rate_percentage(decimal_rate)

            # Assert
            assert result == expected_percentage


class TestJobCompanyHelpersIntegration:
    """Integration tests for job and company helpers working together."""

    def test_realistic_job_data_scenario(self):
        """Test helpers with realistic job posting data."""
        # Arrange - Create realistic job data
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone = timezone

            # Create mock company
            company = Mock()
            company.name = "TechCorp Inc."

            # Create mock jobs with realistic data
            recent_job = Mock()
            recent_job.posted_date = datetime(
                2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc
            )  # 1 day ago
            recent_job.archived = False

            old_job = Mock()
            old_job.posted_date = datetime(
                2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc
            )  # 14 days ago
            old_job.archived = False

            archived_job = Mock()
            archived_job.posted_date = datetime(
                2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc
            )  # 5 days ago
            archived_job.archived = True

            jobs_list = [recent_job, old_job, archived_job]

            # Act - Test various calculations
            company_name = get_job_company_name(company)
            total_jobs = calculate_total_jobs_count(jobs_list)
            active_jobs = calculate_active_jobs_count(jobs_list)
            last_posted = find_last_job_posted(jobs_list)
            recent_job_posted = is_job_recently_posted(recent_job.posted_date)
            old_job_posted = is_job_recently_posted(old_job.posted_date)

            # Assert - Verify realistic scenario calculations
            assert company_name == "TechCorp Inc."
            assert total_jobs == 3
            assert active_jobs == 2  # Only non-archived
            assert last_posted == recent_job.posted_date  # Most recent
            assert recent_job_posted is True  # Within 7 days
            assert old_job_posted is False  # Outside 7 days

    def test_salary_range_formatting_scenarios(self):
        """Test salary range formatting with realistic job posting scenarios."""
        # Common salary scenarios in job postings
        test_scenarios = [
            # Tech industry ranges
            ((70000, 90000), "$70,000 - $90,000", "Junior Developer"),
            ((100000, 130000), "$100,000 - $130,000", "Mid-level Engineer"),
            ((150000, 200000), "$150,000 - $200,000", "Senior Engineer"),
            ((200000, 300000), "$200,000 - $300,000", "Staff Engineer"),
            # Contract/consulting ranges
            ((104000, 156000), "$104,000 - $156,000", "Contract ($50-75/hour)"),
            # Executive ranges
            ((300000, 500000), "$300,000 - $500,000", "Director Level"),
            # Partial information scenarios
            ((80000, None), "From $80,000", "Minimum salary only"),
            ((None, 120000), "Up to $120,000", "Maximum salary only"),
            (None, "Not specified", "No salary information"),
            # Equal min/max (fixed salary)
            ((75000, 75000), "$75,000", "Fixed salary position"),
        ]

        for salary_tuple, expected_format, description in test_scenarios:
            # Act
            result = format_salary_range(salary_tuple)

            # Assert
            assert result == expected_format, f"Failed for {description}"

            # Also test min/max extraction
            if salary_tuple:
                min_salary = get_salary_min(salary_tuple)
                max_salary = get_salary_max(salary_tuple)

                if salary_tuple[0] is not None:
                    assert min_salary == salary_tuple[0]
                if salary_tuple[1] is not None:
                    assert max_salary == salary_tuple[1]

    def test_company_statistics_comprehensive_scenario(self):
        """Test company statistics with comprehensive job data."""
        # Arrange - Create comprehensive company data
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Create jobs with various statuses and dates
        jobs = []

        # Recent active jobs
        for i in range(5):
            job = Mock()
            job.posted_date = now - timedelta(days=i)  # 0-4 days ago
            job.archived = False
            jobs.append(job)

        # Older active jobs
        for i in range(3):
            job = Mock()
            job.posted_date = now - timedelta(days=10 + i)  # 10-12 days ago
            job.archived = False
            jobs.append(job)

        # Archived jobs
        for i in range(2):
            job = Mock()
            job.posted_date = now - timedelta(days=20 + i)  # 20-21 days ago
            job.archived = True
            jobs.append(job)

        # Act
        total_jobs = calculate_total_jobs_count(jobs)
        active_jobs = calculate_active_jobs_count(jobs)
        last_posted = find_last_job_posted(jobs)

        # Assert
        assert total_jobs == 10  # 5 + 3 + 2
        assert active_jobs == 8  # 5 + 3 (only non-archived)
        assert last_posted == now  # Most recent (today)

    def test_success_rate_calculation_realistic_values(self):
        """Test success rate formatting with realistic values."""
        # Realistic success rates for different scenarios
        test_cases = [
            (0.95, 95.0, "High success rate"),
            (0.85, 85.0, "Good success rate"),
            (0.70, 70.0, "Average success rate"),
            (0.50, 50.0, "Moderate success rate"),
            (0.25, 25.0, "Low success rate"),
            (0.05, 5.0, "Very low success rate"),
            (0.875, 87.5, "Precise calculation"),
            (0.333, 33.3, "One third success"),
            (0.667, 66.7, "Two thirds success"),
        ]

        for decimal_rate, expected_percentage, description in test_cases:
            # Act
            result = format_success_rate_percentage(decimal_rate)

            # Assert
            assert result == expected_percentage, f"Failed for {description}"

    def test_date_calculations_timezone_handling(self):
        """Test date calculations handle timezones correctly."""
        # Arrange - Test with different timezones
        utc_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("src.ui.utils.ui_helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = utc_time
            mock_datetime.timezone = timezone

            # Test dates in different timezones (should be normalized to UTC)
            test_dates = [
                datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc),  # UTC
                datetime(
                    2024, 1, 14, 5, 0, 0, tzinfo=timezone(timedelta(hours=-5))
                ),  # EST
                datetime(
                    2024, 1, 14, 15, 0, 0, tzinfo=timezone(timedelta(hours=5))
                ),  # UTC+5
            ]

            for posted_date in test_dates:
                # Act
                days_since = calculate_days_since_posted(posted_date)
                is_recent = is_job_recently_posted(posted_date)

                # Assert - All should calculate to 1 day ago
                assert days_since == 1
                assert is_recent is True

    def test_helpers_handle_concurrent_access(self):
        """Test helper functions work correctly with concurrent access."""
        import queue
        import threading

        results = queue.Queue()

        def worker():
            # Test multiple helper functions concurrently
            for i in range(50):
                # Test salary helpers
                salary_tuple = (50000 + i * 1000, 75000 + i * 1000)
                salary_range = format_salary_range(salary_tuple)
                min_salary = get_salary_min(salary_tuple)
                max_salary = get_salary_max(salary_tuple)

                # Test company helpers
                mock_company = Mock()
                mock_company.name = f"Company {i}"
                company_name = get_job_company_name(mock_company)

                # Test statistics
                success_rate = format_success_rate_percentage(0.5 + (i * 0.01))

                results.put(
                    (salary_range, min_salary, max_salary, company_name, success_rate)
                )

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

        assert len(collected_results) == 150  # 3 threads x 50 operations

        # Verify all results are properly formatted
        for salary_range, min_sal, max_sal, company, success in collected_results:
            assert "$" in salary_range
            assert isinstance(min_sal, int)
            assert isinstance(max_sal, int)
            assert "Company" in company
            assert isinstance(success, float)
            assert 0 <= success <= 100
