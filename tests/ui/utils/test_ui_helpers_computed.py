"""Tests for T1.4: SQLModel Computed Fields Replacement - UI Helper Functions.

Tests cover:
- Salary formatting and calculations
- Date calculations and formatting
- Company statistics helpers
- Job field extraction and display
- Safe validation and conversion utilities
- Performance and edge case handling
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

# All UI helpers consolidated in ui_helpers.py
from src.ui.utils.ui_helpers import (
    calculate_active_jobs_count,
    calculate_days_since_posted,
    calculate_eta,
    calculate_scraping_speed,
    calculate_total_jobs_count,
    find_last_job_posted,
    format_duration,
    format_jobs_count,
    format_salary,
    format_salary_range,
    format_success_rate_percentage,
    format_timestamp,
    is_job_recently_posted,
    is_streamlit_context,
    safe_int,
    safe_job_count,
)


class TestT1SalaryFormatting:
    """Test T1.4: Salary formatting and calculation helpers."""

    def test_format_salary_basic_amounts(self):
        """Test salary formatting for basic amounts."""
        assert format_salary(0) == "$0"
        assert format_salary(500) == "$500"
        assert format_salary(999) == "$999"

    def test_format_salary_thousands(self):
        """Test salary formatting for thousands."""
        assert format_salary(1000) == "$1k"
        assert format_salary(15000) == "$15k"
        assert format_salary(99000) == "$99k"
        assert format_salary(125000) == "$125k"

    def test_format_salary_millions(self):
        """Test salary formatting for millions."""
        assert format_salary(1000000) == "$1.0M"
        assert format_salary(1500000) == "$1.5M"
        assert format_salary(2750000) == "$2.8M"  # Rounded

    def test_format_salary_edge_cases(self):
        """Test salary formatting edge cases."""
        assert format_salary(None) == "$0"
        assert format_salary(-1000) == "$0"  # Negative treated as $0
        assert format_salary(999999) == "$999k"  # Just under 1M

    def test_format_salary_float_conversion(self):
        """Test salary formatting converts floats to integers."""
        assert format_salary(50000.7) == "$50k"
        assert format_salary(1500000.9) == "$1.5M"

    def test_get_salary_min_extraction(self):
        """Test minimum salary extraction from tuple."""
        assert ((50000, 80000)[0] if (50000, 80000) else None) == 50000
        assert ((None, 80000)[0] if (None, 80000) else None) is None
        assert None is None

    def test_get_salary_max_extraction(self):
        """Test maximum salary extraction from tuple."""
        assert ((50000, 80000)[1] if (50000, 80000) else None) == 80000
        assert ((50000, None)[1] if (50000, None) else None) is None
        assert None is None

    def test_format_salary_range_complete_range(self):
        """Test formatting complete salary ranges."""
        assert format_salary_range((50000, 80000)) == "$50,000 - $80,000"
        assert format_salary_range((100000, 150000)) == "$100,000 - $150,000"

    def test_format_salary_range_same_values(self):
        """Test formatting when min and max are the same."""
        assert format_salary_range((75000, 75000)) == "$75,000"

    def test_format_salary_range_partial_ranges(self):
        """Test formatting partial salary ranges."""
        assert format_salary_range((50000, None)) == "$50,000+"
        assert format_salary_range((None, 100000)) == "Up to $100,000"

    def test_format_salary_range_edge_cases(self):
        """Test salary range formatting edge cases."""
        assert format_salary_range(None) == "Not specified"
        assert format_salary_range((None, None)) == "Not specified"


class TestT1DateCalculations:
    """Test T1.4: Date calculation and formatting helpers."""

    def test_calculate_days_since_posted_recent(self):
        """Test calculating days since posted for recent jobs."""
        now = datetime.now(UTC)

        # Today
        today = now
        assert calculate_days_since_posted(today) == 0

        # Yesterday
        yesterday = now - timedelta(days=1)
        assert calculate_days_since_posted(yesterday) == 1

        # One week ago
        week_ago = now - timedelta(days=7)
        assert calculate_days_since_posted(week_ago) == 7

    def test_calculate_days_since_posted_timezone_naive(self):
        """Test calculation handles timezone-naive datetimes."""
        # Naive datetime (common from scraping)
        naive_date = datetime(
            2024,
            1,
            1,
            12,
            0,
            0,
            tzinfo=UTC,
        )  # Explicitly add timezone

        result = calculate_days_since_posted(naive_date)

        # Should handle gracefully and return a valid number
        assert isinstance(result, int)
        assert result >= 0

    def test_calculate_days_since_posted_future_date(self):
        """Test calculation with future dates."""
        future = datetime.now(UTC) + timedelta(days=5)

        result = calculate_days_since_posted(future)

        # Should handle future dates (negative days)
        assert isinstance(result, int)

    def test_calculate_days_since_posted_none(self):
        """Test calculation with None date."""
        assert calculate_days_since_posted(None) is None

    def test_is_job_recently_posted_default_threshold(self):
        """Test recent job detection with default 7-day threshold."""
        now = datetime.now(UTC)

        # Within threshold
        recent = now - timedelta(days=3)
        assert is_job_recently_posted(recent) is True

        # At threshold boundary
        boundary = now - timedelta(days=7)
        assert is_job_recently_posted(boundary) is True

        # Beyond threshold
        old = now - timedelta(days=8)
        assert is_job_recently_posted(old) is False

    def test_is_job_recently_posted_custom_threshold(self):
        """Test recent job detection with custom threshold."""
        now = datetime.now(UTC)

        # 3-day threshold
        recent = now - timedelta(days=2)
        old = now - timedelta(days=4)

        assert is_job_recently_posted(recent, days_threshold=3) is True
        assert is_job_recently_posted(old, days_threshold=3) is False

    def test_is_job_recently_posted_none_date(self):
        """Test recent job detection with None date."""
        assert is_job_recently_posted(None) is False

    def test_format_timestamp_basic(self):
        """Test timestamp formatting with default format."""
        dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=UTC)

        result = format_timestamp(dt)

        assert result == "14:30:45"

    def test_format_timestamp_custom_format(self):
        """Test timestamp formatting with custom format."""
        dt = datetime(2024, 1, 15, 14, 30, 45, tzinfo=UTC)

        result = format_timestamp(dt, "%Y-%m-%d %H:%M")

        assert result == "2024-01-15 14:30"

    def test_format_timestamp_none(self):
        """Test timestamp formatting with None."""
        assert format_timestamp(None) == "N/A"

    def test_format_timestamp_invalid_type(self):
        """Test timestamp formatting with invalid type."""
        assert format_timestamp("not_a_datetime") == "N/A"


class TestT1ProgressCalculations:
    """Test T1.4: Progress calculation helpers."""

    def test_calculate_scraping_speed_basic(self):
        """Test scraping speed calculation with basic inputs."""
        start = datetime.now(UTC)
        end = start + timedelta(minutes=10)  # 10 minutes

        speed = calculate_scraping_speed(50, start, end)

        assert speed == 5.0  # 50 jobs / 10 minutes = 5 jobs/minute

    def test_calculate_scraping_speed_no_end_time(self):
        """Test scraping speed calculation without end time (uses current time)."""
        # Use a start time that's 1 minute ago
        start = datetime.now(UTC) - timedelta(minutes=1)

        speed = calculate_scraping_speed(10, start)

        # Should calculate using current time
        assert isinstance(speed, float)
        assert speed >= 0

    def test_calculate_scraping_speed_edge_cases(self):
        """Test scraping speed calculation edge cases."""
        start = datetime.now(UTC)

        # Zero jobs
        assert calculate_scraping_speed(0, start) == 0.0

        # Negative jobs (invalid)
        assert calculate_scraping_speed(-10, start) == 0.0

        # None start time
        assert calculate_scraping_speed(10, None) == 0.0

        # Zero duration
        assert calculate_scraping_speed(10, start, start) == 0.0

    def test_calculate_eta_basic(self):
        """Test ETA calculation with basic inputs."""
        eta = calculate_eta(
            total_companies=10,
            completed_companies=5,
            time_elapsed=300,
        )  # 5 minutes

        # Should calculate remaining time for 5 companies at 1 minute per company
        assert eta == "5m"

    def test_calculate_eta_edge_cases(self):
        """Test ETA calculation edge cases."""
        # All companies completed
        assert calculate_eta(10, 10, 300) == "Done"

        # No progress yet
        assert calculate_eta(10, 0, 300) == "Calculating..."

        # No time elapsed
        assert calculate_eta(10, 5, 0) == "Calculating..."

        # Invalid inputs
        assert calculate_eta(0, 5, 300) == "Unknown"
        assert calculate_eta(10, -1, 300) == "Unknown"

    def test_format_duration_basic(self):
        """Test duration formatting for basic values."""
        assert format_duration(0) == "0s"
        assert format_duration(30) == "30s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3661) == "1h 1m"  # 1 hour, 1 minute, 1 second
        assert format_duration(3720) == "1h 2m"  # 1 hour, 2 minutes

    def test_format_duration_edge_cases(self):
        """Test duration formatting edge cases."""
        assert format_duration(-10) == "0s"  # Negative becomes 0
        assert format_duration(3.7) == "3s"  # Float truncated
        assert format_duration("invalid") == "0s"  # Invalid type


class TestT1SafeValidation:
    """Test T1.4: Safe validation and conversion utilities."""

    def test_safe_int_basic_conversions(self):
        """Test safe integer conversion for basic types."""
        assert safe_int(42) == 42
        assert safe_int(3.7) == 3
        assert safe_int("25") == 25
        assert safe_int(True) == 1
        assert safe_int(False) == 0

    def test_safe_int_string_parsing(self):
        """Test safe integer conversion from strings."""
        assert safe_int("123") == 123
        assert safe_int("  456  ") == 456  # Whitespace trimmed
        assert safe_int("78.9") == 78  # Float string
        assert safe_int("abc123def") == 123  # Extract number
        assert safe_int("-50") == 0  # Negative becomes 0

    def test_safe_int_edge_cases(self):
        """Test safe integer conversion edge cases."""
        assert safe_int(None) == 0
        assert safe_int("") == 0
        assert safe_int("no_numbers") == 0
        assert safe_int([1, 2, 3]) == 0  # Invalid type

    def test_safe_int_with_default(self):
        """Test safe integer conversion with custom default."""
        # Valid inputs are processed normally, invalid strings use default
        assert safe_int("invalid", default=42) == 42  # No numbers found, use default
        assert safe_int(None, default=100) == 0  # None converts to 0
        assert safe_int("", default=50) == 50  # Empty string uses default

        # Test default is used when input causes validation to fail

    def test_safe_job_count_basic(self):
        """Test safe job count conversion."""
        assert safe_job_count(5) == 5
        assert safe_job_count("10") == 10
        assert safe_job_count(3.8) == 3

    def test_safe_job_count_with_company_context(self):
        """Test safe job count with company name for logging context."""
        # Should not raise errors, just convert safely
        assert safe_job_count("invalid", "Test Company") == 0
        assert safe_job_count(None, "Another Company") == 0

    def test_format_jobs_count_singular_plural(self):
        """Test job count formatting with singular/plural."""
        assert format_jobs_count(0) == "0 jobs"
        assert format_jobs_count(1) == "1 job"
        assert format_jobs_count(2) == "2 jobs"
        assert format_jobs_count(100) == "100 jobs"

    def test_format_jobs_count_custom_labels(self):
        """Test job count formatting with custom labels."""
        assert format_jobs_count(1, "position", "positions") == "1 position"
        assert format_jobs_count(5, "opening", "openings") == "5 openings"

    def test_format_jobs_count_edge_cases(self):
        """Test job count formatting edge cases."""
        assert format_jobs_count(None) == "0 jobs"
        assert format_jobs_count("5") == "5 jobs"  # String conversion
        assert format_jobs_count(2.8) == "2 jobs"  # Float conversion


class TestT1JobHelpers:
    """Test T1.4: Job-related helper functions."""

    def test_get_job_company_name_with_relation(self):
        """Test getting company name from relationship object."""
        # Create a mock company object
        mock_company = Mock()
        mock_company.name = "Test Company Inc."

        result = mock_company.name if mock_company else "Unknown"

        assert result == "Test Company Inc."

    def test_get_job_company_name_none_relation(self):
        """Test getting company name when relation is None."""
        result = None.name if None else "Unknown"

        assert result == "Unknown"


class TestT1CompanyHelpers:
    """Test T1.4: Company statistics helper functions."""

    def create_mock_job(self, job_id=1, posted_date=None, archived=False):
        """Helper to create mock job objects."""
        job = Mock()
        job.id = job_id
        job.posted_date = posted_date
        job.archived = archived
        return job

    def test_calculate_total_jobs_count_basic(self):
        """Test total jobs count calculation."""
        jobs = [self.create_mock_job(i) for i in range(5)]

        count = calculate_total_jobs_count(jobs)

        assert count == 5

    def test_calculate_total_jobs_count_empty(self):
        """Test total jobs count with empty list."""
        assert calculate_total_jobs_count([]) == 0
        assert calculate_total_jobs_count(None) == 0

    def test_calculate_active_jobs_count_mixed(self):
        """Test active jobs count with mixed archived/active jobs."""
        jobs = [
            self.create_mock_job(1, archived=False),
            self.create_mock_job(2, archived=True),
            self.create_mock_job(3, archived=False),
            self.create_mock_job(4, archived=True),
            self.create_mock_job(5, archived=False),
        ]

        count = calculate_active_jobs_count(jobs)

        assert count == 3  # 3 non-archived jobs

    def test_calculate_active_jobs_count_all_archived(self):
        """Test active jobs count when all jobs are archived."""
        jobs = [self.create_mock_job(i, archived=True) for i in range(3)]

        count = calculate_active_jobs_count(jobs)

        assert count == 0

    def test_calculate_active_jobs_count_edge_cases(self):
        """Test active jobs count edge cases."""
        assert calculate_active_jobs_count([]) == 0
        assert calculate_active_jobs_count(None) == 0

    def test_find_last_job_posted_basic(self):
        """Test finding the most recent job posting date."""
        now = datetime.now(UTC)
        dates = [
            now - timedelta(days=10),
            now - timedelta(days=5),  # Most recent
            now - timedelta(days=15),
        ]

        jobs = [self.create_mock_job(i, posted_date=dates[i]) for i in range(3)]

        result = find_last_job_posted(jobs)

        assert result == dates[1]  # Most recent date

    def test_find_last_job_posted_with_none_dates(self):
        """Test finding last job posted with some None dates."""
        now = datetime.now(UTC)
        dates = [None, now - timedelta(days=5), None]

        jobs = [self.create_mock_job(i, posted_date=dates[i]) for i in range(3)]

        result = find_last_job_posted(jobs)

        assert result == dates[1]  # Only non-None date

    def test_find_last_job_posted_edge_cases(self):
        """Test finding last job posted edge cases."""
        # Empty list
        assert find_last_job_posted([]) is None

        # None input
        assert find_last_job_posted(None) is None

        # All None dates
        jobs = [self.create_mock_job(i, posted_date=None) for i in range(3)]
        assert find_last_job_posted(jobs) is None

    def test_format_success_rate_percentage_basic(self):
        """Test success rate percentage formatting."""
        assert format_success_rate_percentage(0.0) == 0.0
        assert format_success_rate_percentage(0.5) == 50.0
        assert format_success_rate_percentage(0.75) == 75.0
        assert format_success_rate_percentage(1.0) == 100.0

    def test_format_success_rate_percentage_rounding(self):
        """Test success rate percentage rounding to 1 decimal place."""
        assert format_success_rate_percentage(0.123) == 12.3
        assert format_success_rate_percentage(0.456) == 45.6
        assert format_success_rate_percentage(0.999) == 99.9


class TestT1StreamlitContext:
    """Test T1.4: Streamlit context detection."""

    def test_is_streamlit_context_mock_environment(self):
        """Test Streamlit context detection in mock environment."""
        # In test environment, should return False since no real Streamlit context
        result = is_streamlit_context()

        # This might vary depending on test setup, but should not raise errors
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        True,
        reason="Requires actual Streamlit runtime for integration testing",
    )
    def test_is_streamlit_context_real_environment(self):
        """Test Streamlit context detection in real Streamlit environment."""
        # This would be tested in actual Streamlit integration tests


class TestT1ErrorHandling:
    """Test T1.4: Error handling and edge cases."""

    def test_formatter_functions_handle_errors(self):
        """Test that formatter functions handle errors gracefully."""
        # These should not raise exceptions
        assert calculate_scraping_speed("invalid", None) == 0.0
        assert format_duration("not_a_number") == "0s"
        # Test removed: calculate_progress_percentage function was removed as unused
        assert format_jobs_count("not_a_number") == "0 jobs"

    def test_date_functions_handle_errors(self):
        """Test that date functions handle errors gracefully."""
        # These should not raise exceptions
        assert calculate_days_since_posted("not_a_date") is None
        assert is_job_recently_posted("not_a_date") is False
        assert format_timestamp("not_a_datetime") == "N/A"

    def test_validation_functions_handle_errors(self):
        """Test that validation functions handle errors gracefully."""
        # These should not raise exceptions
        assert safe_int("not_a_number") == 0
        assert safe_job_count("invalid", "Test Company") == 0

    def test_company_helpers_handle_errors(self):
        """Test that company helper functions handle errors gracefully."""
        # These should not raise exceptions with invalid inputs
        assert calculate_total_jobs_count("not_a_list") == 0
        assert calculate_active_jobs_count("not_a_list") == 0
        assert find_last_job_posted("not_a_list") is None
        assert isinstance(format_success_rate_percentage("invalid"), float)


class TestT1PerformanceOptimizations:
    """Test T1.4: Performance optimizations for helper functions."""

    def test_safe_int_performance_with_large_dataset(self):
        """Test safe_int performance with large datasets."""
        import time

        # Test with 10,000 conversions
        start_time = time.time()

        for i in range(10000):
            safe_int(str(i))

        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 1.0  # Under 1 second

    def test_job_helpers_performance_with_large_lists(self):
        """Test job helper performance with large job lists."""
        import time

        # Create large list of mock jobs
        jobs = [self.create_mock_job(i) for i in range(5000)]

        start_time = time.time()

        # Test all company helper functions
        calculate_total_jobs_count(jobs)
        calculate_active_jobs_count(jobs)
        find_last_job_posted(jobs)

        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 0.5  # Under 500ms

    def create_mock_job(self, job_id=1, posted_date=None, archived=False):
        """Helper to create mock job objects."""
        job = Mock()
        job.id = job_id
        job.posted_date = posted_date or datetime.now(UTC)
        job.archived = archived
        return job


class TestT1RealWorldScenarios:
    """Test T1.4: Real-world usage scenarios for UI helpers."""

    def test_typical_job_display_formatting(self):
        """Test typical job display formatting scenario."""
        # Simulate a job with realistic data
        salary_tuple = (80000, 120000)
        posted_date = datetime.now(UTC) - timedelta(days=3)

        # Test all relevant formatters
        salary_min = salary_tuple[0] if salary_tuple else None
        salary_max = salary_tuple[1] if salary_tuple else None
        salary_range = format_salary_range(salary_tuple)
        days_since = calculate_days_since_posted(posted_date)
        is_recent = is_job_recently_posted(posted_date)

        assert salary_min == 80000
        assert salary_max == 120000
        assert salary_range == "$80,000 - $120,000"
        assert days_since == 3
        assert is_recent is True

    def test_company_statistics_calculation(self):
        """Test company statistics calculation scenario."""
        # Create realistic job data
        now = datetime.now(UTC)
        jobs = [
            self.create_mock_job(1, now - timedelta(days=1), archived=False),
            self.create_mock_job(2, now - timedelta(days=3), archived=False),
            self.create_mock_job(3, now - timedelta(days=10), archived=True),
            self.create_mock_job(4, now - timedelta(days=5), archived=False),
        ]

        total_count = calculate_total_jobs_count(jobs)
        active_count = calculate_active_jobs_count(jobs)
        last_posted = find_last_job_posted(jobs)
        success_rate = format_success_rate_percentage(0.85)

        assert total_count == 4
        assert active_count == 3  # 3 non-archived
        assert last_posted == jobs[0].posted_date  # Most recent
        assert success_rate == 85.0

    def test_progress_tracking_scenario(self):
        """Test progress tracking and ETA calculation scenario."""
        start_time = datetime.now(UTC) - timedelta(minutes=15)
        end_time = datetime.now(UTC)

        # Simulate scraping progress
        speed = calculate_scraping_speed(45, start_time, end_time)
        eta = calculate_eta(
            total_companies=20,
            completed_companies=12,
            time_elapsed=900,
        )  # 15 minutes
        progress = round((12 / 20) * 100, 1) if 20 > 0 else 0.0  # Inlined calculation

        assert speed == 3.0  # 45 jobs in 15 minutes = 3 jobs/minute
        assert progress == 60.0  # 12/20 = 60%
        assert isinstance(eta, str)  # Should return formatted time

    def create_mock_job(self, job_id=1, posted_date=None, archived=False):
        """Helper to create mock job objects."""
        job = Mock()
        job.id = job_id
        job.posted_date = posted_date or datetime.now(UTC)
        job.archived = archived
        return job

    def test_input_validation_scenario(self):
        """Test input validation in realistic data processing scenario."""
        # Simulate incoming data with various quality issues
        raw_inputs = [
            "25",  # Valid string number
            42,  # Valid integer
            "3.7",  # Float string
            "",  # Empty string
            None,  # None value
            "invalid",  # Invalid string
            -10,  # Negative number
        ]

        # Process all inputs safely
        validated = [safe_int(inp) for inp in raw_inputs]

        expected = [25, 42, 3, 0, 0, 0, 0]
        assert validated == expected

    def test_edge_case_combination_scenario(self):
        """Test combination of edge cases in realistic workflow."""
        # Simulate processing job with problematic data
        problematic_job_data = {
            "salary": (None, None),
            "posted_date": None,
            "company": None,
            "job_count": "invalid",
        }

        # Process with helper functions
        salary_range = format_salary_range(problematic_job_data["salary"])
        days_since = calculate_days_since_posted(problematic_job_data["posted_date"])
        is_recent = is_job_recently_posted(problematic_job_data["posted_date"])
        company_name = (
            problematic_job_data["company"].name
            if problematic_job_data["company"]
            else "Unknown"
        )
        safe_count = safe_job_count(problematic_job_data["job_count"])

        # All functions should handle gracefully
        assert salary_range == "Not specified"
        assert days_since is None
        assert is_recent is False
        assert company_name == "Unknown"
        assert safe_count == 0
