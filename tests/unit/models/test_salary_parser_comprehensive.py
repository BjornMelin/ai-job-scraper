"""Comprehensive tests for LibrarySalaryParser class.

This module contains extensive unit tests for the LibrarySalaryParser class,
covering all parsing methods, edge cases, k-suffix patterns, context detection,
and time-based conversions. Tests are designed to achieve 100% coverage of the
salary parsing functionality.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.models import JobSQL, LibrarySalaryParser, SalaryContext, SimplePrice


class TestSalaryContext:
    """Tests for SalaryContext dataclass."""

    def test_salary_context_creation(self):
        """Test SalaryContext creation with default values."""
        context = SalaryContext()
        assert context.is_up_to is False
        assert context.is_from is False
        assert context.is_hourly is False
        assert context.is_monthly is False

    def test_salary_context_with_values(self):
        """Test SalaryContext creation with explicit values."""
        context = SalaryContext(is_up_to=True, is_hourly=True)
        assert context.is_up_to is True
        assert context.is_from is False
        assert context.is_hourly is True
        assert context.is_monthly is False


class TestSimplePrice:
    """Tests for SimplePrice dataclass."""

    def test_simple_price_creation(self):
        """Test SimplePrice creation."""
        price = SimplePrice(amount=Decimal("50000"))
        assert price.amount == Decimal("50000")


class TestLibrarySalaryParser:
    """Comprehensive tests for LibrarySalaryParser class."""

    def test_parse_salary_text_empty_inputs(self):
        """Test parse_salary_text with empty/invalid inputs."""
        # Empty string
        assert LibrarySalaryParser.parse_salary_text("") == (None, None)

        # Whitespace only
        assert LibrarySalaryParser.parse_salary_text("   ") == (None, None)

        # None input
        assert LibrarySalaryParser.parse_salary_text(None) == (None, None)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Basic k-suffix single values
            ("100k", (100000, 100000)),
            ("75K", (75000, 75000)),
            ("125.5k", (125500, 125500)),
            ("$150k", (150000, 150000)),
            # Range patterns with k-suffix
            ("100k-150k", (100000, 150000)),
            ("80k - 120k", (80000, 120000)),
            ("100-120k", (100000, 120000)),  # Shared k suffix
            ("110k-150", (110000, 150000)),  # One-sided k
            # Range patterns with "to"
            ("100k to 150k", (100000, 150000)),
            ("80K TO 120K", (80000, 120000)),
            # Context patterns - "up to"
            ("up to 120k", (None, 120000)),
            ("maximum of $150k", (None, 150000)),
            ("max of 100K", (None, 100000)),
            # Context patterns - "from"
            ("from 100k", (100000, None)),
            ("starting at $120k", (120000, None)),
            ("minimum of 80K", (80000, None)),
            # Without k-suffix but with currency
            ("$100000", (100000, 100000)),
            ("£80,000", (80000, 80000)),
            ("€75000", (75000, 75000)),
            # Hourly rates
            ("$50 per hour", (104000, 104000)),  # 50 * 40 * 52
            ("£25/hour", (52000, 52000)),  # 25 * 40 * 52
            ("$30/hr", (62400, 62400)),
            # Monthly rates
            ("$8000 per month", (96000, 96000)),  # 8000 * 12
            ("£5000/month", (60000, 60000)),
            ("€6000/mo", (72000, 72000)),
        ],
    )
    def test_parse_salary_text_common_patterns(self, text, expected):
        """Test parse_salary_text with common salary patterns."""
        result = LibrarySalaryParser.parse_salary_text(text)
        assert result == expected

    def test_detect_context_patterns(self):
        """Test _detect_context method with various patterns."""
        # Up to patterns
        context = LibrarySalaryParser._detect_context("up to $120k")
        assert context.is_up_to is True
        assert context.is_from is False

        # From patterns
        context = LibrarySalaryParser._detect_context("starting at £80k")
        assert context.is_from is True
        assert context.is_up_to is False

        # Hourly patterns
        context = LibrarySalaryParser._detect_context("$50 per hour")
        assert context.is_hourly is True
        assert context.is_monthly is False

        # Monthly patterns
        context = LibrarySalaryParser._detect_context("$8000 monthly")
        assert context.is_monthly is True
        assert context.is_hourly is False

        # Multiple patterns
        context = LibrarySalaryParser._detect_context("up to $50/hour")
        assert context.is_up_to is True
        assert context.is_hourly is True

    def test_parse_k_suffix_ranges(self):
        """Test _parse_k_suffix_ranges method."""
        # Standard range with shared k
        result = LibrarySalaryParser._parse_k_suffix_ranges("100-120k")
        assert result == (100000, 120000)

        # Both numbers with k
        result = LibrarySalaryParser._parse_k_suffix_ranges("100k-150k")
        assert result == (100000, 150000)

        # One-sided k
        result = LibrarySalaryParser._parse_k_suffix_ranges("110k-150")
        assert result == (110000, 150000)

        # With "to" separator
        result = LibrarySalaryParser._parse_k_suffix_ranges("100k to 150k")
        assert result == (100000, 150000)

        # Decimal values
        result = LibrarySalaryParser._parse_k_suffix_ranges("125.5k-150.5k")
        assert result == (125500, 150500)

        # No k-suffix pattern
        result = LibrarySalaryParser._parse_k_suffix_ranges("100000-150000")
        assert result is None

    @patch("src.models.Price")
    def test_extract_multiple_prices(self, mock_price):
        """Test _extract_multiple_prices method."""
        # Mock Price objects
        mock_price1 = MagicMock()
        mock_price1.amount = 100000
        mock_price2 = MagicMock()
        mock_price2.amount = 150000

        mock_price.fromstring.side_effect = [mock_price1, mock_price2]

        # Test range with indicators
        text = "$100,000 to $150,000 salary range"
        LibrarySalaryParser._extract_multiple_prices(text)

        # Should attempt to parse multiple parts
        assert mock_price.fromstring.call_count >= 1

    def test_apply_context_logic(self):
        """Test _apply_context_logic method."""
        # Up to context
        context = SalaryContext(is_up_to=True)
        result = LibrarySalaryParser._apply_context_logic(120000, context)
        assert result == (None, 120000)

        # From context
        context = SalaryContext(is_from=True)
        result = LibrarySalaryParser._apply_context_logic(100000, context)
        assert result == (100000, None)

        # No special context
        context = SalaryContext()
        result = LibrarySalaryParser._apply_context_logic(150000, context)
        assert result == (150000, 150000)

    def test_convert_time_based_salary(self):
        """Test _convert_time_based_salary method."""
        # Hourly conversion
        values = [50, 30]
        result = LibrarySalaryParser._convert_time_based_salary(
            values, is_hourly=True, is_monthly=False
        )
        assert result == [104000, 62400]  # 50*40*52, 30*40*52

        # Monthly conversion
        values = [8000, 10000]
        result = LibrarySalaryParser._convert_time_based_salary(
            values, is_hourly=False, is_monthly=True
        )
        assert result == [96000, 120000]  # 8000*12, 10000*12

        # No conversion (annual)
        values = [100000, 150000]
        result = LibrarySalaryParser._convert_time_based_salary(
            values, is_hourly=False, is_monthly=False
        )
        assert result == [100000, 150000]

        # Custom work parameters
        values = [40]
        result = LibrarySalaryParser._convert_time_based_salary(
            values,
            is_hourly=True,
            is_monthly=False,
            weekly_hours=35,
            working_weeks_per_year=50,
        )
        assert result == [70000]  # 40 * 35 * 50

    def test_safe_decimal_to_int(self):
        """Test _safe_decimal_to_int method."""
        # Valid decimal string
        result = LibrarySalaryParser._safe_decimal_to_int("125.5")
        assert result == 125

        # Invalid format
        result = LibrarySalaryParser._safe_decimal_to_int("invalid")
        assert result is None

        # Empty string
        result = LibrarySalaryParser._safe_decimal_to_int("")
        assert result is None

    def test_clean_text_for_babel(self):
        """Test _clean_text_for_babel method."""
        # Text with currency symbols
        result = LibrarySalaryParser._clean_text_for_babel("$100k per year")
        assert "$" not in result

        # Text with multiple numbers (range)
        result = LibrarySalaryParser._clean_text_for_babel("100k to 150k annually")
        assert "100" in result
        assert "150" in result

        # Text with single number
        result = LibrarySalaryParser._clean_text_for_babel("$120k annually")
        # Should extract just the number
        assert any(char.isdigit() for char in result)

    def test_apply_k_suffix_multiplication(self):
        """Test _apply_k_suffix_multiplication method."""
        # Text with k suffix
        result = LibrarySalaryParser._apply_k_suffix_multiplication("120k", 120)
        assert result == 120000

        # Text without k suffix
        result = LibrarySalaryParser._apply_k_suffix_multiplication("120000", 120000)
        assert result == 120000

        # Text with K suffix (uppercase)
        result = LibrarySalaryParser._apply_k_suffix_multiplication("100K", 100)
        assert result == 100000

    @patch("src.models.parse_decimal")
    def test_parse_with_babel_fallback(self, mock_parse_decimal):
        """Test _parse_with_babel_fallback method."""
        # Successful decimal parsing without k-suffix
        mock_parse_decimal.return_value = Decimal("100000")
        context = SalaryContext()

        result = LibrarySalaryParser._parse_with_babel_fallback("100000", context)
        assert result == (100000, 100000)

        # With k-suffix - mock should return the base value before k multiplication
        mock_parse_decimal.return_value = Decimal("100")  # Base value before k-suffix
        result = LibrarySalaryParser._parse_with_babel_fallback("100k", context)
        assert result[0] == 100000  # Should multiply by 1000: 100 * 1000 = 100000

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Very large numbers
        result = LibrarySalaryParser.parse_salary_text("500k")
        assert result == (500000, 500000)

        # Very small numbers (should still work)
        result = LibrarySalaryParser.parse_salary_text("1k")
        assert result == (1000, 1000)

        # Malformed input
        result = LibrarySalaryParser.parse_salary_text("abc-xyz")
        assert result == (None, None)

        # Mixed valid/invalid patterns
        result = LibrarySalaryParser.parse_salary_text("100k-abc")
        # Should still try to parse what it can
        assert result is not None

    def test_complex_salary_descriptions(self):
        """Test parsing of complex, realistic salary descriptions."""
        # Job posting style descriptions
        test_cases = [
            ("Salary: $100k - $150k per annum plus benefits", (100000, 150000)),
            ("Competitive salary up to £120,000 DOE", (None, 120000)),
            ("Starting from $80k with bonus potential", (80000, None)),
            ("£50-60k base + equity package", (50000, 60000)),
            ("Hourly rate: $45-65/hour", (93600, 135200)),  # 45*40*52 to 65*40*52
            ("Monthly: €8k-10k gross", (96000, 120000)),  # 8k*12 to 10k*12
        ]

        for description, expected in test_cases:
            result = LibrarySalaryParser.parse_salary_text(description)
            assert result == expected, f"Failed for: {description}"

    def test_international_formats(self):
        """Test parsing of international salary formats."""
        # Different currency symbols
        test_cases = [
            ("£100k", (100000, 100000)),
            ("€120k", (120000, 120000)),
            ("¥5000k", (5000000, 5000000)),
            ("₹800k", (800000, 800000)),
        ]

        for text, expected in test_cases:
            result = LibrarySalaryParser.parse_salary_text(text)
            assert result == expected, f"Failed for: {text}"


class TestJobSQLSalaryIntegration:
    """Test JobSQL salary parsing integration."""

    def test_jobsql_salary_field_validator(self):
        """Test JobSQL salary field validator uses LibrarySalaryParser."""
        # Test with string input
        result = JobSQL.parse_salary("100k-150k")
        assert result == (100000, 150000)

        # Test with tuple input (should pass through)
        result = JobSQL.parse_salary((80000, 120000))
        assert result == (80000, 120000)

        # Test with None input
        result = JobSQL.parse_salary(None)
        assert result == (None, None)

        # Test with empty string
        result = JobSQL.parse_salary("")
        assert result == (None, None)

    def test_jobsql_create_with_salary_parsing(self):
        """Test JobSQL creation with salary parsing."""
        job_data = {
            "title": "Software Engineer",
            "description": "Great role",
            "link": "https://example.com/job/1",
            "salary": "100k-150k per year",
            "location": "San Francisco",
        }

        job = JobSQL.create_validated(**job_data)

        # Should have parsed salary correctly
        assert job.salary == (100000, 150000)

        # Test helper functions for salary display
        from src.ui.utils.ui_helpers import (
            format_salary_range,
            get_salary_max,
            get_salary_min,
        )

        assert get_salary_min(job.salary) == 100000
        assert get_salary_max(job.salary) == 150000
        assert format_salary_range(job.salary) == "$100,000 - $150,000"

    def test_jobsql_salary_helper_functions(self):
        """Test JobSQL salary helper functions (replacement for computed properties)."""
        from src.ui.utils.ui_helpers import (
            format_salary_range,
            get_salary_max,
            get_salary_min,
        )

        # Create job with salary range
        job = JobSQL(
            title="Test Job",
            description="Test",
            link="https://example.com/test",
            salary=(100000, 150000),
            location="Test City",
        )

        assert get_salary_min(job.salary) == 100000
        assert get_salary_max(job.salary) == 150000
        assert format_salary_range(job.salary) == "$100,000 - $150,000"

        # Test single value salary
        job.salary = (120000, 120000)
        assert format_salary_range(job.salary) == "$120,000"

        # Test "from" salary
        job.salary = (100000, None)
        assert format_salary_range(job.salary) == "From $100,000"

        # Test "up to" salary
        job.salary = (None, 150000)
        assert format_salary_range(job.salary) == "Up to $150,000"

        # Test no salary
        job.salary = (None, None)
        assert format_salary_range(job.salary) == "Not specified"
