"""Comprehensive tests for LibrarySalaryParser library-first implementation.

This module provides focused tests for the new LibrarySalaryParser class,
testing each method individually and covering edge cases not covered by
the main model validation tests.
"""

import pytest

from src.models import LibrarySalaryParser, SalaryContext


class TestLibrarySalaryParserDirectly:
    """Test LibrarySalaryParser methods directly for better coverage."""

    def test_detect_context_all_flags(self) -> None:
        """Test context detection for various patterns."""
        # Test up-to patterns
        context = LibrarySalaryParser._detect_context("up to $150k")
        assert context.is_up_to is True
        assert context.is_from is False
        assert context.is_hourly is False
        assert context.is_monthly is False

        # Test from patterns
        context = LibrarySalaryParser._detect_context("starting at $100k")
        assert context.is_from is True
        assert context.is_up_to is False

        # Test hourly patterns
        context = LibrarySalaryParser._detect_context("$50 per hour")
        assert context.is_hourly is True
        assert context.is_monthly is False

        # Test monthly patterns
        context = LibrarySalaryParser._detect_context("$5000 per month")
        assert context.is_monthly is True
        assert context.is_hourly is False

        # Test multiple flags
        context = LibrarySalaryParser._detect_context("up to $50 per hour")
        assert context.is_up_to is True
        assert context.is_hourly is True

    def test_apply_context_logic(self) -> None:
        """Test context application logic."""
        # Test up-to context
        context = SalaryContext(is_up_to=True)
        result = LibrarySalaryParser._apply_context_logic(150000, context)
        assert result == (None, 150000)

        # Test from context
        context = SalaryContext(is_from=True)
        result = LibrarySalaryParser._apply_context_logic(100000, context)
        assert result == (100000, None)

        # Test default context
        context = SalaryContext()
        result = LibrarySalaryParser._apply_context_logic(120000, context)
        assert result == (120000, 120000)

    def test_convert_time_based_salary(self) -> None:
        """Test time-based salary conversions."""
        # Test hourly conversion: hourly * 40 * 52
        result = LibrarySalaryParser._convert_time_based_salary([50], True, False)
        assert result == [104000]  # 50 * 40 * 52

        # Test monthly conversion: monthly * 12
        result = LibrarySalaryParser._convert_time_based_salary([5000], False, True)
        assert result == [60000]  # 5000 * 12

        # Test no conversion (annual)
        result = LibrarySalaryParser._convert_time_based_salary([100000], False, False)
        assert result == [100000]

        # Test multiple values
        result = LibrarySalaryParser._convert_time_based_salary([20, 30], True, False)
        assert result == [41600, 62400]  # 20*40*52, 30*40*52

    def test_safe_decimal_to_int(self) -> None:
        """Test safe decimal conversion."""
        assert LibrarySalaryParser._safe_decimal_to_int("123.45") == 123
        assert LibrarySalaryParser._safe_decimal_to_int("100") == 100
        assert LibrarySalaryParser._safe_decimal_to_int("invalid") is None
        assert LibrarySalaryParser._safe_decimal_to_int("") is None

    def test_safe_decimal_to_float(self) -> None:
        """Test safe decimal to float conversion for k-suffix parsing."""
        # Test decimal precision preservation
        assert LibrarySalaryParser._safe_decimal_to_float("125.5") == 125.5
        assert LibrarySalaryParser._safe_decimal_to_float("150.75") == 150.75
        assert LibrarySalaryParser._safe_decimal_to_float("100") == 100.0
        assert LibrarySalaryParser._safe_decimal_to_float("99.9") == 99.9

        # Test error handling
        assert LibrarySalaryParser._safe_decimal_to_float("invalid") is None
        assert LibrarySalaryParser._safe_decimal_to_float("") is None

    def test_apply_k_suffix_multiplication(self) -> None:
        """Test k-suffix multiplication."""
        # With k suffix
        assert LibrarySalaryParser._apply_k_suffix_multiplication("100k", 100) == 100000
        assert LibrarySalaryParser._apply_k_suffix_multiplication("85.5K", 85) == 85000

        # Without k suffix
        assert LibrarySalaryParser._apply_k_suffix_multiplication("100", 100) == 100
        assert LibrarySalaryParser._apply_k_suffix_multiplication("85000", 85) == 85

    def test_clean_text_for_babel(self) -> None:
        """Test text cleaning for babel parsing."""
        # Remove currency symbols
        cleaned = LibrarySalaryParser._clean_text_for_babel("$100,000")
        assert "$" not in cleaned

        # Remove common phrases
        cleaned = LibrarySalaryParser._clean_text_for_babel(
            "$120k per year plus benefits",
        )
        assert "per year" not in cleaned
        assert "plus benefits" not in cleaned

        # Extract numeric part
        cleaned = LibrarySalaryParser._clean_text_for_babel("Salary: $100,000 DOE")
        assert cleaned in ["100,000", "100000"]  # Either format acceptable

    def test_parse_k_suffix_ranges(self) -> None:
        """Test k-suffix range parsing."""
        # Range with shared k: "100-120k"
        result = LibrarySalaryParser._parse_k_suffix_ranges("100-120k")
        assert result == (100000, 120000)

        # Range with both k: "100k-150k"
        result = LibrarySalaryParser._parse_k_suffix_ranges("100k-150k")
        assert result == (100000, 150000)

        # Range with one-sided k: "100k-120"
        result = LibrarySalaryParser._parse_k_suffix_ranges("100k-120")
        assert result == (100000, 120000)

        # "to" pattern: "100k to 150k"
        result = LibrarySalaryParser._parse_k_suffix_ranges("100k to 150k")
        assert result == (100000, 150000)

        # No k-suffix range
        result = LibrarySalaryParser._parse_k_suffix_ranges("100-120")
        assert result is None

    def test_decimal_precision_k_suffix_ranges(self) -> None:
        """Test decimal precision preservation in k-suffix ranges - THE BUG FIX."""
        # Test the specific bug case: "125.5k-150.5k" should return (125500, 150500)
        result = LibrarySalaryParser._parse_k_suffix_ranges("125.5k-150.5k")
        assert result == (125500, 150500), f"Expected (125500, 150500) but got {result}"

        # Test shared k suffix with decimals: "125.5-150.5k"
        result = LibrarySalaryParser._parse_k_suffix_ranges("125.5-150.5k")
        assert result == (125500, 150500), f"Expected (125500, 150500) but got {result}"

        # Test "to" pattern with decimals: "125.5k to 150.75k"
        result = LibrarySalaryParser._parse_k_suffix_ranges("125.5k to 150.75k")
        assert result == (125500, 150750), f"Expected (125500, 150750) but got {result}"

        # Test one-sided k with decimals: "125.5k-150"
        result = LibrarySalaryParser._parse_k_suffix_ranges("125.5k-150")
        assert result == (125500, 150000), f"Expected (125500, 150000) but got {result}"

        # Test single decimal cases
        result = LibrarySalaryParser._parse_k_suffix_ranges("99.9k-100.1k")
        assert result == (99900, 100100), f"Expected (99900, 100100) but got {result}"

        # Test high precision decimals
        result = LibrarySalaryParser._parse_k_suffix_ranges("85.25k-95.75k")
        assert result == (85250, 95750), f"Expected (85250, 95750) but got {result}"

    def test_extract_multiple_prices(self) -> None:
        """Test multiple price extraction."""
        # Test range with currency
        prices = LibrarySalaryParser._extract_multiple_prices("$100,000 - $150,000")
        assert len(prices) >= 2

        # Test k-suffix handling in price extraction
        prices = LibrarySalaryParser._extract_multiple_prices("100k-150k")
        assert len(prices) >= 2
        assert prices[0].amount == 100000
        assert prices[1].amount == 150000

    def test_parse_single_salary_edge_cases(self) -> None:
        """Test single salary parsing edge cases."""
        # Test with k-suffix and hourly context
        context = SalaryContext(is_hourly=True)
        result = LibrarySalaryParser._parse_single_salary("50k", context)
        expected = 50000 * 40 * 52  # Convert to annual
        assert result == (expected, expected)

        # Test fallback to babel when price-parser fails
        context = SalaryContext()
        result = LibrarySalaryParser._parse_single_salary("random text 100", context)
        # Should parse the number 100 from the text
        assert result == (100, 100)

    @pytest.mark.parametrize(
        ("text", "expected_range"),
        (
            # Test library integration edge cases
            ("£100k-£150k", (100000, 150000)),  # Multiple currency symbols
            ("€85.5k to €95.5k", (85500, 95500)),  # Decimal k values - BUG FIX
            ("¥1000k", (1000000, 1000000)),  # Non-standard currency
            ("₹500k-₹750k", (500000, 750000)),  # Range with Indian Rupees
            # Test decimal precision preservation - THE BUG FIX CASES
            ("125.5k", (125500, 125500)),  # Single decimal k value
            ("150.75K", (150750, 150750)),  # Capital K with decimal
            ("99.9k-100.1k", (99900, 100100)),  # Decimal range
            ("85.25k to 95.75k", (85250, 95750)),  # "to" pattern with decimals
            # Test complex formatting
            ("Salary range: $100,000 - $150,000 per annum", (100000, 150000)),
            ("Compensation: £80k-£120k depending on experience", (80000, 120000)),
            # Test edge cases that should return None
            ("To be determined", (None, None)),
            ("Competitive package", (None, None)),
            ("Contact for details", (None, None)),
        ),
    )
    def test_library_integration_cases(
        self,
        text: str,
        expected_range: tuple[int | None, int | None],
    ) -> None:
        """Test library integration with various real-world formats."""
        result = LibrarySalaryParser.parse_salary_text(text)
        assert result == expected_range

    def test_error_handling(self) -> None:
        """Test error handling in various parsing scenarios."""
        # Test empty and None inputs
        assert LibrarySalaryParser.parse_salary_text("") == (None, None)
        assert LibrarySalaryParser.parse_salary_text("   ") == (None, None)

        # Test invalid k-suffix parsing
        context = SalaryContext()
        result = LibrarySalaryParser._parse_single_salary("invalid.k", context)
        assert result == (None, None)

        # Test babel fallback with invalid input
        result = LibrarySalaryParser._parse_with_babel_fallback(
            "no numbers here",
            context,
        )
        assert result == (None, None)

    def test_performance_patterns(self) -> None:
        """Test patterns that should be fast to parse."""
        # These should be handled by the k-suffix patterns (fastest path)
        fast_patterns = [
            "100k",
            "85.5K",
            "100-120k",
            "100k-150k",
            "110k to 150k",
        ]

        for pattern in fast_patterns:
            result = LibrarySalaryParser.parse_salary_text(pattern)
            assert result != (None, None), f"Should parse: {pattern}"
            assert result[0] is not None or result[1] is not None

    def test_currency_handling(self) -> None:
        """Test various currency symbol handling."""
        currencies = ["$", "£", "€", "¥", "₹", "¢"]

        for currency in currencies:
            text = f"{currency}100k"
            result = LibrarySalaryParser.parse_salary_text(text)
            assert result == (100000, 100000), f"Failed for currency: {currency}"
