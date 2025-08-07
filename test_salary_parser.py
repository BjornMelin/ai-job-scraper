"""Test suite for the enhanced salary parser in JobSQL model."""

import sys

from src.models import JobSQL


class TestSalaryParser:
    """Test suite for JobSQL salary parsing functionality."""

    def test_basic_range_formats(self):
        """Test basic salary range formats."""
        # Standard ranges with 'k' suffix
        assert JobSQL.parse_salary("$100k-150k") == (100000, 150000)
        assert JobSQL.parse_salary("£80,000 - £120,000") == (80000, 120000)
        assert JobSQL.parse_salary("110k to 150k") == (110000, 150000)
        assert JobSQL.parse_salary("€90000-€130000") == (90000, 130000)

    def test_single_values(self):
        """Test single salary values."""
        # Single values should return the same for min and max
        assert JobSQL.parse_salary("$120k") == (120000, 120000)
        assert JobSQL.parse_salary("150000") == (150000, 150000)
        assert JobSQL.parse_salary("85.5k") == (85500, 85500)

    def test_contextual_patterns(self):
        """Test contextual patterns like 'up to' and 'from'."""
        # "up to" patterns should return (None, max)
        assert JobSQL.parse_salary("up to $150k") == (None, 150000)
        assert JobSQL.parse_salary("maximum of £100,000") == (None, 100000)
        assert JobSQL.parse_salary("not more than 120k") == (None, 120000)

        # "from" patterns should return (min, None)
        assert JobSQL.parse_salary("from $110k") == (110000, None)
        assert JobSQL.parse_salary("starting at €80000") == (80000, None)
        assert JobSQL.parse_salary("minimum of 90k") == (90000, None)
        assert JobSQL.parse_salary("at least £75,000") == (75000, None)

    def test_currency_symbols(self):
        """Test various currency symbols."""
        assert JobSQL.parse_salary("$100000") == (100000, 100000)
        assert JobSQL.parse_salary("£85000") == (85000, 85000)
        assert JobSQL.parse_salary("€95000") == (95000, 95000)
        assert JobSQL.parse_salary("¥100000") == (100000, 100000)
        assert JobSQL.parse_salary("₹500000") == (500000, 500000)

    def test_common_phrases(self):
        """Test removal of common salary phrases."""
        assert JobSQL.parse_salary("$110k - $150k per year") == (110000, 150000)
        assert JobSQL.parse_salary("£80,000 per annum") == (80000, 80000)
        assert JobSQL.parse_salary("€100k annually") == (100000, 100000)
        assert JobSQL.parse_salary("$120k plus benefits") == (120000, 120000)
        assert JobSQL.parse_salary("85k depending on experience") == (85000, 85000)
        assert JobSQL.parse_salary("$90k DOE") == (90000, 90000)
        assert JobSQL.parse_salary("£70,000 gross") == (70000, 70000)
        assert JobSQL.parse_salary("$130k before tax") == (130000, 130000)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty or None values
        assert JobSQL.parse_salary(None) == (None, None)
        assert JobSQL.parse_salary("") == (None, None)
        assert JobSQL.parse_salary("   ") == (None, None)

        # Already a tuple
        assert JobSQL.parse_salary((80000, 120000)) == (80000, 120000)

        # Non-parseable strings
        assert JobSQL.parse_salary("competitive") == (None, None)
        assert JobSQL.parse_salary("negotiable") == (None, None)
        assert JobSQL.parse_salary("TBD") == (None, None)

    def test_decimal_handling(self):
        """Test decimal value handling."""
        assert JobSQL.parse_salary("120.5k") == (120500, 120500)
        assert JobSQL.parse_salary("$85.75k - $95.25k") == (85750, 95250)
        assert JobSQL.parse_salary("150.999k") == (150999, 150999)

    def test_comma_handling(self):
        """Test comma handling in numbers."""
        assert JobSQL.parse_salary("100,000") == (100000, 100000)
        assert JobSQL.parse_salary("$1,250,000") == (1250000, 1250000)
        assert JobSQL.parse_salary("80,000 - 120,000") == (80000, 120000)

    def test_shared_k_suffix(self):
        """Test ranges where 'k' applies to both numbers."""
        assert JobSQL.parse_salary("100-120k") == (100000, 120000)
        assert JobSQL.parse_salary("85-95K") == (85000, 95000)
        assert JobSQL.parse_salary("110 - 150k") == (110000, 150000)

    def test_mixed_formats(self):
        """Test mixed and complex formats."""
        assert JobSQL.parse_salary("$100k-$150k per year plus benefits") == (
            100000,
            150000,
        )
        assert JobSQL.parse_salary("From £80,000 to £120,000 per annum") == (
            80000,
            120000,
        )
        assert JobSQL.parse_salary("Starting at $90k, up to $130k DOE") == (
            90000,
            130000,
        )

    def test_hourly_monthly_rates(self):
        """Test handling of hourly and monthly rates."""
        # These should be parsed but the context is removed
        assert JobSQL.parse_salary("$50 per hour") == (50, 50)
        assert JobSQL.parse_salary("£5000 per month") == (5000, 5000)
        assert JobSQL.parse_salary("€30 hourly") == (30, 30)


def run_tests():
    """Run all tests and report results."""
    test_suite = TestSalaryParser()
    methods = [method for method in dir(test_suite) if method.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in methods:
        method = getattr(test_suite, method_name)
        try:
            method()
            print(f"✓ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {method_name}: Unexpected error - {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
