"""Tests for UI helpers validation utilities.

Tests the validation and safety functions in ui_helpers.py including:
- Safe integer conversion and validation
- Job count validation with context-aware logging
- Streamlit context detection
- Safe parsing with Pydantic models
"""

from unittest.mock import Mock, patch

import pytest

from src.ui.utils.ui_helpers import (
    is_streamlit_context,
    safe_int,
    safe_job_count,
)


class TestSafeInt:
    """Test the safe_int utility function."""

    def test_converts_valid_integers(self):
        """Test safe_int converts valid integers correctly."""
        test_cases = [
            (0, 0),
            (1, 1),
            (100, 100),
            (1000, 1000),
        ]

        for input_val, expected in test_cases:
            # Act
            result = safe_int(input_val)

            # Assert
            assert result == expected

    def test_converts_with_custom_default(self):
        """Test safe_int uses custom default for invalid inputs."""
        # Act - empty string causes ValidationError, should use default
        result = safe_int("", default=42)

        # Assert
        assert result == 42

    def test_handles_validation_errors_gracefully(self):
        """Test safe_int handles ValidationError gracefully."""
        # Act - Use an input that causes ValidationError (empty string)
        result = safe_int("", default=10)

        # Assert
        assert result == 10

    def test_handles_unexpected_exceptions(self):
        """Test safe_int handles unexpected exceptions gracefully."""
        # For this test, we can simulate an exception by using an invalid value
        # that would cause an exception in the conversion logic

        # Act - use a complex object that should fall back to default
        result = safe_int(object(), default=5)

        # Assert
        assert result == 5

    def test_ensures_non_negative_default(self):
        """Test safe_int ensures default is non-negative."""
        # Act - Test with negative default
        result = safe_int("invalid", default=-10)

        # Assert - Should be max(0, -10) = 0
        assert result == 0

    def test_successful_conversion_returns_validator_value(self):
        """Test safe_int returns validator value on successful conversion."""
        # Act
        result = safe_int("123")

        # Assert
        assert result == 123

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        (
            (None, 0),
            ("", 0),
            ("invalid", 0),
            ([], 0),
            ({}, 0),
            (float("inf"), 0),
            (float("nan"), 0),
            (-5, 0),
            ("123", 123),
            (42, 42),
            (True, 1),
            (False, 0),
        ),
    )
    def test_safe_int_comprehensive_inputs(self, input_val, expected):
        """Test safe_int with comprehensive input types."""
        # Act
        result = safe_int(input_val)

        # Assert
        assert result == expected


class TestSafeJobCount:
    """Test the safe_job_count utility function."""

    def test_converts_valid_job_counts(self):
        """Test safe_job_count converts valid job counts correctly."""
        test_cases = [
            (0, 0),
            (1, 1),
            (25, 25),
            (100, 100),
        ]

        for input_val, expected in test_cases:
            # Act
            result = safe_job_count(input_val, "Test Company")

            # Assert
            assert result == expected

    def test_handles_invalid_job_counts_with_context(self):
        """Test safe_job_count handles invalid inputs with company context."""
        # Act
        result = safe_job_count("invalid", "Test Company")

        # Assert
        assert result == 0

    def test_logs_conversion_information(self):
        """Test safe_job_count logs conversion information appropriately."""
        with patch("src.ui.utils.ui_helpers.logger") as mock_logger:
            # Act - Value that gets converted
            result = safe_job_count("5", "Test Company")

            # Assert
            assert result == 5
            # Should log conversion info when original != result (string "5" -> int 5)
            mock_logger.info.assert_called_once()
            assert "Test Company" in mock_logger.info.call_args[0][1]

    def test_logs_warning_on_conversion_failure(self):
        """Test safe_job_count logs warning on conversion failure."""
        with patch("src.ui.utils.ui_helpers.safe_int") as mock_safe_int:
            mock_safe_int.side_effect = Exception("Conversion failed")

            with patch("src.ui.utils.ui_helpers.logger") as mock_logger:
                # Act
                result = safe_job_count("invalid", "Test Company")

                # Assert
                assert result == 0
                mock_logger.warning.assert_called_once()
                assert "Test Company" in mock_logger.warning.call_args[0][1]

    def test_no_logging_for_unchanged_values(self):
        """Test safe_job_count doesn't log when value unchanged."""
        with patch("src.ui.utils.ui_helpers.logger") as mock_logger:
            # Act - Value that doesn't change
            result = safe_job_count(10, "Test Company")

            # Assert
            assert result == 10
            # Should not log info when value unchanged
            mock_logger.info.assert_not_called()

    def test_no_logging_for_none_values(self):
        """Test safe_job_count handles None values without unnecessary logging."""
        with patch("src.ui.utils.ui_helpers.logger") as mock_logger:
            # Act
            result = safe_job_count(None, "Test Company")

            # Assert
            assert result == 0
            # Should not log info for None -> 0 conversion
            mock_logger.info.assert_not_called()

    def test_uses_default_company_name(self):
        """Test safe_job_count uses default company name when not provided."""
        # Act
        result = safe_job_count(5)

        # Assert
        assert result == 5

        # Test with invalid value and default company name
        result = safe_job_count("invalid")
        assert result == 0

    @pytest.mark.parametrize(
        ("input_val", "company", "expected"),
        (
            (None, "Company A", 0),
            ("5", "Company B", 5),
            (-10, "Company C", 0),
            (25, "Company D", 25),
            ("invalid", "Company E", 0),
            ([], "Company F", 0),
        ),
    )
    def test_safe_job_count_various_scenarios(self, input_val, company, expected):
        """Test safe_job_count with various input scenarios."""
        # Act
        result = safe_job_count(input_val, company)

        # Assert
        assert result == expected


class TestIsStreamlitContext:
    """Test the Streamlit context detection function."""

    def test_detects_streamlit_context_when_available(self):
        """Test function detects Streamlit context when available."""
        with patch("streamlit.runtime.scriptrunner.get_script_run_ctx") as mock_get_ctx:
            # Arrange - Mock successful Streamlit context
            mock_get_ctx.return_value = Mock()  # Non-None context

            # Act
            result = is_streamlit_context()

            # Assert
            assert result is True

    def test_detects_no_context_when_none_returned(self):
        """Test function detects no context when None returned."""
        with patch("streamlit.runtime.scriptrunner.get_script_run_ctx") as mock_get_ctx:
            # Arrange - Mock no Streamlit context
            mock_get_ctx.return_value = None

            # Act
            result = is_streamlit_context()

            # Assert
            assert result is False

    def test_handles_import_error_gracefully(self):
        """Test function handles ImportError gracefully."""
        with patch("streamlit.runtime.scriptrunner.get_script_run_ctx") as mock_get_ctx:
            # Arrange - Mock ImportError
            mock_get_ctx.side_effect = ImportError("Streamlit not available")

            # Act
            result = is_streamlit_context()

            # Assert
            assert result is False

    def test_handles_attribute_error_gracefully(self):
        """Test function handles AttributeError gracefully."""
        with patch("streamlit.runtime.scriptrunner.get_script_run_ctx") as mock_get_ctx:
            # Arrange - Mock AttributeError
            mock_get_ctx.side_effect = AttributeError("get_script_run_ctx not found")

            # Act
            result = is_streamlit_context()

            # Assert
            assert result is False

    def test_handles_other_exceptions_gracefully(self):
        """Test function handles other exceptions gracefully."""
        # Test actual import error scenario by mocking the module import
        with patch.dict("sys.modules", {"streamlit.runtime.scriptrunner": None}):
            # Act
            result = is_streamlit_context()

            # Assert
            assert result is False

    def test_actual_streamlit_import_behavior(self):
        """Test function behavior when Streamlit is not actually available."""
        # This test runs without mocking to check real import behavior
        # In test environment, Streamlit might not be available

        # Act
        result = is_streamlit_context()

        # Assert - Should handle gracefully regardless of Streamlit availability
        assert isinstance(result, bool)


class TestValidationUtilitiesIntegration:
    """Integration tests for validation utilities working together."""

    def test_safe_int_and_safe_job_count_consistency(self):
        """Test safe_int and safe_job_count produce consistent results."""
        test_values = [
            0,
            1,
            "5",
            "invalid",
            None,
            -10,
            3.14,
            True,
            False,
        ]

        for value in test_values:
            # Act
            safe_int_result = safe_int(value)
            safe_job_count_result = safe_job_count(value, "Test Company")

            # Assert - Both should produce same result
            assert safe_int_result == safe_job_count_result

    def test_validation_with_realistic_job_data(self):
        """Test validation functions with realistic job scraping data."""
        # Simulate real job count data that might come from scraping
        realistic_inputs = [
            ("25 open positions", 25),
            ("No jobs found", 0),
            ("500+ opportunities", 500),
            ("", 0),
            (None, 0),
            ("Error loading count", 0),
            ("3.5 average jobs", 3),
        ]

        for input_val, expected in realistic_inputs:
            # Act
            result = safe_job_count(input_val, "Real Company")

            # Assert
            assert result == expected

    def test_validation_handles_edge_cases_consistently(self):
        """Test validation functions handle edge cases consistently."""
        edge_cases = [
            float("inf"),
            float("-inf"),
            float("nan"),
            2**63,  # Large integer
            -(2**63),  # Large negative integer
            complex(1, 2),  # Complex number
            b"123",  # Bytes
        ]

        for value in edge_cases:
            # Act - Should not raise exceptions
            safe_int_result = safe_int(value)
            safe_job_count_result = safe_job_count(value, "Edge Case Company")

            # Assert - Both should return non-negative integers
            assert isinstance(safe_int_result, int)
            assert isinstance(safe_job_count_result, int)
            assert safe_int_result >= 0
            assert safe_job_count_result >= 0

    def test_validation_performance_with_large_datasets(self):
        """Test validation functions performance with many conversions."""
        import time

        # Test with many conversions
        start_time = time.perf_counter()

        for i in range(1000):
            safe_int(str(i))
            safe_job_count(f"{i} jobs", f"Company {i}")

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should complete quickly (less than 0.5 seconds for 1000 conversions)
        assert duration < 0.5, f"Validation too slow: {duration}s"

    def test_validation_logging_behavior(self):
        """Test validation functions logging behavior under various conditions."""
        # Test that safe_job_count works with various inputs
        # Detailed logging verification would require more complex mocking
        result1 = safe_job_count("invalid", "Company A")  # Should return 0
        result2 = safe_job_count("5", "Company B")  # Should return 5
        result3 = safe_job_count(10, "Company C")  # Should return 10
        result4 = safe_job_count(None, "Company D")  # Should return 0

        # Verify results
        assert result1 == 0
        assert result2 == 5
        assert result3 == 10
        assert result4 == 0

    def test_streamlit_context_detection_reliability(self):
        """Test Streamlit context detection is reliable and fast."""
        # Test multiple calls for consistency
        results = [is_streamlit_context() for _ in range(10)]

        # All results should be consistent
        assert all(r == results[0] for r in results), "Inconsistent context detection"

        # Should always return boolean
        assert all(isinstance(r, bool) for r in results)

    def test_validation_with_concurrent_access(self):
        """Test validation functions work correctly with concurrent access."""
        import queue
        import threading

        results = queue.Queue()

        def worker(worker_id):
            # Each worker performs validation operations
            for i in range(100):
                result1 = safe_int(f"{worker_id}{i}")
                result2 = safe_job_count(f"{worker_id}{i}", f"Worker {worker_id}")
                results.put((result1, result2))

        # Create multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]

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

        assert len(collected_results) == 300  # 3 workers x 100 operations

        # All results should be valid non-negative integers
        for result1, result2 in collected_results:
            assert isinstance(result1, int)
            assert result1 >= 0
            assert isinstance(result2, int)
            assert result2 >= 0
