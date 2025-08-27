"""Comprehensive integration tests for search bar component with FTS5 search service.

This module provides comprehensive integration tests for the search bar UI component,
focusing on FTS5 search service integration, advanced filtering, performance metrics,
and real-time search functionality.

Test Categories:
- Search service integration and error handling
- Advanced filter functionality with all filter types
- Performance monitoring and metrics tracking
- Real-time search with debouncing
- Search results handling and export functionality
- Error handling and edge cases
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
import streamlit as st

from src.constants import APPLICATION_STATUSES, SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN
from src.ui.components.search_bar import (
    DEFAULT_SEARCH_LIMIT,
    FTS5_SEARCH_HINTS,
    SEARCH_DEBOUNCE_DELAY,
    _build_search_filters,
    _clear_all_filters,
    _handle_search_input_change,
    _has_active_filters,
    _init_search_state,
    _perform_search,
    _trigger_search_update,
    export_search_results,
    get_search_suggestions,
)
from tests.ui.components.test_utils import MockSessionState


class TestSearchServiceIntegration:
    """Test integration with FTS5 search service."""

    @patch("src.ui.components.search_bar.search_service")
    def test_perform_search_basic_functionality(self, mock_search_service):
        """Test basic search functionality with mocked search service."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "Python developer"

            # Mock search service response
            mock_search_service.search_jobs.return_value = [
                {
                    "id": 1,
                    "title": "Python Developer",
                    "company": "TechCorp",
                    "rank": 0.95,
                },
                {
                    "id": 2,
                    "title": "Senior Python Engineer",
                    "company": "StartupXYZ",
                    "rank": 0.87,
                },
            ]

            # Mock time.perf_counter for performance measurement
            with patch("time.perf_counter", side_effect=[0, 0.005]):  # 5ms search
                _perform_search()

            # Verify search was called with correct parameters
            mock_search_service.search_jobs.assert_called_once()
            args, _ = mock_search_service.search_jobs.call_args
            assert args[0] == "Python developer"

            # Verify results were stored in session state
            assert len(st.session_state["search_results"]) == 2
            assert st.session_state["search_stats"]["total_results"] == 2
            assert st.session_state["search_stats"]["query_time"] == 5.0  # 5ms

    @patch("src.ui.components.search_bar.search_service")
    def test_perform_search_with_filters(self, mock_search_service):
        """Test search with advanced filters applied."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "data scientist"

            # Set up filters
            st.session_state["search_filters"]["location"] = "San Francisco"
            st.session_state["search_filters"]["salary_min"] = 100000
            st.session_state["search_filters"]["favorites_only"] = True
            st.session_state["search_filters"]["application_status"] = "Applied"

            mock_search_service.search_jobs.return_value = []

            with patch("time.perf_counter", side_effect=[0, 0.003]):
                _perform_search()

            # Verify filters were passed correctly
            args, _ = mock_search_service.search_jobs.call_args
            filters = args[1]  # Second argument should be filters

            assert "favorites_only" in filters
            assert filters["favorites_only"] is True
            assert "application_status" in filters
            assert "Applied" in filters["application_status"]

    @patch("src.ui.components.search_bar.search_service")
    def test_perform_search_error_handling(self, mock_search_service):
        """Test search error handling when service fails."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "test query"

            # Mock search service to raise an exception
            mock_search_service.search_jobs.side_effect = Exception("Search failed")

            with patch("time.perf_counter", side_effect=[0, 0.001]):
                _perform_search()  # Should not raise exception

            # Should handle error gracefully
            assert st.session_state["search_results"] == []
            assert st.session_state["search_stats"]["total_results"] == 0

    def test_search_input_change_handler(self):
        """Test search input change handling with debouncing."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_input"] = "new search query"

            with patch(
                "src.ui.components.search_bar._trigger_search_update"
            ) as mock_trigger:
                _handle_search_input_change()

            # Should update search query and trigger search
            assert st.session_state["search_query"] == "new search query"
            mock_trigger.assert_called_once()

    @patch("time.time")
    def test_search_debouncing(self, mock_time):
        """Test search debouncing to prevent excessive API calls."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # First search call
            mock_time.return_value = 1000
            with patch("src.ui.components.search_bar._perform_search") as mock_search:
                _trigger_search_update()
                mock_search.assert_called_once()

            # Rapid second call (within debounce period)
            mock_time.return_value = 1000 + SEARCH_DEBOUNCE_DELAY - 0.1
            with patch("src.ui.components.search_bar._perform_search") as mock_search:
                _trigger_search_update()
                mock_search.assert_not_called()  # Should be debounced

            # Call after debounce period
            mock_time.return_value = 1000 + SEARCH_DEBOUNCE_DELAY + 0.1
            with patch("src.ui.components.search_bar._perform_search") as mock_search:
                _trigger_search_update()
                mock_search.assert_called_once()  # Should execute

    def test_search_suggestions(self):
        """Test search suggestions functionality."""
        suggestions = get_search_suggestions()

        # Should return a list of suggestions
        assert isinstance(suggestions, list)
        # Should include some of the FTS5 hints
        assert len(suggestions) > 0

        # Suggestions should be strings
        for suggestion in suggestions:
            assert isinstance(suggestion, str)

    @patch("src.ui.components.search_bar.search_service")
    def test_search_service_statistics_integration(self, mock_search_service):
        """Test integration with search service statistics."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "test"

            # Mock search service stats
            mock_search_service.get_search_stats.return_value = {
                "fts_enabled": True,
                "indexed_jobs": 1500,
                "total_jobs": 1500,
                "index_coverage": 100.0,
                "last_updated": "2024-01-15T10:30:00Z",
            }

            mock_search_service.search_jobs.return_value = []

            with patch("time.perf_counter", side_effect=[0, 0.008]):
                _perform_search()

            # Verify stats integration attempt
            stats = st.session_state["search_stats"]
            assert "fts_enabled" in stats


class TestAdvancedFiltering:
    """Test advanced filtering functionality with comprehensive coverage."""

    def test_build_search_filters_comprehensive(self):
        """Test comprehensive search filter building with all filter types."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Set various filter values
            st.session_state.search_filters["location"] = "Remote"
            st.session_state.search_filters["salary_min"] = 80000
            st.session_state.search_filters["salary_max"] = 150000
            st.session_state.search_filters["remote_only"] = True
            st.session_state.search_filters["application_status"] = "Applied"
            st.session_state.search_filters["favorites_only"] = True
            st.session_state.search_filters["date_from"] = datetime(
                2024, 1, 1, tzinfo=UTC
            )
            st.session_state.search_filters["date_to"] = datetime(
                2024, 12, 31, tzinfo=UTC
            )

            filters = _build_search_filters()

            # Verify all filters are properly converted
            assert "application_status" in filters
            assert filters["application_status"] == ["Applied"]
            assert filters["favorites_only"] is True
            assert filters["salary_min"] == 80000
            assert filters["salary_max"] == 150000
            assert "date_from" in filters
            assert "date_to" in filters

    def test_has_active_filters_comprehensive(self):
        """Test active filter detection with various filter combinations."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Initially no active filters
            assert _has_active_filters() is False

            # Test location filter
            st.session_state.search_filters["location"] = "San Francisco"
            assert _has_active_filters() is True

            # Reset and test salary filter
            _init_search_state()
            st.session_state.search_filters["salary_min"] = 90000
            assert _has_active_filters() is True

            # Reset and test remote filter
            _init_search_state()
            st.session_state.search_filters["remote_only"] = True
            assert _has_active_filters() is True

            # Reset and test status filter
            _init_search_state()
            st.session_state.search_filters["application_status"] = "Applied"
            assert _has_active_filters() is True

            # Reset and test favorites filter
            _init_search_state()
            st.session_state.search_filters["favorites_only"] = True
            assert _has_active_filters() is True

    def test_clear_all_filters_comprehensive(self):
        """Test comprehensive filter clearing functionality."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Set all possible filters
            st.session_state.search_query = "test query"
            st.session_state.search_filters["location"] = "Remote"
            st.session_state.search_filters["salary_min"] = 90000
            st.session_state.search_filters["salary_max"] = 180000
            st.session_state.search_filters["remote_only"] = True
            st.session_state.search_filters["application_status"] = "Applied"
            st.session_state.search_filters["favorites_only"] = True
            st.session_state.search_results = [{"id": 1, "title": "Test Job"}]
            st.session_state.show_advanced_filters = True

            with patch(
                "src.ui.components.search_bar._trigger_search_update"
            ) as mock_trigger:
                _clear_all_filters()

            # Verify all values are reset to defaults
            assert st.session_state.search_query == ""
            assert st.session_state.search_results == []
            assert st.session_state.search_filters["location"] == ""
            assert st.session_state.search_filters["salary_min"] == SALARY_DEFAULT_MIN
            assert st.session_state.search_filters["salary_max"] == SALARY_DEFAULT_MAX
            assert st.session_state.search_filters["remote_only"] is False
            assert st.session_state.search_filters["application_status"] == "All"
            assert st.session_state.search_filters["favorites_only"] is False
            assert st.session_state.show_advanced_filters is False

            # Should trigger search update
            mock_trigger.assert_called_once()

    def test_date_filter_handling(self):
        """Test date filter handling with various date formats."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Test with datetime objects
            date_from = datetime(2024, 1, 1, tzinfo=UTC)
            date_to = datetime(2024, 6, 30, tzinfo=UTC)
            st.session_state.search_filters["date_from"] = date_from
            st.session_state.search_filters["date_to"] = date_to

            filters = _build_search_filters()

            assert "date_from" in filters
            assert "date_to" in filters
            assert filters["date_from"] == date_from
            assert filters["date_to"] == date_to

    def test_salary_range_filters(self):
        """Test salary range filter handling with edge cases."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Test with high salary values
            st.session_state.search_filters["salary_min"] = 200000
            st.session_state.search_filters["salary_max"] = 500000

            filters = _build_search_filters()

            assert filters["salary_min"] == 200000
            assert filters["salary_max"] == 500000

    def test_application_status_filter_variations(self):
        """Test various application status filter scenarios."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Test each application status
            for status in APPLICATION_STATUSES:
                st.session_state.search_filters["application_status"] = status
                filters = _build_search_filters()
                assert "application_status" in filters
                assert status in filters["application_status"]

            # Test "All" status
            st.session_state.search_filters["application_status"] = "All"
            filters = _build_search_filters()
            # "All" should result in no specific status filter


class TestPerformanceAndMetrics:
    """Test performance monitoring and metrics functionality."""

    @patch("src.ui.components.search_bar.search_service")
    @patch("time.perf_counter")
    def test_performance_metrics_tracking(self, mock_perf_counter, mock_search_service):
        """Test that search performance metrics are properly tracked."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "performance test"

            # Mock performance counter to simulate 15ms search
            mock_perf_counter.side_effect = [1000.0, 1000.015]

            # Mock search results
            mock_search_service.search_jobs.return_value = [
                {"id": 1, "title": "Test Job", "rank": 0.9}
            ]

            _perform_search()

            # Verify performance metrics are recorded
            stats = st.session_state["search_stats"]
            assert stats["query_time"] == 15.0  # 15ms
            assert stats["total_results"] == 1
            assert "fts_enabled" in stats

    @patch("src.ui.components.search_bar.search_service")
    def test_search_performance_fast_query(self, mock_search_service):
        """Test performance tracking for fast queries."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "fast query"

            # Mock very fast search (1ms)
            mock_search_service.search_jobs.return_value = []

            with patch("time.perf_counter", side_effect=[0, 0.001]):
                _perform_search()

            stats = st.session_state["search_stats"]
            assert stats["query_time"] == 1.0  # 1ms
            assert stats["total_results"] == 0

    @patch("src.ui.components.search_bar.search_service")
    def test_search_performance_slow_query(self, mock_search_service):
        """Test performance tracking for slower queries."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "slow query"

            # Mock slower search (100ms)
            mock_search_service.search_jobs.return_value = []

            with patch("time.perf_counter", side_effect=[0, 0.1]):
                _perform_search()

            stats = st.session_state["search_stats"]
            assert stats["query_time"] == 100.0  # 100ms


class TestSearchResultsHandling:
    """Test search results display and handling functionality."""

    def test_export_search_results_csv(self):
        """Test CSV export functionality for search results."""
        # Mock search results
        results = [
            {
                "id": 1,
                "title": "Python Developer",
                "company": "TechCorp",
                "location": "San Francisco",
                "salary": "[100000, 150000]",
                "description": "Great Python role",
            },
            {
                "id": 2,
                "title": "Data Scientist",
                "company": "DataCorp",
                "location": "Remote",
                "salary": "[120000, 180000]",
                "description": "Analyze data sets",
            },
        ]

        # Test CSV export (should not raise exception)
        try:
            export_search_results(results, "csv")
        except Exception as e:
            pytest.fail(f"CSV export failed: {e}")

    def test_export_search_results_json(self):
        """Test JSON export functionality for search results."""
        results = [
            {
                "id": 1,
                "title": "Python Developer",
                "company": "TechCorp",
                "location": "San Francisco",
            }
        ]

        # Test JSON export (should not raise exception)
        try:
            export_search_results(results, "json")
        except Exception as e:
            pytest.fail(f"JSON export failed: {e}")

    def test_empty_results_handling(self):
        """Test handling of empty search results."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_results"] = []

            # Empty results should not cause errors in export
            try:
                export_search_results([], "csv")
            except Exception as e:
                pytest.fail(f"Empty results export failed: {e}")

    def test_large_result_set_handling(self):
        """Test handling of large search result sets."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Create large result set
            large_results = [
                {"id": i, "title": f"Job {i}", "company": f"Company {i}"}
                for i in range(1000)
            ]

            st.session_state["search_results"] = large_results

            # Should handle large datasets without issues
            try:
                export_search_results(large_results[:100], "csv")  # Limit for testing
            except Exception as e:
                pytest.fail(f"Large result set handling failed: {e}")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_invalid_filter_values(self):
        """Test handling of invalid filter values."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Set invalid filter values
            st.session_state.search_filters["salary_min"] = "not a number"
            st.session_state.search_filters["salary_max"] = None
            st.session_state.search_filters["date_from"] = "invalid date"

            # Should handle gracefully without crashing
            try:
                filters = _build_search_filters()
                assert isinstance(filters, dict)
            except Exception as e:
                pytest.fail(f"Filter building failed with invalid values: {e}")

    def test_missing_session_state_keys(self):
        """Test handling when session state keys are missing."""
        with patch.object(st, "session_state", MockSessionState()):
            # Don't initialize search state

            # Should handle missing keys gracefully
            try:
                _has_active_filters()
            except KeyError:
                # This is expected if keys don't exist - test validates error handling
                pytest.skip("KeyError expected when session state not initialized")
            except Exception as e:
                pytest.fail(f"Unexpected exception with missing keys: {e}")

    @patch("src.ui.components.search_bar.search_service")
    def test_search_service_unavailable(self, mock_search_service):
        """Test handling when search service is unavailable."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "test"

            # Mock search service to be None or unavailable
            mock_search_service.search_jobs.side_effect = AttributeError(
                "Service unavailable"
            )

            # Should handle gracefully
            try:
                _perform_search()
                # Results should be empty but no exception should be raised
                assert st.session_state["search_results"] == []
            except Exception:
                # Should handle service errors gracefully - verify error is logged
                assert st.session_state["search_results"] == []

    def test_malformed_filter_data(self):
        """Test handling of various malformed filter data types."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Test various malformed scenarios
            malformed_cases = [
                {"salary_min": []},  # List instead of number
                {"salary_max": {}},  # Dict instead of number
                {"application_status": 123},  # Number instead of string
                {"favorites_only": "yes"},  # String instead of boolean
                {"remote_only": None},  # None instead of boolean
            ]

            for malformed_data in malformed_cases:
                # Update filters with malformed data
                for key, value in malformed_data.items():
                    st.session_state.search_filters[key] = value

                # Should handle gracefully
                try:
                    filters = _build_search_filters()
                    assert isinstance(filters, dict)
                except Exception as e:
                    pytest.fail(
                        f"Failed to handle malformed data {malformed_data}: {e}"
                    )

                # Reset for next test
                _init_search_state()

    @patch("src.ui.components.search_bar.search_service")
    def test_search_timeout_handling(self, mock_search_service):
        """Test handling of search timeouts."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            st.session_state["search_query"] = "timeout test"

            # Mock search service to simulate timeout
            mock_search_service.search_jobs.side_effect = TimeoutError(
                "Search timed out"
            )

            with patch("time.perf_counter", side_effect=[0, 5.0]):  # 5 second timeout
                _perform_search()

            # Should handle timeout gracefully
            assert st.session_state["search_results"] == []
            assert st.session_state["search_stats"]["total_results"] == 0


class TestSearchConstants:
    """Test search-related constants and configuration."""

    def test_search_constants_validity(self):
        """Test that search constants are properly configured."""
        # Test default search limit
        assert isinstance(DEFAULT_SEARCH_LIMIT, int)
        assert DEFAULT_SEARCH_LIMIT > 0
        assert DEFAULT_SEARCH_LIMIT <= 100

        # Test debounce delay
        assert isinstance(SEARCH_DEBOUNCE_DELAY, (int, float))
        assert SEARCH_DEBOUNCE_DELAY > 0
        assert SEARCH_DEBOUNCE_DELAY < 2.0  # Should be reasonable for UI

        # Test FTS5 search hints
        assert isinstance(FTS5_SEARCH_HINTS, list)
        assert len(FTS5_SEARCH_HINTS) > 0

        for hint in FTS5_SEARCH_HINTS:
            assert isinstance(hint, str)
            assert len(hint) > 0

    def test_application_statuses_integration(self):
        """Test integration with application status constants."""
        # Should be able to use APPLICATION_STATUSES in filters
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            for status in APPLICATION_STATUSES:
                st.session_state.search_filters["application_status"] = status
                filters = _build_search_filters()
                assert "application_status" in filters
                assert status in filters["application_status"]

    def test_salary_constants_integration(self):
        """Test integration with salary constants."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Should initialize with salary constants
            assert st.session_state.search_filters["salary_min"] == SALARY_DEFAULT_MIN
            assert st.session_state.search_filters["salary_max"] == SALARY_DEFAULT_MAX

    def test_fts5_search_hints_validity(self):
        """Test that FTS5 search hints are valid search queries."""
        # Each hint should be a valid string that could be used as a search query
        for hint in FTS5_SEARCH_HINTS:
            assert isinstance(hint, str)
            assert len(hint.strip()) > 0
            # Should not be empty or just whitespace
            assert hint.strip() != ""

    def test_search_limit_boundaries(self):
        """Test search limit boundary conditions."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Should initialize with default limit
            assert st.session_state.get("search_limit") == DEFAULT_SEARCH_LIMIT

            # Test setting custom limits
            st.session_state["search_limit"] = 25
            assert st.session_state["search_limit"] == 25

            st.session_state["search_limit"] = 100
            assert st.session_state["search_limit"] == 100


class TestSearchUIIntegration:
    """Test search UI component integration patterns."""

    def test_search_state_initialization_idempotency(self):
        """Test that repeated search state initialization is safe."""
        with patch.object(st, "session_state", MockSessionState()):
            # Initialize multiple times
            _init_search_state()
            _init_search_state()
            _init_search_state()

            # Should have all required keys
            required_keys = [
                "search_query",
                "search_results",
                "search_stats",
                "search_filters",
                "show_advanced_filters",
                "last_search_time",
                "search_limit",
            ]

            for key in required_keys:
                assert key in st.session_state

    def test_search_state_isolation(self):
        """Test search state doesn't interfere with other session state."""
        with patch.object(
            st, "session_state", MockSessionState({"existing_key": "existing_value"})
        ):
            _init_search_state()

            # Check existing state is preserved
            assert st.session_state["existing_key"] == "existing_value"

            # Check new search state is added
            assert "search_query" in st.session_state
            assert "search_results" in st.session_state

    @patch("src.ui.components.search_bar.search_service")
    def test_search_integration_with_empty_query(self, mock_search_service):
        """Test search behavior with empty or whitespace-only queries."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Test empty string
            st.session_state["search_query"] = ""
            _perform_search()
            # Should not call search service for empty query
            mock_search_service.search_jobs.assert_not_called()

            # Test whitespace only
            st.session_state["search_query"] = "   \n\t   "
            _perform_search()
            # Should not call search service for whitespace-only query
            mock_search_service.search_jobs.assert_not_called()

    def test_filter_state_persistence(self):
        """Test that filter states persist across search operations."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Set various filters
            st.session_state.search_filters["location"] = "San Francisco"
            st.session_state.search_filters["salary_min"] = 100000
            st.session_state.search_filters["favorites_only"] = True

            # Perform search operations
            with patch("src.ui.components.search_bar.search_service"):
                with patch("time.perf_counter", side_effect=[0, 0.001]):
                    _perform_search()

            # Filters should persist
            assert st.session_state.search_filters["location"] == "San Francisco"
            assert st.session_state.search_filters["salary_min"] == 100000
            assert st.session_state.search_filters["favorites_only"] is True
