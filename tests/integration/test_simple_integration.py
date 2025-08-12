"""Simple integration test to validate the approach.

This module tests the background scraping workflow in a test environment where:
- No actual threading occurs (runs synchronously)
- External dependencies are mocked
- Session state behaves predictably
"""

from unittest.mock import patch

from src.ui.utils.background_helpers import (
    is_scraping_active,
    start_background_scraping,
)

# UI test fixtures are auto-loaded from conftest.py files


class TestSimpleIntegration:
    """Simple integration tests to validate the approach."""

    def test_basic_scraping_workflow(
        self,
        mock_session_state,
        prevent_real_system_execution,
    ):
        """Test basic scraping workflow initialization in test environment.

        This test validates that:
        1. Background scraping can be started and returns a task ID
        2. In test environments, scraping runs synchronously and completes immediately
        3. Session state is properly managed throughout the process
        4. Progress tracking is initialized correctly
        """
        # Arrange
        companies = ["TechCorp"]
        scraping_results = {"TechCorp": 10}
        prevent_real_system_execution["scrape_all"].return_value = scraping_results
        prevent_real_system_execution["scrape_all_bg"].return_value = scraping_results

        # Verify initial state
        assert is_scraping_active() is False
        assert mock_session_state.get("task_progress", {}) == {}

        # Act: Start scraping (runs synchronously in test environment)
        with patch(
            "src.ui.utils.background_tasks.JobService.get_active_companies",
            return_value=companies,
        ):
            task_id = start_background_scraping()

        # Assert: Validate task creation and completion
        assert isinstance(task_id, str)
        assert len(task_id) > 0

        # In test environment, scraping runs synchronously and completes immediately
        # So scraping_active should be False after the call returns
        assert is_scraping_active() is False

        # Verify task progress was initialized and remains accessible
        from src.ui.utils.background_helpers import get_scraping_progress

        progress = get_scraping_progress()
        assert task_id in progress
        assert progress[task_id].message == "Starting scraping..."
        assert progress[task_id].progress == 0.0

        # Verify scraping was called with mocked dependencies
        prevent_real_system_execution["scrape_all_bg"].assert_called_once()

        # Verify session state contains expected keys after completion
        assert "scraping_active" in mock_session_state
        assert mock_session_state.get("scraping_active") is False

    def test_mock_service_works(
        self,
        mock_session_state,
    ):
        """Test that mock service works correctly."""
        companies = ["TestCorp"]

        with patch(
            "src.services.job_service.JobService.get_active_companies",
            return_value=companies,
        ) as mock_get_companies:
            from src.services.job_service import JobService

            result = JobService.get_active_companies()
            assert result == companies
            mock_get_companies.assert_called_once()

    def test_session_state_utility_functions(
        self,
        mock_session_state,
    ):
        """Test session state utility functions work correctly.

        Validates that all utility functions properly read from and interact
        with the mocked session state in expected ways.
        """
        from src.ui.utils.background_helpers import (
            get_company_progress,
            get_scraping_progress,
            get_scraping_results,
        )

        # Test initial state - all functions should return empty containers
        assert is_scraping_active() is False
        assert get_scraping_results() == {}
        assert get_scraping_progress() == {}
        assert get_company_progress() == {}

        # Test after setting some values directly in session state
        mock_session_state["scraping_active"] = True
        mock_session_state["scraping_results"] = {"test": 5}
        mock_session_state["task_progress"] = {"task1": "progress"}
        mock_session_state["company_progress"] = {"TechCorp": "status"}

        # Verify functions return expected values from session state
        assert is_scraping_active() is True
        assert get_scraping_results() == {"test": 5}
        assert get_scraping_progress() == {"task1": "progress"}
        assert get_company_progress() == {"TechCorp": "status"}

    def test_synchronous_execution_in_test_environment(
        self,
        mock_session_state,
        prevent_real_system_execution,
    ):
        """Test that scraping executes synchronously in test environments.

        This test specifically validates the test environment detection and
        synchronous execution path, ensuring no threading occurs during testing.
        """
        from src.ui.utils.background_helpers import _is_test_environment

        # Verify we're detected as being in a test environment
        assert _is_test_environment() is True

        # Mock companies to scrape
        companies = ["Company1", "Company2"]
        expected_results = {"Company1": 5, "Company2": 3}
        prevent_real_system_execution["scrape_all_bg"].return_value = expected_results

        with patch(
            "src.ui.utils.background_tasks.JobService.get_active_companies",
            return_value=companies,
        ):
            # Start scraping - should complete synchronously
            task_id = start_background_scraping()

            # Immediately after the call, scraping should be complete
            # (no waiting required as in async/threaded execution)
            assert is_scraping_active() is False
            assert isinstance(task_id, str)

            # Verify the synchronous execution path was used
            # (scrape_all_bg called during start_background_scraping)
            prevent_real_system_execution["scrape_all_bg"].assert_called_once()
