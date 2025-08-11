"""Simple integration test to validate the approach."""

from unittest.mock import patch

from src.ui.utils.background_tasks import (
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
        """Test basic scraping workflow initialization without background threading."""
        # Arrange
        companies = ["TechCorp"]
        scraping_results = {"TechCorp": 10}
        prevent_real_system_execution["scrape_all"].return_value = scraping_results

        # Act: Start scraping (but prevent the background thread from starting)
        with (
            patch(
                "src.ui.utils.background_tasks.JobService.get_active_companies",
                return_value=companies,
            ),
            patch(
                "src.ui.utils.background_tasks.threading.Thread.start"
            ) as mock_thread_start,
        ):
            task_id = start_background_scraping()

            # Verify thread would have been started (but didn't run due to mock)
            mock_thread_start.assert_called_once()

        # Assert: Basic state is correct for task initialization
        assert isinstance(task_id, str)
        assert len(task_id) > 0

        # Check that scraping was marked as active
        assert is_scraping_active() is True

        # Verify task progress was initialized
        from src.ui.utils.background_tasks import get_scraping_progress

        progress = get_scraping_progress()
        assert task_id in progress
        assert progress[task_id].message == "Starting scraping..."

    def test_mock_service_works(
        self,
        mock_session_state,
    ):
        """Test that mock service works correctly."""
        companies = ["TestCorp"]

        with patch(
            "src.ui.utils.background_tasks.JobService.get_active_companies",
            return_value=companies,
        ) as mock_get_companies:
            from src.ui.utils.background_tasks import JobService

            result = JobService.get_active_companies()
            assert result == companies
            mock_get_companies.assert_called_once()

    def test_session_state_utility_functions(
        self,
        mock_session_state,
    ):
        """Test session state utility functions work correctly."""
        from src.ui.utils.background_tasks import (
            get_company_progress,
            get_scraping_progress,
            get_scraping_results,
        )

        # Test initial state
        assert is_scraping_active() is False
        assert get_scraping_results() == {}
        assert get_scraping_progress() == {}
        assert get_company_progress() == {}

        # Test after setting some values
        mock_session_state["scraping_active"] = True
        mock_session_state["scraping_results"] = {"test": 5}
        mock_session_state["task_progress"] = {"task1": "progress"}
        mock_session_state["company_progress"] = {"TechCorp": "status"}

        # Verify functions return expected values
        assert is_scraping_active() is True
        assert get_scraping_results() == {"test": 5}
        assert get_scraping_progress() == {"task1": "progress"}
        assert get_company_progress() == {"TechCorp": "status"}
