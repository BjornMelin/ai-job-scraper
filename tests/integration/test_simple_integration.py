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
        """Test basic scraping workflow without UI rendering."""
        # Arrange
        companies = ["TechCorp"]
        scraping_results = {"TechCorp": 10}
        prevent_real_system_execution["scrape_all"].return_value = scraping_results

        # Act: Start scraping
        with patch(
            "src.ui.utils.background_tasks.JobService.get_active_companies",
            return_value=companies,
        ):
            task_id = start_background_scraping()

        # Assert: Basic state is correct
        assert is_scraping_active() is True
        assert isinstance(task_id, str)
        assert len(task_id) > 0

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
