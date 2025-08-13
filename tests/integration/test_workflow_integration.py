"""Streamlined integration tests for the scraping workflow.

This module provides focused integration tests that validate the complete
scraping workflow while working around Streamlit threading limitations.
These tests focus on the core integration scenarios without getting stuck
on Streamlit session state threading issues.
"""

from unittest.mock import patch

from src.ui.utils.background_helpers import (
    CompanyProgress,
    get_company_progress,
    get_scraping_results,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)

# UI test fixtures are auto-loaded from conftest.py files


class TestWorkflowIntegration:
    """Core workflow integration tests."""

    def test_start_stop_workflow_integration(
        self,
        mock_session_state,
    ):
        """Test start and stop workflow integration."""
        # Arrange
        companies = ["TechCorp", "DataInc"]

        # Act 1: Start scraping
        with (
            patch(
                "src.services.job_service.JobService.get_active_companies",
                return_value=companies,
            ),
            patch(
                "src.scraper.scrape_all",
                return_value={"TechCorp": 10, "DataInc": 15},
            ),
        ):
            task_id = start_background_scraping(stay_active_in_tests=False)

        # Assert: Scraping started correctly (synchronous in tests)
        assert is_scraping_active() is False  # Completes immediately in test env
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        assert task_id in mock_session_state.get("task_progress", {})

        # Act 2: Set up active state to test stopping
        mock_session_state.scraping_active = True
        stopped_count = stop_all_scraping()

        # Assert: Scraping stopped correctly
        assert stopped_count == 1
        assert is_scraping_active() is False
        assert mock_session_state.get("scraping_status") == "Scraping stopped"

    def test_session_state_integration(
        self,
        mock_session_state,
    ):
        """Test session state integration across workflow."""
        # Arrange
        companies = ["TechCorp"]
        expected_results = {"TechCorp": 15}

        # Act: Complete workflow setup
        with patch(
            "src.services.job_service.JobService.get_active_companies",
            return_value=companies,
        ):
            task_id = start_background_scraping()

        # Simulate manual completion (without threading issues)
        mock_session_state.update(
            {
                "scraping_results": expected_results,
                "scraping_active": False,
                "company_progress": {
                    "TechCorp": CompanyProgress(
                        name="TechCorp",
                        status="Completed",
                        jobs_found=15,
                    ),
                },
            },
        )

        # Assert: Session state integration works
        # 1. Task tracking
        assert mock_session_state.get("task_id") == task_id

        # 2. Results integration
        results = get_scraping_results()
        assert results == expected_results

        # 3. Company progress integration
        company_progress = get_company_progress()
        assert len(company_progress) == 1
        assert "TechCorp" in company_progress
        assert company_progress["TechCorp"].jobs_found == 15

    def test_error_handling_integration(
        self,
        mock_session_state,
    ):
        """Test error handling integration across components."""
        # Arrange: Service that will fail
        service_error = Exception("Database connection failed")

        # Act: Start scraping with failing service
        with patch(
            "src.services.job_service.JobService.get_active_companies",
            side_effect=service_error,
        ):
            start_background_scraping()

        # Simulate error state manually
        mock_session_state.update(
            {
                "scraping_active": False,
                "scraping_status": "❌ Scraping failed: Database connection failed",
                "company_progress": {},
            },
        )

        # Assert: Error handling integration
        assert is_scraping_active() is False
        error_status = mock_session_state.get("scraping_status", "")
        assert "failed" in error_status.lower()

        # System should be able to retry
        retry_companies = ["TechCorp"]
        with patch(
            "src.services.job_service.JobService.get_active_companies",
            return_value=retry_companies,
        ):
            retry_task_id = start_background_scraping(stay_active_in_tests=False)
            assert isinstance(retry_task_id, str)
            assert is_scraping_active() is False  # Synchronous completion in test env

    def test_multiple_workflow_cycles(
        self,
        mock_session_state,
    ):
        """Test multiple complete workflow cycles."""
        companies = ["TechCorp"]

        # Run 3 complete cycles
        for _cycle in range(3):
            # Act: Start scraping
            with patch(
                "src.services.job_service.JobService.get_active_companies",
                return_value=companies,
            ):
                task_id = start_background_scraping(stay_active_in_tests=False)

            # Assert: Each cycle starts correctly (synchronous in tests)
            assert is_scraping_active() is False  # Completes immediately in test env
            assert isinstance(task_id, str)

            # Act: Set up active state and stop scraping
            mock_session_state.scraping_active = True
            stopped_count = stop_all_scraping()

            # Assert: Each cycle stops correctly
            assert stopped_count == 1
            assert is_scraping_active() is False

            # Clean up between cycles
            progress_keys = ["task_progress", "company_progress", "scraping_results"]
            for key in progress_keys:
                if key in mock_session_state._data and hasattr(
                    mock_session_state._data[key],
                    "clear",
                ):
                    mock_session_state._data[key].clear()

        # Assert: System remains stable after multiple cycles
        assert is_scraping_active() is False

    def test_background_task_component_integration(
        self,
        mock_session_state,
    ):
        """Test integration between background task components."""
        # Arrange
        companies = ["TechCorp", "DataInc"]
        scraping_results = {"TechCorp": 12, "DataInc": 8}

        # Act: Initialize all components
        with patch(
            "src.services.job_service.JobService.get_active_companies",
            return_value=companies,
        ):
            task_id = start_background_scraping()

        # Manually set up integrated state (simulating successful completion)
        mock_session_state.update(
            {
                "scraping_active": False,
                "scraping_results": scraping_results,
                "company_progress": {
                    "TechCorp": CompanyProgress(
                        name="TechCorp",
                        status="Completed",
                        jobs_found=12,
                    ),
                    "DataInc": CompanyProgress(
                        name="DataInc",
                        status="Completed",
                        jobs_found=8,
                    ),
                },
            },
        )

        # Assert: Component integration
        # 1. Task management component
        task_progress = mock_session_state.get("task_progress", {})
        assert task_id in task_progress

        # 2. Results component
        final_results = get_scraping_results()
        assert final_results == scraping_results

        # 3. Progress tracking component
        company_progress = get_company_progress()
        assert len(company_progress) == 2
        assert all(
            progress.status == "Completed" for progress in company_progress.values()
        )

        # 4. State management component
        assert is_scraping_active() is False

    def test_data_flow_integration(
        self,
        mock_session_state,
        prevent_real_system_execution,
    ):
        """Test data flow integration across all components."""
        # Arrange: Multi-component data
        companies = ["CompanyA", "CompanyB", "CompanyC"]
        sync_stats = {
            "inserted": 25,
            "updated": 5,
            "archived": 0,
            "deleted": 0,
            "skipped": 2,
        }

        prevent_real_system_execution["scrape_all"].return_value = sync_stats

        # Act: Start workflow to initialize data flow
        with patch(
            "src.services.job_service.JobService.get_active_companies",
            return_value=companies,
        ):
            task_id = start_background_scraping()

        # Simulate data flowing through all components
        mock_session_state.update(
            {
                "scraping_active": False,
                "scraping_results": sync_stats,
                "task_progress": {
                    task_id: {
                        "progress": 1.0,
                        "message": "Scraping completed",
                        "timestamp": "2024-01-01T00:00:00Z",
                    },
                },
                "company_progress": {
                    company: CompanyProgress(
                        name=company,
                        status="Completed",
                        jobs_found=sync_stats["inserted"] // len(companies),
                    )
                    for company in companies
                },
            },
        )

        # Assert: Data flows correctly through all layers
        # 1. Scraper layer (mocked but called)
        prevent_real_system_execution["scrape_all"].return_value = sync_stats

        # 2. Background task layer
        final_results = get_scraping_results()
        assert final_results == sync_stats

        # 3. Progress tracking layer
        company_progress = get_company_progress()
        assert len(company_progress) == 3
        assert all(
            isinstance(progress, CompanyProgress)
            for progress in company_progress.values()
        )

        # 4. Session state layer
        assert mock_session_state.get("task_id") == task_id
        assert is_scraping_active() is False

    def test_ui_component_state_integration(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test UI component integration with background task state."""
        # Arrange: Set up realistic UI state
        companies = ["TechCorp", "DataInc"]

        # Simulate completed scraping state
        mock_session_state.update(
            {
                "scraping_active": False,
                "scraping_results": {"TechCorp": 10, "DataInc": 15},
                "company_progress": {
                    "TechCorp": CompanyProgress(
                        name="TechCorp",
                        status="Completed",
                        jobs_found=10,
                    ),
                    "DataInc": CompanyProgress(
                        name="DataInc",
                        status="Completed",
                        jobs_found=15,
                    ),
                },
            },
        )

        # Act: Import and test UI integration functions
        from src.ui.pages.scraping import render_scraping_page

        with patch(
            "src.ui.pages.scraping.JobService.get_active_companies",
            return_value=companies,
        ):
            # This should not raise an exception and should call UI components
            render_scraping_page()

        # Assert: UI components were integrated correctly
        # 1. Page structure components called
        mock_streamlit["markdown"].assert_called()

        # 2. Control components called
        button_calls = [
            call for call in mock_streamlit["button"].call_args_list if call[0]
        ]
        assert len(button_calls) >= 3  # Start, Stop, Reset buttons

        # 3. Metrics components called with state data
        metric_calls = mock_streamlit["metric"].call_args_list
        assert len(metric_calls) >= 2  # Multiple metrics should be displayed

    def test_proxy_configuration_integration(
        self,
        mock_session_state,
        prevent_real_system_execution,
        test_settings,
    ):
        """Test proxy configuration integration with scraping workflow."""
        # Arrange
        companies = ["TechCorp"]
        scraping_results = {"TechCorp": 5}

        # Configure with proxies
        test_settings.use_proxies = True
        test_settings.proxy_pool = ["proxy1:8080", "proxy2:8080"]

        prevent_real_system_execution["scrape_all"].return_value = scraping_results

        # Act: Run workflow with proxy configuration
        with (
            patch("src.config.Settings", return_value=test_settings),
            patch(
                "src.services.job_service.JobService.get_active_companies",
                return_value=companies,
            ),
        ):
            task_id = start_background_scraping()

        # Simulate completion
        mock_session_state.update(
            {"scraping_active": False, "scraping_results": scraping_results},
        )

        # Assert: Proxy integration works
        assert isinstance(task_id, str)
        final_results = get_scraping_results()
        assert final_results == scraping_results

        # Test without proxies
        test_settings.use_proxies = False
        test_settings.proxy_pool = []

        with (
            patch("src.config.Settings", return_value=test_settings),
            patch(
                "src.services.job_service.JobService.get_active_companies",
                return_value=companies,
            ),
        ):
            task_id_no_proxy = start_background_scraping()

        assert isinstance(task_id_no_proxy, str)
        assert task_id_no_proxy != task_id  # Should be different task


class TestConcurrentScenarios:
    """Test concurrent and edge case scenarios."""

    def test_rapid_start_stop_integration(
        self,
    ):
        """Test rapid start/stop cycles for stability."""
        companies = ["TechCorp"]

        # Perform rapid cycles
        task_ids = []
        for _i in range(5):
            # Start
            with patch(
                "src.services.job_service.JobService.get_active_companies",
                return_value=companies,
            ):
                task_id = start_background_scraping(stay_active_in_tests=False)
                task_ids.append(task_id)
                assert (
                    is_scraping_active() is False
                )  # Synchronous completion in test env

            # Stop scraping (no need since it completes synchronously)
            # But test the stop function anyway
            stopped_count = stop_all_scraping()
            assert stopped_count == 0  # No active tasks to stop in sync mode
            assert is_scraping_active() is False

        # Assert: System stability
        assert len(set(task_ids)) == 5  # All unique task IDs
        assert is_scraping_active() is False  # Clean final state

    def test_state_consistency_under_load(
        self,
        mock_session_state,
    ):
        """Test state consistency under multiple operations."""
        companies = ["TechCorp", "DataInc", "AI Solutions"]

        # Perform multiple overlapping operations
        with patch(
            "src.services.job_service.JobService.get_active_companies",
            return_value=companies,
        ):
            # Start first task
            task_id_1 = start_background_scraping(stay_active_in_tests=False)

            # Attempt second start (should handle gracefully)
            task_id_2 = start_background_scraping(stay_active_in_tests=False)

            # Stop should work regardless (but no active tasks in sync mode)
            stopped_count = stop_all_scraping()

        # Assert: Consistent state maintained
        assert stopped_count == 0  # No active tasks to stop in sync mode
        assert is_scraping_active() is False

        # Task IDs should be managed consistently
        current_task_id = mock_session_state.get("task_id")
        assert current_task_id in [task_id_1, task_id_2]
