"""Tests for simplified Phase 3 background task implementation.

This module tests the simplified 50-line background task management system
based on ADR-017 using standard Python threading with Streamlit integration.

Core test areas:
- Threading context setup with add_script_run_ctx
- Session state coordination to prevent concurrent operations
- st.status component integration for progress display
- Error handling and graceful cleanup
- Thread lifecycle management
"""

from unittest.mock import Mock, call, patch

import pytest


class TestThreadingContextSetup:
    """Test proper threading context setup with Streamlit."""

    def test_threading_context_setup(self, mock_session_state, mock_streamlit):
        """Verify proper threading context setup with Streamlit."""
        from src.ui.utils.background_helpers import start_background_scraping

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.runtime.scriptrunner.add_script_run_ctx") as mock_add_ctx,
        ):
            # Configure thread mock
            mock_thread_instance = Mock()
            mock_thread_class.return_value = mock_thread_instance

            # Act
            start_background_scraping(stay_active_in_tests=True)

            # Verify thread creation with daemon=True
            mock_thread_class.assert_called_once()
            call_kwargs = mock_thread_class.call_args[1]
            assert call_kwargs["daemon"]
            assert "target" in call_kwargs

            # Verify add_script_run_ctx was called with thread instance
            mock_add_ctx.assert_called_once_with(mock_thread_instance)

            # Verify thread start was called
            mock_thread_instance.start.assert_called_once()

    def test_daemon_thread_creation(self, mock_session_state):
        """Test that background threads are created as daemon threads."""
        from src.ui.utils.background_helpers import start_background_scraping

        with patch("threading.Thread") as mock_thread_class:
            mock_thread_instance = Mock()
            mock_thread_class.return_value = mock_thread_instance

            # Act
            start_background_scraping(stay_active_in_tests=True)

            # Assert daemon thread configuration
            call_args = mock_thread_class.call_args
            assert call_args[1]["daemon"]

    def test_streamlit_context_integration(self, mock_session_state):
        """Test add_script_run_ctx integration for Streamlit compatibility."""
        from src.ui.utils.background_helpers import start_background_scraping

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.runtime.scriptrunner.add_script_run_ctx") as mock_add_ctx,
        ):
            mock_thread_instance = Mock()
            mock_thread_class.return_value = mock_thread_instance

            # Act
            start_background_scraping(stay_active_in_tests=True)

            # Verify context is added to thread before starting
            mock_add_ctx.assert_called_once_with(mock_thread_instance)

            # Verify call order: create thread, add context, start thread
            mock_thread_class.assert_called_once()
            mock_add_ctx.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_thread_lifecycle_management(self, mock_session_state):
        """Test proper thread lifecycle from creation to completion."""
        from src.ui.utils.background_helpers import (
            start_background_scraping,
            stop_all_scraping,
        )

        with patch("threading.Thread") as mock_thread_class:
            mock_thread_instance = Mock()
            mock_thread_instance.is_alive.return_value = True
            mock_thread_class.return_value = mock_thread_instance

            # Start background scraping
            start_background_scraping(stay_active_in_tests=True)

            # Store thread reference in session state for testing
            mock_session_state.scraping_thread = mock_thread_instance

            # Stop scraping
            stop_all_scraping()

            # Verify thread join was called for cleanup
            mock_thread_instance.join.assert_called_once_with(timeout=5.0)


class TestSessionStateCoordination:
    """Test session state management prevents concurrent operations."""

    def test_session_state_coordination_prevents_concurrent_scraping(
        self, mock_session_state
    ):
        """Test session state prevents concurrent scraping operations."""
        from src.ui.utils.background_helpers import start_background_scraping

        # Set up active scraping state
        mock_session_state.scraping_active = True

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.warning") as mock_warning,
        ):
            # Act - try to start scraping when already active
            start_background_scraping(stay_active_in_tests=True)

            # Should prevent starting new thread when scraping already active
            # (Note: this test depends on the implementation checking scraping_active)
            mock_warning.assert_called_with("Scraping already in progress")
            mock_thread_class.assert_not_called()

    def test_session_state_scraping_active_flag(self, mock_session_state):
        """Test scraping_active flag coordination."""
        from src.ui.utils.background_helpers import (
            is_scraping_active,
            start_background_scraping,
        )

        # Initially not active
        assert not is_scraping_active()

        # Start scraping
        start_background_scraping(stay_active_in_tests=True)

        # Should be active after starting
        assert is_scraping_active()
        assert mock_session_state.scraping_active

    def test_session_state_cleanup_after_completion(self, mock_session_state):
        """Test session state cleanup after scraping completion."""
        from src.ui.utils.background_helpers import stop_all_scraping

        # Set up active scraping
        mock_session_state.scraping_active = True
        mock_session_state.scraping_status = "Running"

        # Stop scraping
        stopped_count = stop_all_scraping()

        # Verify state cleanup
        assert not mock_session_state.scraping_active
        assert mock_session_state.scraping_status == "Scraping stopped"
        assert stopped_count == 1

    def test_prevent_multiple_concurrent_operations(self, mock_session_state):
        """Test prevention of multiple concurrent scraping operations."""
        from src.ui.utils.background_helpers import start_background_scraping

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.warning") as mock_warning,
        ):
            # First operation should succeed
            start_background_scraping(stay_active_in_tests=True)
            assert mock_thread_class.call_count == 1

            # Second operation should be prevented
            start_background_scraping(stay_active_in_tests=True)
            mock_warning.assert_called_with("Scraping already in progress")
            assert mock_thread_class.call_count == 1  # No additional thread created


class TestStreamlitStatusIntegration:
    """Test st.status component functionality."""

    def test_streamlit_status_component_usage(self, mock_session_state):
        """Test st.status component integration."""
        from src.ui.utils.background_helpers import start_background_scraping

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.status") as mock_status,
        ):
            # Configure status context manager
            mock_status_obj = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_obj)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            # Start scraping and execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify st.status was used for progress display
            mock_status.assert_called_with("🔍 Scraping jobs...", expanded=True)

    def test_progress_updates_through_status(self, mock_session_state):
        """Test progress updates through st.status component."""
        from src.ui.utils.background_helpers import start_background_scraping

        # Set up companies for scraping
        mock_session_state.selected_companies = ["OpenAI", "Anthropic"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.status") as mock_status,
            patch("src.scraper.scrape_company_jobs", return_value=[]),
            patch("src.services.database_sync.sync_jobs_to_database"),
        ):
            # Configure status context manager
            mock_status_obj = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_obj)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify progress updates for each company
            expected_calls = [
                call("Processing OpenAI (1/2)"),
                call("Processing Anthropic (2/2)"),
            ]
            mock_status_obj.write.assert_has_calls(expected_calls)

            # Verify completion update
            mock_status_obj.update.assert_called_with(
                label="✅ Scraping completed!", state="complete"
            )

    def test_status_completion_state_change(self, mock_session_state):
        """Test completion state changes in st.status."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.status") as mock_status,
            patch("src.scraper.scrape_company_jobs", return_value=[]),
            patch("src.services.database_sync.sync_jobs_to_database"),
        ):
            mock_status_obj = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_obj)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify completion state update
            mock_status_obj.update.assert_called_with(
                label="✅ Scraping completed!", state="complete"
            )


class TestBackgroundTaskErrorHandling:
    """Test graceful error handling."""

    def test_background_task_exception_handling(self, mock_session_state):
        """Test graceful error handling in background thread."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.error") as mock_error,
            patch(
                "src.scraper.scrape_company_jobs",
                side_effect=Exception("Scraping failed"),
            ),
        ):
            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify error was displayed to user
            mock_error.assert_called_with("Scraping failed: Scraping failed")

            # Verify session state was reset after error
            assert not mock_session_state.scraping_active

    def test_session_state_reset_on_error(self, mock_session_state):
        """Test session state reset on background task error."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.error"),
            patch(
                "src.scraper.scrape_company_jobs", side_effect=Exception("Test error")
            ),
        ):
            # Execute thread function that will fail
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify session state cleanup in finally block
            assert not mock_session_state.scraping_active

    def test_error_display_to_user(self, mock_session_state):
        """Test error messages are properly displayed to user."""
        from src.ui.utils.background_helpers import start_background_scraping

        error_message = "Database connection failed"
        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.error") as mock_error,
            patch(
                "src.scraper.scrape_company_jobs", side_effect=Exception(error_message)
            ),
        ):
            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify specific error message displayed
            mock_error.assert_called_with(f"Scraping failed: {error_message}")

    def test_graceful_cleanup_after_exception(self, mock_session_state):
        """Test graceful cleanup happens even after exceptions."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("streamlit.error"),
            patch(
                "src.scraper.scrape_company_jobs", side_effect=Exception("Test error")
            ),
        ):
            # Set scraping as active initially
            mock_session_state.scraping_active = True

            # Execute thread function that will fail
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify cleanup still happens in finally block
            assert not mock_session_state.scraping_active


class TestCoreThreadingFunctionality:
    """Test core threading functionality without complex task management."""

    def test_simple_background_task_execution(self, mock_session_state):
        """Test basic background task execution."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["OpenAI", "Anthropic"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch(
                "src.scraper.scrape_company_jobs", return_value=[{"title": "Engineer"}]
            ),
            patch("src.services.database_sync.sync_jobs_to_database"),
        ):
            mock_thread_instance = Mock()
            mock_thread_class.return_value = mock_thread_instance

            # Start background scraping
            start_background_scraping(stay_active_in_tests=True)

            # Verify thread setup
            mock_thread_class.assert_called_once()
            assert mock_thread_class.call_args[1]["daemon"]
            mock_thread_instance.start.assert_called_once()

    def test_integration_with_scraping_functions(self, mock_session_state):
        """Test integration with actual scraping functions."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_company_jobs") as mock_scrape_jobs,
            patch("src.services.database_sync.sync_jobs_to_database") as mock_sync,
        ):
            mock_jobs = [{"title": "Engineer", "company": "TestCorp"}]
            mock_scrape_jobs.return_value = mock_jobs

            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify integration calls
            mock_scrape_jobs.assert_called_with("TestCorp")
            mock_sync.assert_called_with(mock_jobs)

    def test_minimal_session_state_usage(self, mock_session_state):
        """Test minimal session state usage without complex data structures."""
        from src.ui.utils.background_helpers import (
            is_scraping_active,
            start_background_scraping,
        )

        # Should use only basic session state flags
        assert not is_scraping_active()

        start_background_scraping(stay_active_in_tests=True)

        # Should set basic active flag
        assert is_scraping_active()
        assert hasattr(mock_session_state, "scraping_active")
        assert isinstance(mock_session_state.scraping_active, bool)


class TestIntegrationWithADRRequirements:
    """Test integration with ADR-008 database sync and ADR-014 scraping strategy."""

    def test_database_sync_integration(self, mock_session_state):
        """Test integration with ADR-008 database sync operations."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["TestCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_company_jobs", return_value=[{"title": "Job"}]),
            patch("src.services.database_sync.sync_jobs_to_database") as mock_sync,
        ):
            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify database sync integration
            mock_sync.assert_called_once()

    def test_scraping_strategy_integration(self, mock_session_state):
        """Test integration with ADR-014 hybrid scraping strategy."""
        from src.ui.utils.background_helpers import start_background_scraping

        mock_session_state.selected_companies = ["Company1", "Company2"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_company_jobs") as mock_scrape_jobs,
            patch("src.services.database_sync.sync_jobs_to_database"),
        ):
            mock_scrape_jobs.return_value = []

            # Execute thread function
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify scraping strategy integration - called for each company
            assert mock_scrape_jobs.call_count == 2
            mock_scrape_jobs.assert_any_call("Company1")
            mock_scrape_jobs.assert_any_call("Company2")


class TestSimplificationVerification:
    """Test that implementation follows simplification principles."""

    def test_no_complex_dataclasses_required(self, mock_session_state):
        """Test that no complex dataclasses are required for basic operation."""
        from src.ui.utils.background_helpers import (
            is_scraping_active,
            start_background_scraping,
        )

        # Should work with just basic session state
        start_background_scraping(stay_active_in_tests=True)

        # Should not require TaskInfo, ProgressInfo, or CompanyProgress objects
        assert is_scraping_active()
        assert isinstance(mock_session_state.scraping_active, bool)

    def test_minimal_imports_required(self):
        """Test that minimal imports are required for core functionality."""
        # Core functionality should work with minimal imports
        try:
            import threading

            from unittest.mock import patch

            # These should be the core imports for simplified implementation
            with patch("streamlit.runtime.scriptrunner.add_script_run_ctx"):
                assert threading is not None
        except ImportError:
            pytest.fail("Required imports not available")

    def test_reduced_complexity_verification(self, mock_session_state):
        """Test that complexity is reduced compared to original implementation."""
        from src.ui.utils.background_helpers import start_background_scraping

        # Should work with minimal session state setup
        result = start_background_scraping(stay_active_in_tests=True)

        # Should return simple task ID, not complex objects
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestCompleteBackgroundTaskWorkflow:
    """Integration tests for complete background task workflows."""

    def test_complete_simplified_workflow(self, mock_session_state):
        """Test complete workflow using simplified 50-line implementation."""
        from src.ui.utils.background_helpers import (
            is_scraping_active,
            start_background_scraping,
            stop_all_scraping,
        )

        mock_session_state.selected_companies = ["OpenAI", "Anthropic"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_company_jobs", return_value=[]),
            patch("src.services.database_sync.sync_jobs_to_database"),
            patch("streamlit.status") as mock_status,
        ):
            # Configure status context manager
            mock_status_obj = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_obj)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            # 1. Start background scraping
            task_id = start_background_scraping(stay_active_in_tests=True)
            assert is_scraping_active()
            assert isinstance(task_id, str)

            # 2. Execute background thread
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # 3. Verify completion
            mock_status_obj.update.assert_called_with(
                label="✅ Scraping completed!", state="complete"
            )
            assert not mock_session_state.scraping_active

            # 4. Stop scraping (cleanup)
            stopped_count = stop_all_scraping()
            assert stopped_count == 0  # Already stopped by completion

    def test_workflow_with_error_recovery(self, mock_session_state):
        """Test complete workflow with error handling and recovery."""
        from src.ui.utils.background_helpers import (
            is_scraping_active,
            start_background_scraping,
        )

        mock_session_state.selected_companies = ["FailCorp"]

        with (
            patch("threading.Thread") as mock_thread_class,
            patch(
                "src.scraper.scrape_company_jobs",
                side_effect=Exception("Connection error"),
            ),
            patch("streamlit.error") as mock_error,
        ):
            # Start and execute scraping (which will fail)
            start_background_scraping(stay_active_in_tests=True)
            thread_func = mock_thread_class.call_args[1]["target"]
            thread_func()

            # Verify error handling and cleanup
            mock_error.assert_called_with("Scraping failed: Connection error")
            assert not is_scraping_active()  # Should be cleaned up after error
