"""Tests for simplified background helpers - Phase 3 implementation.

This module tests the simplified background helpers focused on core functionality:
- Throttled rerun functionality with interval control
- Simple session state management without complex dataclasses
- Basic background task handling with minimal custom code
- Test environment detection for synchronous execution

Focus: 50-line simplified implementation per ADR-017
"""

import time

from unittest.mock import Mock, patch

from src.ui.utils.background_helpers import (
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
    throttled_rerun,
)


class TestThrottledRerunFunctionality:
    """Test throttled rerun functionality - core utility for implementation."""

    def test_throttled_rerun_respects_interval(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test throttled rerun respects the specified interval."""
        # First call should trigger rerun
        throttled_rerun("test_key", 1.0, should_rerun=True)

        mock_streamlit["rerun"].assert_called_once()
        assert "test_key" in mock_session_state

        # Reset mock
        mock_streamlit["rerun"].reset_mock()

        # Immediate second call should not trigger rerun (within interval)
        throttled_rerun("test_key", 1.0, should_rerun=True)

        mock_streamlit["rerun"].assert_not_called()

    def test_throttled_rerun_allows_after_interval(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test throttled rerun allows execution after interval passes."""
        # First call
        throttled_rerun("test_key", 0.1, should_rerun=True)  # 100ms interval
        mock_streamlit["rerun"].assert_called_once()

        # Wait for interval to pass
        time.sleep(0.15)  # 150ms > 100ms interval

        # Reset mock
        mock_streamlit["rerun"].reset_mock()

        # Second call should now trigger rerun
        throttled_rerun("test_key", 0.1, should_rerun=True)
        mock_streamlit["rerun"].assert_called_once()

    def test_throttled_rerun_respects_should_rerun_flag(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test throttled rerun respects the should_rerun flag."""
        # should_rerun=False should never trigger rerun
        throttled_rerun("test_key", 1.0, should_rerun=False)

        mock_streamlit["rerun"].assert_not_called()
        assert "test_key" not in mock_session_state

    def test_throttled_rerun_different_keys_independent(
        self,
        mock_session_state,
        mock_streamlit,
    ):
        """Test different session keys are throttled independently."""
        # First key
        throttled_rerun("key1", 1.0, should_rerun=True)
        assert mock_streamlit["rerun"].call_count == 1

        # Different key should not be throttled
        throttled_rerun("key2", 1.0, should_rerun=True)
        assert mock_streamlit["rerun"].call_count == 2


class TestSimplifiedSessionStateManagement:
    """Test simplified session state operations without complex dataclasses."""

    def test_is_scraping_active_reads_session_state(self, mock_session_state):
        """Test is_scraping_active reads from session state."""
        mock_session_state.scraping_active = True

        result = is_scraping_active()

        assert result

    def test_is_scraping_active_default_false(self, mock_session_state):
        """Test is_scraping_active defaults to False."""
        result = is_scraping_active()
        assert not result

    def test_session_state_coordination_basic_flags(self, mock_session_state):
        """Test basic session state flag coordination."""
        # Should start as inactive
        assert not is_scraping_active()

        # Set active manually
        mock_session_state.scraping_active = True
        assert is_scraping_active()

        # Clear active
        mock_session_state.scraping_active = False
        assert not is_scraping_active()


class TestSimplifiedBackgroundTaskHandling:
    """Test simplified background task handling without complex managers."""

    def test_start_background_scraping_basic_state_setup(self, mock_session_state):
        """Test start_background_scraping sets basic session state."""
        task_id = start_background_scraping(stay_active_in_tests=True)

        # Should return a valid UUID-like string
        assert task_id is not None
        assert len(task_id) > 0
        assert isinstance(task_id, str)

        # Should set basic active flag
        assert mock_session_state.scraping_active

    def test_start_background_scraping_test_mode_sync(self, mock_session_state):
        """Test start_background_scraping executes synchronously in test mode."""
        start_background_scraping(stay_active_in_tests=False)

        # In test mode without stay_active, should complete synchronously
        assert not mock_session_state.scraping_active
        assert "scraping_results" in mock_session_state

    def test_start_background_scraping_test_mode_async(self, mock_session_state):
        """Test start_background_scraping stays active in test mode when requested."""
        start_background_scraping(stay_active_in_tests=True)

        # Should stay active when requested
        assert mock_session_state.scraping_active

    def test_stop_all_scraping_cleans_basic_state(self, mock_session_state):
        """Test stop_all_scraping cleans up basic session state."""
        # Set up active scraping state
        mock_session_state.scraping_active = True
        # Set up thread reference that will be cleaned up
        from unittest.mock import Mock

        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_session_state.scraping_thread = mock_thread

        stopped_count = stop_all_scraping()

        assert stopped_count == 1
        assert not mock_session_state.scraping_active
        assert mock_session_state.scraping_status == "Scraping stopped"

    def test_stop_all_scraping_no_active_scraping(self, mock_session_state):
        """Test stop_all_scraping when no scraping is active."""
        stopped_count = stop_all_scraping()

        assert stopped_count == 0


class TestTestEnvironmentDetection:
    """Test detection of test environment for conditional behavior."""

    def test_test_environment_handling(self, mock_session_state):
        """Test that test environment behavior works correctly."""
        # In test mode, background scraping should complete synchronously
        task_id = start_background_scraping(stay_active_in_tests=False)

        # Should complete synchronously in test mode
        assert not mock_session_state.scraping_active
        assert isinstance(task_id, str)


class TestSimplifiedWorkflowIntegration:
    """Test simplified workflow without complex task management."""

    def test_typical_scraping_workflow_simplified(self, mock_session_state):
        """Test typical scraping workflow using simplified helpers."""
        # Start scraping
        task_id = start_background_scraping(stay_active_in_tests=True)
        assert is_scraping_active()
        assert isinstance(task_id, str)

        # Stop scraping
        stopped = stop_all_scraping()
        assert stopped == 1
        assert not is_scraping_active()

    def test_throttled_rerun_first_call_triggers(
        self, mock_session_state, mock_streamlit
    ):
        """Test first throttled rerun call triggers."""
        throttled_rerun("ui_refresh", 1.0, should_rerun=True)
        assert mock_streamlit["rerun"].call_count == 1

    def test_throttled_rerun_subsequent_calls_throttled(
        self, mock_session_state, mock_streamlit
    ):
        """Test subsequent throttled rerun calls are throttled."""
        # First call triggers
        throttled_rerun("ui_refresh", 1.0, should_rerun=True)
        assert mock_streamlit["rerun"].call_count == 1

        # Subsequent calls are throttled
        for _ in range(4):
            throttled_rerun("ui_refresh", 1.0, should_rerun=True)

        # Should still be 1 - no additional calls triggered
        assert mock_streamlit["rerun"].call_count == 1

    def test_integration_with_background_task_state(
        self, mock_session_state, mock_streamlit
    ):
        """Test integration between throttled rerun and simplified task state."""
        # Start background task
        start_background_scraping(stay_active_in_tests=True)

        # Use throttled rerun for UI updates
        throttled_rerun("scraping_ui", 0.1, should_rerun=is_scraping_active())

        # Should trigger rerun since scraping is active
        mock_streamlit["rerun"].assert_called_once()

        # Stop scraping
        stop_all_scraping()

        # Reset mock
        mock_streamlit["rerun"].reset_mock()

        # Wait for throttle to clear
        time.sleep(0.2)

        # Throttled rerun should not trigger when scraping is inactive
        throttled_rerun("scraping_ui", 0.1, should_rerun=is_scraping_active())

        mock_streamlit["rerun"].assert_not_called()


class TestSimplificationCompliance:
    """Test that implementation follows simplification principles from ADR-017."""

    def test_minimal_session_state_usage(self, mock_session_state):
        """Test minimal session state usage without complex objects."""
        # Should work with basic boolean flags
        start_background_scraping(stay_active_in_tests=True)

        # Should use simple boolean flags, not complex dataclasses
        assert isinstance(mock_session_state.scraping_active, bool)
        assert is_scraping_active()

    def test_no_complex_task_management_required(self, mock_session_state):
        """Test that complex task management objects are not required."""
        # Basic workflow should work without TaskInfo, ProgressInfo, etc.
        task_id = start_background_scraping(stay_active_in_tests=True)

        # Should return simple string ID
        assert isinstance(task_id, str)

        # Should work with basic session state
        assert is_scraping_active()

        # Cleanup should work simply
        stopped = stop_all_scraping()
        assert stopped == 1

    def test_reduced_complexity_verification(self, mock_session_state):
        """Test that complexity is reduced per ADR-017 requirements."""
        # Should work with minimal function calls
        task_id = start_background_scraping(stay_active_in_tests=True)
        assert isinstance(task_id, str)

        # Should have simple state management
        assert is_scraping_active()

        # Should clean up simply
        stop_all_scraping()
        assert not is_scraping_active()


class TestCoreFunctionalityOnly:
    """Test only core functionality needed for 50-line implementation."""

    def test_essential_functions_work(self, mock_session_state):
        """Test that essential functions work without complex dependencies."""
        # Core functions should work
        assert callable(start_background_scraping)
        assert callable(stop_all_scraping)
        assert callable(is_scraping_active)
        assert callable(throttled_rerun)

    def test_basic_threading_coordination(self, mock_session_state):
        """Test basic threading coordination without complex objects."""
        with patch(
            "src.ui.utils.background_helpers.threading.Thread"
        ) as mock_thread_class:
            mock_thread_instance = Mock()
            mock_thread_class.return_value = mock_thread_instance

            # Should create and start thread
            start_background_scraping(stay_active_in_tests=True)

            # Should use daemon threads
            assert mock_thread_class.call_args[1]["daemon"] is True
            mock_thread_instance.start.assert_called_once()

    def test_streamlit_integration_basics(self, mock_session_state):
        """Test basic Streamlit integration without complex components."""
        with (
            patch(
                "src.ui.utils.background_helpers.threading.Thread"
            ) as mock_thread_class,
            patch("src.ui.utils.background_helpers.add_script_run_ctx") as mock_add_ctx,
        ):
            mock_thread_instance = Mock()
            mock_thread_class.return_value = mock_thread_instance

            # Should integrate with Streamlit context
            start_background_scraping(stay_active_in_tests=True)

            # Should add script run context
            mock_add_ctx.assert_called_once_with(mock_thread_instance)


class TestPerformanceAndStability:
    """Test performance characteristics of simplified implementation."""

    def test_fast_operations_with_simple_state(self, mock_session_state):
        """Test that operations are fast with simplified state management."""
        import time

        start_time = time.time()

        # Perform basic operations without threading (test mode)
        for _ in range(100):
            start_background_scraping(stay_active_in_tests=False)  # Sync test mode
            is_scraping_active()
            stop_all_scraping()

        end_time = time.time()

        # Should complete quickly with simple state in sync mode
        assert (end_time - start_time) < 1.0  # Under 1 second for 100 cycles

    def test_memory_efficiency_simple_state(self, mock_session_state):
        """Test memory efficiency with simple state objects."""
        # Multiple operations shouldn't create complex objects
        task_ids = []
        for _ in range(10):
            task_id = start_background_scraping(
                stay_active_in_tests=False
            )  # Sync test mode
            task_ids.append(task_id)
            stop_all_scraping()

        # Should just use simple strings and booleans
        assert all(isinstance(tid, str) for tid in task_ids)
        # Session state should remain simple
        assert isinstance(mock_session_state.scraping_active, bool)

    def test_concurrent_state_access_stability(self, mock_session_state):
        """Test stability under concurrent access patterns."""
        # Rapid state changes should be stable in sync mode
        for _ in range(20):
            start_background_scraping(stay_active_in_tests=False)  # Sync test mode
            assert not is_scraping_active()  # Already completed in sync mode

            stop_all_scraping()
            assert not is_scraping_active()
