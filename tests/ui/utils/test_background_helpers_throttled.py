"""Tests for T1.3: Auto-refresh Fragment Elimination - Throttled Rerun Functionality.

Tests cover:
- Throttled rerun functionality with interval control
- Session state task management operations
- Background task handling without custom managers
- Test environment detection and synchronous execution
- Thread safety and atomic operations
"""

import threading
import time

from datetime import UTC, datetime
from unittest.mock import Mock, patch

from src.ui.utils.background_helpers import (
    CompanyProgress,
    ProgressInfo,
    TaskInfo,
    _atomic_check_and_set,
    _execute_test_scraping,
    _is_test_environment,
    add_task,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    get_task,
    is_scraping_active,
    remove_task,
    start_background_scraping,
    stop_all_scraping,
    throttled_rerun,
)


class TestT1ThrottledRerunFunctionality:
    """Test T1.3: Auto-refresh Fragment Elimination - Throttled Rerun."""

    def test_throttled_rerun_respects_interval(
        self, mock_session_state, mock_streamlit
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
        self, mock_session_state, mock_streamlit
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
        self, mock_session_state, mock_streamlit
    ):
        """Test throttled rerun respects the should_rerun flag."""
        # should_rerun=False should never trigger rerun
        throttled_rerun("test_key", 1.0, should_rerun=False)

        mock_streamlit["rerun"].assert_not_called()
        assert "test_key" not in mock_session_state

    def test_throttled_rerun_uses_default_interval(
        self, mock_session_state, mock_streamlit
    ):
        """Test throttled rerun uses default interval when not specified."""
        throttled_rerun("test_key", should_rerun=True)

        mock_streamlit["rerun"].assert_called_once()

        # Verify default interval is used
        timestamp = mock_session_state.get("test_key")
        assert timestamp is not None

    def test_throttled_rerun_handles_zero_interval(
        self, mock_session_state, mock_streamlit
    ):
        """Test throttled rerun handles zero interval appropriately."""
        # Zero interval should always allow rerun
        throttled_rerun("test_key", 0.0, should_rerun=True)
        mock_streamlit["rerun"].assert_called_once()

        mock_streamlit["rerun"].reset_mock()

        # Immediate second call should also trigger (no throttling)
        throttled_rerun("test_key", 0.0, should_rerun=True)
        mock_streamlit["rerun"].assert_called_once()

    def test_throttled_rerun_handles_negative_interval(
        self, mock_session_state, mock_streamlit
    ):
        """Test throttled rerun treats negative intervals as zero."""
        throttled_rerun("test_key", -1.0, should_rerun=True)
        mock_streamlit["rerun"].assert_called_once()

        mock_streamlit["rerun"].reset_mock()

        # Second call should immediately trigger (negative treated as zero)
        throttled_rerun("test_key", -1.0, should_rerun=True)
        mock_streamlit["rerun"].assert_called_once()

    def test_throttled_rerun_different_keys_independent(
        self, mock_session_state, mock_streamlit
    ):
        """Test different session keys are throttled independently."""
        # First key
        throttled_rerun("key1", 1.0, should_rerun=True)
        assert mock_streamlit["rerun"].call_count == 1

        # Different key should not be throttled
        throttled_rerun("key2", 1.0, should_rerun=True)
        assert mock_streamlit["rerun"].call_count == 2


class TestSessionStateTaskManagement:
    """Test direct session state operations for task management."""

    def test_add_task_stores_in_session_state(self, mock_session_state):
        """Test add_task stores task info in session state."""
        task_info = TaskInfo(
            task_id="test-123",
            status="running",
            progress=0.5,
            message="Test task",
            timestamp=datetime.now(UTC),
        )

        add_task("test-123", task_info)

        assert "tasks" in mock_session_state
        assert "test-123" in mock_session_state.tasks
        assert mock_session_state.tasks["test-123"] == task_info

    def test_add_task_initializes_tasks_dict(self, mock_session_state):
        """Test add_task initializes tasks dictionary if not present."""
        # Ensure tasks key doesn't exist
        assert "tasks" not in mock_session_state

        task_info = TaskInfo(
            task_id="init-test",
            status="pending",
            progress=0.0,
            message="Init task",
            timestamp=datetime.now(UTC),
        )

        add_task("init-test", task_info)

        assert "tasks" in mock_session_state
        assert isinstance(mock_session_state.tasks, dict)

    def test_get_task_retrieves_from_session_state(self, mock_session_state):
        """Test get_task retrieves task info from session state."""
        task_info = TaskInfo(
            task_id="get-test",
            status="completed",
            progress=1.0,
            message="Get task",
            timestamp=datetime.now(UTC),
        )

        # Set up session state
        mock_session_state.tasks = {"get-test": task_info}

        retrieved = get_task("get-test")

        assert retrieved == task_info

    def test_get_task_returns_none_for_missing_task(self, mock_session_state):
        """Test get_task returns None for non-existent task."""
        mock_session_state.tasks = {}

        result = get_task("missing-task")

        assert result is None

    def test_get_task_handles_missing_tasks_dict(self, mock_session_state):
        """Test get_task handles missing tasks dictionary gracefully."""
        # Ensure tasks key doesn't exist
        assert "tasks" not in mock_session_state

        result = get_task("any-task")

        assert result is None

    def test_remove_task_removes_from_session_state(self, mock_session_state):
        """Test remove_task removes task from session state."""
        task_info = TaskInfo(
            task_id="remove-test",
            status="completed",
            progress=1.0,
            message="Remove task",
            timestamp=datetime.now(UTC),
        )

        mock_session_state.tasks = {"remove-test": task_info, "keep-test": task_info}

        remove_task("remove-test")

        assert "remove-test" not in mock_session_state.tasks
        assert "keep-test" in mock_session_state.tasks

    def test_remove_task_handles_missing_task(self, mock_session_state):
        """Test remove_task handles missing task gracefully."""
        mock_session_state.tasks = {
            "existing": TaskInfo("existing", "done", 1.0, "msg", datetime.now(UTC))
        }

        # Should not raise error
        remove_task("missing")

        # Existing task should remain
        assert "existing" in mock_session_state.tasks

    def test_remove_task_handles_missing_tasks_dict(self, mock_session_state):
        """Test remove_task handles missing tasks dictionary gracefully."""
        # Should not raise error
        remove_task("any-task")


class TestAtomicOperations:
    """Test atomic session state operations for thread safety."""

    def test_atomic_check_and_set_success(self, mock_session_state):
        """Test atomic check and set succeeds when values match."""
        mock_session_state["test_key"] = "expected_value"

        result = _atomic_check_and_set("test_key", "expected_value", "new_value")

        assert result is True
        assert mock_session_state.get("test_key") == "new_value"

    def test_atomic_check_and_set_failure(self, mock_session_state):
        """Test atomic check and set fails when values don't match."""
        mock_session_state["test_key"] = "current_value"

        result = _atomic_check_and_set("test_key", "expected_value", "new_value")

        assert result is False
        assert mock_session_state.get("test_key") == "current_value"

    def test_atomic_check_and_set_missing_key(self, mock_session_state):
        """Test atomic check and set handles missing key."""
        result = _atomic_check_and_set("missing_key", None, "new_value")

        assert result is True
        assert mock_session_state.get("missing_key") == "new_value"

    def test_atomic_check_and_set_thread_safety(self, mock_session_state):
        """Test atomic check and set is thread-safe."""
        mock_session_state["counter"] = 0
        results = []

        def worker():
            # Try to increment counter atomically
            for _ in range(10):
                current = mock_session_state.get("counter", 0)
                success = _atomic_check_and_set("counter", current, current + 1)
                results.append(success)

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # At least some operations should succeed
        assert any(results)


class TestTestEnvironmentDetection:
    """Test detection of test environment for conditional behavior."""

    def test_is_test_environment_detects_pytest(self):
        """Test detection works when pytest is in sys.modules."""
        result = _is_test_environment()
        # Should be True since we're running under pytest
        assert result is True

    @patch("sys.modules", {"pytest": Mock()})
    def test_is_test_environment_pytest_module(self):
        """Test detection when pytest module is present."""
        result = _is_test_environment()
        assert result is True

    @patch("sys.modules", {"unittest": Mock()})
    def test_is_test_environment_unittest_module(self):
        """Test detection when unittest module is present."""
        result = _is_test_environment()
        assert result is True

    def test_is_test_environment_session_state_flag(self, mock_session_state):
        """Test detection using session state test flag."""
        mock_session_state._test_mode = True

        result = _is_test_environment()
        assert result is True


class TestBackgroundTaskHandling:
    """Test background task handling without custom task managers."""

    def test_is_scraping_active_reads_session_state(self, mock_session_state):
        """Test is_scraping_active reads from session state."""
        mock_session_state["scraping_active"] = True

        result = is_scraping_active()

        assert result is True

    def test_is_scraping_active_default_false(self, mock_session_state):
        """Test is_scraping_active defaults to False."""
        result = is_scraping_active()
        assert result is False

    def test_get_scraping_results_reads_session_state(self, mock_session_state):
        """Test get_scraping_results reads from session state."""
        expected_results = {
            "inserted": 5,
            "updated": 3,
            "archived": 1,
            "deleted": 0,
            "skipped": 2,
        }
        mock_session_state["scraping_results"] = expected_results

        result = get_scraping_results()

        assert result == expected_results

    def test_get_scraping_results_default_empty(self, mock_session_state):
        """Test get_scraping_results defaults to empty dict."""
        result = get_scraping_results()
        assert result == {}

    def test_start_background_scraping_initializes_state(self, mock_session_state):
        """Test start_background_scraping initializes session state properly."""
        task_id = start_background_scraping(stay_active_in_tests=True)

        # Should return a valid UUID
        assert task_id is not None
        assert len(task_id) > 0

        # Should initialize session state
        assert "task_progress" in mock_session_state
        assert task_id in mock_session_state.task_progress
        assert mock_session_state.scraping_active is True
        assert mock_session_state.scraping_trigger is True

    def test_start_background_scraping_test_mode_sync(self, mock_session_state):
        """Test start_background_scraping executes synchronously in test mode."""
        start_background_scraping(stay_active_in_tests=False)

        # In test mode without stay_active, should complete synchronously
        assert mock_session_state.scraping_active is False
        assert mock_session_state.scraping_status == "Scraping completed"
        assert "scraping_results" in mock_session_state

    def test_start_background_scraping_test_mode_async(self, mock_session_state):
        """Test start_background_scraping stays active in test mode when requested."""
        start_background_scraping(stay_active_in_tests=True)

        # Should stay active when requested
        assert mock_session_state.scraping_active is True
        assert mock_session_state._test_stay_active is True

    def test_stop_all_scraping_cleans_state(self, mock_session_state):
        """Test stop_all_scraping cleans up session state."""
        # Set up active scraping state
        mock_session_state["scraping_active"] = True

        stopped_count = stop_all_scraping()

        assert stopped_count == 1
        assert mock_session_state.scraping_active is False
        assert mock_session_state.scraping_status == "Scraping stopped"

    def test_stop_all_scraping_no_active_scraping(self, mock_session_state):
        """Test stop_all_scraping when no scraping is active."""
        stopped_count = stop_all_scraping()

        assert stopped_count == 0

    def test_get_scraping_progress_reads_session_state(self, mock_session_state):
        """Test get_scraping_progress reads from session state."""
        progress_data = {
            "task-1": ProgressInfo(0.5, "In progress", datetime.now(UTC)),
            "task-2": ProgressInfo(1.0, "Complete", datetime.now(UTC)),
        }
        mock_session_state["task_progress"] = progress_data

        result = get_scraping_progress()

        assert result == progress_data

    def test_get_company_progress_reads_session_state(self, mock_session_state):
        """Test get_company_progress reads from session state."""
        company_data = {
            "company1": CompanyProgress("Company 1", "Complete", 10),
            "company2": CompanyProgress("Company 2", "In Progress", 5),
        }
        mock_session_state["company_progress"] = company_data

        result = get_company_progress()

        assert result == company_data


class TestExecuteTestScraping:
    """Test synchronous test scraping execution."""

    def test_execute_test_scraping_updates_progress(self, mock_session_state):
        """Test _execute_test_scraping updates progress to completion."""
        task_id = "test-task-123"

        # Set up initial progress
        mock_session_state["task_progress"] = {
            task_id: ProgressInfo(0.0, "Starting", datetime.now(UTC))
        }

        _execute_test_scraping(task_id)

        # Should update progress to complete
        progress = mock_session_state.task_progress[task_id]
        assert progress.progress == 1.0
        assert progress.message == "Scraping completed"

    def test_execute_test_scraping_sets_inactive(self, mock_session_state):
        """Test _execute_test_scraping sets scraping as inactive."""
        _execute_test_scraping("test-task")

        assert mock_session_state.scraping_active is False
        assert mock_session_state.scraping_status == "Scraping completed"

    def test_execute_test_scraping_creates_mock_results(self, mock_session_state):
        """Test _execute_test_scraping creates mock scraping results."""
        _execute_test_scraping("test-task")

        assert "scraping_results" in mock_session_state
        results = mock_session_state.scraping_results

        # Should have all expected keys
        expected_keys = ["inserted", "updated", "archived", "deleted", "skipped"]
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], int)
            assert results[key] >= 0

    def test_execute_test_scraping_handles_missing_progress(self, mock_session_state):
        """Test _execute_test_scraping handles missing task progress gracefully."""
        # Don't set up task_progress

        _execute_test_scraping("missing-task")

        # Should still complete other operations
        assert mock_session_state.scraping_active is False
        assert "scraping_results" in mock_session_state


class TestDataClasses:
    """Test data classes for background task management."""

    def test_company_progress_creation(self):
        """Test CompanyProgress dataclass creation and defaults."""
        progress = CompanyProgress("Test Company")

        assert progress.name == "Test Company"
        assert progress.status == "Pending"
        assert progress.jobs_found == 0
        assert progress.start_time is None
        assert progress.end_time is None
        assert progress.error is None

    def test_company_progress_with_values(self):
        """Test CompanyProgress with custom values."""
        start_time = datetime.now(UTC)
        end_time = datetime.now(UTC)

        progress = CompanyProgress(
            name="Complete Company",
            status="Complete",
            jobs_found=25,
            start_time=start_time,
            end_time=end_time,
            error=None,
        )

        assert progress.name == "Complete Company"
        assert progress.status == "Complete"
        assert progress.jobs_found == 25
        assert progress.start_time == start_time
        assert progress.end_time == end_time

    def test_progress_info_creation(self):
        """Test ProgressInfo dataclass creation."""
        timestamp = datetime.now(UTC)
        info = ProgressInfo(0.75, "Almost done", timestamp)

        assert info.progress == 0.75
        assert info.message == "Almost done"
        assert info.timestamp == timestamp

    def test_task_info_creation(self):
        """Test TaskInfo dataclass creation."""
        timestamp = datetime.now(UTC)
        task = TaskInfo("task-123", "running", 0.5, "Processing", timestamp)

        assert task.task_id == "task-123"
        assert task.status == "running"
        assert task.progress == 0.5
        assert task.message == "Processing"
        assert task.timestamp == timestamp


class TestT1RealisticScenarios:
    """Test realistic usage scenarios for T1.3 background helpers."""

    def test_typical_scraping_workflow(self, mock_session_state):
        """Test typical scraping workflow using simplified background helpers."""
        # Start scraping
        task_id = start_background_scraping(stay_active_in_tests=True)
        assert is_scraping_active()

        # Check initial progress
        progress = get_scraping_progress()
        assert task_id in progress

        # Stop scraping
        stopped = stop_all_scraping()
        assert stopped == 1
        assert not is_scraping_active()

    def test_throttled_rerun_in_ui_loop(self, mock_session_state, mock_streamlit):
        """Test throttled rerun in typical UI refresh loop."""
        # Simulate UI refresh loop with throttling
        for i in range(10):
            # Only should trigger rerun on first call
            throttled_rerun("ui_refresh", 1.0, should_rerun=True)

            if i == 0:
                # First call should trigger
                assert mock_streamlit["rerun"].call_count == 1
            else:
                # Subsequent calls should be throttled
                assert mock_streamlit["rerun"].call_count == 1

    def test_concurrent_task_management(self, mock_session_state):
        """Test concurrent task management operations."""
        # Add multiple tasks concurrently
        tasks = []
        for i in range(5):
            task_info = TaskInfo(
                task_id=f"task-{i}",
                status="running",
                progress=i * 0.2,
                message=f"Task {i}",
                timestamp=datetime.now(UTC),
            )
            add_task(f"task-{i}", task_info)
            tasks.append(task_info)

        # Verify all tasks were added
        for i in range(5):
            retrieved = get_task(f"task-{i}")
            assert retrieved == tasks[i]

        # Remove some tasks
        remove_task("task-1")
        remove_task("task-3")

        # Verify correct tasks remain
        assert get_task("task-0") is not None
        assert get_task("task-1") is None
        assert get_task("task-2") is not None
        assert get_task("task-3") is None
        assert get_task("task-4") is not None

    def test_background_helpers_integration(self, mock_session_state, mock_streamlit):
        """Test integration between throttled rerun and task management."""
        # Start background task
        start_background_scraping(stay_active_in_tests=True)

        # Use throttled rerun for UI updates
        throttled_rerun("scraping_ui", 0.5, should_rerun=is_scraping_active())

        # Should trigger rerun since scraping is active
        mock_streamlit["rerun"].assert_called_once()

        # Stop scraping
        stop_all_scraping()

        # Reset mock
        mock_streamlit["rerun"].reset_mock()

        # Throttled rerun should not trigger when scraping is inactive
        throttled_rerun("scraping_ui", 0.5, should_rerun=is_scraping_active())

        mock_streamlit["rerun"].assert_not_called()

    def test_performance_with_frequent_operations(self, mock_session_state):
        """Test performance with frequent task operations."""
        import time

        start_time = time.time()

        # Perform many task operations
        for i in range(1000):
            task_info = TaskInfo(f"perf-{i}", "test", 0.5, "msg", datetime.now(UTC))
            add_task(f"perf-{i}", task_info)
            get_task(f"perf-{i}")
            if i % 2 == 0:
                remove_task(f"perf-{i}")

        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 1.0  # Under 1 second for 1000 operations
