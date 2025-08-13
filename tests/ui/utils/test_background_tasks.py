"""Tests for background task management utilities.

Tests the background task system including task creation, progress tracking,
company progress management, and integration with Streamlit session state.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.ui.utils.background_helpers import (
    CompanyProgress,
    ProgressInfo,
    TaskInfo,
    add_task,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    get_task,
    is_scraping_active,
    remove_task,
    start_background_scraping,
    stop_all_scraping,
)


class TestCompanyProgress:
    """Test the CompanyProgress dataclass functionality."""

    def test_company_progress_default_values(self):
        """Test CompanyProgress creates with correct default values."""
        # Act
        progress = CompanyProgress(name="TechCorp")

        # Assert
        assert progress.name == "TechCorp"
        assert progress.status == "Pending"
        assert progress.jobs_found == 0
        assert progress.start_time is None
        assert progress.end_time is None
        assert progress.error is None

    def test_company_progress_with_custom_values(self):
        """Test CompanyProgress accepts custom values."""
        # Arrange
        start_time = datetime.now(UTC)
        end_time = datetime.now(UTC)

        # Act
        progress = CompanyProgress(
            name="DataCorp",
            status="Completed",
            jobs_found=25,
            start_time=start_time,
            end_time=end_time,
            error="Connection timeout",
        )

        # Assert
        assert progress.name == "DataCorp"
        assert progress.status == "Completed"
        assert progress.jobs_found == 25
        assert progress.start_time == start_time
        assert progress.end_time == end_time
        assert progress.error == "Connection timeout"

    def test_company_progress_can_be_updated(self):
        """Test CompanyProgress fields can be updated after creation."""
        # Arrange
        progress = CompanyProgress(name="TechCorp")

        # Act
        progress.status = "Scraping"
        progress.jobs_found = 10
        progress.start_time = datetime.now(UTC)

        # Assert
        assert progress.status == "Scraping"
        assert progress.jobs_found == 10
        assert progress.start_time is not None


class TestProgressInfo:
    """Test the ProgressInfo dataclass functionality."""

    def test_progress_info_creation(self):
        """Test ProgressInfo creates correctly with all fields."""
        # Arrange
        timestamp = datetime.now(UTC)

        # Act
        info = ProgressInfo(
            progress=0.75, message="Processing companies...", timestamp=timestamp
        )

        # Assert
        assert info.progress == 0.75
        assert info.message == "Processing companies..."
        assert info.timestamp == timestamp


class TestTaskInfo:
    """Test the TaskInfo dataclass functionality."""

    def test_task_info_creation(self):
        """Test TaskInfo creates correctly with all fields."""
        # Arrange
        timestamp = datetime.now(UTC)

        # Act
        task = TaskInfo(
            task_id="task-123",
            status="running",
            progress=0.5,
            message="Scraping in progress",
            timestamp=timestamp,
        )

        # Assert
        assert task.task_id == "task-123"
        assert task.status == "running"
        assert task.progress == 0.5
        assert task.message == "Scraping in progress"
        assert task.timestamp == timestamp


@pytest.mark.usefixtures("mock_session_state")
class TestSessionStateTaskFunctions:
    """Test the session state task management functions."""

    def test_add_task_stores_task_info(self, mock_session_state):
        """Test adding a task stores it in session state."""
        # Arrange
        task_info = TaskInfo(
            task_id="task-123",
            status="running",
            progress=0.0,
            message="Starting",
            timestamp=datetime.now(UTC),
        )

        # Act
        add_task("task-123", task_info)

        # Assert
        assert "tasks" in mock_session_state._data
        assert "task-123" in mock_session_state._data["tasks"]
        assert mock_session_state._data["tasks"]["task-123"] == task_info

    def test_get_task_returns_correct_task(self, mock_session_state):
        """Test getting a task returns the correct TaskInfo."""
        # Arrange
        task_info = TaskInfo(
            task_id="task-123",
            status="running",
            progress=0.5,
            message="In progress",
            timestamp=datetime.now(UTC),
        )
        mock_session_state.update({"tasks": {"task-123": task_info}})

        # Act
        retrieved_task = get_task("task-123")

        # Assert
        assert retrieved_task == task_info

    def test_get_task_returns_none_for_missing_task(self, mock_session_state):
        """Test getting a non-existent task returns None."""
        # Act
        retrieved_task = get_task("nonexistent-task")

        # Assert
        assert retrieved_task is None

    def test_remove_task_deletes_task(self, mock_session_state):
        """Test removing a task deletes it from session state."""
        # Arrange
        task_info = TaskInfo(
            task_id="task-123",
            status="completed",
            progress=1.0,
            message="Done",
            timestamp=datetime.now(UTC),
        )
        mock_session_state.update({"tasks": {"task-123": task_info}})

        # Act
        remove_task("task-123")

        # Assert
        assert "task-123" not in mock_session_state._data.get("tasks", {})

    def test_remove_nonexistent_task_does_not_error(self, mock_session_state):
        """Test removing a non-existent task doesn't raise an error."""
        # Act & Assert - Should not raise exception
        remove_task("nonexistent-task")


@pytest.mark.usefixtures("mock_session_state")
class TestBackgroundTaskStateFunctions:
    """Test background task state management functions."""

    def test_is_scraping_active_returns_false_by_default(self):
        """Test is_scraping_active returns False when not set."""
        # Act
        result = is_scraping_active()

        # Assert
        assert result is False

    def test_is_scraping_active_returns_session_state_value(self, mock_session_state):
        """Test is_scraping_active returns value from session state."""
        # Arrange
        mock_session_state.update({"scraping_active": True})

        # Act
        result = is_scraping_active()

        # Assert
        assert result is True

    def test_get_scraping_results_returns_empty_dict_by_default(self):
        """Test get_scraping_results returns empty dict when not set."""
        # Act
        result = get_scraping_results()

        # Assert
        assert result == {}

    def test_get_scraping_results_returns_session_state_value(self, mock_session_state):
        """Test get_scraping_results returns value from session state."""
        # Arrange
        results = {"TechCorp": 25, "DataCorp": 15}
        mock_session_state.update({"scraping_results": results})

        # Act
        result = get_scraping_results()

        # Assert
        assert result == results

    def test_get_scraping_progress_returns_empty_dict_by_default(self):
        """Test get_scraping_progress returns empty dict when not set."""
        # Act
        result = get_scraping_progress()

        # Assert
        assert result == {}

    def test_get_scraping_progress_returns_session_state_value(
        self, mock_session_state
    ):
        """Test get_scraping_progress returns value from session state."""
        # Arrange
        progress = {
            "task-123": ProgressInfo(
                progress=0.5,
                message="In progress",
                timestamp=datetime.now(UTC),
            )
        }
        mock_session_state.update({"task_progress": progress})

        # Act
        result = get_scraping_progress()

        # Assert
        assert result == progress

    def test_get_company_progress_returns_empty_dict_by_default(self):
        """Test get_company_progress returns empty dict when not set."""
        # Act
        result = get_company_progress()

        # Assert
        assert result == {}

    def test_get_company_progress_returns_session_state_value(self, mock_session_state):
        """Test get_company_progress returns value from session state."""
        # Arrange
        progress = {
            "TechCorp": CompanyProgress(
                name="TechCorp", status="Completed", jobs_found=25
            )
        }
        mock_session_state.update({"company_progress": progress})

        # Act
        result = get_company_progress()

        # Assert
        assert result == progress


class TestStartBackgroundScraping:
    """Test the start_background_scraping function."""

    def test_start_background_scraping_initializes_session_state(
        self, mock_session_state
    ):
        """Test start_background_scraping initializes required session state."""
        # Act
        task_id = start_background_scraping()

        # Assert
        assert mock_session_state.get("scraping_active") is True
        assert "task_progress" in mock_session_state._data
        assert task_id in mock_session_state.get("task_progress")
        assert mock_session_state.get("task_id") == task_id

    def test_start_background_scraping_creates_progress_info(self, mock_session_state):
        """Test start_background_scraping creates initial progress information."""
        # Act
        task_id = start_background_scraping()

        # Assert
        task_progress = mock_session_state.get("task_progress")
        progress_info = task_progress[task_id]

        assert isinstance(progress_info, ProgressInfo)
        assert progress_info.progress == 0.0
        assert progress_info.message == "Starting scraping..."
        assert isinstance(progress_info.timestamp, datetime)

    def test_start_background_scraping_sets_trigger_flag(self, mock_session_state):
        """Test start_background_scraping sets scraping trigger flag."""
        # Act
        start_background_scraping()

        # Assert
        assert mock_session_state.get("scraping_trigger") is True

    def test_start_background_scraping_returns_task_id(self, mock_session_state):
        """Test start_background_scraping returns a valid task ID."""
        # Act
        task_id = start_background_scraping()

        # Assert
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        # Should also store the task ID in session state
        assert mock_session_state.get("task_id") == task_id
        # Should be a UUID-like string
        assert len(task_id.split("-")) == 5


class TestStopAllScraping:
    """Test the stop_all_scraping function."""

    def test_stop_all_scraping_stops_active_scraping(self, mock_session_state):
        """Test stop_all_scraping stops active scraping and returns count."""
        # Arrange
        mock_session_state.update({"scraping_active": True})

        # Act
        stopped_count = stop_all_scraping()

        # Assert
        assert mock_session_state.get("scraping_active") is False
        assert mock_session_state.get("scraping_status") == "Scraping stopped"
        assert stopped_count == 1

    def test_stop_all_scraping_returns_zero_when_not_active(self, mock_session_state):
        """Test stop_all_scraping returns 0 when scraping is not active."""
        # Arrange
        mock_session_state.update({"scraping_active": False})

        # Act
        stopped_count = stop_all_scraping()

        # Assert
        assert stopped_count == 0

    def test_stop_all_scraping_cleans_up_thread_reference(self, mock_session_state):
        """Test stop_all_scraping cleans up thread references."""
        # Arrange
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_session_state.update(
            {"scraping_active": True, "scraping_thread": mock_thread}
        )

        # Act
        stop_all_scraping()

        # Assert
        mock_thread.join.assert_called_once_with(timeout=5.0)
        assert "scraping_thread" not in mock_session_state._data

    def test_stop_all_scraping_handles_dead_thread_gracefully(self, mock_session_state):
        """Test stop_all_scraping handles already-dead threads gracefully."""
        # Arrange
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        mock_session_state.update(
            {"scraping_active": True, "scraping_thread": mock_thread}
        )

        # Act
        stopped_count = stop_all_scraping()

        # Assert
        mock_thread.join.assert_not_called()  # Should not join dead thread
        assert stopped_count == 1


class TestStartScrapingFunction:
    """Test the start_background_scraping function and background task execution."""

    def test_start_background_scraping_sets_active_state(self, mock_session_state):
        """Test start_background_scraping sets scraping to active state."""
        # Act
        start_background_scraping()

        # Assert
        assert mock_session_state.get("scraping_active") is True
        assert mock_session_state.get("scraping_status") == "Initializing scraping..."

    def test_start_background_scraping_with_test_parameter(self, mock_session_state):
        """Test start_background_scraping accepts stay_active_in_tests parameter."""
        # Act - Function should accept the parameter without error
        task_id = start_background_scraping(stay_active_in_tests=True)

        # Assert - Function should still work and return a valid task ID
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        assert mock_session_state.get("scraping_active") is True

    def test_start_background_scraping_initializes_task_progress(
        self, mock_session_state
    ):
        """Test start_background_scraping initializes task progress tracking."""
        # Act
        task_id = start_background_scraping()

        # Assert
        task_progress = mock_session_state.get("task_progress")
        assert task_progress is not None
        assert task_id in task_progress
        progress_info = task_progress[task_id]
        assert progress_info.progress == 0.0
        assert "Starting scraping" in progress_info.message

    def test_start_background_scraping_handles_multiple_calls(self, mock_session_state):
        """Test start_background_scraping handles multiple calls gracefully."""
        # Act - Call function multiple times
        task_id_1 = start_background_scraping()
        task_id_2 = start_background_scraping()

        # Assert - Both tasks should be different and tracked
        assert task_id_1 != task_id_2
        assert isinstance(task_id_1, str)
        assert isinstance(task_id_2, str)

        # Latest task ID should be stored
        assert mock_session_state.get("task_id") == task_id_2

        # Both should have task progress
        task_progress = mock_session_state.get("task_progress")
        assert task_id_1 in task_progress
        assert task_id_2 in task_progress

    def test_start_background_scraping_sets_all_required_flags(
        self, mock_session_state
    ):
        """Test start_background_scraping sets all required session state flags."""
        # Act
        task_id = start_background_scraping()

        # Assert - All required flags should be set
        assert mock_session_state.get("scraping_trigger") is True
        assert mock_session_state.get("scraping_active") is True
        assert mock_session_state.get("scraping_status") == "Initializing scraping..."
        assert mock_session_state.get("task_id") == task_id

        # Task progress should be initialized
        task_progress = mock_session_state.get("task_progress")
        assert task_progress is not None
        assert task_id in task_progress

        progress_info = task_progress[task_id]
        assert progress_info.progress == 0.0
        assert "Starting scraping" in progress_info.message

    def test_start_background_scraping_handles_scraping_errors_gracefully(
        self, mock_session_state, mock_job_service
    ):
        """Test start_background_scraping handles scraping errors without crashing."""
        # Arrange
        mock_job_service.get_active_companies.return_value = ["TechCorp"]

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_all", side_effect=Exception("Scraping failed")),
        ):
            mock_thread_class.return_value.start = Mock()
            start_background_scraping()

            # Execute the thread function
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

        # Assert - Should handle error gracefully
        assert mock_session_state.get("scraping_active") is False
        company_progress = mock_session_state.get("company_progress")
        if company_progress and "TechCorp" in company_progress:
            assert company_progress["TechCorp"].status == "Error"

    def test_start_background_scraping_stores_scraping_results(
        self, mock_session_state, mock_job_service
    ):
        """Test start_background_scraping stores final scraping results."""
        # Arrange
        mock_job_service.get_active_companies.return_value = ["TechCorp", "DataCorp"]
        scraping_results = {"TechCorp": 25, "DataCorp": 15}

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_all", return_value=scraping_results),
        ):
            mock_thread_class.return_value.start = Mock()
            start_background_scraping()

            # Execute the thread function
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

        # Assert
        stored_results = mock_session_state.get("scraping_results")
        assert stored_results == scraping_results


class TestBackgroundTaskIntegration:
    """Integration tests for complete background task workflows."""

    def test_complete_background_scraping_workflow(
        self, mock_session_state, mock_job_service
    ):
        """Test complete workflow from start to finish."""
        # Arrange
        mock_job_service.get_active_companies.return_value = ["TechCorp"]
        scraping_results = {"TechCorp": 20}

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_all", return_value=scraping_results),
        ):
            mock_thread_class.return_value.start = Mock()

            # Start scraping
            task_id = start_background_scraping()

            # Execute background thread
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

            # Stop scraping
            stopped_count = stop_all_scraping()

        # Assert complete workflow
        # 1. Task was created and started
        assert isinstance(task_id, str)
        assert mock_session_state.get("task_id") == task_id

        # 2. Company progress was tracked
        company_progress = mock_session_state.get("company_progress")
        assert "TechCorp" in company_progress
        assert company_progress["TechCorp"].jobs_found == 20

        # 3. Results were stored
        assert mock_session_state.get("scraping_results") == scraping_results

        # 4. Scraping was stopped
        assert stopped_count == 1
        assert mock_session_state.get("scraping_active") is False

    def test_concurrent_task_management(self, mock_session_state):
        """Test session state can handle multiple concurrent task operations."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = TaskInfo(
                task_id=f"task-{i}",
                status="running",
                progress=i * 0.1,
                message=f"Task {i} running",
                timestamp=datetime.now(UTC),
            )
            tasks.append(task)

        # Act
        for i, task in enumerate(tasks):
            add_task(f"task-{i}", task)

        # Assert
        assert len(mock_session_state._data.get("tasks", {})) == 3
        for i in range(3):
            retrieved = get_task(f"task-{i}")
            assert retrieved == tasks[i]

    def test_background_task_error_recovery(self, mock_session_state, mock_job_service):
        """Test background task system recovers from errors."""
        # Arrange
        mock_job_service.get_active_companies.side_effect = Exception("Database error")

        # Act
        with patch("threading.Thread") as mock_thread_class:
            mock_thread_class.return_value.start = Mock()

            # Start scraping (which will fail)
            task_id = start_background_scraping()

            # Execute thread (which should handle error)
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

        # Assert - System should remain stable after error
        assert isinstance(task_id, str)
        assert mock_session_state.get("scraping_active") is False
        # Task progress should still be accessible
        task_progress = mock_session_state.get("task_progress")
        assert task_id in task_progress

    def test_session_state_cleanup_on_stop(self, mock_session_state):
        """Test proper cleanup of session state when stopping scraping."""
        # Arrange
        mock_session_state.update(
            {
                "scraping_active": True,
                "scraping_status": "Running",
                "scraping_thread": Mock(),
            }
        )

        # Act
        stop_all_scraping()

        # Assert
        assert mock_session_state.get("scraping_active") is False
        assert mock_session_state.get("scraping_status") == "Scraping stopped"
        assert "scraping_thread" not in mock_session_state._data
