"""Tests for background task management utilities.

Tests the background task system including task creation, progress tracking,
company progress management, and integration with Streamlit session state.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.ui.utils.background_tasks import (
    BackgroundTaskManager,
    CompanyProgress,
    ProgressInfo,
    StreamlitTaskManager,
    TaskInfo,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    get_task_manager,
    is_scraping_active,
    start_background_scraping,
    start_scraping,
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
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)

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
        progress.start_time = datetime.now(timezone.utc)

        # Assert
        assert progress.status == "Scraping"
        assert progress.jobs_found == 10
        assert progress.start_time is not None


class TestProgressInfo:
    """Test the ProgressInfo dataclass functionality."""

    def test_progress_info_creation(self):
        """Test ProgressInfo creates correctly with all fields."""
        # Arrange
        timestamp = datetime.now(timezone.utc)

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
        timestamp = datetime.now(timezone.utc)

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


class TestBackgroundTaskManager:
    """Test the BackgroundTaskManager functionality."""

    def test_task_manager_initialization(self):
        """Test task manager initializes with empty task dictionary."""
        # Act
        manager = BackgroundTaskManager()

        # Assert
        assert manager.tasks == {}

    def test_add_task_stores_task_info(self):
        """Test adding a task stores it in the manager."""
        # Arrange
        manager = BackgroundTaskManager()
        task_info = TaskInfo(
            task_id="task-123",
            status="running",
            progress=0.0,
            message="Starting",
            timestamp=datetime.now(timezone.utc),
        )

        # Act
        manager.add_task("task-123", task_info)

        # Assert
        assert "task-123" in manager.tasks
        assert manager.tasks["task-123"] == task_info

    def test_get_task_returns_correct_task(self):
        """Test getting a task returns the correct TaskInfo."""
        # Arrange
        manager = BackgroundTaskManager()
        task_info = TaskInfo(
            task_id="task-123",
            status="running",
            progress=0.5,
            message="In progress",
            timestamp=datetime.now(timezone.utc),
        )
        manager.add_task("task-123", task_info)

        # Act
        retrieved_task = manager.get_task("task-123")

        # Assert
        assert retrieved_task == task_info

    def test_get_task_returns_none_for_missing_task(self):
        """Test getting a non-existent task returns None."""
        # Arrange
        manager = BackgroundTaskManager()

        # Act
        retrieved_task = manager.get_task("nonexistent-task")

        # Assert
        assert retrieved_task is None

    def test_remove_task_deletes_task(self):
        """Test removing a task deletes it from the manager."""
        # Arrange
        manager = BackgroundTaskManager()
        task_info = TaskInfo(
            task_id="task-123",
            status="completed",
            progress=1.0,
            message="Done",
            timestamp=datetime.now(timezone.utc),
        )
        manager.add_task("task-123", task_info)

        # Act
        manager.remove_task("task-123")

        # Assert
        assert "task-123" not in manager.tasks

    def test_remove_nonexistent_task_does_not_error(self):
        """Test removing a non-existent task doesn't raise an error."""
        # Arrange
        manager = BackgroundTaskManager()

        # Act & Assert - Should not raise exception
        manager.remove_task("nonexistent-task")


class TestStreamlitTaskManager:
    """Test the StreamlitTaskManager (inherits from BackgroundTaskManager)."""

    def test_streamlit_task_manager_inherits_functionality(self):
        """Test StreamlitTaskManager inherits all base functionality."""
        # Arrange
        manager = StreamlitTaskManager()
        task_info = TaskInfo(
            task_id="task-123",
            status="running",
            progress=0.3,
            message="Processing",
            timestamp=datetime.now(timezone.utc),
        )

        # Act
        manager.add_task("task-123", task_info)
        retrieved_task = manager.get_task("task-123")

        # Assert
        assert retrieved_task == task_info


import pytest


@pytest.mark.usefixtures("_mock_session_state")
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

    def test_get_task_manager_creates_and_returns_manager(self, mock_session_state):
        """Test get_task_manager creates and stores manager in session state."""
        # Act
        manager = get_task_manager()

        # Assert
        assert isinstance(manager, StreamlitTaskManager)
        assert mock_session_state.get("task_manager") == manager

    def test_get_task_manager_returns_existing_manager(self, mock_session_state):
        """Test get_task_manager returns existing manager from session state."""
        # Arrange
        existing_manager = StreamlitTaskManager()
        mock_session_state.update({"task_manager": existing_manager})

        # Act
        manager = get_task_manager()

        # Assert
        assert manager == existing_manager

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
                timestamp=datetime.now(timezone.utc),
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
        with patch("src.ui.utils.background_tasks.start_scraping"):
            task_id = start_background_scraping()

        # Assert
        assert mock_session_state.get("scraping_active") is True
        assert "task_progress" in mock_session_state._data
        assert task_id in mock_session_state.get("task_progress")
        assert mock_session_state.get("task_id") == task_id

    def test_start_background_scraping_creates_progress_info(self, mock_session_state):
        """Test start_background_scraping creates initial progress information."""
        # Act
        with patch("src.ui.utils.background_tasks.start_scraping"):
            task_id = start_background_scraping()

        # Assert
        task_progress = mock_session_state.get("task_progress")
        progress_info = task_progress[task_id]

        assert isinstance(progress_info, ProgressInfo)
        assert progress_info.progress == 0.0
        assert progress_info.message == "Starting scraping..."
        assert isinstance(progress_info.timestamp, datetime)

    def test_start_background_scraping_calls_start_scraping(self):
        """Test start_background_scraping calls the start_scraping function."""
        # Act
        with patch("src.ui.utils.background_tasks.start_scraping") as mock_start:
            start_background_scraping()

        # Assert
        mock_start.assert_called_once()

    def test_start_background_scraping_returns_task_id(self):
        """Test start_background_scraping returns a valid task ID."""
        # Act
        with patch("src.ui.utils.background_tasks.start_scraping"):
            task_id = start_background_scraping()

        # Assert
        assert isinstance(task_id, str)
        assert len(task_id) > 0
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
    """Test the start_scraping function and background task execution."""

    def test_start_scraping_sets_active_state(self, mock_session_state):
        """Test start_scraping sets scraping to active state."""
        # Act
        with (
            patch("src.ui.utils.background_tasks.JobService"),
            patch("threading.Thread") as mock_thread_class,
        ):
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            start_scraping()

        # Assert
        assert mock_session_state.get("scraping_active") is True
        assert mock_session_state.get("scraping_status") == "Initializing scraping..."

    def test_start_scraping_creates_background_thread(self, mock_session_state):
        """Test start_scraping creates and starts background thread."""
        # Act
        with (
            patch("src.ui.utils.background_tasks.JobService"),
            patch("threading.Thread") as mock_thread_class,
        ):
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            start_scraping()

        # Assert
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()
        assert mock_session_state.get("scraping_thread") == mock_thread

    def test_start_scraping_handles_no_active_companies(
        self, mock_session_state, mock_job_service
    ):
        """Test start_scraping handles case with no active companies."""
        # Arrange
        mock_job_service.get_active_companies.return_value = []

        # Act
        with patch("threading.Thread") as mock_thread_class:
            mock_thread_class.return_value.start = Mock()
            start_scraping()

            # Execute the thread function directly to test its behavior
            thread_call_args = mock_thread_class.call_args
            thread_target = thread_call_args[1]["target"]
            thread_target()

        # Assert - Should handle gracefully without crashing
        assert mock_session_state.get("scraping_active") is False

    def test_start_scraping_initializes_company_progress(
        self, mock_session_state, mock_job_service
    ):
        """Test start_scraping initializes company progress tracking."""
        # Arrange
        mock_job_service["background_tasks"].get_active_companies.return_value = [
            "TechCorp",
            "DataCorp",
        ]

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch(
                "src.scraper.scrape_all", return_value={"TechCorp": 25, "DataCorp": 15}
            ),
        ):
            mock_thread_class.return_value.start = Mock()
            start_scraping()

            # Execute the thread function
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

        # Assert
        company_progress = mock_session_state.get("company_progress")
        assert "TechCorp" in company_progress
        assert "DataCorp" in company_progress

        assert company_progress["TechCorp"].name == "TechCorp"
        assert company_progress["DataCorp"].name == "DataCorp"

    def test_start_scraping_updates_company_progress_with_results(
        self, mock_session_state, mock_job_service
    ):
        """Test start_scraping updates company progress with scraping results."""
        # Arrange
        mock_job_service.get_active_companies.return_value = ["TechCorp"]
        scraping_results = {"TechCorp": 30}

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_all", return_value=scraping_results),
        ):
            mock_thread_class.return_value.start = Mock()
            start_scraping()

            # Execute the thread function
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

        # Assert
        company_progress = mock_session_state.get("company_progress")
        techcorp_progress = company_progress["TechCorp"]

        assert techcorp_progress.status == "Completed"
        assert techcorp_progress.jobs_found == 30
        assert techcorp_progress.end_time is not None

    def test_start_scraping_handles_scraping_errors_gracefully(
        self, mock_session_state, mock_job_service
    ):
        """Test start_scraping handles scraping errors without crashing."""
        # Arrange
        mock_job_service.get_active_companies.return_value = ["TechCorp"]

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_all", side_effect=Exception("Scraping failed")),
        ):
            mock_thread_class.return_value.start = Mock()
            start_scraping()

            # Execute the thread function
            thread_target = mock_thread_class.call_args[1]["target"]
            thread_target()

        # Assert - Should handle error gracefully
        assert mock_session_state.get("scraping_active") is False
        company_progress = mock_session_state.get("company_progress")
        if company_progress and "TechCorp" in company_progress:
            assert company_progress["TechCorp"].status == "Error"

    def test_start_scraping_stores_scraping_results(
        self, mock_session_state, mock_job_service
    ):
        """Test start_scraping stores final scraping results in session state."""
        # Arrange
        mock_job_service.get_active_companies.return_value = ["TechCorp", "DataCorp"]
        scraping_results = {"TechCorp": 25, "DataCorp": 15}

        # Act
        with (
            patch("threading.Thread") as mock_thread_class,
            patch("src.scraper.scrape_all", return_value=scraping_results),
        ):
            mock_thread_class.return_value.start = Mock()
            start_scraping()

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

    def test_concurrent_task_management(self):
        """Test task manager can handle multiple concurrent operations."""
        # Arrange
        manager = get_task_manager()

        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = TaskInfo(
                task_id=f"task-{i}",
                status="running",
                progress=i * 0.1,
                message=f"Task {i} running",
                timestamp=datetime.now(timezone.utc),
            )
            tasks.append(task)

        # Act
        for i, task in enumerate(tasks):
            manager.add_task(f"task-{i}", task)

        # Assert
        assert len(manager.tasks) == 3
        for i in range(3):
            retrieved = manager.get_task(f"task-{i}")
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
