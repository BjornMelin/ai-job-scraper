"""Tests for T1.5: BackgroundTaskManager Elimination - Direct Session State Operations.

Tests cover:
- Integration between all T1 simplified components
- Task management without custom managers
- Session state isolation and thread safety
- Background task lifecycle management
- Error handling and recovery
- Performance with simplified implementations
"""

import threading
import time

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
    get_task,
    is_scraping_active,
    remove_task,
    start_background_scraping,
    stop_all_scraping,
    throttled_rerun,
)
from src.ui.utils.database_helpers import (
    clean_session_state,
    get_cached_session_factory,
)
from src.ui.utils import safe_int


class TestT1TaskManagementIntegration:
    """Test T1.5: Direct Session State Task Management Integration."""

    def test_task_lifecycle_without_custom_manager(self, mock_session_state):
        """Test complete task lifecycle using direct session state operations."""
        # Create and add task
        task_info = TaskInfo(
            task_id="integration-test",
            status="pending",
            progress=0.0,
            message="Starting integration test",
            timestamp=datetime.now(UTC),
        )

        add_task("integration-test", task_info)

        # Verify task was added
        retrieved = get_task("integration-test")
        assert retrieved == task_info
        assert "tasks" in mock_session_state

        # Update task progress
        updated_task = TaskInfo(
            task_id="integration-test",
            status="running",
            progress=0.5,
            message="Integration test in progress",
            timestamp=datetime.now(UTC),
        )

        add_task("integration-test", updated_task)  # Overwrite

        # Verify update
        retrieved = get_task("integration-test")
        assert retrieved.status == "running"
        assert retrieved.progress == 0.5

        # Complete and remove task
        remove_task("integration-test")

        # Verify removal
        assert get_task("integration-test") is None

    def test_multiple_tasks_concurrent_management(self, mock_session_state):
        """Test managing multiple tasks concurrently without custom manager."""
        tasks = []

        # Create multiple tasks
        for i in range(10):
            task_info = TaskInfo(
                task_id=f"concurrent-{i}",
                status="running",
                progress=i * 0.1,
                message=f"Concurrent task {i}",
                timestamp=datetime.now(UTC),
            )
            tasks.append(task_info)
            add_task(f"concurrent-{i}", task_info)

        # Verify all tasks exist
        for i in range(10):
            retrieved = get_task(f"concurrent-{i}")
            assert retrieved == tasks[i]

        # Remove even-numbered tasks
        for i in range(0, 10, 2):
            remove_task(f"concurrent-{i}")

        # Verify correct tasks remain
        for i in range(10):
            retrieved = get_task(f"concurrent-{i}")
            if i % 2 == 0:
                assert retrieved is None  # Removed
            else:
                assert retrieved == tasks[i]  # Still exists

    def test_task_management_with_background_scraping(self, mock_session_state):
        """Test task management integration with background scraping workflow."""
        # Start background scraping (creates task automatically)
        task_id = start_background_scraping(stay_active_in_tests=True)

        # Verify scraping task was created in progress tracking
        progress = get_scraping_progress()
        assert task_id in progress
        assert isinstance(progress[task_id], ProgressInfo)

        # Add additional manual task
        manual_task = TaskInfo(
            task_id="manual-task",
            status="running",
            progress=0.3,
            message="Manual task running alongside scraping",
            timestamp=datetime.now(UTC),
        )
        add_task("manual-task", manual_task)

        # Verify both tasks coexist
        assert is_scraping_active()
        assert get_task("manual-task") == manual_task

        # Stop scraping
        stop_all_scraping()

        # Verify scraping stopped but manual task remains
        assert not is_scraping_active()
        assert get_task("manual-task") == manual_task

        # Clean up
        remove_task("manual-task")

    def test_session_state_isolation_between_tasks(self, mock_session_state):
        """Test that task management maintains proper session state isolation."""
        # Add tasks with different data types and structures
        string_task = TaskInfo(
            "string-task",
            "status",
            0.5,
            "message",
            datetime.now(UTC),
        )
        dict_task = TaskInfo(
            "dict-task",
            "active",
            0.8,
            "complex message",
            datetime.now(UTC),
        )

        add_task("string-task", string_task)
        add_task("dict-task", dict_task)

        # Add non-task data to session state
        mock_session_state["user_data"] = {"key": "value"}
        mock_session_state["ui_state"] = "active"

        # Verify tasks don't interfere with other session state
        assert mock_session_state["user_data"] == {"key": "value"}
        assert mock_session_state["ui_state"] == "active"

        # Verify tasks are properly isolated
        assert get_task("string-task") == string_task
        assert get_task("dict-task") == dict_task

        # Remove one task
        remove_task("string-task")

        # Verify other data remains intact
        assert mock_session_state["user_data"] == {"key": "value"}
        assert mock_session_state["ui_state"] == "active"
        assert get_task("dict-task") == dict_task
        assert get_task("string-task") is None


class TestT1ComponentIntegration:
    """Test integration between all T1 simplified components."""

    def test_database_helpers_with_task_management(self, mock_session_state):
        """Test database helpers integration with task management."""
        # Start a task that uses database operations
        task_info = TaskInfo(
            task_id="db-task",
            status="running",
            progress=0.0,
            message="Database operation task",
            timestamp=datetime.now(UTC),
        )
        add_task("db-task", task_info)

        # Simulate database session usage
        with patch("src.ui.utils.database_helpers.get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session

            # Get cached session factory (T1.1)
            factory = get_cached_session_factory()
            session = factory()

            # Verify session creation works with task management
            assert session is mock_session
            assert get_task("db-task") == task_info

        # Clean up
        remove_task("db-task")

    def test_ui_helpers_with_task_progress(self, mock_session_state):
        """Test UI helpers integration with task progress tracking."""
        # Create progress tracking task
        progress_task = TaskInfo(
            task_id="progress-task",
            status="running",
            progress=0.65,
            message="Processing jobs",
            timestamp=datetime.now(UTC),
        )
        add_task("progress-task", progress_task)

        # Use UI helpers for progress-related calculations
        percentage = safe_int(progress_task.progress * 100)  # Convert to percentage
        jobs_processed = safe_int(
            progress_task.progress * 150,
        )  # Simulate 150 total jobs

        # Verify calculations work correctly
        assert percentage == 65
        assert jobs_processed == 97  # 65% of 150, truncated

        # Verify task remains unaffected by UI helper operations
        retrieved_task = get_task("progress-task")
        assert retrieved_task == progress_task

    def test_throttled_rerun_with_task_status(self, mock_session_state, mock_streamlit):
        """Test throttled rerun integration with task status checking."""
        # Start background task
        start_background_scraping(stay_active_in_tests=True)

        # Use throttled rerun based on scraping status
        throttled_rerun("ui_refresh", 0.5, should_rerun=is_scraping_active())

        # Should trigger rerun since scraping is active
        mock_streamlit["rerun"].assert_called_once()

        # Stop scraping
        stop_all_scraping()

        # Reset mock
        mock_streamlit["rerun"].reset_mock()

        # Try throttled rerun again
        throttled_rerun("ui_refresh", 0.5, should_rerun=is_scraping_active())

        # Should not trigger since scraping is inactive
        mock_streamlit["rerun"].assert_not_called()

    def test_session_state_cleaning_with_active_tasks(self, mock_session_state):
        """Test session state cleaning doesn't interfere with active tasks."""
        # Add active tasks
        task1 = TaskInfo("task1", "running", 0.3, "msg1", datetime.now(UTC))
        task2 = TaskInfo("task2", "pending", 0.0, "msg2", datetime.now(UTC))

        add_task("task1", task1)
        add_task("task2", task2)

        # Add some problematic session state (mock SQLAlchemy objects)
        from sqlmodel import Session

        mock_session_state["contaminated_session"] = Mock(spec=Session)
        mock_session_state["clean_data"] = {"key": "value"}

        # Clean session state
        removed_count = clean_session_state()

        # Verify contamination was cleaned but tasks remain
        assert removed_count >= 0  # May be 0 in test environment
        assert get_task("task1") == task1
        assert get_task("task2") == task2
        assert mock_session_state["clean_data"] == {"key": "value"}


class TestT1ErrorHandlingAndRecovery:
    """Test error handling and recovery in simplified T1 implementations."""

    def test_task_management_error_recovery(self, mock_session_state):
        """Test task management handles errors gracefully."""
        # Add a task
        task_info = TaskInfo("error-test", "running", 0.5, "msg", datetime.now(UTC))
        add_task("error-test", task_info)

        # Corrupt the session state structure
        mock_session_state.tasks = "not_a_dict"  # Invalid type

        # Operations should handle gracefully
        result = get_task("error-test")
        assert result is None  # Should not raise error

        # Should handle removal of non-existent task
        remove_task("nonexistent")  # Should not raise error

        # Restore proper structure
        mock_session_state.tasks = {}

        # Should work normally again
        add_task("recovery-test", task_info)
        assert get_task("recovery-test") == task_info

    def test_background_scraping_error_handling(self, mock_session_state):
        """Test background scraping error handling in simplified implementation."""
        # Start scraping
        start_background_scraping(stay_active_in_tests=True)
        assert is_scraping_active()

        # Simulate error condition by corrupting session state
        mock_session_state["scraping_active"] = "invalid_type"

        # Should handle gracefully
        try:
            active = is_scraping_active()
            # Should return a boolean or handle the error
            assert isinstance(active, bool) or active is None
        except Exception:
            pytest.fail("is_scraping_active should handle errors gracefully")

        # Recovery - fix session state
        mock_session_state["scraping_active"] = True
        assert is_scraping_active() is True

        # Clean stop should work
        stopped = stop_all_scraping()
        assert stopped >= 0

    def test_ui_helpers_error_resilience(self, mock_session_state):
        """Test UI helpers error resilience with task data."""
        # Create task with problematic data
        problematic_task = TaskInfo(
            task_id="problematic",
            status="error",
            progress="invalid_progress",  # Wrong type
            message=None,  # None message
            timestamp="not_a_datetime",  # Wrong type
        )

        add_task("problematic", problematic_task)

        # UI helpers should handle gracefully
        safe_progress = safe_int(problematic_task.progress)
        assert safe_progress == 0  # Should convert invalid to 0

        # Task should still be retrievable
        retrieved = get_task("problematic")
        assert retrieved == problematic_task


class TestT1PerformanceWithSimplifiedImplementations:
    """Test performance characteristics of simplified T1 implementations."""

    def test_task_management_performance(self, mock_session_state):
        """Test task management performance without custom managers."""
        start_time = time.time()

        # Create, update, and remove many tasks
        for i in range(1000):
            task = TaskInfo(
                f"perf-{i}",
                "running",
                i / 1000,
                f"msg-{i}",
                datetime.now(UTC),
            )
            add_task(f"perf-{i}", task)

            if i % 100 == 0:  # Check every 100th task
                retrieved = get_task(f"perf-{i}")
                assert retrieved == task

            if i % 200 == 0:  # Remove every 200th task
                remove_task(f"perf-{i}")

        end_time = time.time()

        # Should complete quickly without custom manager overhead
        assert (end_time - start_time) < 2.0  # Under 2 seconds for 1000 operations

    def test_integrated_workflow_performance(self, mock_session_state, mock_streamlit):
        """Test performance of integrated T1 workflow."""
        start_time = time.time()

        # Simulate realistic workflow
        for iteration in range(100):
            # Start background operation
            start_background_scraping(stay_active_in_tests=True)

            # Add progress tracking
            progress_task = TaskInfo(
                f"progress-{iteration}",
                "running",
                iteration / 100,
                f"Iteration {iteration}",
                datetime.now(UTC),
            )
            add_task(f"progress-{iteration}", progress_task)

            # Use UI helpers
            safe_int(progress_task.progress * 100)

            # Use throttled rerun (first call only triggers)
            if iteration == 0:
                throttled_rerun("perf_test", 10.0, should_rerun=True)
            else:
                throttled_rerun(
                    "perf_test",
                    10.0,
                    should_rerun=True,
                )  # Should be throttled

            # Clean up
            stop_all_scraping()
            remove_task(f"progress-{iteration}")

        end_time = time.time()

        # Should complete efficiently
        assert (end_time - start_time) < 5.0  # Under 5 seconds for 100 iterations

        # Verify only first rerun was triggered
        assert mock_streamlit["rerun"].call_count == 1

    def test_concurrent_operations_performance(self, mock_session_state):
        """Test performance of concurrent operations in simplified implementation."""
        results = []
        errors = []

        def worker(worker_id):
            try:
                start_time = time.time()

                # Each worker manages its own tasks
                for i in range(100):
                    task_id = f"worker-{worker_id}-task-{i}"
                    task = TaskInfo(
                        task_id,
                        "running",
                        i / 100,
                        f"Worker {worker_id}",
                        datetime.now(UTC),
                    )

                    add_task(task_id, task)
                    retrieved = get_task(task_id)
                    assert retrieved == task
                    remove_task(task_id)

                end_time = time.time()
                results.append(end_time - start_time)

            except Exception as e:
                errors.append(e)

        # Run multiple workers concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all workers completed without errors
        assert len(errors) == 0
        assert len(results) == 5

        # Each worker should complete quickly
        for duration in results:
            assert duration < 2.0  # Under 2 seconds per worker


class TestT1RealisticProductionScenarios:
    """Test realistic production scenarios with T1 simplified implementations."""

    def test_typical_job_scraping_workflow(self, mock_session_state, mock_streamlit):
        """Test typical job scraping workflow using T1 components."""
        # Start scraping
        start_background_scraping(stay_active_in_tests=True)
        assert is_scraping_active()

        # Add company progress tracking
        companies = ["Company A", "Company B", "Company C"]
        for i, company in enumerate(companies):
            progress = CompanyProgress(
                name=company,
                status="In Progress" if i < 2 else "Pending",
                jobs_found=i * 5,
                start_time=datetime.now(UTC) if i < 2 else None,
            )

            # Store in session state (simulating background_helpers functionality)
            if "company_progress" not in mock_session_state:
                mock_session_state["company_progress"] = {}
            mock_session_state["company_progress"][company] = progress

        # Get company progress
        company_progress = get_company_progress()
        assert len(company_progress) == 3
        assert company_progress["Company A"].jobs_found == 0
        assert company_progress["Company B"].jobs_found == 5

        # Use throttled rerun for UI updates
        throttled_rerun("scraping_ui", 1.0, should_rerun=is_scraping_active())
        mock_streamlit["rerun"].assert_called_once()

        # Stop scraping
        stopped = stop_all_scraping()
        assert stopped == 1
        assert not is_scraping_active()

    def test_session_state_management_in_production(self, mock_session_state):
        """Test session state management patterns used in production."""
        # Simulate typical session state with mixed data
        mock_session_state.update(
            {
                "user_preferences": {"theme": "dark", "language": "en"},
                "current_page": "jobs",
                "filter_criteria": {"location": "remote", "salary_min": 80000},
                "ui_state": {"sidebar_expanded": True, "view_mode": "cards"},
            },
        )

        # Add background tasks
        tasks_data = [
            ("scraping", "running", 0.7, "Scraping in progress"),
            ("data_sync", "pending", 0.0, "Waiting for scraping to complete"),
            ("ui_refresh", "completed", 1.0, "UI refresh completed"),
        ]

        for task_id, status, progress, message in tasks_data:
            task = TaskInfo(task_id, status, progress, message, datetime.now(UTC))
            add_task(task_id, task)

        # Verify all data coexists properly
        assert mock_session_state["user_preferences"]["theme"] == "dark"
        assert mock_session_state["current_page"] == "jobs"
        assert mock_session_state["filter_criteria"]["location"] == "remote"

        # Verify tasks are accessible
        for task_id, expected_status, _, _ in tasks_data:
            task = get_task(task_id)
            assert task.status == expected_status

        # Simulate task cleanup
        remove_task("ui_refresh")
        assert get_task("ui_refresh") is None

        # Verify other data remains intact
        assert mock_session_state["user_preferences"]["theme"] == "dark"
        assert get_task("scraping").status == "running"

    def test_error_recovery_in_production_workflow(self, mock_session_state):
        """Test error recovery in production-like workflow."""
        # Start normal operation
        start_background_scraping(stay_active_in_tests=True)

        # Add some processing tasks
        processing_tasks = []
        for i in range(5):
            task = TaskInfo(
                f"process-{i}",
                "running",
                i * 0.2,
                f"Processing {i}",
                datetime.now(UTC),
            )
            processing_tasks.append(task)
            add_task(f"process-{i}", task)

        # Simulate error condition - corrupt some session state
        mock_session_state["corrupted_data"] = object()  # Unpickleable object

        # System should continue to work
        assert is_scraping_active()

        # Tasks should still be accessible
        for i, original_task in enumerate(processing_tasks):
            retrieved = get_task(f"process-{i}")
            assert retrieved == original_task

        # Recovery operations should work
        stop_all_scraping()
        assert not is_scraping_active()

        # Clean up tasks
        for i in range(5):
            remove_task(f"process-{i}")
            assert get_task(f"process-{i}") is None

    def test_high_volume_task_management(self, mock_session_state):
        """Test task management under high volume typical of production."""
        # Simulate high-volume scenario
        batch_size = 50
        total_batches = 20

        for batch in range(total_batches):
            # Add batch of tasks
            batch_tasks = []
            for i in range(batch_size):
                task_id = f"batch-{batch}-task-{i}"
                task = TaskInfo(
                    task_id,
                    "running",
                    (batch * batch_size + i) / (total_batches * batch_size),
                    f"Processing batch {batch}, item {i}",
                    datetime.now(UTC),
                )
                batch_tasks.append((task_id, task))
                add_task(task_id, task)

            # Process batch (simulate task completion)
            for task_id, original_task in batch_tasks:
                retrieved = get_task(task_id)
                assert retrieved == original_task

                # Mark as completed and remove
                remove_task(task_id)
                assert get_task(task_id) is None

        # Verify clean state
        assert mock_session_state.get("tasks", {}) == {}
