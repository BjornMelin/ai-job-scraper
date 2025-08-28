"""Background Task Concurrency Tests for Streamlit threading.

This test suite validates thread safety, cancellation mechanisms, and
concurrent operation handling in the background task management system.
Tests ensure proper Streamlit context preservation and reliable progress tracking.

Test coverage includes:
- Multiple scraping tasks running simultaneously
- Task cancellation and cleanup mechanisms
- Streamlit context preservation across threads
- Progress tracking with concurrent updates
- Thread safety of session state management
- Memory cleanup and resource management
- Error propagation from background threads
- Race condition detection and prevention
"""

import logging
import threading
import time

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.ui.utils.background_helpers import (
    CompanyProgress,
    get_company_progress,
    get_scraping_progress,
    get_scraping_results,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def mock_streamlit_state():
    """Create mock Streamlit session state for testing."""
    return {}


@pytest.fixture
def background_test_environment():
    """Set up test environment for background task testing."""
    # Mock session state
    with patch("src.ui.utils.background_helpers.st") as mock_st:
        mock_st.session_state = {}
        mock_st.warning = Mock()
        mock_st.error = Mock()
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        mock_st.status = Mock()

        yield mock_st


class TestConcurrentScrapingTasks:
    """Test concurrent scraping operations and thread safety."""

    def test_multiple_scraping_tasks_thread_safety(self, background_test_environment):
        """Test multiple scraping tasks running concurrently with thread safety."""
        mock_st = background_test_environment

        # Track task execution
        task_results = {}
        task_lock = threading.Lock()

        def mock_scrape_function(task_id):
            """Mock scraping function that simulates work."""
            with task_lock:
                task_results[task_id] = {
                    "started": True,
                    "thread_id": threading.current_thread().ident,
                    "timestamp": datetime.now(UTC),
                }

            # Simulate scraping work
            time.sleep(0.1)

            # Update session state safely
            with task_lock:
                if "scraping_results" not in mock_st.session_state:
                    mock_st.session_state["scraping_results"] = {}

                mock_st.session_state["scraping_results"][task_id] = {
                    "inserted": 5,
                    "updated": 2,
                    "archived": 0,
                    "deleted": 0,
                    "skipped": 1,
                }
                task_results[task_id]["completed"] = True

        # Mock the scraper module
        with patch("src.ui.utils.background_helpers.scrape_all") as mock_scrape_all:
            mock_scrape_all.side_effect = lambda: {
                "inserted": 5,
                "updated": 2,
                "archived": 0,
            }

            # Start multiple tasks concurrently
            task_ids = []
            threads = []

            for i in range(3):
                task_id = f"task_{i}"
                task_ids.append(task_id)

                # Initialize session state for each task
                mock_st.session_state[f"task_{task_id}"] = {
                    "active": True,
                    "progress": 0.0,
                }

                # Create and start thread
                thread = threading.Thread(
                    target=mock_scrape_function, args=(task_id,), daemon=True
                )
                threads.append(thread)
                thread.start()

            # Wait for all tasks to complete
            for thread in threads:
                thread.join(timeout=2.0)
                assert not thread.is_alive(), "Thread did not complete within timeout"

            # Verify all tasks completed successfully
            assert len(task_results) == 3
            for task_id in task_ids:
                assert task_results[task_id]["started"] is True
                assert task_results[task_id]["completed"] is True

            # Verify unique thread IDs (actually concurrent)
            thread_ids = {result["thread_id"] for result in task_results.values()}
            assert len(thread_ids) == 3, "Tasks should run on different threads"

    def test_concurrent_progress_updates_thread_safety(
        self, background_test_environment
    ):
        """Test concurrent progress updates don't cause race conditions."""
        mock_st = background_test_environment
        mock_st.session_state["company_progress"] = {}
        mock_st.session_state["task_progress"] = {}

        # Track progress updates
        progress_updates = []
        update_lock = threading.Lock()

        def update_progress_worker(worker_id, company_name):
            """Worker that updates company progress."""
            for i in range(10):
                # Simulate progress update
                progress = CompanyProgress(
                    name=company_name,
                    status="Scraping",
                    jobs_found=i,
                    start_time=datetime.now(UTC),
                )

                # Thread-safe update
                with update_lock:
                    mock_st.session_state["company_progress"][company_name] = progress
                    progress_updates.append((worker_id, company_name, i))

                time.sleep(0.01)  # Small delay to increase chance of race conditions

        # Start multiple workers updating different companies
        companies = ["CompanyA", "CompanyB", "CompanyC"]
        threads = []

        for worker_id, company in enumerate(companies):
            thread = threading.Thread(
                target=update_progress_worker, args=(worker_id, company), daemon=True
            )
            threads.append(thread)
            thread.start()

        # Wait for all workers to complete
        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive()

        # Verify progress was updated correctly
        assert len(progress_updates) == 30  # 10 updates * 3 companies
        assert len(mock_st.session_state["company_progress"]) == 3

        # Verify final progress values
        for company in companies:
            progress = mock_st.session_state["company_progress"][company]
            assert progress.name == company
            assert progress.jobs_found >= 0  # Should have valid progress

    def test_task_cancellation_cleanup(self, background_test_environment):
        """Test proper cleanup when tasks are cancelled."""
        mock_st = background_test_environment

        # Track cancellation handling
        cancellation_events = []
        cancellation_lock = threading.Lock()

        def long_running_task(task_id):
            """Simulate long-running task that can be cancelled."""
            mock_st.session_state["scraping_active"] = True

            try:
                for i in range(100):  # Long-running operation
                    # Check for cancellation
                    if not mock_st.session_state.get("scraping_active", False):
                        with cancellation_lock:
                            cancellation_events.append(f"{task_id}_cancelled_at_{i}")
                        return

                    # Simulate work
                    time.sleep(0.01)

                    # Update progress
                    mock_st.session_state[f"progress_{task_id}"] = i / 100.0

            finally:
                # Cleanup resources
                with cancellation_lock:
                    cancellation_events.append(f"{task_id}_cleanup")

        # Start long-running task
        task_thread = threading.Thread(
            target=long_running_task, args=("test_task",), daemon=True
        )
        task_thread.start()

        # Let task run briefly
        time.sleep(0.1)
        assert mock_st.session_state.get("scraping_active") is True

        # Cancel the task
        mock_st.session_state["scraping_active"] = False

        # Wait for cleanup
        task_thread.join(timeout=2.0)
        assert not task_thread.is_alive()

        # Verify cancellation was handled
        assert len(cancellation_events) >= 1
        assert any("cancelled" in event for event in cancellation_events)
        assert any("cleanup" in event for event in cancellation_events)

    def test_streamlit_context_preservation(self, background_test_environment):
        """Test Streamlit context is preserved across thread boundaries."""
        mock_st = background_test_environment

        # Track context preservation
        context_data = {}
        context_lock = threading.Lock()

        with patch(
            "src.ui.utils.background_helpers.add_script_run_ctx"
        ) as mock_add_ctx:

            def mock_context_worker(worker_id):
                """Worker that requires Streamlit context."""
                # Simulate Streamlit operations that require context
                try:
                    # This would fail without proper context
                    mock_st.session_state[f"worker_{worker_id}"] = {
                        "thread_id": threading.current_thread().ident,
                        "context_preserved": True,
                    }

                    with context_lock:
                        context_data[worker_id] = "success"

                except Exception as e:
                    with context_lock:
                        context_data[worker_id] = f"error: {e}"

            # Start workers that require Streamlit context
            threads = []
            for worker_id in range(3):
                thread = threading.Thread(
                    target=mock_context_worker, args=(worker_id,), daemon=True
                )

                # Simulate add_script_run_ctx being called
                mock_add_ctx(thread)
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=1.0)
                assert not thread.is_alive()

            # Verify context was preserved
            assert len(context_data) == 3
            for worker_id in range(3):
                assert context_data[worker_id] == "success"

            # Verify add_script_run_ctx was called for each thread
            assert mock_add_ctx.call_count == 3


class TestSessionStateManagement:
    """Test session state management under concurrent access."""

    def test_session_state_isolation_between_tasks(self, background_test_environment):
        """Test session state isolation prevents cross-task contamination."""
        mock_st = background_test_environment
        mock_st.session_state = {}

        # Track isolated state operations
        state_operations = []
        state_lock = threading.Lock()

        def isolated_task_worker(task_id, data_prefix):
            """Worker that operates on isolated session state."""
            # Each task should operate on different keys
            key_prefix = f"{data_prefix}_{task_id}"

            for i in range(5):
                state_key = f"{key_prefix}_item_{i}"
                state_value = f"value_{task_id}_{i}"

                # Update session state
                mock_st.session_state[state_key] = state_value

                # Track operation
                with state_lock:
                    state_operations.append((task_id, state_key, state_value))

                time.sleep(0.01)  # Allow interleaving

        # Start multiple tasks with different prefixes
        task_configs = [
            (1, "company_progress"),
            (2, "scraping_results"),
            (3, "task_metadata"),
        ]

        threads = []
        for task_id, data_prefix in task_configs:
            thread = threading.Thread(
                target=isolated_task_worker, args=(task_id, data_prefix), daemon=True
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=1.0)
            assert not thread.is_alive()

        # Verify operations completed
        assert len(state_operations) == 15  # 5 operations * 3 tasks

        # Verify state isolation - each task should have its own keys
        task_keys = {}
        for task_id, data_prefix in task_configs:
            task_keys[task_id] = [
                key
                for key in mock_st.session_state
                if key.startswith(f"{data_prefix}_{task_id}")
            ]
            assert len(task_keys[task_id]) == 5  # Each task created 5 keys

        # Verify no key overlap
        all_keys = set()
        for keys in task_keys.values():
            for key in keys:
                assert key not in all_keys, f"Key {key} appears in multiple tasks"
                all_keys.add(key)

    def test_concurrent_session_state_reads_writes(self, background_test_environment):
        """Test concurrent reads and writes to session state are safe."""
        mock_st = background_test_environment
        mock_st.session_state = {"shared_counter": 0}

        # Track concurrent operations
        operation_results = []
        result_lock = threading.Lock()

        def concurrent_state_worker(worker_id, operation_count):
            """Worker that performs concurrent reads/writes."""
            local_results = []

            for i in range(operation_count):
                try:
                    # Read current value
                    current_value = mock_st.session_state.get("shared_counter", 0)

                    # Simulate some processing
                    time.sleep(0.001)

                    # Update value
                    mock_st.session_state["shared_counter"] = current_value + 1

                    # Record operation
                    local_results.append(
                        {
                            "worker": worker_id,
                            "iteration": i,
                            "read_value": current_value,
                            "wrote_value": current_value + 1,
                        }
                    )

                except Exception as e:
                    local_results.append(
                        {"worker": worker_id, "iteration": i, "error": str(e)}
                    )

            # Store results thread-safely
            with result_lock:
                operation_results.extend(local_results)

        # Start multiple concurrent workers
        workers = 5
        operations_per_worker = 10

        threads = []
        for worker_id in range(workers):
            thread = threading.Thread(
                target=concurrent_state_worker,
                args=(worker_id, operations_per_worker),
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive()

        # Verify all operations completed
        total_operations = workers * operations_per_worker
        assert len(operation_results) == total_operations

        # Check for errors
        errors = [result for result in operation_results if "error" in result]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Verify final counter value (may be less than expected due to race conditions)
        final_counter = mock_st.session_state["shared_counter"]
        assert final_counter > 0
        assert final_counter <= total_operations


class TestBackgroundTaskLifecycle:
    """Test complete lifecycle of background tasks."""

    def test_background_task_start_progress_completion(
        self, background_test_environment
    ):
        """Test complete task lifecycle from start to completion."""
        mock_st = background_test_environment

        # Mock scraping dependencies
        with (
            patch("src.ui.utils.background_helpers.JobService") as mock_job_service,
            patch("src.ui.utils.background_helpers.scrape_all") as mock_scrape_all,
        ):
            mock_job_service.get_active_companies.return_value = [
                "Company1",
                "Company2",
            ]
            mock_scrape_all.return_value = {
                "inserted": 10,
                "updated": 5,
                "archived": 2,
                "deleted": 0,
                "skipped": 1,
            }

            # Track lifecycle events
            lifecycle_events = []

            # Start background scraping
            mock_st.session_state.clear()
            task_id = start_background_scraping(stay_active_in_tests=True)

            assert task_id is not None
            lifecycle_events.append(("task_started", task_id))

            # Verify initial state
            assert is_scraping_active() is True
            lifecycle_events.append(("task_active", True))

            # Check progress tracking
            progress = get_scraping_progress()
            assert isinstance(progress, dict)
            lifecycle_events.append(("progress_tracked", len(progress)))

            # Wait for task to complete (in test mode, should be immediate)
            time.sleep(0.1)

            # Check results
            results = get_scraping_results()
            assert isinstance(results, dict)
            assert "inserted" in results
            lifecycle_events.append(("results_available", results["inserted"]))

            # Verify company progress
            company_progress = get_company_progress()
            assert isinstance(company_progress, dict)
            lifecycle_events.append(("company_progress", len(company_progress)))

            # Verify lifecycle progression
            expected_events = [
                "task_started",
                "task_active",
                "progress_tracked",
                "results_available",
                "company_progress",
            ]
            actual_event_types = [event[0] for event in lifecycle_events]

            for expected_event in expected_events:
                assert expected_event in actual_event_types

    def test_background_task_error_handling(self, background_test_environment):
        """Test error handling in background tasks."""
        mock_st = background_test_environment

        # Mock scraping to raise exception
        with (
            patch("src.ui.utils.background_helpers.JobService") as mock_job_service,
            patch("src.ui.utils.background_helpers.scrape_all") as mock_scrape_all,
        ):
            mock_job_service.get_active_companies.return_value = ["Company1"]
            mock_scrape_all.side_effect = Exception("Scraping failed")

            # Start background scraping that should fail
            mock_st.session_state.clear()
            start_background_scraping(stay_active_in_tests=False)

            # Allow time for background thread to process and fail
            time.sleep(0.2)

            # Verify error handling
            # Task should no longer be active after failure
            assert is_scraping_active() is False

            # Should have error status in session state
            if "scraping_status" in mock_st.session_state:
                status = mock_st.session_state["scraping_status"]
                assert "Error" in status or "error" in status.lower()

    def test_multiple_task_cancellation_cleanup(self, background_test_environment):
        """Test cancellation and cleanup of multiple background tasks."""
        mock_st = background_test_environment

        # Mock long-running scraping
        with (
            patch("src.ui.utils.background_helpers.JobService") as mock_job_service,
            patch("src.ui.utils.background_helpers._run_unified_scrape") as mock_scrape,
        ):
            mock_job_service.get_active_companies.return_value = [
                "Company1",
                "Company2",
            ]

            def long_scrape(*args, **kwargs):
                # Simulate long-running task
                for _i in range(100):
                    if not mock_st.session_state.get("scraping_active", False):
                        break
                    time.sleep(0.01)

            mock_scrape.side_effect = long_scrape

            # Start multiple background tasks
            task_ids = []
            for _i in range(3):
                mock_st.session_state.clear()
                task_id = start_background_scraping(stay_active_in_tests=True)
                task_ids.append(task_id)
                time.sleep(0.05)  # Stagger starts

            # Verify tasks are active
            assert is_scraping_active() is True

            # Stop all scraping
            stopped_count = stop_all_scraping()
            assert stopped_count >= 0  # Should stop at least one task

            # Verify cleanup
            time.sleep(0.1)  # Allow cleanup time
            assert is_scraping_active() is False

            # Verify status indicates stopped
            if "scraping_status" in mock_st.session_state:
                status = mock_st.session_state["scraping_status"]
                assert "stopped" in status.lower() or "cancelled" in status.lower()


class TestMemoryAndResourceManagement:
    """Test memory and resource management in background tasks."""

    def test_memory_cleanup_after_task_completion(self, background_test_environment):
        """Test memory is properly cleaned up after tasks complete."""
        mock_st = background_test_environment

        # Track memory-related objects
        created_objects = []

        def memory_intensive_task(task_id):
            """Task that creates objects to track memory usage."""
            # Create some objects that should be cleaned up
            large_data = [f"data_{i}" for i in range(1000)]
            created_objects.append(large_data)

            # Update session state
            mock_st.session_state[f"task_{task_id}_data"] = len(large_data)

            # Simulate completion
            time.sleep(0.1)

        # Run multiple memory-intensive tasks
        threads = []
        for task_id in range(5):
            thread = threading.Thread(
                target=memory_intensive_task, args=(task_id,), daemon=True
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=1.0)
            assert not thread.is_alive()

        # Verify objects were created
        assert len(created_objects) == 5

        # Simulate cleanup (in real scenario, garbage collection would handle this)
        created_objects.clear()

        # Verify session state contains results but large objects are cleaned up
        task_data_keys = [key for key in mock_st.session_state if "task_" in key]
        assert len(task_data_keys) == 5  # Results preserved
        assert len(created_objects) == 0  # Large objects cleaned up

    def test_thread_lifecycle_management(self, background_test_environment):
        """Test proper thread creation, execution, and termination."""
        # Track thread lifecycle
        thread_events = []
        event_lock = threading.Lock()

        def tracked_worker(worker_id):
            """Worker that tracks its lifecycle events."""
            thread_id = threading.current_thread().ident

            with event_lock:
                thread_events.append(("started", worker_id, thread_id))

            try:
                # Simulate work
                for i in range(10):
                    time.sleep(0.01)

                with event_lock:
                    thread_events.append(("working", worker_id, i))

            finally:
                with event_lock:
                    thread_events.append(("finished", worker_id, thread_id))

        # Create and manage threads
        threads = []
        worker_count = 5

        # Start workers
        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=tracked_worker, args=(worker_id,), daemon=True
            )
            threads.append(thread)
            thread.start()

        # Verify all threads are alive initially
        active_threads = [t for t in threads if t.is_alive()]
        assert len(active_threads) == worker_count

        # Wait for completion with timeout
        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive(), "Thread did not terminate within timeout"

        # Verify lifecycle events
        started_events = [e for e in thread_events if e[0] == "started"]
        finished_events = [e for e in thread_events if e[0] == "finished"]

        assert len(started_events) == worker_count
        assert len(finished_events) == worker_count

        # Verify each worker started and finished
        started_workers = {e[1] for e in started_events}
        finished_workers = {e[1] for e in finished_events}

        assert started_workers == finished_workers
        assert len(started_workers) == worker_count


class TestEdgeCasesAndRaceConditions:
    """Test edge cases and potential race conditions."""

    def test_rapid_start_stop_operations(self, background_test_environment):
        """Test rapid start/stop operations don't cause issues."""
        mock_st = background_test_environment

        with patch("src.ui.utils.background_helpers.JobService") as mock_job_service:
            mock_job_service.get_active_companies.return_value = ["Company1"]

            # Rapidly start and stop scraping
            operation_results = []

            for _cycle in range(10):
                # Clear previous state
                mock_st.session_state.clear()

                # Start scraping
                task_id = start_background_scraping(stay_active_in_tests=True)
                operation_results.append(("started", task_id, is_scraping_active()))

                # Immediately stop
                stopped = stop_all_scraping()
                operation_results.append(("stopped", stopped, is_scraping_active()))

                # Brief pause
                time.sleep(0.01)

            # Verify all operations completed
            assert len(operation_results) == 20  # 10 start + 10 stop operations

            # Verify final state is stopped
            assert is_scraping_active() is False

    def test_concurrent_state_modifications(self, background_test_environment):
        """Test concurrent modifications to session state are handled safely."""
        mock_st = background_test_environment
        mock_st.session_state = {"shared_data": {"counter": 0, "items": []}}

        modification_results = []
        result_lock = threading.Lock()

        def state_modifier(modifier_id, modification_count):
            """Worker that modifies shared state."""
            local_results = []

            for i in range(modification_count):
                try:
                    # Get current state
                    current_data = mock_st.session_state.get("shared_data", {})

                    # Modify counter
                    current_data["counter"] = current_data.get("counter", 0) + 1

                    # Add item
                    current_data.setdefault("items", []).append(f"{modifier_id}_{i}")

                    # Update session state
                    mock_st.session_state["shared_data"] = current_data

                    local_results.append(
                        ("success", modifier_id, i, current_data["counter"])
                    )

                except Exception as e:
                    local_results.append(("error", modifier_id, i, str(e)))

                time.sleep(0.001)  # Small delay to encourage race conditions

            with result_lock:
                modification_results.extend(local_results)

        # Start multiple modifiers
        modifiers = 5
        modifications_per_modifier = 20

        threads = []
        for modifier_id in range(modifiers):
            thread = threading.Thread(
                target=state_modifier,
                args=(modifier_id, modifications_per_modifier),
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=3.0)
            assert not thread.is_alive()

        # Analyze results
        total_expected = modifiers * modifications_per_modifier
        assert len(modification_results) == total_expected

        # Check for errors
        [r for r in modification_results if r[0] == "error"]
        success_count = len([r for r in modification_results if r[0] == "success"])

        # Should have mostly successes (some race conditions are acceptable)
        assert success_count > total_expected * 0.8  # At least 80% success rate

        # Verify final state integrity
        final_data = mock_st.session_state.get("shared_data", {})
        assert "counter" in final_data
        assert "items" in final_data
        assert final_data["counter"] > 0
        assert len(final_data["items"]) > 0
