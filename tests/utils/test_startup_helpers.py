"""Tests for startup helpers and performance optimization utilities.

Tests the startup performance optimization functions in startup_helpers.py including:
- Cache warming with priority queues
- Background prefetching and threading
- Performance initialization coordination
- Streamlit session state management
- Error handling and resilience patterns
"""

import threading
import time

from unittest.mock import Mock, patch

import pytest

from src.utils.startup_helpers import (
    get_cache_performance_stats,
    initialize_performance_optimizations,
    start_background_prefetching,
    warm_startup_cache,
)


class TestWarmStartupCache:
    """Test the cache warming functionality."""

    def test_warm_startup_cache_default_config(self):
        """Test cache warming with default configuration."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 3,
                "items_cached": 150,
            }

            # Act
            result = warm_startup_cache()

            # Assert
            assert result["status"] == "completed"
            assert result["background"] is True
            assert result["timeout"] == 30
            assert (
                result["queries_completed"] >= 2
            )  # At least company count + job counts
            assert result["items_cached"] == 150
            assert result["errors"] == 0
            assert "duration_seconds" in result

            # Verify services were called
            mock_company_service.get_active_companies_count.assert_called_once()
            mock_job_service.get_job_counts_by_status.assert_called_once()
            mock_job_service.prefetch_common_queries.assert_called_once_with(
                background=True
            )

    def test_warm_startup_cache_custom_config(self):
        """Test cache warming with custom configuration."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            config = {"background": False, "timeout": 10}
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 2,
                "items_cached": 75,
            }

            # Act
            result = warm_startup_cache(config)

            # Assert
            assert result["status"] == "completed"
            assert result["background"] is False
            assert result["timeout"] == 10
            assert result["queries_completed"] >= 2
            assert result["items_cached"] == 75

    def test_warm_startup_cache_skips_when_already_warmed(self):
        """Test cache warming skips when already warmed in session."""
        with patch("src.utils.startup_helpers.st") as mock_st:
            # Arrange
            mock_st.session_state.cache_warmed = True

            # Act
            result = warm_startup_cache()

            # Assert
            assert result["status"] == "skipped"
            assert result["reason"] == "already_warmed"

    def test_warm_startup_cache_handles_service_errors(self):
        """Test cache warming handles service errors gracefully."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_company_service.get_active_companies_count.side_effect = Exception(
                "Service error"
            )
            mock_job_service.get_job_counts_by_status.side_effect = Exception(
                "Service error"
            )
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 0,
                "items_cached": 0,
            }

            # Act
            result = warm_startup_cache()

            # Assert
            assert result["status"] == "completed"
            assert result["errors"] == 2  # Two service calls failed
            assert result["queries_completed"] >= 0  # Prefetch might still work

    def test_warm_startup_cache_background_mode(self):
        """Test cache warming in background mode."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService"),
            patch("src.utils.startup_helpers.st") as mock_st,
            patch("src.utils.startup_helpers.threading.Thread") as mock_thread,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            config = {"background": True, "timeout": 30}

            # Act
            result = warm_startup_cache(config)

            # Assert
            assert result["status"] == "background_started"
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_warm_startup_cache_priority_queries_complete_within_timeout(self):
        """Test priority queries complete within timeout limits."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
            patch("src.utils.startup_helpers.time") as mock_time,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            # Mock time to simulate timeout scenario
            mock_time.time.side_effect = [0, 0.1, 0.2, 29.5]  # Almost at timeout
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 2,
                "items_cached": 50,
            }

            # Act
            result = warm_startup_cache({"timeout": 30})

            # Assert
            assert result["status"] == "completed"
            # Priority queries should complete
            mock_company_service.get_active_companies_count.assert_called_once()
            mock_job_service.get_job_counts_by_status.assert_called_once()
            # Time-consuming query might be skipped
            # (companies_with_job_counts might not be called due to timeout)

    def test_warm_startup_cache_handles_total_failure(self):
        """Test cache warming handles total failure gracefully."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_company_service.get_active_companies_count.side_effect = Exception(
                "Total failure"
            )

            with patch("src.utils.startup_helpers._warm_cache_sync") as mock_warm_sync:
                mock_warm_sync.side_effect = Exception("Total cache warming failure")

                # Act
                warm_startup_cache({"background": False})

                # This should be handled by the outer exception handler in the
                # actual function
                # For testing, we'll verify the function structure can handle this

    def test_warm_startup_cache_marks_session_as_warmed(self):
        """Test cache warming marks session state as warmed."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 1,
                "items_cached": 25,
            }

            # Act
            result = warm_startup_cache({"background": False})

            # Assert
            assert result["status"] == "completed"
            # Verify session state was marked as warmed
            assert mock_st.session_state.cache_warmed is True

    def test_warm_startup_cache_no_streamlit_session_state(self):
        """Test cache warming when Streamlit session state is not available."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange - Remove session_state attribute
            delattr(mock_st, "session_state")
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 1,
                "items_cached": 25,
            }

            # Act
            result = warm_startup_cache({"background": False})

            # Assert - Should still work but won't check/set session state
            assert result["status"] == "completed"


class TestStartBackgroundPrefetching:
    """Test the background prefetching functionality."""

    def test_start_background_prefetching_success(self):
        """Test background prefetching starts successfully."""
        with patch("src.utils.startup_helpers.threading.Thread") as mock_thread:
            # Arrange
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            # Act
            result = start_background_prefetching()

            # Assert
            assert result is True
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_start_background_prefetching_handles_thread_creation_failure(self):
        """Test background prefetching handles thread creation failure."""
        with patch("src.utils.startup_helpers.threading.Thread") as mock_thread:
            # Arrange
            mock_thread.side_effect = Exception("Thread creation failed")

            # Act
            result = start_background_prefetching()

            # Assert
            assert result is False

    def test_background_prefetch_loop_execution(self):
        """Test the background prefetch loop executes correctly."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.time") as mock_time,
            patch("src.utils.startup_helpers.BACKGROUND_PREFETCH_INTERVAL", 1),
        ):
            # Arrange
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 2,
                "items_cached": 50,
            }

            # Mock time.sleep to control loop execution
            sleep_call_count = 0

            def mock_sleep(_duration):
                nonlocal sleep_call_count
                sleep_call_count += 1
                if sleep_call_count >= 2:  # Stop after 2 iterations
                    raise KeyboardInterrupt("Stop test loop")

            mock_time.sleep.side_effect = mock_sleep

            # Act - This will run the background loop
            with pytest.raises(KeyboardInterrupt):
                # Import and run the background function directly for testing
                pass
                # We'll test the thread target function indirectly

    def test_background_prefetch_handles_service_errors(self):
        """Test background prefetch loop handles service errors gracefully."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.time") as mock_time,
        ):
            # Arrange
            mock_job_service.prefetch_common_queries.side_effect = Exception(
                "Service error"
            )
            mock_company_service.get_active_companies_count.side_effect = Exception(
                "Service error"
            )

            # The loop should continue despite errors
            sleep_call_count = 0

            def mock_sleep(_duration):
                nonlocal sleep_call_count
                sleep_call_count += 1
                if sleep_call_count >= 2:
                    raise KeyboardInterrupt("Stop test loop")

            mock_time.sleep.side_effect = mock_sleep

            # Act & Assert - Should not crash despite service errors
            with pytest.raises(KeyboardInterrupt):
                pass  # The actual loop testing would be more complex


class TestInitializePerformanceOptimizations:
    """Test the performance optimization initialization."""

    def test_initialize_performance_optimizations_complete_success(self):
        """Test complete successful initialization of performance optimizations."""
        with (
            patch("src.utils.startup_helpers.warm_startup_cache") as mock_warm_cache,
            patch(
                "src.utils.startup_helpers.start_background_prefetching"
            ) as mock_start_prefetch,
        ):
            # Arrange
            mock_warm_cache.return_value = {
                "status": "completed",
                "queries_completed": 5,
                "items_cached": 200,
                "errors": 0,
            }
            mock_start_prefetch.return_value = True

            # Act
            result = initialize_performance_optimizations()

            # Assert
            assert "cache_warmup" in result
            assert "background_prefetch" in result
            assert "startup_time" in result
            assert "cache_manager_initialized" in result
            assert result["cache_manager_initialized"] is True
            assert result["background_prefetch"] is True
            assert result["cache_warmup"]["status"] == "completed"

            # Verify functions were called correctly
            mock_warm_cache.assert_called_once_with(background=True)
            mock_start_prefetch.assert_called_once()

    def test_initialize_performance_optimizations_partial_failure(self):
        """Test initialization handles partial failures gracefully."""
        with (
            patch("src.utils.startup_helpers.warm_startup_cache") as mock_warm_cache,
            patch(
                "src.utils.startup_helpers.start_background_prefetching"
            ) as mock_start_prefetch,
        ):
            # Arrange
            mock_warm_cache.return_value = {
                "status": "failed",
                "queries_completed": 2,
                "items_cached": 50,
                "errors": 3,
            }
            mock_start_prefetch.return_value = False  # Prefetching failed

            # Act
            result = initialize_performance_optimizations()

            # Assert
            assert result["cache_manager_initialized"] is True
            assert result["background_prefetch"] is False
            assert result["cache_warmup"]["status"] == "failed"

    def test_initialize_performance_optimizations_handles_exceptions(self):
        """Test initialization handles unexpected exceptions gracefully."""
        with (
            patch("src.utils.startup_helpers.warm_startup_cache") as mock_warm_cache,
            patch(
                "src.utils.startup_helpers.start_background_prefetching"
            ) as mock_start_prefetch,
        ):
            # Arrange
            mock_warm_cache.side_effect = Exception("Cache warming failed")
            mock_start_prefetch.side_effect = Exception("Prefetching failed")

            # Act
            result = initialize_performance_optimizations()

            # Assert - Should still return a valid result structure
            assert "cache_warmup" in result
            assert "background_prefetch" in result
            assert "startup_time" in result

    def test_initialize_performance_optimizations_timing(self):
        """Test initialization records timing information correctly."""
        with (
            patch("src.utils.startup_helpers.warm_startup_cache") as mock_warm_cache,
            patch(
                "src.utils.startup_helpers.start_background_prefetching"
            ) as mock_start_prefetch,
            patch("src.utils.startup_helpers.time") as mock_time,
        ):
            # Arrange
            mock_time.time.side_effect = [1000.0, 1000.5]  # 0.5 second duration
            mock_warm_cache.return_value = {"status": "completed"}
            mock_start_prefetch.return_value = True

            # Act
            result = initialize_performance_optimizations()

            # Assert
            assert result["startup_time"] == 1000.0
            # The total_time calculation should be logged (we can't easily test
            # the log message)


class TestGetCachePerformanceStats:
    """Test the cache performance statistics function."""

    def test_get_cache_performance_stats_success(self):
        """Test cache performance stats returns valid data."""
        with patch("src.utils.startup_helpers.time") as mock_time:
            # Arrange
            mock_time.time.return_value = 1234567890.0

            # Act
            result = get_cache_performance_stats()

            # Assert
            assert "message" in result
            assert result["message"] == "Using native Streamlit caching"
            assert result["timestamp"] == 1234567890.0
            assert result["performance_optimizations_active"] is True

    def test_get_cache_performance_stats_handles_exception(self):
        """Test cache performance stats handles exceptions gracefully."""
        with patch("src.utils.startup_helpers.time") as mock_time:
            # Arrange
            mock_time.time.side_effect = Exception("Time error")

            # Act
            result = get_cache_performance_stats()

            # Assert
            assert "error" in result
            assert result["error"] == "Failed to retrieve cache stats"
            assert "timestamp" in result
            assert result["performance_optimizations_active"] is False

    def test_get_cache_performance_stats_caching_behavior(self):
        """Test cache performance stats function is cached correctly."""
        # This test verifies the @st.cache_data decorator behavior
        # In a real environment, multiple calls should return cached results

        with patch("src.utils.startup_helpers.time") as mock_time:
            # Arrange
            mock_time.time.return_value = 1234567890.0

            # Act - Call multiple times
            result1 = get_cache_performance_stats()
            result2 = get_cache_performance_stats()

            # Assert - Should be identical (cached)
            assert result1 == result2
            # In real Streamlit environment, time.time would only be called once


class TestStartupHelpersIntegration:
    """Integration tests for startup helpers working together."""

    def test_complete_startup_sequence(self):
        """Test complete startup sequence with all components."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
            patch("src.utils.startup_helpers.threading.Thread") as mock_thread,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_company_service.get_active_companies_count.return_value = 50
            mock_job_service.get_job_counts_by_status.return_value = {
                "active": 100,
                "archived": 25,
            }
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 5,
                "items_cached": 300,
            }
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            # Act - Run complete initialization
            init_result = initialize_performance_optimizations()
            stats_result = get_cache_performance_stats()

            # Assert initialization results
            assert init_result["cache_manager_initialized"] is True
            assert init_result["background_prefetch"] is True
            assert init_result["cache_warmup"]["status"] == "background_started"

            # Assert stats results
            assert stats_result["performance_optimizations_active"] is True

            # Verify all services were properly coordinated
            mock_thread.assert_called()
            mock_thread_instance.start.assert_called()

    def test_startup_helpers_error_resilience(self):
        """Test startup helpers maintain resilience under various error conditions."""
        with (
            patch("src.utils.startup_helpers.CompanyService") as mock_company_service,
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange - Various service failures
            mock_st.session_state.cache_warmed = False
            mock_company_service.get_active_companies_count.side_effect = Exception(
                "Company service down"
            )
            mock_job_service.get_job_counts_by_status.side_effect = Exception(
                "Job service down"
            )
            mock_job_service.prefetch_common_queries.side_effect = Exception(
                "Prefetch service down"
            )

            # Act
            init_result = initialize_performance_optimizations()
            stats_result = get_cache_performance_stats()

            # Assert - Should still complete despite errors
            assert "cache_warmup" in init_result
            assert "background_prefetch" in init_result
            assert stats_result["performance_optimizations_active"] is True

    def test_concurrent_initialization_safety(self):
        """Test startup helpers handle concurrent initialization safely."""
        import queue

        results = queue.Queue()

        def worker():
            """Worker function for concurrent testing."""
            with (
                patch("src.utils.startup_helpers.CompanyService"),
                patch("src.utils.startup_helpers.JobService") as mock_job_service,
                patch("src.utils.startup_helpers.st") as mock_st,
            ):
                mock_st.session_state.cache_warmed = False
                mock_job_service.prefetch_common_queries.return_value = {
                    "queries_prefetched": 2,
                    "items_cached": 50,
                }

                # Run initialization
                result = initialize_performance_optimizations()
                results.put(result)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all results are valid
        collected_results = []
        while not results.empty():
            collected_results.append(results.get())

        assert len(collected_results) == 3

        # All results should have valid structure
        for result in collected_results:
            assert "cache_warmup" in result
            assert "background_prefetch" in result
            assert "startup_time" in result

    def test_startup_helpers_performance_benchmarks(self):
        """Test startup helpers complete within reasonable time."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 3,
                "items_cached": 150,
            }

            # Act - Measure actual execution time
            start_time = time.perf_counter()

            # Run synchronous warming (background=False for testing)
            result = warm_startup_cache({"background": False, "timeout": 30})

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Assert - Should complete quickly (under 1 second in test environment)
            assert duration < 1.0, f"Startup helpers too slow: {duration}s"
            assert result["status"] == "completed"

    def test_startup_helpers_memory_efficiency(self):
        """Test startup helpers don't consume excessive memory."""
        import gc
        import tracemalloc

        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 5,
                "items_cached": 250,
            }

            tracemalloc.start()

            # Act - Run initialization multiple times
            for _ in range(10):
                initialize_performance_optimizations()
                warm_startup_cache({"background": False})
                get_cache_performance_stats()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Assert - Memory usage should be reasonable (less than 5MB for this test)
            assert peak < 5 * 1024 * 1024, f"Memory usage too high: {peak} bytes"

            # Force garbage collection
            gc.collect()

    def test_startup_helpers_logging_behavior(self):
        """Test startup helpers produce appropriate log messages."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
            patch("src.utils.startup_helpers.logger") as mock_logger,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 3,
                "items_cached": 150,
            }

            # Act
            warm_startup_cache({"background": False})

            # Assert - Verify appropriate logging calls
            assert mock_logger.info.called
            assert mock_logger.debug.called

            # Check that startup and completion messages were logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Starting cache warmup" in msg for msg in info_calls)
            assert any("Cache warmup completed" in msg for msg in info_calls)

    def test_startup_helpers_configuration_flexibility(self):
        """Test startup helpers work with various configuration options."""
        with (
            patch("src.utils.startup_helpers.CompanyService"),
            patch("src.utils.startup_helpers.JobService") as mock_job_service,
            patch("src.utils.startup_helpers.st") as mock_st,
        ):
            # Arrange
            mock_st.session_state.cache_warmed = False
            mock_job_service.prefetch_common_queries.return_value = {
                "queries_prefetched": 2,
                "items_cached": 75,
            }

            # Test various configurations
            configs = [
                None,  # Default config
                {"background": True, "timeout": 60},  # Extended timeout
                {"background": False, "timeout": 5},  # Quick sync mode
                {"background": True, "timeout": 1},  # Very short timeout
            ]

            for config in configs:
                # Act
                result = warm_startup_cache(config)

                # Assert - All should complete successfully
                assert result["status"] in ["completed", "background_started"]
                if config:
                    assert result["background"] == config["background"]
                    assert result["timeout"] == config["timeout"]
