"""Integration tests for Settings page functionality.

Tests cover:
- End-to-end settings workflow
- Background task integration with limits
- Session state integration
- Settings persistence across operations
"""

import time

from unittest import mock

from src.ui.pages.settings import load_settings, save_settings
from src.ui.utils.background_helpers import (
    get_company_progress,
    get_scraping_results,
    is_scraping_active,
    start_background_scraping,
)


class TestSettingsIntegration:
    """Integration tests for settings functionality."""

    def setup_method(self, method):
        """Reset session state before each test."""
        # Session state will be cleared by the fixture automatically

    def test_settings_to_scraping_workflow(self, mock_session_state):
        """Test complete workflow from settings to scraping."""
        # Step 1: Configure settings
        settings = {
            "llm_provider": "OpenAI",
            "max_jobs_per_company": 25,
        }
        save_settings(settings)

        # Verify settings are saved
        loaded_settings = load_settings()
        assert loaded_settings["max_jobs_per_company"] == 25

        # Step 2: Mock scraping with settings
        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["Tech Corp", "AI Startup"]

            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                mock_scrape_all.return_value = {
                    "Tech Corp": 25,
                    "AI Startup": 20,
                    "inserted": 45,
                    "updated": 0,
                    "archived": 0,
                    "deleted": 0,
                    "skipped": 0,
                }

                # Start scraping (this should use the settings)
                start_background_scraping()

                # Wait for background thread to process
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Verify scraping used the limit from settings
                mock_scrape_all.assert_called_once_with(25)

                # Verify results
                results = get_scraping_results()
                assert isinstance(results, dict)

    def test_settings_persistence_across_sessions(self, mock_session_state):
        """Test settings persist across different operations."""
        # Initial settings
        initial_settings = {
            "llm_provider": "Groq",
            "max_jobs_per_company": 75,
        }
        save_settings(initial_settings)

        # Simulate session activity
        mock_session_state["some_other_key"] = "some_value"
        mock_session_state["another_key"] = 123

        # Load settings again
        loaded_settings = load_settings()
        assert loaded_settings["llm_provider"] == "Groq"
        assert loaded_settings["max_jobs_per_company"] == 75

        # Update only one setting
        updated_settings = loaded_settings.copy()
        updated_settings["max_jobs_per_company"] = 100
        save_settings(updated_settings)

        # Verify update
        final_settings = load_settings()
        assert final_settings["llm_provider"] == "Groq"  # Unchanged
        assert final_settings["max_jobs_per_company"] == 100  # Updated

    def test_background_task_session_state_integration(self, mock_session_state):
        """Test background tasks properly read from session state."""
        # Set up job limit in session state
        mock_session_state["max_jobs_per_company"] = 30

        # Mock active companies
        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["Company A", "Company B"]

            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                mock_scrape_all.return_value = {
                    "Company A": 30,
                    "Company B": 25,
                }

                # Start background task
                start_background_scraping()

                # Wait for completion
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Verify the session state value was used
                mock_scrape_all.assert_called_once_with(30)

    def test_scraping_with_different_limits(self, mock_session_state):
        """Test scraping behavior with different job limits."""
        test_cases = [
            {"limit": 10, "expected": 10},
            {"limit": 50, "expected": 50},
            {"limit": 100, "expected": 100},
        ]

        for case in test_cases:
            # Set the limit - use the mock session state directly
            mock_session_state["max_jobs_per_company"] = case["limit"]

            with mock.patch(
                "src.services.job_service.JobService.get_active_companies",
            ) as mock_companies:
                mock_companies.return_value = ["Test Company"]

                with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                    mock_scrape_all.return_value = {"Test Company": case["limit"]}

                    # Start scraping
                    start_background_scraping()

                    # Wait for completion
                    timeout = 3
                    start_time = time.time()
                    while is_scraping_active() and (time.time() - start_time) < timeout:
                        time.sleep(0.1)

                    # Verify correct limit was passed
                    mock_scrape_all.assert_called_once_with(case["expected"])

    def test_company_progress_tracking_integration(self, mock_session_state):
        """Test company progress tracking with settings integration."""
        # Set up settings
        mock_session_state["max_jobs_per_company"] = 20

        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["Progress Corp", "Track Inc"]

            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                mock_scrape_all.return_value = {
                    "Progress Corp": 18,
                    "Track Inc": 20,
                }

                # Start scraping
                start_background_scraping()

                # Check progress during execution
                progress = get_company_progress()

                # Should have entries for both companies
                company_names = list(progress.keys())
                assert "Progress Corp" in company_names
                assert "Track Inc" in company_names

                # Wait for completion
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Check final progress
                final_progress = get_company_progress()
                for company_progress in final_progress.values():
                    assert company_progress.status in ["Completed", "Error"]
                    assert hasattr(company_progress, "jobs_found")

    def test_settings_validation_integration(self, mock_session_state):
        """Test settings validation in integration context."""
        # Test various edge cases (must include all required keys)
        edge_cases = [
            {"llm_provider": "OpenAI", "max_jobs_per_company": 0},  # Zero limit
            {"llm_provider": "Groq", "max_jobs_per_company": -5},  # Negative limit
            {
                "llm_provider": "OpenAI",
                "max_jobs_per_company": 1000,
            },  # Very large limit
        ]

        for case in edge_cases:
            # Edge case loop - session state handled by fixture

            # Save the edge case setting
            save_settings(case)

            # Load and verify
            loaded = load_settings()
            assert loaded["max_jobs_per_company"] == case["max_jobs_per_company"]

            # Test with background task
            with mock.patch(
                "src.services.job_service.JobService.get_active_companies",
            ) as mock_companies:
                mock_companies.return_value = ["Edge Case Corp"]

                with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                    mock_scrape_all.return_value = {}

                    start_background_scraping()

                    # Should not crash, even with edge case values
                    timeout = 3
                    start_time = time.time()
                    while is_scraping_active() and (time.time() - start_time) < timeout:
                        time.sleep(0.1)

                    # Verify the edge case value was handled correctly
                    # Invalid values (0, negative) replaced with default
                    expected_value = case["max_jobs_per_company"]
                    if expected_value < 1:  # Invalid values get replaced with default
                        from src.scraper_company_pages import (
                            DEFAULT_MAX_JOBS_PER_COMPANY,
                        )

                        expected_value = DEFAULT_MAX_JOBS_PER_COMPANY
                    mock_scrape_all.assert_called_once_with(expected_value)

    def test_concurrent_settings_operations(self, mock_session_state):
        """Test concurrent settings operations don't interfere."""
        # Set initial state
        save_settings({"max_jobs_per_company": 50})

        # Simulate concurrent operations
        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["Concurrent Corp"]

            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                mock_scrape_all.return_value = {"Concurrent Corp": 50}

                # Start background scraping
                start_background_scraping()

                # # In test environment, scraping runs synchronously, so update settings
                # # after
                save_settings({"max_jobs_per_company": 75})

                # Wait for scraping to complete (should be immediate in test mode)
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Verify the original value was used for scraping
                mock_scrape_all.assert_called_once_with(50)

                # Verify new setting is saved
                current_settings = load_settings()
                assert current_settings["max_jobs_per_company"] == 75

    def test_error_handling_integration(self, mock_session_state):
        """Test error handling in settings and scraping integration."""
        # Set up settings
        save_settings({"max_jobs_per_company": 40})

        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["Error Corp"]

            # Mock scraping to raise exception
            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                mock_scrape_all.side_effect = Exception("Scraping failed")

                # Start scraping
                start_background_scraping()

                # Wait for error handling
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Should not be active after error
                assert not is_scraping_active()

                # Settings should still be intact
                settings = load_settings()
                assert settings["max_jobs_per_company"] == 40

    def test_session_state_cleanup(self, mock_session_state):
        """Test session state is properly cleaned up after operations."""
        # Set initial state with various keys
        save_settings(
            {
                "llm_provider": "OpenAI",
                "max_jobs_per_company": 35,
            },
        )
        mock_session_state["temp_key"] = "temp_value"
        mock_session_state["another_temp"] = 123

        # Run scraping operation
        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["Cleanup Corp"]

            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                mock_scrape_all.return_value = {"Cleanup Corp": 35}

                start_background_scraping()

                # Wait for completion
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Settings should still be available
                settings = load_settings()
                assert settings["llm_provider"] == "OpenAI"
                assert settings["max_jobs_per_company"] == 35

                # Temporary keys should still exist (they're not cleaned)
                assert mock_session_state.get("temp_key") == "temp_value"
                assert mock_session_state.get("another_temp") == 123


class TestSettingsWithRealScenarios:
    """Test settings with realistic user scenarios."""

    def setup_method(self, method):
        """Reset session state before each test."""
        # Session state will be cleared by the fixture automatically

    def test_user_changes_limit_during_scraping(self, mock_session_state):
        """Test user changing job limit while scraping is active."""
        # Start with initial limit
        save_settings({"max_jobs_per_company": 30})

        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            mock_companies.return_value = ["User Corp"]

            with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                # Simulate slow scraping
                def slow_scrape(limit):
                    time.sleep(0.2)  # Small delay to simulate work
                    return {"User Corp": limit}

                mock_scrape_all.side_effect = slow_scrape

                # Start scraping
                start_background_scraping()

                # Immediately change settings (user action)
                save_settings({"max_jobs_per_company": 60})

                # Wait for scraping completion
                timeout = 5
                start_time = time.time()
                while is_scraping_active() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                # Scraping should have used original limit (30)
                mock_scrape_all.assert_called_once_with(30)

                # New setting should be saved for next time
                current_settings = load_settings()
                assert current_settings["max_jobs_per_company"] == 60

    def test_multiple_scraping_sessions(self, mock_session_state):
        """Test multiple scraping sessions with different settings."""
        sessions = [
            {"limit": 20, "companies": ["Session1 Corp"]},
            {"limit": 40, "companies": ["Session2 Corp", "Session2 Inc"]},
            {"limit": 15, "companies": ["Session3 Corp"]},
        ]

        for session in sessions:
            # Session loop - state handled by fixture

            # Set session-specific limit
            save_settings({"max_jobs_per_company": session["limit"]})

            with mock.patch(
                "src.services.job_service.JobService.get_active_companies",
            ) as mock_companies:
                mock_companies.return_value = session["companies"]

                with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                    mock_scrape_all.return_value = dict.fromkeys(
                        session["companies"],
                        session["limit"],
                    )

                    # Start scraping
                    start_background_scraping()

                    # Wait for completion
                    timeout = 5
                    start_time = time.time()
                    while is_scraping_active() and (time.time() - start_time) < timeout:
                        time.sleep(0.1)

                    # Verify correct limit was used
                    mock_scrape_all.assert_called_once_with(session["limit"])

                    # Verify results
                    results = get_scraping_results()
                    for company in session["companies"]:
                        assert company in results or len(results) > 0

    def test_settings_backup_and_restore(self, mock_session_state):
        """Test backing up and restoring settings."""
        # Original settings
        original_settings = {
            "llm_provider": "Groq",
            "max_jobs_per_company": 45,
        }
        save_settings(original_settings)

        # Backup settings
        backup = load_settings()

        # Change settings
        changed_settings = {
            "llm_provider": "OpenAI",
            "max_jobs_per_company": 80,
        }
        save_settings(changed_settings)

        # Verify changes
        current = load_settings()
        assert current["llm_provider"] == "OpenAI"
        assert current["max_jobs_per_company"] == 80

        # Restore from backup
        save_settings(backup)

        # Verify restoration
        restored = load_settings()
        assert restored["llm_provider"] == "Groq"
        assert restored["max_jobs_per_company"] == 45

    def test_settings_with_no_active_companies(self, mock_session_state):
        """Test settings behavior when no companies are active."""
        save_settings({"max_jobs_per_company": 25})

        with mock.patch(
            "src.services.job_service.JobService.get_active_companies",
        ) as mock_companies:
            # No active companies
            mock_companies.return_value = []

            # Start scraping
            start_background_scraping()

            # Wait for quick completion
            timeout = 3
            start_time = time.time()
            while is_scraping_active() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            # Should complete quickly with no work
            assert not is_scraping_active()

            # Settings should remain unchanged
            settings = load_settings()
            assert settings["max_jobs_per_company"] == 25

    def test_extreme_workflow_stress_test(self, mock_session_state):
        """Test settings under extreme workflow conditions."""
        # Rapid setting changes
        limits = [10, 25, 50, 75, 100, 5, 30]

        for i, limit in enumerate(limits):
            # Rapid-fire setting changes
            save_settings({"max_jobs_per_company": limit})

            # Verify each change
            current = load_settings()
            assert current["max_jobs_per_company"] == limit

            if i % 2 == 0:  # Test scraping on even iterations
                with mock.patch(
                    "src.services.job_service.JobService.get_active_companies",
                ) as mock_companies:
                    mock_companies.return_value = [f"Stress{i} Corp"]

                    with mock.patch("src.scraper.scrape_all") as mock_scrape_all:
                        mock_scrape_all.return_value = {f"Stress{i} Corp": limit}

                        start_background_scraping()

                        # Quick completion
                        timeout = 2
                        start_time = time.time()
                        while (
                            is_scraping_active()
                            and (time.time() - start_time) < timeout
                        ):
                            time.sleep(0.05)

                        # Should use correct limit
                        mock_scrape_all.assert_called_once_with(limit)

        # Final verification
        final_settings = load_settings()
        assert final_settings["max_jobs_per_company"] == limits[-1]
