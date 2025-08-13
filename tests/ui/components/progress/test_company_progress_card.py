"""Tests for Company Progress Card component.

Tests the rendering, styling, and data display functionality of the
CompanyProgressCard component used in the scraping dashboard.
"""

from datetime import UTC, datetime, timezone
from unittest.mock import patch

from src.ui.components.progress.company_progress_card import (
    CompanyProgressCard,
    render_company_progress_card,
)
from src.ui.utils.background_helpers import CompanyProgress


class TestCompanyProgressCardRendering:
    """Test the basic rendering functionality of the company progress card."""

    def test_card_renders_with_bordered_container(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test that the progress card creates a bordered container."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Scraping", jobs_found=15)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        container_calls = mock_streamlit["container"].call_args_list
        assert len(container_calls) > 0
        assert container_calls[0].kwargs["border"] is True

    def test_card_displays_company_name_and_status(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card displays company name with appropriate emoji and status."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Scraping", jobs_found=15)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        markdown_calls = mock_streamlit["markdown"].call_args_list

        # Check company name with scraping emoji is displayed
        company_name_call = next(
            call
            for call in markdown_calls
            if "🔄 TechCorp" in call.args[0] and "**" in call.args[0]
        )
        assert company_name_call is not None
        assert company_name_call.kwargs["help"] == "Status: Scraping"

        # Check status badge is rendered
        status_badge_call = next(
            call
            for call in markdown_calls
            if "SCRAPING" in call.args[0] and "unsafe_allow_html" in call.kwargs
        )
        assert status_badge_call is not None

    def test_card_displays_different_status_configurations(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card displays appropriate styling for different status values."""
        # Test cases for different statuses
        status_tests = [
            ("Pending", "⏳", "#6c757d"),
            ("Scraping", "🔄", "#007bff"),
            ("Completed", "✅", "#28a745"),
            ("Error", "❌", "#dc3545"),
        ]

        for status, emoji, color in status_tests:
            # Arrange
            progress = CompanyProgress(name="TestCorp", status=status, jobs_found=10)
            card = CompanyProgressCard()

            # Reset mock calls
            mock_streamlit["markdown"].reset_mock()

            # Act
            card.render(progress)

            # Assert
            markdown_calls = mock_streamlit["markdown"].call_args_list

            # Check emoji in company name
            company_name_call = next(
                call for call in markdown_calls if f"{emoji} TestCorp" in call.args[0]
            )
            assert company_name_call is not None

            # Check status badge color
            status_badge_call = next(
                call
                for call in markdown_calls
                if color in call.args[0] and status.upper() in call.args[0]
            )
            assert status_badge_call is not None

    def test_card_shows_error_message_for_error_status(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card shows error message when status is Error."""
        # Arrange
        progress = CompanyProgress(
            name="TechCorp",
            status="Error",
            jobs_found=0,
            error="Connection timeout",
        )
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        error_calls = mock_streamlit["error"].call_args_list
        assert len(error_calls) > 0
        assert "Error: Connection timeout" in error_calls[0].args[0]

    def test_card_handles_missing_error_field_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card handles Error status without error message gracefully."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Error", jobs_found=0)
        card = CompanyProgressCard()

        # Act & Assert - Should not raise exception
        card.render(progress)

        # Error section should still be rendered properly
        markdown_calls = mock_streamlit["markdown"].call_args_list
        assert len(markdown_calls) > 0  # Basic rendering still works


class TestCompanyProgressCardProgressBar:
    """Test the progress bar rendering logic."""

    def test_completed_status_shows_100_percent_progress(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test completed status shows 100% progress."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Completed", jobs_found=25)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        progress_calls = mock_streamlit["progress"].call_args_list
        assert len(progress_calls) > 0
        assert progress_calls[0].args[0] == 1.0  # 100% progress
        assert progress_calls[0].kwargs["text"] == "Completed"

    def test_scraping_status_shows_time_based_progress(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test scraping status shows time-based progress estimation."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        progress = CompanyProgress(
            name="TechCorp",
            status="Scraping",
            jobs_found=15,
            start_time=start_time,
        )
        card = CompanyProgressCard()

        with patch("src.ui.utils.background_helpers.datetime") as mock_datetime:
            # Mock current time to be 1 minute after start (should show ~50% progress)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 1, tzinfo=UTC)
            mock_datetime.timezone = timezone

            # Act
            card.render(progress)

            # Assert
            progress_calls = mock_streamlit["progress"].call_args_list
            assert len(progress_calls) > 0

            # Should show some progress based on elapsed time
            progress_value = progress_calls[0].args[0]
            assert 0.0 < progress_value <= 0.9  # Between 0% and 90%
            assert "Scraping..." in progress_calls[0].kwargs["text"]

    def test_scraping_without_start_time_shows_minimal_progress(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test scraping status without start time shows minimal progress."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Scraping", jobs_found=15)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        progress_calls = mock_streamlit["progress"].call_args_list
        assert len(progress_calls) > 0
        assert progress_calls[0].args[0] == 0.1  # Minimal progress indication
        assert progress_calls[0].kwargs["text"] == "Scraping..."

    def test_error_status_shows_zero_progress(self, mock_streamlit, mock_session_state):
        """Test error status shows zero progress."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Error", jobs_found=0)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        progress_calls = mock_streamlit["progress"].call_args_list
        assert len(progress_calls) > 0
        assert progress_calls[0].args[0] == 0.0  # No progress
        assert progress_calls[0].kwargs["text"] == "Failed"

    def test_pending_status_shows_zero_progress(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test pending status shows zero progress with waiting message."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Pending", jobs_found=0)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert
        progress_calls = mock_streamlit["progress"].call_args_list
        assert len(progress_calls) > 0
        assert progress_calls[0].args[0] == 0.0  # No progress
        assert progress_calls[0].kwargs["text"] == "Waiting to start"


class TestCompanyProgressCardMetrics:
    """Test the metrics section rendering."""

    def test_metrics_display_jobs_found_and_speed(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test metrics section displays jobs found and scraping speed."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 10, 5, tzinfo=UTC)  # 5 minutes
        progress = CompanyProgress(
            name="TechCorp",
            status="Completed",
            jobs_found=30,
            start_time=start_time,
            end_time=end_time,
        )
        card = CompanyProgressCard()

        with patch(
            "src.ui.components.progress.company_progress_card.calculate_scraping_speed",
        ) as mock_calc_speed:
            mock_calc_speed.return_value = 6.0  # 6 jobs per minute

            # Act
            card.render(progress)

            # Assert
            metric_calls = mock_streamlit["metric"].call_args_list

            # Check Jobs Found metric
            jobs_metric_call = next(
                call for call in metric_calls if call.kwargs["label"] == "Jobs Found"
            )
            assert jobs_metric_call.kwargs["value"] == 30

            # Check Speed metric
            speed_metric_call = next(
                call for call in metric_calls if call.kwargs["label"] == "Speed"
            )
            assert speed_metric_call.kwargs["value"] == "6.0 /min"

            # Verify speed calculation was called with correct parameters
            mock_calc_speed.assert_called_once_with(30, start_time, end_time)

    def test_metrics_handle_zero_speed_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test metrics display handles zero or N/A speed values."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Pending", jobs_found=0)
        card = CompanyProgressCard()

        with patch(
            "src.ui.components.progress.company_progress_card.calculate_scraping_speed",
        ) as mock_calc_speed:
            mock_calc_speed.return_value = 0.0  # No speed calculated

            # Act
            card.render(progress)

            # Assert
            metric_calls = mock_streamlit["metric"].call_args_list

            # Check Speed metric shows N/A
            speed_metric_call = next(
                call for call in metric_calls if call.kwargs["label"] == "Speed"
            )
            assert speed_metric_call.kwargs["value"] == "N/A"


class TestCompanyProgressCardTimingInfo:
    """Test the timing information display."""

    def test_timing_shows_start_and_completion_info(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test timing section shows start, end, and duration information."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 10, 3, tzinfo=UTC)
        progress = CompanyProgress(
            name="TechCorp",
            status="Completed",
            jobs_found=30,
            start_time=start_time,
            end_time=end_time,
        )
        card = CompanyProgressCard()

        with (
            patch(
                "src.ui.components.progress.company_progress_card.format_timestamp",
            ) as mock_format_time,
            patch(
                "src.ui.components.progress.company_progress_card.format_duration",
            ) as mock_format_duration,
        ):
            mock_format_time.side_effect = ["10:00:00", "10:03:00"]
            mock_format_duration.return_value = "3m 0s"

            # Act
            card.render(progress)

            # Assert
            caption_calls = mock_streamlit["caption"].call_args_list
            assert len(caption_calls) > 0

            timing_text = caption_calls[0].args[0]
            assert "Started: 10:00:00" in timing_text
            assert "Completed: 10:03:00" in timing_text
            assert "Duration: 3m 0s" in timing_text

    def test_timing_shows_elapsed_time_for_active_scraping(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test timing shows elapsed time for active scraping."""
        # Arrange
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        progress = CompanyProgress(
            name="TechCorp",
            status="Scraping",
            jobs_found=15,
            start_time=start_time,
        )
        card = CompanyProgressCard()

        with (
            patch(
                "src.ui.components.progress.company_progress_card.format_timestamp",
            ) as mock_format_time,
            patch(
                "src.ui.components.progress.company_progress_card.format_duration",
            ) as mock_format_duration,
            patch("src.ui.utils.background_helpers.datetime") as mock_datetime,
        ):
            # Mock current time to be 2 minutes after start
            mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 2, tzinfo=UTC)
            mock_datetime.timezone = timezone

            mock_format_time.return_value = "10:00:00"
            mock_format_duration.return_value = "2m 0s"

            # Act
            card.render(progress)

            # Assert
            caption_calls = mock_streamlit["caption"].call_args_list
            assert len(caption_calls) > 0

            timing_text = caption_calls[0].args[0]
            assert "Started: 10:00:00" in timing_text
            assert "Elapsed: 2m 0s" in timing_text

    def test_timing_handles_missing_times_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test timing section handles missing timestamp data gracefully."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Pending", jobs_found=0)
        card = CompanyProgressCard()

        # Act
        card.render(progress)

        # Assert - No caption should be created when no timing info available
        caption_calls = mock_streamlit["caption"].call_args_list
        # Either no calls, or if there are calls, they shouldn't contain timing info
        if caption_calls:
            for call in caption_calls:
                timing_text = call.args[0]
                assert "Started:" not in timing_text


class TestCompanyProgressCardErrorHandling:
    """Test error handling and edge cases."""

    def test_card_handles_rendering_exceptions_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
        mock_logging,
    ):
        """Test card handles internal rendering exceptions gracefully."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Scraping", jobs_found=15)
        card = CompanyProgressCard()

        # Mock container to raise an exception
        mock_streamlit["container"].side_effect = Exception("Rendering error")

        # Act - Should not raise exception
        card.render(progress)

        # Assert error message is shown instead of crashing
        error_calls = mock_streamlit["error"].call_args_list
        assert len(error_calls) > 0
        assert "Error displaying progress for TechCorp" in error_calls[0].args[0]

    def test_card_handles_unknown_status_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card handles unknown status values gracefully."""
        # Arrange - Use unknown status
        progress = CompanyProgress(name="TechCorp", status="Unknown", jobs_found=15)
        card = CompanyProgressCard()

        # Act - Should not crash
        card.render(progress)

        # Assert - Should fall back to Pending configuration
        markdown_calls = mock_streamlit["markdown"].call_args_list

        # Should use pending emoji as fallback
        company_name_call = next(
            call
            for call in markdown_calls
            if "⏳ TechCorp" in call.args[0]  # Pending emoji
        )
        assert company_name_call is not None

    def test_card_handles_negative_jobs_found_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card handles edge case of negative jobs found."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Error", jobs_found=-5)
        card = CompanyProgressCard()

        # Act - Should not crash
        card.render(progress)

        # Assert - Should still render the card
        metric_calls = mock_streamlit["metric"].call_args_list
        jobs_metric_call = next(
            call for call in metric_calls if call.kwargs["label"] == "Jobs Found"
        )
        assert jobs_metric_call.kwargs["value"] == -5  # Display the actual value


class TestConvenienceFunction:
    """Test the convenience function for rendering progress cards."""

    def test_render_company_progress_card_function(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test the convenience function creates and renders a card."""
        # Arrange
        progress = CompanyProgress(name="TechCorp", status="Completed", jobs_found=25)

        # Act
        render_company_progress_card(progress)

        # Assert - Card should be rendered (container created)
        container_calls = mock_streamlit["container"].call_args_list
        assert len(container_calls) > 0
        assert container_calls[0].kwargs["border"] is True

        # Assert - Company name should be displayed
        markdown_calls = mock_streamlit["markdown"].call_args_list
        company_name_call = next(
            call
            for call in markdown_calls
            if "✅ TechCorp" in call.args[0]  # Completed status emoji
        )
        assert company_name_call is not None


class TestCompanyProgressCardIntegration:
    """Integration tests for the complete card functionality."""

    def test_complete_card_rendering_workflow(self, mock_streamlit, mock_session_state):
        """Test complete card rendering with all components."""
        # Arrange - Create a realistic progress object
        start_time = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 10, 5, tzinfo=UTC)
        progress = CompanyProgress(
            name="TechCorp",
            status="Completed",
            jobs_found=45,
            start_time=start_time,
            end_time=end_time,
        )

        with (
            patch(
                "src.ui.components.progress.company_progress_card.calculate_scraping_speed",
            ) as mock_speed,
            patch(
                "src.ui.components.progress.company_progress_card.format_timestamp",
            ) as mock_timestamp,
            patch(
                "src.ui.components.progress.company_progress_card.format_duration",
            ) as mock_duration,
        ):
            mock_speed.return_value = 9.0
            mock_timestamp.side_effect = ["10:00:00", "10:05:00"]
            mock_duration.return_value = "5m 0s"

            # Act
            render_company_progress_card(progress)

            # Assert - All major components should be rendered

            # 1. Container with border
            container_calls = mock_streamlit["container"].call_args_list
            assert len(container_calls) > 0
            assert container_calls[0].kwargs["border"] is True

            # 2. Company name and status
            markdown_calls = mock_streamlit["markdown"].call_args_list
            assert any("✅ TechCorp" in call.args[0] for call in markdown_calls)
            assert any("COMPLETED" in call.args[0] for call in markdown_calls)

            # 3. Progress bar
            progress_calls = mock_streamlit["progress"].call_args_list
            assert len(progress_calls) > 0
            assert progress_calls[0].args[0] == 1.0  # 100% for completed

            # 4. Metrics
            metric_calls = mock_streamlit["metric"].call_args_list
            jobs_metric = next(
                call for call in metric_calls if call.kwargs["label"] == "Jobs Found"
            )
            assert jobs_metric.kwargs["value"] == 45

            speed_metric = next(
                call for call in metric_calls if call.kwargs["label"] == "Speed"
            )
            assert speed_metric.kwargs["value"] == "9.0 /min"

            # 5. Timing information
            caption_calls = mock_streamlit["caption"].call_args_list
            assert len(caption_calls) > 0
            timing_text = caption_calls[0].args[0]
            assert "Started: 10:00:00" in timing_text
            assert "Completed: 10:05:00" in timing_text
            assert "Duration: 5m 0s" in timing_text

    def test_card_responsiveness_with_long_company_names(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test card handles very long company names gracefully."""
        # Arrange
        long_name = "Very Long Company Name That Might Cause UI Issues Inc."
        progress = CompanyProgress(name=long_name, status="Scraping", jobs_found=10)

        # Act - Should not cause layout issues
        render_company_progress_card(progress)

        # Assert - Card should still render properly
        markdown_calls = mock_streamlit["markdown"].call_args_list
        company_name_call = next(
            call for call in markdown_calls if long_name in call.args[0]
        )
        assert company_name_call is not None
