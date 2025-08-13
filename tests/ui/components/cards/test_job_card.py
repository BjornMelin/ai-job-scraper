"""Tests for Job Card component functionality.

Tests job card rendering, interactive controls, and user actions for
individual job postings in both grid and list views.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.schemas import Job
from src.ui.components.cards.job_card import (
    _format_posted_date,
    _handle_favorite_toggle,
    _handle_status_change,
    _handle_view_details,
    render_job_card,
    render_jobs_grid,
    render_jobs_list,
)


class TestJobCardRendering:
    """Test individual job card rendering functionality."""

    def test_render_job_card_displays_basic_info(self, mock_streamlit, sample_job_dto):
        """Test job card displays title, company, location and description."""
        # Act
        render_job_card(sample_job_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]

        # Verify job information is displayed
        assert any("Senior AI Engineer" in call for call in markdown_calls)
        assert any("Tech Corp" in call for call in markdown_calls)
        assert any("San Francisco, CA" in call for call in markdown_calls)

    def test_render_job_card_displays_status_badge(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test job card displays status badge with correct styling."""
        # Act
        render_job_card(sample_job_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        status_badge_calls = [call for call in markdown_calls if "status-badge" in call]

        assert status_badge_calls
        assert any("New" in call for call in status_badge_calls)

    def test_render_job_card_shows_favorite_star_when_favorited(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test job card shows star icon when job is favorited."""
        # Arrange
        sample_job_dto.favorite = True

        # Act
        render_job_card(sample_job_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        assert any("⭐" in call for call in markdown_calls)

    def test_render_job_card_hides_favorite_star_when_not_favorited(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test job card hides star icon when job is not favorited."""
        # Arrange
        sample_job_dto.favorite = False

        # Act
        render_job_card(sample_job_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        assert all("⭐" not in call for call in markdown_calls)

    def test_render_job_card_creates_status_selectbox(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test job card creates status selectbox with correct options."""
        # Act
        render_job_card(sample_job_dto)

        # Assert
        selectbox_calls = mock_streamlit["selectbox"].call_args_list

        # Find status selectbox call
        status_call = next(
            (call for call in selectbox_calls if call.args[0] == "Status"),
            None,
        )

        assert status_call is not None
        expected_options = ["New", "Interested", "Applied", "Rejected"]
        assert status_call.args[1] == expected_options
        assert status_call.kwargs["key"] == f"status_{sample_job_dto.id}"

    def test_render_job_card_creates_favorite_button(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test job card creates favorite toggle button."""
        # Act
        render_job_card(sample_job_dto)

        # Assert
        button_calls = mock_streamlit["button"].call_args_list

        # Find favorite button call
        favorite_call = next(
            (
                call
                for call in button_calls
                if call.kwargs.get("key") == f"favorite_{sample_job_dto.id}"
            ),
            None,
        )

        assert favorite_call is not None
        expected_icon = "❤️" if sample_job_dto.favorite else "🤍"
        assert favorite_call.args[0] == expected_icon

    def test_render_job_card_creates_view_details_button(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test job card creates view details button."""
        # Act
        render_job_card(sample_job_dto)

        # Assert
        button_calls = mock_streamlit["button"].call_args_list

        # Find view details button call
        details_call = next(
            (call for call in button_calls if call.args[0] == "View Details"),
            None,
        )

        assert details_call is not None
        assert details_call.kwargs["key"] == f"details_{sample_job_dto.id}"

    def test_render_job_card_truncates_long_description(self, mock_streamlit):
        """Test job card truncates very long job descriptions."""
        # Arrange
        long_description = "A" * 300  # 300 characters
        job = Job(
            id=1,
            company_id=1,
            company="Test Co",
            title="Test Job",
            description=long_description,
            link="https://test.com",
            location="Test City",
            content_hash="hash",
        )

        # Act
        render_job_card(job)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        description_calls = [call for call in markdown_calls if "..." in call]

        # Should truncate to 200 chars plus "..."
        assert any(len(call) <= 203 for call in description_calls if "A" in call)


class TestJobCardInteractions:
    """Test job card interactive functionality."""

    def test_handle_status_change_updates_job_status(self, mock_session_state):
        """Test status change handler updates job status via service."""
        # Arrange
        job_id = 1
        new_status = "Applied"
        mock_session_state[f"status_{job_id}"] = new_status

        with (
            patch("src.ui.components.cards.job_card.JobService") as mock_job_service,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_job_service.update_job_status.return_value = True

            # Act
            _handle_status_change(job_id)

            # Assert
            mock_job_service.update_job_status.assert_called_once_with(
                job_id,
                new_status,
            )
            mock_rerun.assert_called_once()

    def test_handle_status_change_handles_service_failure(self, mock_session_state):
        """Test status change handler handles service failure gracefully."""
        # Arrange
        job_id = 1
        mock_session_state[f"status_{job_id}"] = "Applied"

        with (
            patch("src.ui.components.cards.job_card.JobService") as mock_job_service,
            patch("streamlit.error") as mock_error,
            patch("streamlit.rerun"),
        ):
            mock_job_service.update_job_status.side_effect = Exception("Database error")

            # Act
            _handle_status_change(job_id)

            # Assert
            mock_error.assert_called_once_with("Failed to update job status")

    def test_handle_favorite_toggle_toggles_favorite_status(self):
        """Test favorite toggle handler toggles favorite status."""
        # Arrange
        job_id = 1

        with (
            patch("src.ui.components.cards.job_card.JobService") as mock_job_service,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_job_service.toggle_favorite.return_value = True

            # Act
            _handle_favorite_toggle(job_id)

            # Assert
            mock_job_service.toggle_favorite.assert_called_once_with(job_id)
            mock_rerun.assert_called_once()

    def test_handle_favorite_toggle_handles_service_failure(self):
        """Test favorite toggle handler handles service failure gracefully."""
        # Arrange
        job_id = 1

        with (
            patch("src.ui.components.cards.job_card.JobService") as mock_job_service,
            patch("streamlit.error") as mock_error,
            patch("streamlit.rerun"),
        ):
            mock_job_service.toggle_favorite.side_effect = Exception("Database error")

            # Act
            _handle_favorite_toggle(job_id)

            # Assert
            mock_error.assert_called_once_with("Failed to toggle favorite")

    def test_handle_view_details_sets_session_state(self, mock_session_state):
        """Test view details handler sets job ID in session state."""
        # Arrange
        job_id = 123

        # Act
        _handle_view_details(job_id)

        # Assert
        assert mock_session_state["view_job_id"] == job_id


class TestDateFormatting:
    """Test posted date formatting functionality."""

    def test_format_posted_date_returns_today_for_current_date(self):
        """Test date formatting returns 'Today' for current date."""
        # Arrange
        current_date = datetime.now(UTC)

        # Act
        result = _format_posted_date(current_date)

        # Assert
        assert result == "Today"

    def test_format_posted_date_returns_yesterday_for_one_day_ago(self):
        """Test date formatting returns 'Yesterday' for one day ago."""
        # Arrange
        yesterday = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = yesterday.replace(day=yesterday.day - 1)

        # Act
        result = _format_posted_date(yesterday)

        # Assert
        assert result == "Yesterday"

    def test_format_posted_date_returns_days_ago_for_multiple_days(self):
        """Test date formatting returns 'N days ago' for multiple days."""
        # Arrange
        days_ago = 5
        old_date = datetime.now(UTC).replace(day=datetime.now(UTC).day - days_ago)

        # Act
        result = _format_posted_date(old_date)

        # Assert
        assert result == f"{days_ago} days ago"

    def test_format_posted_date_handles_string_input(self):
        """Test date formatting handles string date input."""
        # Arrange
        date_string = "2024-01-15"

        # Act
        result = _format_posted_date(date_string)

        # Assert
        # Should return some number of days ago (exact number depends on current date)
        assert "days ago" in result or result in ["Today", "Yesterday"]

    def test_format_posted_date_handles_invalid_string(self):
        """Test date formatting handles invalid string input gracefully."""
        # Arrange
        invalid_date = "not-a-date"

        # Act
        result = _format_posted_date(invalid_date)

        # Assert
        assert result == ""

    def test_format_posted_date_handles_none_input(self):
        """Test date formatting handles None input gracefully."""
        # Arrange & Act
        result = _format_posted_date(None)

        # Assert
        assert result == ""


class TestJobGridRendering:
    """Test job grid rendering functionality."""

    def test_render_jobs_grid_creates_columns_layout(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test jobs grid creates proper column layout."""
        # Act
        render_jobs_grid(sample_jobs_dto, num_columns=3)

        # Assert
        # Should create columns for each row of jobs
        columns_calls = mock_streamlit["columns"].call_args_list
        assert columns_calls  # At least one row of columns

    def test_render_jobs_grid_handles_empty_job_list(self, mock_streamlit):
        """Test jobs grid handles empty job list gracefully."""
        # Act
        render_jobs_grid([], num_columns=3)

        # Assert
        mock_streamlit["info"].assert_called_once_with("No jobs to display.")

    def test_render_jobs_grid_respects_column_count(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test jobs grid respects specified column count."""
        # Act
        render_jobs_grid(sample_jobs_dto, num_columns=2)

        # Assert
        columns_calls = mock_streamlit["columns"].call_args_list
        # First call = grid columns (2), rest = within job cards
        grid_columns_call = columns_calls[0]  # First call is the grid layout
        assert grid_columns_call.args[0] == 2

    @pytest.mark.parametrize(
        ("num_jobs", "num_columns", "expected_rows"),
        (
            (3, 3, 1),  # 3 jobs in 3 columns = 1 row
            (4, 3, 2),  # 4 jobs in 3 columns = 2 rows
            (6, 3, 2),  # 6 jobs in 3 columns = 2 rows
            (7, 3, 3),  # 7 jobs in 3 columns = 3 rows
        ),
    )
    def test_render_jobs_grid_calculates_correct_rows(
        self,
        mock_streamlit,
        num_jobs,
        num_columns,
        expected_rows,
    ):
        """Test jobs grid calculates correct number of rows for different job counts."""
        # Arrange
        jobs = [
            Job(
                id=i,
                company_id=1,
                company=f"Company {i}",
                title=f"Job {i}",
                description="Test job",
                link="https://test.com",
                location="Test City",
                content_hash=f"hash{i}",
            )
            for i in range(1, num_jobs + 1)
        ]

        # Act
        render_jobs_grid(jobs, num_columns=num_columns)

        # Assert
        columns_calls = mock_streamlit["columns"].call_args_list
        # Filter for grid-level columns calls (those with gap="medium" parameter)
        grid_calls = [
            call
            for call in columns_calls
            if len(call.kwargs) > 0 and call.kwargs.get("gap") == "medium"
        ]
        assert len(grid_calls) == expected_rows


class TestJobListRendering:
    """Test job list rendering functionality."""

    def test_render_jobs_list_renders_all_jobs(self, mock_streamlit, sample_jobs_dto):
        """Test jobs list renders all provided jobs."""
        # Act
        render_jobs_list(sample_jobs_dto)

        # Assert
        # Should render job cards for all jobs
        container_calls = mock_streamlit["container"].call_args_list
        # Each job creates a container with border=True
        job_containers = [
            call for call in container_calls if call.kwargs.get("border") is True
        ]
        assert len(job_containers) >= len(sample_jobs_dto)

    def test_render_jobs_list_handles_empty_job_list(self, mock_streamlit):
        """Test jobs list handles empty job list gracefully."""
        # Act
        render_jobs_list([])

        # Assert
        mock_streamlit["info"].assert_called_once_with("No jobs to display.")

    def test_render_jobs_list_adds_separators_between_jobs(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test jobs list adds separators between job cards."""
        # Act
        render_jobs_list(sample_jobs_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        separator_calls = [call for call in markdown_calls if call == "---"]

        # Should have separators between jobs (one less than number of jobs)
        assert len(separator_calls) >= len(sample_jobs_dto) - 1


class TestJobCardIntegration:
    """Integration tests for complete job card workflows."""

    def test_complete_job_status_update_workflow(
        self,
        mock_streamlit,
        mock_session_state,
        sample_job_dto,
    ):
        """Test complete workflow of updating job status."""
        # Arrange
        job_id = sample_job_dto.id
        new_status = "Applied"
        mock_session_state[f"status_{job_id}"] = new_status

        with (
            patch("src.ui.components.cards.job_card.JobService") as mock_job_service,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_job_service.update_job_status.return_value = True

            # Act - Simulate status change during card rendering
            render_job_card(sample_job_dto)
            _handle_status_change(job_id)

            # Assert
            # 1. Status selectbox was created
            selectbox_calls = mock_streamlit["selectbox"].call_args_list
            status_selectbox = next(
                call for call in selectbox_calls if call.args[0] == "Status"
            )
            assert status_selectbox.kwargs["key"] == f"status_{job_id}"

            # 2. Service was called to update status
            mock_job_service.update_job_status.assert_called_once_with(
                job_id,
                new_status,
            )

            # 3. UI was refreshed
            mock_rerun.assert_called_once()

    def test_complete_favorite_toggle_workflow(
        self,
        mock_streamlit,
        mock_session_state,
        sample_job_dto,
    ):
        """Test complete workflow of toggling favorite status."""
        # Arrange
        job_id = sample_job_dto.id

        with (
            patch("src.ui.components.cards.job_card.JobService") as mock_job_service,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_job_service.toggle_favorite.return_value = True

            # Act - Simulate favorite toggle during card rendering
            render_job_card(sample_job_dto)
            _handle_favorite_toggle(job_id)

            # Assert
            # 1. Favorite button was created
            button_calls = mock_streamlit["button"].call_args_list
            favorite_button = next(
                call
                for call in button_calls
                if call.kwargs.get("key") == f"favorite_{job_id}"
            )
            assert favorite_button is not None

            # 2. Service was called to toggle favorite
            mock_job_service.toggle_favorite.assert_called_once_with(job_id)

            # 3. UI was refreshed
            mock_rerun.assert_called_once()

    def test_complete_view_details_workflow(
        self,
        mock_streamlit,
        mock_session_state,
        sample_job_dto,
    ):
        """Test complete workflow of viewing job details."""
        # Arrange
        job_id = sample_job_dto.id

        # Act - Simulate view details click during card rendering
        render_job_card(sample_job_dto)
        _handle_view_details(job_id)

        # Assert
        # 1. View details button was created
        button_calls = mock_streamlit["button"].call_args_list
        details_button = next(
            call for call in button_calls if call.args[0] == "View Details"
        )
        assert details_button.kwargs["key"] == f"details_{job_id}"

        # 2. Session state was updated with job ID
        assert mock_session_state["view_job_id"] == job_id

    def test_render_jobs_grid_with_fewer_jobs_than_columns(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test jobs grid handles case with fewer jobs than columns correctly."""
        # Arrange - 2 jobs with 3 columns
        jobs = sample_jobs_dto[:2]  # Use first 2 jobs

        # Act
        render_jobs_grid(jobs, num_columns=3)

        # Assert - should create 1 row with 3 columns, use first 2
        columns_calls = mock_streamlit["columns"].call_args_list
        # First call should be for the grid columns
        grid_call = columns_calls[0]
        assert grid_call.args[0] == 3  # 3 columns requested

        # Verify render_job_card was called for each job
        with patch("src.ui.components.cards.job_card.render_job_card") as mock_render:
            render_jobs_grid(jobs, num_columns=3)
            assert mock_render.call_count == 2  # Only 2 jobs rendered
