"""Comprehensive tests for the Job Card UI component.

This module tests the Job Card functionality including:
- Individual job card rendering with all elements
- Interactive controls (status dropdown, favorite toggle, view details button)
- Date formatting and timezone handling
- HTML escaping for security
- Grid and list layout rendering
- Service integration for status/favorite updates
- Error handling and edge cases
- Callback functions and session state updates
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tests.utils.streamlit_utils import StreamlitComponentTester

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


@pytest.fixture
def sample_job():
    """Create a sample Job object for testing."""
    return Job(
        id=1,
        title="Senior Python Developer",
        company="Tech Corp",
        location="San Francisco, CA",
        description="Exciting Python role with machine learning focus. Build scalable web applications using Django and Flask. Work with data science team on ML pipelines.",
        posted_date=datetime.now(UTC) - timedelta(days=2),
        last_seen=datetime.now(UTC),
        favorite=False,
        application_status="New",
        notes="Interesting opportunity",
        link="https://example.com/job1",
    )


@pytest.fixture
def sample_jobs_list():
    """Create a list of sample Job objects for testing."""
    now = datetime.now(UTC)
    return [
        Job(
            id=1,
            title="Senior Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            description="Build scalable Python applications",
            posted_date=now - timedelta(days=1),
            last_seen=now,
            favorite=True,
            application_status="Applied",
            notes="Great opportunity",
            link="https://example.com/job1",
        ),
        Job(
            id=2,
            title="Data Scientist",
            company="AI Startup",
            location="Remote",
            description="Work on cutting-edge ML models and data pipelines",
            posted_date=now - timedelta(days=3),
            last_seen=now,
            favorite=False,
            application_status="Interested",
            notes="",
            link="https://example.com/job2",
        ),
        Job(
            id=3,
            title="Full Stack Engineer",
            company="Scale Inc",
            location="New York, NY",
            description="Build modern web applications with React and Node.js",
            posted_date=now - timedelta(days=5),
            last_seen=now,
            favorite=True,
            application_status="New",
            notes="Remote friendly",
            link="https://example.com/job3",
        ),
    ]


class TestJobCardRendering:
    """Test individual job card rendering functionality."""

    def test_render_job_card_basic_structure(self, sample_job):
        """Test basic job card structure and content."""
        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ) as mock_format_date,
        ):
            # Mock container context manager
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            # Mock columns
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            # Mock interactive elements
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(sample_job)

            # Verify container with border is created
            mock_container.assert_called_once_with(border=True)

            # Verify date formatting is called
            mock_format_date.assert_called_once_with(sample_job.posted_date)

            # Verify content is rendered
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]

            # Check title
            assert any("Senior Python Developer" in call for call in markdown_calls)

            # Check company and location
            assert any(
                "Tech Corp" in call and "San Francisco, CA" in call
                for call in markdown_calls
            )

            # Check description preview (truncated)
            assert any(
                "Exciting Python role with machine learning focus" in call
                for call in markdown_calls
            )

    def test_render_job_card_with_long_description(self, sample_job):
        """Test job card truncates long descriptions correctly."""
        # Create job with very long description
        long_job = sample_job.model_copy()
        long_job.description = "A" * 300  # 300 characters

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(long_job)

            # Verify description is truncated with ellipsis
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            description_calls = [call for call in markdown_calls if len(call) > 100]
            assert any(
                call.endswith("...") and len(call) <= 203 for call in description_calls
            )  # 200 chars + "..."

    def test_render_job_card_favorite_indicator(self, sample_job):
        """Test favorite job displays star indicator."""
        favorite_job = sample_job.model_copy()
        favorite_job.favorite = True

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(favorite_job)

            # Verify favorite star is displayed
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            assert any("⭐" in call for call in markdown_calls)

    def test_render_job_card_status_badge_rendering(self, sample_job):
        """Test status badge is rendered with correct HTML."""
        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(sample_job)

            # Verify status badge HTML is rendered
            markdown_calls = mock_markdown.call_args_list
            status_badge_calls = [
                call for call in markdown_calls if call[1].get("unsafe_allow_html")
            ]
            assert len(status_badge_calls) > 0

            # Check status badge content
            badge_html = status_badge_calls[0][0][0]
            assert 'class="status-badge status-new"' in badge_html
            assert "New" in badge_html

    def test_render_job_card_html_escaping(self):
        """Test job card properly escapes HTML in user content."""
        # Create job with potentially malicious content
        malicious_job = Job(
            id=999,
            title='<script>alert("XSS")</script>Fake Job',
            company="<img src=x onerror=alert(1)>Evil Corp",
            location="<b>Dangerous</b> City",
            description="Normal description",
            posted_date=datetime.now(UTC),
            last_seen=datetime.now(UTC),
            favorite=False,
            application_status="New",
            notes="",
            link="https://example.com/evil",
        )

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="Today",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(malicious_job)

            # Verify HTML is escaped
            markdown_calls = [
                call[0][0]
                for call in mock_markdown.call_args_list
                if not call[1].get("unsafe_allow_html")
            ]

            # Check that dangerous HTML is escaped
            content_calls = " ".join(markdown_calls)
            assert "&lt;script&gt;" in content_calls
            assert "&lt;img src=" in content_calls
            assert "&lt;b&gt;" in content_calls


class TestJobCardInteractiveElements:
    """Test interactive elements in job cards."""

    def test_render_job_card_status_selectbox(self, sample_job):
        """Test status selectbox is rendered with correct options and callback."""
        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(sample_job)

            # Verify selectbox is called with correct parameters
            mock_selectbox.assert_called_once()
            selectbox_call = mock_selectbox.call_args

            # Check selectbox parameters
            assert selectbox_call[0][0] == "Status"  # Label
            assert selectbox_call[1]["key"] == f"status_{sample_job.id}"
            assert selectbox_call[1]["on_change"] == _handle_status_change
            assert selectbox_call[1]["args"] == (sample_job.id,)

            # Check current status index
            from src.constants import APPLICATION_STATUSES

            assert selectbox_call[1]["index"] == APPLICATION_STATUSES.index("New")

    def test_render_job_card_favorite_button(self, sample_job):
        """Test favorite toggle button is rendered correctly."""
        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(sample_job)

            # Verify buttons are called
            button_calls = mock_button.call_args_list
            assert len(button_calls) == 2  # Favorite and View Details buttons

            # Check favorite button (first call)
            favorite_button_call = button_calls[0]
            assert favorite_button_call[0][0] == "🤍"  # Not favorited
            assert favorite_button_call[1]["key"] == f"favorite_{sample_job.id}"
            assert favorite_button_call[1]["help"] == "Toggle favorite"
            assert favorite_button_call[1]["on_click"] == _handle_favorite_toggle
            assert favorite_button_call[1]["args"] == (sample_job.id,)

    def test_render_job_card_favorite_button_favorited(self, sample_job):
        """Test favorite button shows heart icon when job is favorited."""
        favorite_job = sample_job.model_copy()
        favorite_job.favorite = True

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "Applied"
            mock_button.return_value = False

            tester.run_component(favorite_job)

            # Check favorite button shows heart icon
            button_calls = mock_button.call_args_list
            favorite_button_call = button_calls[0]
            assert favorite_button_call[0][0] == "❤️"  # Favorited

    def test_render_job_card_view_details_button(self, sample_job):
        """Test view details button is rendered correctly."""
        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            tester.run_component(sample_job)

            # Check view details button (second call)
            button_calls = mock_button.call_args_list
            details_button_call = button_calls[1]
            assert details_button_call[0][0] == "View Details"
            assert details_button_call[1]["key"] == f"details_{sample_job.id}"
            assert details_button_call[1]["on_click"] == _handle_view_details
            assert details_button_call[1]["args"] == (sample_job.id,)


class TestDateFormatting:
    """Test date formatting functionality."""

    def test_format_posted_date_today(self):
        """Test formatting for job posted today."""
        today = datetime.now(UTC)
        result = _format_posted_date(today)
        assert result == "Today"

    def test_format_posted_date_yesterday(self):
        """Test formatting for job posted yesterday."""
        yesterday = datetime.now(UTC) - timedelta(days=1)
        result = _format_posted_date(yesterday)
        assert result == "Yesterday"

    def test_format_posted_date_days_ago(self):
        """Test formatting for job posted multiple days ago."""
        five_days_ago = datetime.now(UTC) - timedelta(days=5)
        result = _format_posted_date(five_days_ago)
        assert result == "5 days ago"

    def test_format_posted_date_future_date(self):
        """Test formatting for future date (edge case)."""
        future_date = datetime.now(UTC) + timedelta(days=3)
        result = _format_posted_date(future_date)
        assert result == "In 3 days"

    def test_format_posted_date_string_input(self):
        """Test formatting with string date input."""
        date_string = "2024-01-15"
        result = _format_posted_date(date_string)
        # Should parse the string and calculate difference
        assert result != ""
        assert "days ago" in result or result in ["Today", "Yesterday"]

    def test_format_posted_date_invalid_string(self):
        """Test formatting with invalid string date."""
        invalid_string = "not-a-date"
        result = _format_posted_date(invalid_string)
        assert result == ""

    def test_format_posted_date_none_input(self):
        """Test formatting with None input."""
        result = _format_posted_date(None)
        assert result == ""

    def test_format_posted_date_naive_datetime(self):
        """Test formatting with timezone-naive datetime."""
        naive_date = datetime(2024, 1, 15, 10, 30)  # No timezone info

        with patch("src.ui.components.cards.job_card.logger") as mock_logger:
            result = _format_posted_date(naive_date)

            # Should convert to UTC and return valid result
            assert result != ""
            # Should log the conversion
            mock_logger.debug.assert_called()

    def test_format_posted_date_timezone_aware_datetime(self):
        """Test formatting with timezone-aware datetime."""
        aware_date = datetime.now(UTC) - timedelta(days=3)
        result = _format_posted_date(aware_date)
        assert result == "3 days ago"

    def test_format_posted_date_unexpected_type(self):
        """Test formatting with unexpected input type."""
        unexpected_input = 12345

        with patch("src.ui.components.cards.job_card.logger") as mock_logger:
            result = _format_posted_date(unexpected_input)

            assert result == ""
            mock_logger.warning.assert_called()


class TestCallbackFunctions:
    """Test callback functions for interactive elements."""

    def test_handle_status_change_success(self):
        """Test successful status change callback."""
        tester = StreamlitComponentTester(_handle_status_change)

        job_id = 1
        new_status = "Applied"
        tester.set_session_state(**{f"status_{job_id}": new_status})

        with (
            patch(
                "src.services.job_service.JobService.update_job_status"
            ) as mock_update,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component(job_id)

            # Verify service is called
            mock_update.assert_called_once_with(job_id, new_status)

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_handle_status_change_no_status_in_session(self):
        """Test status change callback when no status in session state."""
        tester = StreamlitComponentTester(_handle_status_change)

        job_id = 1
        # Don't set status in session state

        with (
            patch(
                "src.services.job_service.JobService.update_job_status"
            ) as mock_update,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component(job_id)

            # Verify service is not called
            mock_update.assert_not_called()
            mock_rerun.assert_not_called()

    def test_handle_status_change_service_exception(self):
        """Test status change handles service exceptions."""
        tester = StreamlitComponentTester(_handle_status_change)

        job_id = 1
        new_status = "Applied"
        tester.set_session_state(**{f"status_{job_id}": new_status})

        with (
            patch(
                "src.services.job_service.JobService.update_job_status",
                side_effect=Exception("Service error"),
            ),
            patch("streamlit.error") as mock_error,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component(job_id)

            # Verify error is displayed
            mock_error.assert_called_once_with("Failed to update job status")

            # Verify rerun is not called due to error
            mock_rerun.assert_not_called()

    def test_handle_favorite_toggle_success(self):
        """Test successful favorite toggle callback."""
        tester = StreamlitComponentTester(_handle_favorite_toggle)

        job_id = 1

        with (
            patch("src.services.job_service.JobService.toggle_favorite") as mock_toggle,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component(job_id)

            # Verify service is called
            mock_toggle.assert_called_once_with(job_id)

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_handle_favorite_toggle_service_exception(self):
        """Test favorite toggle handles service exceptions."""
        tester = StreamlitComponentTester(_handle_favorite_toggle)

        job_id = 1

        with (
            patch(
                "src.services.job_service.JobService.toggle_favorite",
                side_effect=Exception("Database error"),
            ),
            patch("streamlit.error") as mock_error,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component(job_id)

            # Verify error is displayed
            mock_error.assert_called_once_with("Failed to toggle favorite")

            # Verify rerun is not called due to error
            mock_rerun.assert_not_called()

    def test_handle_view_details(self):
        """Test view details callback sets session state correctly."""
        tester = StreamlitComponentTester(_handle_view_details)

        job_id = 42

        tester.run_component(job_id)

        # Verify view_job_id is set in session state
        state = tester.get_session_state()
        assert state.get("view_job_id") == job_id


class TestJobGridRendering:
    """Test jobs grid rendering functionality."""

    def test_render_jobs_grid_with_jobs(self, sample_jobs_list):
        """Test jobs grid renders correctly with job data."""
        tester = StreamlitComponentTester(render_jobs_grid)

        with (
            patch("src.ui.styles.styles.apply_job_grid_styles") as mock_apply_styles,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch(
                "src.ui.components.cards.job_card.render_job_card"
            ) as mock_render_card,
        ):
            # Mock columns for grid layout
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            # Mock container context manager
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            tester.run_component(sample_jobs_list, num_columns=3)

            # Verify styles are applied
            mock_apply_styles.assert_called_once()

            # Verify columns are created (one call for first row)
            mock_columns.assert_called_once_with(3, gap="medium")

            # Verify render_job_card is called for each job
            assert mock_render_card.call_count == len(sample_jobs_list)

    def test_render_jobs_grid_empty_list(self):
        """Test jobs grid handles empty job list."""
        tester = StreamlitComponentTester(render_jobs_grid)

        with (
            patch("streamlit.info") as mock_info,
            patch("src.ui.styles.styles.apply_job_grid_styles") as mock_apply_styles,
        ):
            tester.run_component([], num_columns=3)

            # Verify info message is displayed
            mock_info.assert_called_once_with("No jobs to display.")

            # Verify styles are not applied for empty list
            mock_apply_styles.assert_not_called()

    def test_render_jobs_grid_different_column_counts(self, sample_jobs_list):
        """Test jobs grid with different column configurations."""
        column_counts = [1, 2, 3, 4]

        for num_cols in column_counts:
            tester = StreamlitComponentTester(render_jobs_grid)

            with (
                patch("src.ui.styles.styles.apply_job_grid_styles"),
                patch("streamlit.columns") as mock_columns,
                patch("streamlit.container") as mock_container,
                patch("streamlit.markdown"),
                patch("src.ui.components.cards.job_card.render_job_card"),
            ):
                mock_columns.return_value = [Mock() for _ in range(num_cols)]
                mock_container.return_value.__enter__ = Mock()
                mock_container.return_value.__exit__ = Mock(return_value=None)

                tester.run_component(sample_jobs_list, num_columns=num_cols)

                # Verify correct number of columns is used
                mock_columns.assert_called_with(num_cols, gap="medium")

    def test_render_jobs_grid_row_spacing(self, sample_jobs_list):
        """Test jobs grid adds spacing between rows."""
        # Test with more jobs than columns to ensure multiple rows
        extended_jobs = sample_jobs_list * 2  # 6 jobs total

        tester = StreamlitComponentTester(render_jobs_grid)

        with (
            patch("src.ui.styles.styles.apply_job_grid_styles"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("src.ui.components.cards.job_card.render_job_card"),
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            tester.run_component(extended_jobs, num_columns=3)

            # Verify columns are called multiple times (for multiple rows)
            assert mock_columns.call_count == 2  # 6 jobs / 3 columns = 2 rows

            # Verify spacing markdown is added between rows
            spacing_calls = [
                call
                for call in mock_markdown.call_args_list
                if len(call) > 0 and "job-card-grid" in str(call[0])
            ]
            assert len(spacing_calls) == 1  # One spacing call between rows


class TestJobListRendering:
    """Test jobs list rendering functionality."""

    def test_render_jobs_list_with_jobs(self, sample_jobs_list):
        """Test jobs list renders correctly with job data."""
        tester = StreamlitComponentTester(render_jobs_list)

        with (
            patch(
                "src.ui.components.cards.job_card.render_job_card"
            ) as mock_render_card,
            patch("streamlit.markdown") as mock_markdown,
        ):
            tester.run_component(sample_jobs_list)

            # Verify render_job_card is called for each job
            assert mock_render_card.call_count == len(sample_jobs_list)

            # Verify separators are added between jobs
            separator_calls = [
                call
                for call in mock_markdown.call_args_list
                if len(call) > 0 and call[0][0] == "---"
            ]
            assert len(separator_calls) == len(sample_jobs_list)

    def test_render_jobs_list_empty_list(self):
        """Test jobs list handles empty job list."""
        tester = StreamlitComponentTester(render_jobs_list)

        with (
            patch("streamlit.info") as mock_info,
            patch(
                "src.ui.components.cards.job_card.render_job_card"
            ) as mock_render_card,
        ):
            tester.run_component([])

            # Verify info message is displayed
            mock_info.assert_called_once_with("No jobs to display.")

            # Verify no job cards are rendered
            mock_render_card.assert_not_called()

    def test_render_jobs_list_single_job(self, sample_job):
        """Test jobs list with single job."""
        tester = StreamlitComponentTester(render_jobs_list)

        with (
            patch(
                "src.ui.components.cards.job_card.render_job_card"
            ) as mock_render_card,
            patch("streamlit.markdown") as mock_markdown,
        ):
            tester.run_component([sample_job])

            # Verify single job is rendered
            mock_render_card.assert_called_once_with(sample_job)

            # Verify separator is added
            separator_calls = [
                call
                for call in mock_markdown.call_args_list
                if len(call) > 0 and call[0][0] == "---"
            ]
            assert len(separator_calls) == 1


class TestJobCardEdgeCases:
    """Test edge cases and error conditions."""

    def test_render_job_card_with_missing_fields(self):
        """Test job card handles missing optional fields gracefully."""
        minimal_job = Job(
            id=1,
            title="Minimal Job",
            company="Test Company",
            location=None,  # Missing location
            description="",  # Empty description
            posted_date=None,  # Missing date
            last_seen=datetime.now(UTC),
            favorite=False,
            application_status="New",
            notes=None,  # Missing notes
            link="https://example.com",
        )

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date", return_value=""
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            # Should not raise exception
            tester.run_component(minimal_job)

            # Verify basic structure is still rendered
            assert mock_markdown.call_count > 0

    def test_render_job_card_with_special_characters(self):
        """Test job card handles special characters correctly."""
        special_job = Job(
            id=1,
            title="Job with Spëcïál Çhäracters & Symbols!",
            company="Tëch Çorp™",
            location="São Paulo, Brasil",
            description="Description with émojis 🚀 and unicode ⚡",
            posted_date=datetime.now(UTC),
            last_seen=datetime.now(UTC),
            favorite=False,
            application_status="New",
            notes="Notes with special chars: ñ, ü, ç",
            link="https://example.com",
        )

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="Today",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            # Should handle special characters without errors
            tester.run_component(special_job)

            # Verify content is rendered
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            content = " ".join(markdown_calls)

            # Check that special characters are preserved (not HTML entities)
            assert "Spëcïál" in content
            assert "🚀" in content
            assert "⚡" in content

    def test_render_job_card_with_invalid_status(self, sample_job):
        """Test job card handles invalid application status."""
        invalid_job = sample_job.model_copy()
        invalid_job.application_status = "InvalidStatus"

        tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "InvalidStatus"
            mock_button.return_value = False

            tester.run_component(invalid_job)

            # Verify selectbox uses index 0 (default) for invalid status
            selectbox_call = mock_selectbox.call_args
            assert selectbox_call[1]["index"] == 0


class TestJobCardIntegration:
    """Integration tests for job card components working together."""

    def test_complete_job_card_interaction_workflow(self, sample_job):
        """Test complete interaction workflow from card rendering to callbacks."""
        # Test the entire flow from rendering to user interaction

        # 1. Render the job card
        card_tester = StreamlitComponentTester(render_job_card)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.button") as mock_button,
            patch(
                "src.ui.components.cards.job_card._format_posted_date",
                return_value="2 days ago",
            ),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.return_value = "New"
            mock_button.return_value = False

            card_tester.run_component(sample_job)

            # Verify all interactive elements are set up correctly
            assert mock_selectbox.called
            assert mock_button.call_count == 2  # Favorite and details buttons

        # 2. Test status change callback
        status_tester = StreamlitComponentTester(_handle_status_change)
        status_tester.set_session_state(**{f"status_{sample_job.id}": "Applied"})

        with (
            patch(
                "src.services.job_service.JobService.update_job_status"
            ) as mock_update,
            patch("streamlit.rerun"),
        ):
            status_tester.run_component(sample_job.id)
            mock_update.assert_called_with(sample_job.id, "Applied")

        # 3. Test favorite toggle callback
        favorite_tester = StreamlitComponentTester(_handle_favorite_toggle)

        with (
            patch("src.services.job_service.JobService.toggle_favorite") as mock_toggle,
            patch("streamlit.rerun"),
        ):
            favorite_tester.run_component(sample_job.id)
            mock_toggle.assert_called_with(sample_job.id)

        # 4. Test view details callback
        details_tester = StreamlitComponentTester(_handle_view_details)
        details_tester.run_component(sample_job.id)

        state = details_tester.get_session_state()
        assert state.get("view_job_id") == sample_job.id

    def test_jobs_grid_and_list_render_same_jobs_differently(self, sample_jobs_list):
        """Test that grid and list rendering produce different layouts for same jobs."""
        # Test grid rendering
        grid_tester = StreamlitComponentTester(render_jobs_grid)

        with (
            patch("src.ui.styles.styles.apply_job_grid_styles"),
            patch("streamlit.columns") as mock_grid_columns,
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown"),
            patch("src.ui.components.cards.job_card.render_job_card"),
        ):
            mock_grid_columns.return_value = [Mock(), Mock(), Mock()]
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            grid_tester.run_component(sample_jobs_list, num_columns=3)

            # Grid uses columns
            mock_grid_columns.assert_called()

        # Test list rendering
        list_tester = StreamlitComponentTester(render_jobs_list)

        with (
            patch("src.ui.components.cards.job_card.render_job_card"),
            patch("streamlit.markdown") as mock_list_markdown,
        ):
            list_tester.run_component(sample_jobs_list)

            # List uses separators between jobs
            separator_calls = [
                call
                for call in mock_list_markdown.call_args_list
                if len(call) > 0 and call[0][0] == "---"
            ]
            assert len(separator_calls) == len(sample_jobs_list)

    def test_error_recovery_in_callback_functions(self, sample_job):
        """Test error recovery and user feedback in callback functions."""
        # Test that callback errors don't break the UI and provide user feedback

        # Test status change with service error
        status_tester = StreamlitComponentTester(_handle_status_change)
        status_tester.set_session_state(**{f"status_{sample_job.id}": "Applied"})

        with (
            patch(
                "src.services.job_service.JobService.update_job_status",
                side_effect=Exception("Database connection failed"),
            ),
            patch("streamlit.error") as mock_error,
        ):
            # Should handle error gracefully
            status_tester.run_component(sample_job.id)

            # Verify error message is shown to user
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to update job status" in error_message

        # Test favorite toggle with service error
        favorite_tester = StreamlitComponentTester(_handle_favorite_toggle)

        with (
            patch(
                "src.services.job_service.JobService.toggle_favorite",
                side_effect=Exception("Network timeout"),
            ),
            patch("streamlit.error") as mock_error,
        ):
            # Should handle error gracefully
            favorite_tester.run_component(sample_job.id)

            # Verify error message is shown to user
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Failed to toggle favorite" in error_message
