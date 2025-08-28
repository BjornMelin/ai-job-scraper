"""Comprehensive tests for the Jobs page UI component.

This module tests the Jobs page functionality including:
- Page rendering and component display
- Session state management and URL sync
- User interaction flows (refresh, tab switching, modal display)
- Fragment auto-refresh behavior
- Job filtering and statistics
- Error states and edge cases
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tests.utils.streamlit_utils import StreamlitComponentTester

from src.schemas import Job
from src.ui.pages.jobs import (
    _execute_scraping_safely,
    _get_applied_jobs,
    _get_favorites_jobs,
    _get_filtered_jobs,
    _handle_job_details_modal,
    _handle_refresh_jobs,
    _render_action_bar,
    _render_active_sources_metric,
    _render_job_display,
    _render_job_tabs,
    _render_last_refresh_status,
    _render_metric_cards,
    _render_page_header,
    _render_progress_visualization,
    _render_statistics_dashboard,
    render_jobs_page,
    show_job_details_modal,
)


@pytest.fixture
def sample_jobs():
    """Create sample Job objects for testing."""
    now = datetime.now(UTC)
    return [
        Job(
            id=1,
            title="Senior Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            description="Exciting Python role with ML focus",
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
            description="Work on cutting-edge ML models",
            posted_date=now - timedelta(days=3),
            last_seen=now,
            favorite=False,
            application_status="New",
            notes="",
            link="https://example.com/job2",
        ),
        Job(
            id=3,
            title="Full Stack Engineer",
            company="Scale Inc",
            location="New York, NY",
            description="Build scalable web applications",
            posted_date=now - timedelta(days=2),
            last_seen=now,
            favorite=True,
            application_status="Interested",
            notes="Remote friendly",
            link="https://example.com/job3",
        ),
    ]


class TestJobsPageInitialization:
    """Test jobs page initialization and basic rendering."""

    def test_render_jobs_page_basic_structure(self, sample_jobs):
        """Test basic page structure renders correctly."""
        tester = StreamlitComponentTester(render_jobs_page)

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=sample_jobs),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch("src.ui.pages.jobs.validate_url_params", return_value={}),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
        ):
            # Set initial session state
            tester.set_session_state(
                filters={
                    "keyword": "",
                    "company": [],
                    "date_from": None,
                    "date_to": None,
                },
                selected_tab="all",
            )

            # Run the component
            tester.run_component()

            # Verify session state is properly initialized
            state = tester.get_session_state()
            assert "filters" in state
            assert "selected_tab" in state

    def test_render_jobs_page_with_url_validation_errors(self):
        """Test page handles URL validation errors gracefully."""
        tester = StreamlitComponentTester(render_jobs_page)

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=[]),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch(
                "src.ui.pages.jobs.validate_url_params",
                return_value={"invalid_param": "Invalid value"},
            ),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
            patch("streamlit.warning") as mock_warning,
        ):
            tester.set_session_state(filters={}, selected_tab="all")
            tester.run_component()

            # Verify warning is displayed for invalid URL params
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            assert "URL parameters are invalid" in call_args

    def test_render_jobs_page_no_jobs_found(self):
        """Test page displays appropriate message when no jobs found."""
        tester = StreamlitComponentTester(render_jobs_page)

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=[]),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch("src.ui.pages.jobs.validate_url_params", return_value={}),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
            patch("streamlit.info") as mock_info,
        ):
            tester.set_session_state(filters={}, selected_tab="all")
            tester.run_component()

            # Verify info message is displayed
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "No jobs found" in call_args


class TestPageHeader:
    """Test page header rendering."""

    def test_render_page_header_displays_title_and_timestamp(self):
        """Test page header displays title and timestamp correctly."""
        tester = StreamlitComponentTester(_render_page_header)

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown") as mock_markdown,
        ):
            # Mock columns context managers
            mock_col1 = Mock()
            mock_col2 = Mock()
            mock_columns.return_value = [mock_col1, mock_col2]

            tester.run_component()

            # Verify columns are created correctly
            mock_columns.assert_called_with([3, 1])

            # Verify markdown content includes title and timestamp
            assert mock_markdown.call_count >= 2
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            assert any("AI Job Tracker" in call for call in markdown_calls)
            assert any("Last updated" in call for call in markdown_calls)


class TestActionBar:
    """Test action bar functionality."""

    def test_render_action_bar_creates_refresh_button(self):
        """Test action bar renders refresh button with correct properties."""
        tester = StreamlitComponentTester(_render_action_bar)

        with (
            patch("streamlit.container"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
            patch("src.ui.pages.jobs._render_last_refresh_status"),
            patch("src.ui.pages.jobs._render_active_sources_metric"),
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_button.return_value = False

            tester.run_component()

            # Verify button is created with correct properties
            mock_button.assert_called()
            button_call = mock_button.call_args
            assert "🔄 Refresh Jobs" in button_call[0]
            assert button_call[1]["type"] == "primary"
            assert button_call[1]["use_container_width"] is True

    def test_refresh_button_triggers_scraping(self, sample_jobs):
        """Test refresh button click triggers scraping workflow."""
        tester = StreamlitComponentTester(_render_action_bar)

        with (
            patch("streamlit.container"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
            patch("src.ui.pages.jobs._handle_refresh_jobs") as mock_handle_refresh,
            patch("src.ui.pages.jobs._render_last_refresh_status"),
            patch("src.ui.pages.jobs._render_active_sources_metric"),
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_button.return_value = True  # Simulate button click

            tester.run_component()

            # Verify refresh handler is called
            mock_handle_refresh.assert_called_once()


class TestJobRefreshWorkflow:
    """Test job refresh functionality."""

    def test_handle_refresh_jobs_successful_execution(self):
        """Test successful job refresh execution."""
        tester = StreamlitComponentTester(_handle_refresh_jobs)
        sync_stats = {
            "inserted": 5,
            "updated": 3,
            "archived": 1,
            "deleted": 0,
            "skipped": 2,
        }

        with (
            patch(
                "src.services.company_service.CompanyService.get_active_companies_count",
                return_value=10,
            ),
            patch(
                "src.ui.pages.jobs._execute_scraping_safely", return_value=sync_stats
            ),
            patch("streamlit.status") as mock_status,
            patch("streamlit.rerun") as mock_rerun,
        ):
            # Mock the status context manager
            mock_status_context = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            tester.run_component()

            # Verify last_scrape is updated in session state
            state = tester.get_session_state()
            assert "last_scrape" in state
            assert isinstance(state["last_scrape"], datetime)

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_handle_refresh_jobs_scraping_failure(self):
        """Test job refresh handles scraping failures gracefully."""
        tester = StreamlitComponentTester(_handle_refresh_jobs)

        with (
            patch(
                "src.services.company_service.CompanyService.get_active_companies_count",
                return_value=10,
            ),
            patch(
                "src.ui.pages.jobs._execute_scraping_safely",
                side_effect=Exception("Scraping failed"),
            ),
            patch("streamlit.status") as mock_status,
        ):
            # Mock the status context manager
            mock_status_context = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            tester.run_component()

            # Verify error status is displayed
            assert mock_status.call_count >= 2  # Both running and error status

    def test_execute_scraping_safely_successful(self):
        """Test execute_scraping_safely handles async scraping correctly."""
        sync_stats = {
            "inserted": 3,
            "updated": 2,
            "archived": 0,
            "deleted": 0,
            "skipped": 1,
        }

        with patch("asyncio.run", return_value=sync_stats) as mock_asyncio_run:
            result = _execute_scraping_safely()

            # Verify asyncio.run is called with scrape_all
            mock_asyncio_run.assert_called_once()
            assert result == sync_stats

    def test_execute_scraping_safely_handles_exception(self):
        """Test execute_scraping_safely handles exceptions properly."""
        with patch("asyncio.run", side_effect=Exception("Async error")):
            with pytest.raises(Exception) as exc_info:
                _execute_scraping_safely()

            assert "Async error" in str(exc_info.value)


class TestJobFiltering:
    """Test job filtering functionality."""

    def test_get_filtered_jobs_with_filters(self, sample_jobs):
        """Test job filtering with session state filters."""
        tester = StreamlitComponentTester(_get_filtered_jobs)

        # Set up session state with filters
        tester.set_session_state(
            filters={
                "keyword": "python",
                "company": ["Tech Corp"],
                "date_from": datetime.now(UTC) - timedelta(days=7),
                "date_to": datetime.now(UTC),
            }
        )

        with patch(
            "src.services.job_service.JobService.get_filtered_jobs",
            return_value=sample_jobs,
        ) as mock_get_jobs:
            tester.run_component()

            # Verify JobService is called with correct filters
            mock_get_jobs.assert_called_once()
            call_args = mock_get_jobs.call_args[0][0]
            assert call_args["text_search"] == "python"
            assert call_args["company"] == ["Tech Corp"]
            assert "date_from" in call_args
            assert "date_to" in call_args

    def test_get_filtered_jobs_handles_service_exception(self):
        """Test job filtering handles service exceptions gracefully."""
        tester = StreamlitComponentTester(_get_filtered_jobs)

        tester.set_session_state(filters={})

        with patch(
            "src.services.job_service.JobService.get_filtered_jobs",
            side_effect=Exception("DB error"),
        ):
            result = tester.run_component()

            # Should return empty list on exception
            assert result == []

    def test_get_favorites_jobs(self, sample_jobs):
        """Test favorites job filtering."""
        favorites = [job for job in sample_jobs if job.favorite]

        tester = StreamlitComponentTester(_get_favorites_jobs)
        tester.set_session_state(filters={})

        with patch(
            "src.services.job_service.JobService.get_filtered_jobs",
            return_value=favorites,
        ) as mock_get_jobs:
            tester.run_component()

            # Verify favorites_only filter is set
            call_args = mock_get_jobs.call_args[0][0]
            assert call_args["favorites_only"] is True

    def test_get_applied_jobs(self, sample_jobs):
        """Test applied job filtering."""
        applied = [job for job in sample_jobs if job.application_status == "Applied"]

        tester = StreamlitComponentTester(_get_applied_jobs)
        tester.set_session_state(filters={})

        with patch(
            "src.services.job_service.JobService.get_filtered_jobs",
            return_value=applied,
        ) as mock_get_jobs:
            tester.run_component()

            # Verify application_status filter is set
            call_args = mock_get_jobs.call_args[0][0]
            assert call_args["application_status"] == ["Applied"]


class TestJobTabs:
    """Test job tab functionality."""

    def test_render_job_tabs_calls_fragment(self):
        """Test job tabs rendering calls the appropriate fragment."""
        tester = StreamlitComponentTester(_render_job_tabs)

        with patch("src.ui.pages.jobs._job_list_fragment") as mock_fragment:
            tester.run_component()

            # Verify fragment is called
            mock_fragment.assert_called_once()

    def test_render_job_display_card_view(self, sample_jobs):
        """Test job display in card view mode."""
        tester = StreamlitComponentTester(_render_job_display)

        with (
            patch(
                "src.ui.ui_rendering.select_view_mode", return_value=("Card", 3)
            ) as mock_select_view,
            patch("src.ui.ui_rendering.apply_view_mode") as mock_apply_view,
        ):
            tester.run_component(sample_jobs, "all")

            # Verify view mode functions are called
            mock_select_view.assert_called_with("all")
            mock_apply_view.assert_called_with(sample_jobs, "Card", 3)

    def test_render_job_display_empty_list(self):
        """Test job display with empty job list."""
        tester = StreamlitComponentTester(_render_job_display)

        with (
            patch("src.ui.ui_rendering.select_view_mode") as mock_select_view,
            patch("src.ui.ui_rendering.apply_view_mode") as mock_apply_view,
        ):
            tester.run_component([], "all")

            # Should return early without calling view functions
            mock_select_view.assert_not_called()
            mock_apply_view.assert_not_called()


class TestJobDetailsModal:
    """Test job details modal functionality."""

    def test_show_job_details_modal_renders_all_sections(self, sample_jobs):
        """Test job details modal renders all required sections."""
        job = sample_jobs[0]
        tester = StreamlitComponentTester(show_job_details_modal)

        with (
            patch("src.ui.ui_rendering.render_job_header") as mock_header,
            patch("src.ui.ui_rendering.render_job_status") as mock_status,
            patch(
                "src.ui.ui_rendering.render_notes_section", return_value="test notes"
            ) as mock_notes,
            patch("src.ui.ui_rendering.render_job_description") as mock_description,
            patch("src.ui.ui_rendering.render_action_buttons") as mock_actions,
        ):
            tester.run_component(job)

            # Verify all sections are rendered
            mock_header.assert_called_once_with(job)
            mock_status.assert_called_once_with(job)
            mock_notes.assert_called_once_with(job)
            mock_description.assert_called_once_with(job)
            mock_actions.assert_called_once_with(job, "test notes")

    def test_handle_job_details_modal_with_valid_job(self, sample_jobs):
        """Test modal handling with valid job ID."""
        tester = StreamlitComponentTester(_handle_job_details_modal)

        # Set up session state with view_job_id
        tester.set_session_state(view_job_id=1)

        with patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal:
            tester.run_component(sample_jobs)

            # Verify modal is shown with correct job
            mock_show_modal.assert_called_once()
            called_job = mock_show_modal.call_args[0][0]
            assert called_job.id == 1

    def test_handle_job_details_modal_job_not_found(self, sample_jobs):
        """Test modal handling when job is not found in current list."""
        tester = StreamlitComponentTester(_handle_job_details_modal)

        # Set up session state with non-existent job ID
        tester.set_session_state(view_job_id=999)

        with patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal:
            tester.run_component(sample_jobs)

            # Modal should not be shown and view_job_id should be cleared
            mock_show_modal.assert_not_called()
            state = tester.get_session_state()
            assert state.get("view_job_id") is None


class TestStatisticsDashboard:
    """Test statistics dashboard functionality."""

    def test_render_statistics_dashboard_with_jobs(self, sample_jobs):
        """Test statistics dashboard renders with job data."""
        tester = StreamlitComponentTester(_render_statistics_dashboard)

        with (
            patch("src.ui.pages.jobs._render_metric_cards") as mock_metrics,
            patch("src.ui.pages.jobs._render_progress_visualization") as mock_progress,
            patch("streamlit.markdown"),
        ):
            tester.run_component(sample_jobs)

            # Verify dashboard sections are rendered
            mock_metrics.assert_called_once()
            mock_progress.assert_called_once()

            # Check that metrics are calculated correctly
            metric_call_kwargs = mock_metrics.call_args[1]
            assert metric_call_kwargs["total_jobs"] == 3
            assert metric_call_kwargs["applied"] == 1
            assert metric_call_kwargs["favorites"] == 2

    def test_render_metric_cards_calculations(self, sample_jobs):
        """Test metric cards calculate statistics correctly."""
        tester = StreamlitComponentTester(_render_metric_cards)

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.metric") as mock_metric,
        ):
            mock_columns.return_value = [Mock() for _ in range(6)]

            tester.run_component(
                total_jobs=3,
                new_jobs=1,
                interested=1,
                applied=1,
                favorites=2,
                rejected=0,
            )

            # Verify metrics are displayed
            assert mock_metric.call_count == 6

    def test_render_progress_visualization(self):
        """Test progress visualization renders correctly."""
        tester = StreamlitComponentTester(_render_progress_visualization)

        with (
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.progress") as mock_progress,
            patch("streamlit.metric") as mock_metric,
        ):
            mock_columns.return_value = [Mock(), Mock()]

            tester.run_component(
                total_jobs=10, new_jobs=4, interested=3, applied=2, rejected=1
            )

            # Verify progress bars and application rate metric
            assert mock_progress.call_count == 4  # One for each status
            mock_metric.assert_called_once()


class TestLastRefreshStatus:
    """Test last refresh status display."""

    def test_render_last_refresh_status_recent(self):
        """Test last refresh status displays recent refresh correctly."""
        tester = StreamlitComponentTester(_render_last_refresh_status)

        # Set recent refresh time (30 minutes ago)
        recent_time = datetime.now(UTC) - timedelta(minutes=30)
        tester.set_session_state(last_scrape=recent_time)

        with patch("streamlit.info") as mock_info:
            tester.run_component()

            # Verify info message shows minutes
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "30 minute" in call_args

    def test_render_last_refresh_status_hours_ago(self):
        """Test last refresh status displays hours correctly."""
        tester = StreamlitComponentTester(_render_last_refresh_status)

        # Set refresh time 2 hours ago
        hours_ago = datetime.now(UTC) - timedelta(hours=2)
        tester.set_session_state(last_scrape=hours_ago)

        with patch("streamlit.info") as mock_info:
            tester.run_component()

            # Verify info message shows hours
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "2 hour" in call_args

    def test_render_last_refresh_status_no_refresh(self):
        """Test last refresh status when no refresh has occurred."""
        tester = StreamlitComponentTester(_render_last_refresh_status)

        # Don't set last_scrape in session state
        with patch("streamlit.info") as mock_info:
            tester.run_component()

            # Verify default message
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "No recent refresh" in call_args


class TestActiveSourcesMetric:
    """Test active sources metric display."""

    def test_render_active_sources_metric_success(self):
        """Test active sources metric displays correctly."""
        tester = StreamlitComponentTester(_render_active_sources_metric)

        with (
            patch(
                "src.services.company_service.CompanyService.get_active_companies_count",
                return_value=5,
            ),
            patch("streamlit.metric") as mock_metric,
        ):
            tester.run_component()

            # Verify metric is displayed with correct count
            mock_metric.assert_called_once_with("Active Sources", 5)

    def test_render_active_sources_metric_service_error(self):
        """Test active sources metric handles service errors."""
        tester = StreamlitComponentTester(_render_active_sources_metric)

        with (
            patch(
                "src.services.company_service.CompanyService.get_active_companies_count",
                side_effect=Exception("Service error"),
            ),
            patch("streamlit.metric") as mock_metric,
        ):
            tester.run_component()

            # Verify metric shows 0 on error
            mock_metric.assert_called_once_with("Active Sources", 0)


class TestSessionStateManagement:
    """Test session state management in jobs page."""

    def test_tab_selection_persistence(self):
        """Test tab selection persists across page interactions."""
        tester = StreamlitComponentTester(render_jobs_page)

        # Initialize with specific tab
        tester.set_session_state(selected_tab="favorites", filters={})

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=[]),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch("src.ui.pages.jobs.validate_url_params", return_value={}),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
        ):
            tester.run_component()

            # Verify tab selection is maintained
            state = tester.get_session_state()
            assert state["selected_tab"] == "favorites"

    def test_filters_state_preservation(self):
        """Test filter state is preserved across interactions."""
        initial_filters = {
            "keyword": "python developer",
            "company": ["Tech Corp", "AI Startup"],
            "date_from": datetime.now(UTC) - timedelta(days=30),
            "date_to": datetime.now(UTC),
        }

        tester = StreamlitComponentTester(render_jobs_page)
        tester.set_session_state(filters=initial_filters, selected_tab="all")

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=[]),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch("src.ui.pages.jobs.validate_url_params", return_value={}),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
        ):
            tester.run_component()

            # Verify filters are maintained
            state = tester.get_session_state()
            assert state["filters"]["keyword"] == "python developer"
            assert state["filters"]["company"] == ["Tech Corp", "AI Startup"]

    def test_view_job_id_cleanup(self, sample_jobs):
        """Test view_job_id is properly cleaned up when job not found."""
        tester = StreamlitComponentTester(_handle_job_details_modal)

        # Set invalid job ID
        tester.set_session_state(view_job_id=999)

        tester.run_component(sample_jobs)

        # Verify job ID is cleared
        state = tester.get_session_state()
        assert state.get("view_job_id") is None


class TestErrorHandling:
    """Test error handling in jobs page components."""

    def test_page_handles_missing_session_state_gracefully(self):
        """Test page initializes properly with missing session state."""
        tester = StreamlitComponentTester(render_jobs_page)

        # Don't initialize session state
        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=[]),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch("src.ui.pages.jobs.validate_url_params", return_value={}),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
        ):
            # Should not raise exception
            tester.run_component()

            state = tester.get_session_state()
            # Basic state should be initialized
            assert isinstance(state, dict)

    def test_scraping_handles_invalid_response_format(self):
        """Test scraping handles invalid response format gracefully."""
        tester = StreamlitComponentTester(_handle_refresh_jobs)

        # Return invalid format (not dict)
        with (
            patch(
                "src.services.company_service.CompanyService.get_active_companies_count",
                return_value=10,
            ),
            patch(
                "src.ui.pages.jobs._execute_scraping_safely",
                return_value="invalid_format",
            ),
            patch("streamlit.status") as mock_status,
            patch("streamlit.error") as mock_error,
        ):
            mock_status_context = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            tester.run_component()

            # Verify error is displayed
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            assert "unexpected data format" in call_args


class TestFragmentBehavior:
    """Test Streamlit fragment behavior."""

    def test_job_list_fragment_auto_refresh(self, sample_jobs):
        """Test job list fragment can be configured for auto-refresh."""
        # This tests the fragment decorator is properly applied
        from src.ui.pages.jobs import _job_list_fragment

        # Verify the fragment has the expected attributes
        assert hasattr(_job_list_fragment, "__wrapped__")
        # Fragment should have run_every parameter set to "30s"

        tester = StreamlitComponentTester(_job_list_fragment)

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=sample_jobs),
            patch("src.ui.pages.jobs._get_favorites_jobs", return_value=[]),
            patch("src.ui.pages.jobs._get_applied_jobs", return_value=[]),
            patch("streamlit.button") as mock_button,
            patch("streamlit.columns") as mock_columns,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            mock_button.return_value = False

            tester.set_session_state(selected_tab="all")
            tester.run_component()

            # Fragment should execute without errors
            state = tester.get_session_state()
            assert "selected_tab" in state


# Integration tests combining multiple components
class TestJobsPageIntegration:
    """Integration tests for jobs page components working together."""

    def test_full_page_render_with_jobs(self, sample_jobs):
        """Test complete page renders correctly with job data."""
        tester = StreamlitComponentTester(render_jobs_page)

        with (
            patch("src.ui.pages.jobs._get_filtered_jobs", return_value=sample_jobs),
            patch(
                "src.ui.pages.jobs._get_favorites_jobs", return_value=[sample_jobs[0]]
            ),
            patch("src.ui.pages.jobs._get_applied_jobs", return_value=[sample_jobs[0]]),
            patch("src.ui.pages.jobs.render_sidebar"),
            patch("src.ui.pages.jobs.validate_url_params", return_value={}),
            patch("src.ui.pages.jobs.sync_tab_from_url"),
            patch("src.ui.pages.jobs.background_task_status_fragment"),
            patch("src.ui.ui_rendering.select_view_mode", return_value=("Card", 3)),
            patch("src.ui.ui_rendering.apply_view_mode"),
        ):
            tester.set_session_state(
                filters={"keyword": "", "company": []}, selected_tab="all"
            )

            # Should render without errors
            tester.run_component()

            state = tester.get_session_state()
            assert state["selected_tab"] == "all"

    def test_refresh_workflow_integration(self, sample_jobs):
        """Test complete refresh workflow from button click to page update."""
        sync_stats = {
            "inserted": 2,
            "updated": 1,
            "archived": 0,
            "deleted": 0,
            "skipped": 1,
        }

        tester = StreamlitComponentTester(_handle_refresh_jobs)

        with (
            patch(
                "src.services.company_service.CompanyService.get_active_companies_count",
                return_value=5,
            ),
            patch(
                "src.ui.pages.jobs._execute_scraping_safely", return_value=sync_stats
            ),
            patch("streamlit.status") as mock_status,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_status_context = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)

            tester.run_component()

            # Verify complete workflow
            state = tester.get_session_state()
            assert "last_scrape" in state
            assert isinstance(state["last_scrape"], datetime)
            mock_rerun.assert_called_once()

    def test_modal_display_workflow(self, sample_jobs):
        """Test complete modal display workflow."""
        tester = StreamlitComponentTester(_handle_job_details_modal)

        # Set up to show modal for first job
        tester.set_session_state(view_job_id=1)

        with patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal:
            tester.run_component(sample_jobs)

            # Verify correct job is passed to modal
            mock_show_modal.assert_called_once()
            displayed_job = mock_show_modal.call_args[0][0]
            assert displayed_job.id == 1
            assert displayed_job.title == "Senior Python Developer"
