"""Tests for Job Browser page functionality.

Tests job browsing, filtering, grid/list views, modal functionality,
and complete user workflows for the jobs page.
"""

from unittest.mock import patch

import pytest

from src.schemas import Job
from src.ui.pages.jobs import (
    _get_filtered_jobs,
    _handle_job_details_modal,
    _handle_refresh_jobs,
    _render_action_bar,
    _render_job_tabs,
    _render_page_header,
    _render_statistics_dashboard,
    render_jobs_page,
)


class TestJobDetailsModal:
    """Test job details modal functionality."""

    def test_show_job_details_modal_displays_job_info(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test modal displays complete job information."""
        # Act - Test the underlying functions directly
        from src.ui.ui_rendering import (
            render_job_description,
            render_job_header,
            render_job_status,
            render_notes_section,
        )

        # Call the functions directly to test their behavior
        render_job_header(sample_job_dto)
        render_job_status(sample_job_dto)
        render_notes_section(sample_job_dto)
        render_job_description(sample_job_dto)

        # Assert markdown calls contain job information
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]

        # Check that key job information is displayed
        html_content = " ".join(markdown_calls)
        assert sample_job_dto.title in html_content
        assert sample_job_dto.company in html_content
        assert sample_job_dto.location in html_content
        assert sample_job_dto.description in html_content

    def test_show_job_details_modal_displays_status_with_icon(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test modal displays application status with appropriate icon."""
        # Arrange
        sample_job_dto.application_status = "Applied"

        # Act - Test helper function directly
        from src.ui.ui_rendering import render_job_status

        render_job_status(sample_job_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        status_calls = [
            call for call in markdown_calls if "Status:" in call and "🟢" in call
        ]
        assert status_calls

    def test_show_job_details_modal_creates_notes_text_area(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test modal creates text area for job notes."""
        # Act - Test helper function directly
        from src.ui.ui_rendering import render_notes_section

        render_notes_section(sample_job_dto)

        # Assert
        text_area_calls = mock_streamlit["text_area"].call_args_list
        notes_call = text_area_calls[0]  # Should be the first (and only) text area

        assert notes_call.args[0] == "Your notes about this position"
        assert notes_call.kwargs["key"] == f"modal_notes_{sample_job_dto.id}"
        assert notes_call.kwargs["value"] == (sample_job_dto.notes or "")

    def test_show_job_details_modal_creates_action_buttons(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test modal creates save notes, apply now, and close buttons."""
        # Act - Test helper function directly
        from src.ui.ui_rendering import render_action_buttons

        render_action_buttons(sample_job_dto, "test notes")

        # Assert
        button_calls = mock_streamlit["button"].call_args_list
        button_labels = [call.args[0] for call in button_calls]

        assert "Save Notes" in button_labels
        assert "Close" in button_labels

    def test_show_job_details_modal_creates_apply_link_button(
        self,
        mock_streamlit,
        sample_job_dto,
    ):
        """Test modal creates apply now link button when job has link."""
        # Arrange
        sample_job_dto.link = "https://example.com/apply"

        # Act - Test helper function directly
        from src.ui.ui_rendering import render_action_buttons

        render_action_buttons(sample_job_dto, "test notes")

        # Assert
        link_button_calls = mock_streamlit["link_button"].call_args_list
        assert link_button_calls

        apply_button = link_button_calls[0]
        assert apply_button.args[0] == "Apply Now"
        assert apply_button.args[1] == sample_job_dto.link

    # Note: Save job notes functionality is now handled through data editor
    # These tests are removed as the function _save_job_notes no longer exists


class TestJobDetailsModalHandling:
    """Test job details modal handling in context of jobs page."""

    def test_handle_job_details_modal_shows_modal_when_job_selected(
        self,
        mock_streamlit,  # pylint: disable=unused-argument
        sample_jobs_dto,
        mock_session_state,
    ):
        """Test modal handler shows modal when job is selected."""
        # Arrange
        mock_session_state["view_job_id"] = sample_jobs_dto[0].id

        with patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal:
            # Act
            _handle_job_details_modal(sample_jobs_dto)

            # Assert
            mock_show_modal.assert_called_once_with(sample_jobs_dto[0])

    def test_handle_job_details_modal_clears_selection_when_job_not_found(
        self,
        sample_jobs_dto,
        mock_session_state,
    ):
        """Test modal handler clears selection when job not found in current list."""
        # Arrange
        mock_session_state["view_job_id"] = 999  # Non-existent job ID

        # Act
        _handle_job_details_modal(sample_jobs_dto)

        # Assert
        assert mock_session_state["view_job_id"] is None

    def test_handle_job_details_modal_does_nothing_when_no_job_selected(
        self,
        sample_jobs_dto,
        mock_session_state,
    ):
        """Test modal handler does nothing when no job is selected."""
        # Arrange - no view_job_id in session state

        with patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal:
            # Act
            _handle_job_details_modal(sample_jobs_dto)

            # Assert
            mock_show_modal.assert_not_called()


class TestJobPageHeader:
    """Test job page header rendering."""

    def test_render_page_header_displays_title_and_description(self, mock_streamlit):
        """Test page header displays title and descriptive text."""
        # Act
        _render_page_header()

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]

        # Check for title and description in HTML content
        html_content = " ".join(markdown_calls)
        assert "AI Job Tracker" in html_content
        assert "Track and manage your job applications efficiently" in html_content

    def test_render_page_header_displays_last_updated_time(self, mock_streamlit):
        """Test page header displays current timestamp."""
        # Act
        _render_page_header()

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        html_content = " ".join(markdown_calls)

        # Should contain "Last updated:" text
        assert "Last updated:" in html_content


class TestJobActionBar:
    """Test job action bar functionality."""

    def test_render_action_bar_creates_refresh_button(
        self,
        mock_streamlit,
        mock_company_service,
    ):
        """Test action bar creates refresh jobs button."""
        # Arrange
        mock_company_service["jobs_page"].get_active_companies_count.return_value = 3

        # Act
        _render_action_bar()

        # Assert
        button_calls = mock_streamlit["button"].call_args_list
        refresh_button = next(
            call for call in button_calls if "🔄 Refresh Jobs" in call.args[0]
        )

        assert refresh_button is not None
        assert refresh_button.kwargs["type"] == "primary"

    def test_render_action_bar_displays_active_sources_metric(
        self,
        mock_streamlit,
        mock_company_service,
    ):
        """Test action bar displays active sources count."""
        # Arrange
        mock_company_service["jobs_page"].get_active_companies_count.return_value = 5

        # Act
        _render_action_bar()

        # Assert
        metric_calls = mock_streamlit["metric"].call_args_list
        active_sources_metric = next(
            call for call in metric_calls if call.args[0] == "Active Sources"
        )

        assert active_sources_metric.args[1] == 5

    def test_render_action_bar_handles_service_failure_for_active_sources(
        self,
        mock_streamlit,
        mock_company_service,
    ):
        """Test action bar handles service failure gracefully for active sources."""
        # Arrange
        mock_company_service[
            "jobs_page"
        ].get_active_companies_count.side_effect = Exception("Database error")

        # Act
        _render_action_bar()

        # Assert
        metric_calls = mock_streamlit["metric"].call_args_list
        active_sources_metric = next(
            call for call in metric_calls if call.args[0] == "Active Sources"
        )

        # Should default to 0 on error
        assert active_sources_metric.args[1] == 0


class TestJobRefresh:
    """Test job refresh functionality."""

    def test_handle_refresh_jobs_executes_scraping(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test refresh handler executes scraping process."""
        # Arrange
        mock_sync_stats = {
            "inserted": 5,
            "updated": 3,
            "archived": 1,
            "deleted": 0,
            "skipped": 2,
        }

        with patch(
            "src.ui.pages.jobs._execute_scraping_safely",
            return_value=mock_sync_stats,
        ):
            # Act
            _handle_refresh_jobs()

            # Assert
            # Should show success message with sync statistics
            success_calls = mock_streamlit["success"].call_args_list
            assert success_calls

            success_message = success_calls[0].args[0]
            assert (
                "Success! Processed 8 jobs" in success_message
            )  # 5 inserted + 3 updated
            assert "Inserted: 5" in success_message
            assert "Updated: 3" in success_message
            assert "Archived: 1" in success_message

    def test_handle_refresh_jobs_handles_scraping_failure(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test refresh handler handles scraping failure gracefully."""
        # Arrange
        with patch(
            "src.ui.pages.jobs._execute_scraping_safely",
            side_effect=Exception("Scraping failed"),
        ):
            # Act
            _handle_refresh_jobs()

            # Assert
            mock_streamlit["error"].assert_called_with("❌ Scrape failed")

    def test_handle_refresh_jobs_validates_sync_stats_format(
        self,
        mock_streamlit,
        mock_session_state,
    ):
        """Test refresh handler validates sync stats are in expected format."""
        # Arrange - return invalid format
        with patch(
            "src.ui.pages.jobs._execute_scraping_safely",
            return_value="invalid_format",
        ):
            # Act
            _handle_refresh_jobs()

            # Assert
            error_calls = mock_streamlit["error"].call_args_list
            error_messages = [call.args[0] for call in error_calls]
            assert any("unexpected data format" in msg for msg in error_messages)


class TestJobTabs:
    """Test job tabs rendering and functionality."""

    def test_render_job_tabs_creates_three_tabs_with_counts(
        self,
        mock_streamlit,
        sample_jobs_dto,
        mock_job_service,
        mock_session_state,
    ):
        """Test job tabs creates all jobs, favorites, and applied tabs with counts."""
        # Arrange - configure service mocks to return filtered data and session state
        favorites = [j for j in sample_jobs_dto if j.favorite]
        applied = [j for j in sample_jobs_dto if j.application_status == "Applied"]

        # Initialize session state filters
        mock_session_state.filters = {}

        mock_job_service["jobs_page"].get_filtered_jobs.side_effect = lambda filters: (
            favorites
            if filters.get("favorites_only")
            else applied
            if filters.get("application_status") == ["Applied"]
            else sample_jobs_dto
        )

        # Act
        _render_job_tabs(sample_jobs_dto)

        # Assert
        tabs_calls = mock_streamlit["tabs"].call_args_list
        assert tabs_calls

        tab_labels = tabs_calls[0].args[0]
        assert len(tab_labels) == 3

        # Check tab labels contain correct counts
        # Based on sample_jobs_dto: 4 total, 2 favorites, 1 applied
        assert "All Jobs 📋 (4)" in tab_labels
        assert "Favorites ⭐ (2)" in tab_labels
        assert "Applied ✅ (1)" in tab_labels

    def test_render_job_tabs_shows_empty_state_for_no_favorites(self, mock_streamlit):
        """Test favorites tab shows empty state when no favorites exist."""
        # Arrange - jobs with no favorites
        jobs = [
            Job(
                id=1,
                company_id=1,
                company="Test Co",
                title="Test Job",
                description="Test description",
                link="https://test.com",
                location="Test City",
                content_hash="hash",
                favorite=False,
                application_status="New",
            ),
        ]

        # Act
        _render_job_tabs(jobs)

        # Assert
        info_calls = mock_streamlit["info"].call_args_list
        favorites_info = [
            call for call in info_calls if "No favorite jobs yet" in call.args[0]
        ]
        assert favorites_info

    def test_render_job_tabs_shows_empty_state_for_no_applied(self, mock_streamlit):
        """Test applied tab shows empty state when no applied jobs exist."""
        # Arrange - jobs with no applied status
        jobs = [
            Job(
                id=1,
                company_id=1,
                company="Test Co",
                title="Test Job",
                description="Test description",
                link="https://test.com",
                location="Test City",
                content_hash="hash",
                favorite=False,
                application_status="New",
            ),
        ]

        # Act
        _render_job_tabs(jobs)

        # Assert
        info_calls = mock_streamlit["info"].call_args_list
        applied_info = [
            call for call in info_calls if "No applications yet" in call.args[0]
        ]
        assert applied_info


class TestJobStatistics:
    """Test job statistics dashboard."""

    def test_render_statistics_dashboard_displays_metrics(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test statistics dashboard displays all job metrics."""
        # Act
        _render_statistics_dashboard(sample_jobs_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]

        # Check for dashboard title
        assert any("📊 Dashboard" in call for call in markdown_calls)

        # Check for metric values in HTML content
        html_content = " ".join(markdown_calls)
        assert "4" in html_content  # Total jobs
        assert "2" in html_content  # Favorites (2 jobs marked as favorite)

    def test_render_statistics_dashboard_calculates_application_rate(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test statistics dashboard calculates correct application rate."""
        # Act
        _render_statistics_dashboard(sample_jobs_dto)

        # Assert
        markdown_calls = [
            call.args[0] for call in mock_streamlit["markdown"].call_args_list
        ]
        html_content = " ".join(markdown_calls)

        # Should calculate application rate (1 applied out of 4 total = 25%)
        assert "25.0%" in html_content or "25%" in html_content

    def test_render_statistics_dashboard_displays_progress_bars(
        self,
        mock_streamlit,
        sample_jobs_dto,
    ):
        """Test statistics dashboard displays progress bars for job statuses."""
        # Act
        _render_statistics_dashboard(sample_jobs_dto)

        # Assert
        progress_calls = mock_streamlit["progress"].call_args_list

        # Should have progress bars for each status
        assert len(progress_calls) >= 4  # New, Interested, Applied, Rejected


class TestJobFiltering:
    """Test job filtering functionality."""

    def test_get_filtered_jobs_calls_service_with_filters(
        self,
        mock_session_state,
        mock_job_service,
    ):
        """Test job filtering calls service with correct filter parameters."""
        # Arrange
        mock_session_state.filters = {
            "keyword": "engineer",
            "company": ["Tech Corp"],
            "date_from": None,
            "date_to": None,
        }
        mock_job_service["jobs_page"].get_filtered_jobs.return_value = []

        # Act
        _get_filtered_jobs()

        # Assert
        mock_job_service["jobs_page"].get_filtered_jobs.assert_called_once()
        call_args = mock_job_service["jobs_page"].get_filtered_jobs.call_args[0][0]

        assert call_args["text_search"] == "engineer"
        assert call_args["company"] == ["Tech Corp"]
        assert call_args["favorites_only"] is False
        assert call_args["include_archived"] is False

    def test_get_filtered_jobs_handles_service_failure(
        self,
        mock_session_state,
        mock_job_service,
    ):
        """Test job filtering handles service failure gracefully."""
        # Arrange
        mock_session_state.filters = {}
        mock_job_service["jobs_page"].get_filtered_jobs.side_effect = Exception(
            "Database error",
        )

        # Act
        result = _get_filtered_jobs()

        # Assert
        assert result == []


class TestJobsPageIntegration:
    """Integration tests for complete jobs page workflows."""

    def _setup_jobs_page_test(
        self,
        mock_session_state,
        mock_job_service,
        mock_company_service,
        jobs,
        active_companies=3,
    ):
        """Common setup for jobs page tests."""
        mock_session_state.filters = {}
        mock_job_service["jobs_page"].get_filtered_jobs.return_value = jobs
        mock_company_service[
            "jobs_page"
        ].get_active_companies_count.return_value = active_companies

    def test_render_jobs_page_displays_complete_interface(  # pylint: disable=R0917
        self,
        mock_streamlit,
        mock_session_state,
        mock_job_service,
        mock_company_service,
        sample_jobs_dto,
        ensure_proper_job_data_types,
    ):
        """Test complete jobs page renders all components."""
        # Arrange - ensure proper data types to prevent MagicMock string errors
        validated_jobs = ensure_proper_job_data_types(sample_jobs_dto)
        self._setup_jobs_page_test(
            mock_session_state,
            mock_job_service,
            mock_company_service,
            validated_jobs,
        )

        with (
            patch("src.ui.pages.jobs.render_sidebar") as mock_render_sidebar,
            patch(
                "src.ui.pages.jobs._get_filtered_jobs",
                return_value=validated_jobs,
            ) as mock_get_filtered,
            patch("src.ui.pages.jobs._handle_refresh_jobs"),
        ):
            # Act
            render_jobs_page()

            # Assert
            # Verify that mock functions are called correctly
            mock_render_sidebar.assert_called_once()
            mock_get_filtered.assert_called_once()

            # Should render page header - inline variable usage
            assert "AI Job Tracker" in " ".join(
                call.args[0] for call in mock_streamlit["markdown"].call_args_list
            )

            # Should render action bar - inline variable usage
            assert any(
                "🔄 Refresh Jobs" in label
                for label in [
                    call.args[0] for call in mock_streamlit["button"].call_args_list
                ]
            )

            # Should render job tabs
            assert mock_streamlit["tabs"].call_args_list

    def test_render_jobs_page_handles_no_jobs_gracefully(
        self,
        mock_streamlit,
        mock_session_state,
        mock_job_service,
        mock_company_service,
    ):
        """Test jobs page handles empty job list gracefully."""
        # Arrange
        self._setup_jobs_page_test(
            mock_session_state,
            mock_job_service,
            mock_company_service,
            [],
            active_companies=0,
        )

        with (
            patch("src.ui.pages.jobs.render_sidebar") as mock_render_sidebar,
            patch(
                "src.ui.pages.jobs._get_filtered_jobs",
                return_value=[],
            ) as mock_get_filtered,
            patch("src.ui.pages.jobs._handle_refresh_jobs"),
        ):
            # Act
            render_jobs_page()

            # Assert
            # Verify that mock functions are called correctly
            mock_render_sidebar.assert_called_once()
            mock_get_filtered.assert_called_once()

            # Assert - inline variable usage
            assert [
                call
                for call in mock_streamlit["info"].call_args_list
                if "No jobs found" in call.args[0]
            ]

    def test_complete_job_refresh_workflow(self, mock_streamlit, mock_session_state):
        """Test complete job refresh workflow with realistic scraping results."""
        # Arrange
        mock_sync_stats = {
            "inserted": 10,
            "updated": 5,
            "archived": 2,
            "deleted": 1,
            "skipped": 3,
        }

        with (
            patch(
                "src.ui.pages.jobs._execute_scraping_safely",
                return_value=mock_sync_stats,
            ),
            patch("streamlit.rerun") as mock_rerun,
        ):
            # Act
            _handle_refresh_jobs()

            # Assert
            # 1. Spinner was shown
            spinner_calls = mock_streamlit["spinner"].call_args_list
            assert spinner_calls
            assert "🔍 Searching for new jobs..." in spinner_calls[0].args[0]

            # 2. Success message was displayed
            success_calls = mock_streamlit["success"].call_args_list
            assert success_calls
            success_message = success_calls[0].args[0]
            assert "Success! Processed 15 jobs" in success_message  # 10 + 5

            # 3. Page was refreshed
            mock_rerun.assert_called_once()

    @pytest.fixture
    def job_factory(self):
        """Factory fixture for creating Job instances with defaults."""

        def _create_job(**kwargs):
            defaults = {
                "id": 1,
                "company_id": 1,
                "company": "Test",
                "title": "Test Job",
                "description": "Test description",
                "link": "https://test.com",
                "location": "Test Location",
                "content_hash": "hash123",
                "favorite": False,
                "application_status": "New",
            }
            return Job(**(defaults | kwargs))

        return _create_job

    @pytest.fixture
    def no_jobs_scenario(self):
        """Test scenario with no jobs."""
        return {"jobs": [], "expected_favorites": 0, "expected_applied": 0}

    @pytest.fixture
    def single_applied_favorite_scenario(self, job_factory):
        """Test scenario with one job that is both applied and favorite."""
        jobs = [job_factory(id=1, favorite=True, application_status="Applied")]
        return {"jobs": jobs, "expected_favorites": 1, "expected_applied": 1}

    @pytest.fixture
    def mixed_jobs_scenario(self, job_factory):
        """Test scenario with mixed job statuses."""
        jobs = [
            job_factory(id=1, favorite=False, application_status="New"),
            job_factory(id=2, favorite=True, application_status="Rejected"),
        ]
        return {"jobs": jobs, "expected_favorites": 1, "expected_applied": 0}

    @pytest.mark.parametrize(
        "scenario_name",
        ("no_jobs_scenario", "single_applied_favorite_scenario", "mixed_jobs_scenario"),
    )
    def test_job_tabs_calculate_correct_counts(
        self,
        mock_streamlit,
        mock_session_state,
        mock_job_service,
        request,
        scenario_name,
    ):
        """Test job tabs calculate correct counts for various job combinations."""
        # Arrange
        scenario = request.getfixturevalue(scenario_name)
        jobs = scenario["jobs"]
        expected_favorites = scenario["expected_favorites"]
        expected_applied = scenario["expected_applied"]

        # Initialize session state filters
        mock_session_state.filters = {}

        # Set up service mocks to return filtered data
        favorites = [j for j in jobs if j.favorite] if jobs else []
        applied = [j for j in jobs if j.application_status == "Applied"] if jobs else []

        mock_job_service["jobs_page"].get_filtered_jobs.side_effect = lambda filters: (
            favorites
            if filters.get("favorites_only")
            else applied
            if filters.get("application_status") == ["Applied"]
            else jobs
        )

        # Act
        _render_job_tabs(jobs)

        # Assert
        if jobs:
            tabs_calls = mock_streamlit["tabs"].call_args_list
            tab_labels = tabs_calls[0].args[0]

            # Extract counts from tab labels
            favorites_label = next(
                label for label in tab_labels if "Favorites" in label
            )
            applied_label = next(label for label in tab_labels if "Applied" in label)

            assert f"({expected_favorites})" in favorites_label
            assert f"({expected_applied})" in applied_label
