"""Focused tests for the Sidebar UI component.

This module tests the Sidebar functionality including:
- Search filters (company, keyword, date range, salary range)
- View settings (list/card view toggles)
- Company management (adding and editing companies)
- Session state management for filters
- URL synchronization with filters
- Error handling for service failures
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tests.utils.streamlit_utils import StreamlitComponentTester

from src.schemas import Company
from src.ui.components.sidebar import (
    _display_salary_range,
    _get_company_list,
    _handle_add_company,
    _render_search_filters,
    _render_view_settings,
    render_sidebar,
)


@pytest.fixture
def sample_companies():
    """Create sample Company objects for testing."""
    return [
        Company(
            id=1, name="Tech Corp", url="https://techcorp.com/careers", active=True
        ),
        Company(id=2, name="AI Startup", url="https://aistartup.com/jobs", active=True),
        Company(id=3, name="Scale Inc", url="https://scale.com/careers", active=False),
    ]


@pytest.fixture
def default_filters():
    """Create default filter state for testing."""
    now = datetime.now(UTC)
    return {
        "company": [],
        "keyword": "",
        "date_from": now - timedelta(days=30),
        "date_to": now,
        "salary_min": 0,
        "salary_max": 750000,
    }


class TestSidebarRendering:
    """Test main sidebar rendering functionality."""

    def test_render_sidebar_basic_structure(self):
        """Test sidebar renders all main sections."""
        tester = StreamlitComponentTester(render_sidebar)

        with (
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.divider") as mock_divider,
            patch("src.ui.components.sidebar._render_search_filters") as mock_filters,
            patch("src.ui.components.sidebar._render_view_settings") as mock_view,
            patch(
                "src.ui.components.sidebar._render_company_management"
            ) as mock_company,
            patch("src.ui.utils.url_state.sync_filters_from_url") as mock_sync,
        ):
            # Mock sidebar context manager
            mock_sidebar.return_value.__enter__ = Mock()
            mock_sidebar.return_value.__exit__ = Mock(return_value=None)

            # Use manual context mocking instead of mock_streamlit_app
            with patch("src.ui.components.sidebar.st") as mock_st:
                mock_st.sidebar = mock_sidebar.return_value
                mock_st.divider = mock_divider

                tester.run_component()

                # Verify URL sync is called
                mock_sync.assert_called_once()

                # Verify all sections are rendered
                mock_filters.assert_called_once()
                mock_view.assert_called_once()
                mock_company.assert_called_once()

                # Verify dividers are added between sections
                assert mock_divider.call_count == 2


class TestSearchFilters:
    """Test search filters functionality."""

    def test_render_search_filters_basic_layout(
        self, default_filters, sample_companies
    ):
        """Test search filters render correctly."""
        tester = StreamlitComponentTester(_render_search_filters)
        tester.set_session_state(filters=default_filters)

        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.container") as mock_container,
            patch("streamlit.multiselect") as mock_multiselect,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.slider") as mock_slider,
            patch("streamlit.button") as mock_button,
            patch("src.ui.components.sidebar._get_company_list") as mock_get_companies,
            patch("src.ui.components.sidebar._display_salary_range"),
            patch("src.ui.utils.url_state.update_url_from_filters") as mock_update_url,
        ):
            # Mock company list
            mock_get_companies.return_value = [c.name for c in sample_companies]

            # Mock container context manager
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            # Mock columns
            mock_columns.return_value = [Mock(), Mock()]

            # Mock form inputs
            mock_multiselect.return_value = []
            mock_text_input.return_value = ""
            mock_date_input.return_value = datetime.now(UTC).date()
            mock_slider.return_value = (0, 750000)
            mock_button.return_value = False

            tester.run_component()

            # Verify header is rendered
            header_calls = [
                call
                for call in mock_markdown.call_args_list
                if "🔍 Search & Filter" in str(call)
            ]
            assert len(header_calls) == 1

            # Verify company multiselect is created
            mock_multiselect.assert_called_once()
            multiselect_args = mock_multiselect.call_args
            assert multiselect_args[0][0] == "Filter by Company"
            assert multiselect_args[1]["options"] == [c.name for c in sample_companies]

            # Verify keyword search input
            mock_text_input.assert_called_once()
            text_input_args = mock_text_input.call_args
            assert text_input_args[0][0] == "Search Keywords"

            # Verify date inputs
            assert mock_date_input.call_count == 2

            # Verify salary slider
            mock_slider.assert_called_once()
            slider_args = mock_slider.call_args
            assert slider_args[0][0] == "Annual Salary Range"
            assert slider_args[1]["min_value"] == 0
            assert slider_args[1]["max_value"] == 750000

            # Verify URL sync is called
            mock_update_url.assert_called_once()

    def test_search_filters_company_selection_updates_state(self, default_filters):
        """Test company selection updates session state."""
        tester = StreamlitComponentTester(_render_search_filters)
        tester.set_session_state(filters=default_filters)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.multiselect") as mock_multiselect,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.slider") as mock_slider,
            patch("streamlit.button") as mock_button,
            patch("src.ui.components.sidebar._get_company_list") as mock_get_companies,
            patch("src.ui.components.sidebar._display_salary_range"),
            patch("src.ui.utils.url_state.update_url_from_filters"),
            patch("streamlit.markdown"),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]

            # Mock selected companies
            selected_companies = ["Tech Corp", "AI Startup"]
            mock_multiselect.return_value = selected_companies
            mock_text_input.return_value = ""
            mock_date_input.return_value = datetime.now(UTC).date()
            mock_slider.return_value = (0, 750000)
            mock_button.return_value = False
            mock_get_companies.return_value = ["Tech Corp", "AI Startup", "Scale Inc"]

            tester.run_component()

            # Verify companies are updated in session state
            updated_state = tester.get_session_state()
            assert updated_state["filters"]["company"] == selected_companies

    def test_search_filters_clear_button_functionality(self, default_filters):
        """Test clear filters button resets state."""
        tester = StreamlitComponentTester(_render_search_filters)

        # Set some modified filters
        modified_filters = default_filters.copy()
        modified_filters["company"] = ["Tech Corp"]
        modified_filters["keyword"] = "python"
        tester.set_session_state(filters=modified_filters)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.multiselect") as mock_multiselect,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.slider") as mock_slider,
            patch("streamlit.button") as mock_button,
            patch("src.ui.components.sidebar._get_company_list") as mock_get_companies,
            patch("src.ui.components.sidebar._display_salary_range"),
            patch("src.ui.utils.url_state.update_url_from_filters"),
            patch("src.ui.state.session_state.clear_filters") as mock_clear,
            patch("src.ui.utils.url_state.clear_url_params") as mock_clear_url,
            patch("streamlit.rerun") as mock_rerun,
            patch("streamlit.markdown"),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]
            mock_multiselect.return_value = []
            mock_text_input.return_value = ""
            mock_date_input.return_value = datetime.now(UTC).date()
            mock_slider.return_value = (0, 750000)
            mock_button.return_value = True  # Clear button clicked
            mock_get_companies.return_value = ["Tech Corp"]

            tester.run_component()

            # Verify clear functions are called
            mock_clear.assert_called_once()
            mock_clear_url.assert_called_once()
            mock_rerun.assert_called_once()


class TestViewSettings:
    """Test view settings functionality."""

    def test_render_view_settings_list_view_toggle(self):
        """Test list view toggle functionality."""
        tester = StreamlitComponentTester(_render_view_settings)
        tester.set_session_state(view_mode="Card")  # Start in card view

        with (
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            # First button (List View) clicked, second button (Card View) not clicked
            mock_button.side_effect = [True, False]

            tester.run_component()

            # Verify view mode is changed to List
            updated_state = tester.get_session_state()
            assert updated_state["view_mode"] == "List"

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_render_view_settings_card_view_toggle(self):
        """Test card view toggle functionality."""
        tester = StreamlitComponentTester(_render_view_settings)
        tester.set_session_state(view_mode="List")  # Start in list view

        with (
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
            patch("streamlit.rerun") as mock_rerun,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            # First button (List View) not clicked, second button (Card View) clicked
            mock_button.side_effect = [False, True]

            tester.run_component()

            # Verify view mode is changed to Card
            updated_state = tester.get_session_state()
            assert updated_state["view_mode"] == "Card"

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_render_view_settings_button_states(self):
        """Test view settings buttons show correct primary/secondary states."""
        tester = StreamlitComponentTester(_render_view_settings)
        tester.set_session_state(view_mode="Card")

        with (
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            mock_button.return_value = False

            tester.run_component()

            # Verify button states based on current view mode
            button_calls = mock_button.call_args_list

            # List view button should be secondary (not active)
            list_button_call = button_calls[0]
            assert list_button_call[1]["type"] == "secondary"

            # Card view button should be primary (active)
            card_button_call = button_calls[1]
            assert card_button_call[1]["type"] == "primary"


class TestCompanyManagement:
    """Test company management functionality."""

    def test_handle_add_company_success(self):
        """Test successful company addition."""
        tester = StreamlitComponentTester(_handle_add_company)

        with (
            patch(
                "src.services.company_service.CompanyService.add_company"
            ) as mock_add,
            patch("streamlit.success") as mock_success,
            patch("streamlit.rerun") as mock_rerun,
        ):
            tester.run_component("NewCorp", "https://newcorp.com/careers")

            # Verify company service is called
            mock_add.assert_called_once_with("NewCorp", "https://newcorp.com/careers")

            # Verify success message
            mock_success.assert_called_once()
            success_message = mock_success.call_args[0][0]
            assert "Added NewCorp successfully!" in success_message

            # Verify rerun is called
            mock_rerun.assert_called_once()

    def test_handle_add_company_validation_errors(self):
        """Test company addition validation errors."""
        tester = StreamlitComponentTester(_handle_add_company)

        with patch("streamlit.error") as mock_error:
            # Test empty name
            tester.run_component("", "https://example.com")
            mock_error.assert_called_with("Please fill in both fields")

            # Test empty URL
            tester.run_component("Test Company", "")
            mock_error.assert_called_with("Please fill in both fields")

            # Test invalid URL format
            tester.run_component("Test Company", "invalid-url")
            mock_error.assert_called_with("URL must start with http:// or https://")

    def test_handle_add_company_service_error(self):
        """Test company addition handles service errors."""
        tester = StreamlitComponentTester(_handle_add_company)

        with (
            patch(
                "src.services.company_service.CompanyService.add_company"
            ) as mock_add,
            patch("streamlit.error") as mock_error,
        ):
            # Test ValueError from service
            mock_add.side_effect = ValueError("Company already exists")
            tester.run_component("ExistingCorp", "https://existing.com")

            mock_error.assert_called_with("❌ Company already exists")

            # Test general exception
            mock_add.side_effect = Exception("Database error")
            tester.run_component("TestCorp", "https://test.com")

            error_calls = [
                call
                for call in mock_error.call_args_list
                if "Failed to add company" in str(call)
            ]
            assert len(error_calls) == 1


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_company_list_success(self, sample_companies):
        """Test successful company list retrieval."""
        tester = StreamlitComponentTester(_get_company_list)

        with patch(
            "src.services.company_service.CompanyService.get_all_companies"
        ) as mock_get:
            mock_get.return_value = sample_companies

            result = tester.run_component()

            expected_names = [c.name for c in sample_companies]
            assert result == expected_names

    def test_get_company_list_service_error(self):
        """Test company list retrieval handles service errors."""
        tester = StreamlitComponentTester(_get_company_list)

        with patch(
            "src.services.company_service.CompanyService.get_all_companies"
        ) as mock_get:
            mock_get.side_effect = Exception("Database connection failed")

            result = tester.run_component()

            # Should return empty list on error
            assert result == []

    def test_display_salary_range_no_filter(self):
        """Test salary range display with no filtering."""
        tester = StreamlitComponentTester(_display_salary_range)

        with patch("streamlit.caption") as mock_caption:
            tester.run_component((0, 750000))  # Default range

            # Verify "all salary ranges" message
            mock_caption.assert_called_once_with("💰 Showing all salary ranges")

    def test_display_salary_range_with_filter(self):
        """Test salary range display with filtering."""
        tester = StreamlitComponentTester(_display_salary_range)

        with patch("streamlit.caption") as mock_caption:
            tester.run_component((80000, 150000))  # Filtered range

            # Verify selected range is shown
            caption_calls = [call[0][0] for call in mock_caption.call_args_list]
            range_calls = [call for call in caption_calls if "Selected:" in call]
            assert len(range_calls) == 1
            assert "$80k" in range_calls[0]
            assert "$150k" in range_calls[0]

    def test_display_salary_range_high_value(self):
        """Test salary range display with high-value positions."""
        tester = StreamlitComponentTester(_display_salary_range)

        with patch("streamlit.caption") as mock_caption:
            tester.run_component((100000, 750000))  # High-value range

            # Verify high-value indicator is shown
            caption_calls = [call[0][0] for call in mock_caption.call_args_list]
            high_value_calls = [
                call
                for call in caption_calls
                if "Including all positions above" in call
            ]
            assert len(high_value_calls) == 1


class TestErrorHandling:
    """Test error handling across sidebar components."""

    def test_search_filters_handle_service_failures(self):
        """Test search filters handle company service failures gracefully."""
        tester = StreamlitComponentTester(_render_search_filters)
        tester.set_session_state(filters={"company": [], "keyword": ""})

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.multiselect") as mock_multiselect,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.slider") as mock_slider,
            patch("streamlit.button") as mock_button,
            patch("src.ui.components.sidebar._get_company_list") as mock_get_companies,
            patch("src.ui.components.sidebar._display_salary_range"),
            patch("src.ui.utils.url_state.update_url_from_filters"),
            patch("streamlit.markdown"),
        ):
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]

            # Company service failure returns empty list
            mock_get_companies.return_value = []

            mock_multiselect.return_value = []
            mock_text_input.return_value = ""
            mock_date_input.return_value = datetime.now(UTC).date()
            mock_slider.return_value = (0, 750000)
            mock_button.return_value = False

            # Should not raise exception despite service failure
            tester.run_component()

            # Verify multiselect is still created with empty options
            mock_multiselect.assert_called_once()
            multiselect_args = mock_multiselect.call_args
            assert multiselect_args[1]["options"] == []
