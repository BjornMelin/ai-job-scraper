"""Comprehensive tests for the Search Bar UI component.

This module tests the Search Bar functionality including:
- Search input and FTS5 integration with real-time search
- Advanced filters (location, salary, remote, status, dates)
- Search results rendering in cards and list views
- Session state management for search queries and filters
- Performance metrics display and search statistics
- Modal handling for job details from search results
- Error handling and edge cases
- Filter management and clearing functionality
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tests.utils.streamlit_utils import StreamlitComponentTester

from src.schemas import Job
from src.ui.components.search_bar import (
    _build_search_filters,
    _clear_all_filters,
    _handle_search_input_change,
    _handle_search_modal,
    _has_active_filters,
    _init_search_state,
    _perform_search,
    _render_advanced_filters,
    _render_empty_state,
    _render_search_input,
    _render_search_results,
    _render_search_results_cards,
    _render_search_results_list,
    _render_search_status,
    _trigger_search_update,
    render_job_search,
)


@pytest.fixture
def sample_search_job():
    """Create a sample Job object for search testing."""
    return Job(
        id=1,
        title="Senior Python Developer",
        company="Tech Corp",
        location="San Francisco, CA",
        description="Build scalable Python applications with Django and Flask. Work with data science team on ML pipelines.",
        posted_date=datetime.now(UTC) - timedelta(days=2),
        last_seen=datetime.now(UTC),
        favorite=False,
        application_status="New",
        notes="Interesting remote-friendly opportunity",
        link="https://example.com/job1",
        rank=0.85,  # Relevance score for search results
    )


@pytest.fixture
def sample_search_results():
    """Create a list of sample Job objects with search rankings."""
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
            notes="",
            link="https://example.com/job1",
            rank=0.95,
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
            application_status="Interested",
            notes="",
            link="https://example.com/job2",
            rank=0.78,
        ),
        Job(
            id=3,
            title="Machine Learning Engineer",
            company="ML Corp",
            location="New York, NY",
            description="Build ML pipelines and infrastructure",
            posted_date=now - timedelta(days=5),
            last_seen=now,
            favorite=False,
            application_status="New",
            notes="",
            link="https://example.com/job3",
            rank=0.65,
        ),
    ]


class TestSearchStateManagement:
    """Test search state initialization and management."""

    def test_init_search_state_default_values(self):
        """Test search state initialization with default values."""
        tester = StreamlitComponentTester(_init_search_state)

        # Initialize empty session state
        tester.clear_session_state()
        tester.run_component()

        state = tester.get_session_state()

        # Check required keys exist
        assert "search_query" in state
        assert "search_results" in state
        assert "search_stats" in state
        assert "search_filters" in state
        assert "show_advanced_filters" in state
        assert "last_search_time" in state
        assert "search_limit" in state

        # Check default values
        assert state["search_query"] == ""
        assert state["search_results"] == []
        assert state["show_advanced_filters"] is False
        assert state["search_limit"] == 50  # DEFAULT_SEARCH_LIMIT

        # Check search stats structure
        stats = state["search_stats"]
        assert stats["query_time"] == 0
        assert stats["total_results"] == 0
        assert stats["fts_enabled"] is False

        # Check search filters structure
        filters = state["search_filters"]
        assert filters["location"] == ""
        assert filters["remote_only"] is False
        assert filters["application_status"] == "All"
        assert filters["favorites_only"] is False
        assert isinstance(filters["date_from"], datetime)
        assert isinstance(filters["date_to"], datetime)

    def test_init_search_state_preserves_existing_values(self):
        """Test initialization doesn't overwrite existing session state."""
        tester = StreamlitComponentTester(_init_search_state)

        # Set some existing values
        existing_query = "python developer"
        tester.set_session_state(search_query=existing_query, other_key="preserve_me")

        tester.run_component()

        state = tester.get_session_state()

        # Existing values should be preserved
        assert state["search_query"] == existing_query
        assert state["other_key"] == "preserve_me"

        # New defaults should be added
        assert "search_results" in state
        assert "search_filters" in state

    def test_has_active_filters_default_false(self):
        """Test _has_active_filters returns False with default filters."""
        tester = StreamlitComponentTester(_has_active_filters)

        # Initialize with default state
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        result = tester.run_component()
        assert result is False

    def test_has_active_filters_true_with_modifications(self):
        """Test _has_active_filters returns True when filters are modified."""
        tester = StreamlitComponentTester(_has_active_filters)

        # Initialize with default state
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Modify some filters
        state = tester.get_session_state()
        state["search_filters"]["remote_only"] = True
        state["search_filters"]["location"] = "San Francisco"
        state["search_filters"]["application_status"] = "Applied"

        result = tester.run_component()
        assert result is True


class TestSearchInput:
    """Test search input rendering and interaction."""

    def test_render_search_input_creates_text_input(self):
        """Test search input component creates text input with correct properties."""
        tester = StreamlitComponentTester(_render_search_input)

        # Initialize search state
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.button") as mock_button,
        ):
            # Mock columns
            mock_columns.return_value = [Mock(), Mock()]

            # Mock expander context manager
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            mock_button.return_value = False

            tester.run_component()

            # Verify text input is created
            mock_text_input.assert_called_once()
            text_input_call = mock_text_input.call_args

            # Check text input parameters
            assert text_input_call[1]["label"] == "Search Jobs"
            assert text_input_call[1]["key"] == "search_input"
            assert text_input_call[1]["on_change"] == _handle_search_input_change
            assert "FTS5 powered" in text_input_call[1]["placeholder"]
            assert "FTS5" in text_input_call[1]["help"]

    def test_render_search_input_shows_hints_when_empty(self):
        """Test search input shows FTS5 hints when search query is empty."""
        tester = StreamlitComponentTester(_render_search_input)

        # Initialize with empty search query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input"),
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.button") as mock_button,
            patch("streamlit.markdown"),
        ):
            mock_columns.return_value = [Mock(), Mock()]
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            mock_button.return_value = False

            tester.run_component()

            # Verify expander is created for search tips
            mock_expander.assert_called()
            expander_call = mock_expander.call_args
            assert "Search Tips & Examples" in expander_call[0][0]

            # Verify search hint buttons are created
            assert (
                mock_button.call_count >= 5
            )  # At least one hint button + filter button

    def test_render_search_input_filter_button_toggle(self):
        """Test filter button toggles advanced filters display."""
        tester = StreamlitComponentTester(_render_search_input)

        # Initialize search state
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input"),
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.button") as mock_button,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            mock_button.return_value = False

            tester.run_component()

            # Find filter toggle button call
            button_calls = mock_button.call_args_list
            filter_button_calls = [
                call for call in button_calls if call[1].get("key") == "toggle_filters"
            ]

            assert len(filter_button_calls) == 1
            filter_button_call = filter_button_calls[0]
            assert "Filters" in filter_button_call[0][0]


class TestAdvancedFilters:
    """Test advanced filters rendering and functionality."""

    def test_render_advanced_filters_hidden_by_default(self):
        """Test advanced filters are hidden when show_advanced_filters is False."""
        tester = StreamlitComponentTester(_render_advanced_filters)

        # Initialize with show_advanced_filters = False
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Should return early without rendering anything
        with patch("streamlit.expander") as mock_expander:
            tester.run_component()
            mock_expander.assert_not_called()

    def test_render_advanced_filters_shows_when_enabled(self):
        """Test advanced filters render when show_advanced_filters is True."""
        tester = StreamlitComponentTester(_render_advanced_filters)

        # Initialize and enable advanced filters
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(show_advanced_filters=True)

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.checkbox") as mock_checkbox,
            patch("streamlit.number_input") as mock_number_input,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.button") as mock_button,
            patch("streamlit.markdown"),
        ):
            # Mock expander context manager
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            # Mock columns
            mock_columns.return_value = [Mock(), Mock()]

            # Mock filter inputs
            mock_text_input.return_value = ""
            mock_checkbox.return_value = False
            mock_number_input.return_value = 0
            mock_selectbox.return_value = "All"
            mock_date_input.return_value = datetime.now().date()
            mock_button.return_value = False

            tester.run_component()

            # Verify all filter controls are created
            mock_expander.assert_called_once_with("Filter Options", expanded=True)
            assert mock_text_input.call_count == 1  # Location input
            assert mock_checkbox.call_count == 2  # Remote only, Favorites only
            assert mock_number_input.call_count == 2  # Salary min/max
            assert mock_selectbox.call_count == 1  # Application status
            assert mock_date_input.call_count == 2  # Date from/to
            assert mock_button.call_count == 2  # Search, Clear buttons

    def test_render_advanced_filters_location_filter(self):
        """Test location filter input in advanced filters."""
        tester = StreamlitComponentTester(_render_advanced_filters)

        # Initialize and enable advanced filters
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(show_advanced_filters=True)

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.checkbox") as mock_checkbox,
            patch("streamlit.number_input") as mock_number_input,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.button") as mock_button,
            patch("streamlit.markdown"),
        ):
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]
            mock_text_input.return_value = ""
            mock_checkbox.return_value = False
            mock_number_input.return_value = 0
            mock_selectbox.return_value = "All"
            mock_date_input.return_value = datetime.now().date()
            mock_button.return_value = False

            tester.run_component()

            # Check location text input
            location_calls = [
                call
                for call in mock_text_input.call_args_list
                if call[0][0] == "Location"
            ]
            assert len(location_calls) == 1
            location_call = location_calls[0]
            assert "San Francisco, Remote" in location_call[1]["placeholder"]

    def test_render_advanced_filters_salary_range(self):
        """Test salary range inputs in advanced filters."""
        tester = StreamlitComponentTester(_render_advanced_filters)

        # Initialize and enable advanced filters
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(show_advanced_filters=True)

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.checkbox") as mock_checkbox,
            patch("streamlit.number_input") as mock_number_input,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.button") as mock_button,
            patch("streamlit.markdown"),
        ):
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]
            mock_text_input.return_value = ""
            mock_checkbox.return_value = False
            mock_number_input.return_value = 0
            mock_selectbox.return_value = "All"
            mock_date_input.return_value = datetime.now().date()
            mock_button.return_value = False

            tester.run_component()

            # Check salary inputs
            salary_calls = mock_number_input.call_args_list
            assert len(salary_calls) == 2

            # Check minimum salary input
            min_salary_call = next(
                call for call in salary_calls if call[0][0] == "Minimum Salary"
            )
            assert min_salary_call[1]["min_value"] == 0
            assert min_salary_call[1]["max_value"] == 500000
            assert min_salary_call[1]["step"] == 5000

            # Check maximum salary input
            max_salary_call = next(
                call for call in salary_calls if call[0][0] == "Maximum Salary"
            )
            assert max_salary_call[1]["min_value"] == 0
            assert max_salary_call[1]["max_value"] == 500000
            assert max_salary_call[1]["step"] == 5000


class TestSearchResults:
    """Test search results rendering and display."""

    def test_render_search_results_empty_state_no_query(self):
        """Test search results shows welcome message when no query."""
        tester = StreamlitComponentTester(_render_search_results)

        # Initialize with empty search query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("src.ui.components.search_bar._render_search_status"),
            patch(
                "src.ui.components.search_bar._render_empty_state"
            ) as mock_empty_state,
        ):
            tester.run_component()

            # Should render empty state for no results
            mock_empty_state.assert_called_once()

    def test_render_search_results_with_results(self, sample_search_results):
        """Test search results display when results exist."""
        tester = StreamlitComponentTester(_render_search_results)

        # Initialize with search results
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(
            search_results=sample_search_results,
            search_stats={"query_time": 125.5, "total_results": 3, "fts_enabled": True},
        )

        with (
            patch("src.ui.components.search_bar._render_search_status") as mock_status,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.selectbox") as mock_selectbox,
            patch(
                "src.ui.components.search_bar._render_search_results_cards"
            ) as mock_cards,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.side_effect = ["Cards", 50]  # View mode and results limit

            tester.run_component()

            # Verify results display components are called
            mock_status.assert_called_once()
            mock_markdown.assert_called()

            # Check that results count is displayed
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            result_count_calls = [
                call for call in markdown_calls if "Found 3 jobs" in call
            ]
            assert len(result_count_calls) == 1

            # Check that cards view is rendered
            mock_cards.assert_called_once()

    def test_render_search_results_view_mode_selection(self, sample_search_results):
        """Test view mode selection between Cards and List."""
        tester = StreamlitComponentTester(_render_search_results)

        # Initialize with search results
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_results=sample_search_results)

        with (
            patch("src.ui.components.search_bar._render_search_status"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown"),
            patch("streamlit.selectbox") as mock_selectbox,
            patch(
                "src.ui.components.search_bar._render_search_results_list"
            ) as mock_list,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.side_effect = ["List", 50]  # List view mode

            tester.run_component()

            # Verify list view is rendered
            mock_list.assert_called_once()

    def test_render_search_results_cards_with_relevance_scores(
        self, sample_search_results
    ):
        """Test search results cards display with relevance scores."""
        tester = StreamlitComponentTester(_render_search_results_cards)

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.caption") as mock_caption,
            patch(
                "src.ui.components.cards.job_card.render_job_card"
            ) as mock_render_card,
        ):
            # Mock columns for card layout
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            tester.run_component(sample_search_results)

            # Verify relevance scores are displayed
            assert mock_caption.call_count == len(sample_search_results)

            # Check that relevance scores are formatted correctly
            caption_calls = [call[0][0] for call in mock_caption.call_args_list]
            assert any("Relevance: 0.9" in call for call in caption_calls)
            assert any("Relevance: 0.8" in call for call in caption_calls)
            assert any("Relevance: 0.7" in call for call in caption_calls)

            # Verify job cards are rendered
            assert mock_render_card.call_count == len(sample_search_results)

    def test_render_search_results_list_with_details_buttons(
        self, sample_search_results
    ):
        """Test search results list view with view details buttons."""
        tester = StreamlitComponentTester(_render_search_results_list)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown"),
            patch("streamlit.metric") as mock_metric,
            patch("streamlit.button") as mock_button,
        ):
            # Mock container context manager
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            # Mock columns for layout
            mock_columns.return_value = [Mock(), Mock()]
            mock_button.return_value = False

            tester.run_component(sample_search_results)

            # Verify containers are created for each job
            assert mock_container.call_count == len(sample_search_results)

            # Verify view details buttons are created
            button_calls = mock_button.call_args_list
            details_buttons = [
                call for call in button_calls if call[0][0] == "View Details"
            ]
            assert len(details_buttons) == len(sample_search_results)

            # Verify relevance metrics are displayed
            assert mock_metric.call_count == len(sample_search_results)


class TestSearchFunctionality:
    """Test search execution and filter building."""

    def test_perform_search_with_empty_query(self):
        """Test search with empty query clears results."""
        tester = StreamlitComponentTester(_perform_search)

        # Initialize with empty query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_query="")

        tester.run_component()

        state = tester.get_session_state()
        assert state["search_results"] == []
        assert state["search_stats"]["query_time"] == 0
        assert state["search_stats"]["total_results"] == 0

    def test_perform_search_with_query(self, sample_search_results):
        """Test search execution with valid query."""
        tester = StreamlitComponentTester(_perform_search)

        # Initialize with search query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_query="python developer", search_limit=50)

        with (
            patch(
                "src.services.search_service.search_service.search_jobs"
            ) as mock_search,
            patch(
                "src.services.search_service.search_service.get_search_stats"
            ) as mock_stats,
            patch(
                "src.ui.components.search_bar._build_search_filters"
            ) as mock_build_filters,
        ):
            # Mock search service responses
            mock_search.return_value = sample_search_results
            mock_stats.return_value = {"fts_enabled": True}
            mock_build_filters.return_value = {}

            tester.run_component()

            # Verify search service was called
            mock_search.assert_called_once_with(
                query="python developer", filters={}, limit=50
            )

            # Verify results are stored in session state
            state = tester.get_session_state()
            assert state["search_results"] == sample_search_results
            assert state["search_stats"]["total_results"] == len(sample_search_results)
            assert state["search_stats"]["fts_enabled"] is True

    def test_perform_search_handles_service_exception(self):
        """Test search handles service exceptions gracefully."""
        tester = StreamlitComponentTester(_perform_search)

        # Initialize with search query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_query="python developer")

        with (
            patch(
                "src.services.search_service.search_service.search_jobs"
            ) as mock_search,
            patch(
                "src.ui.components.search_bar._build_search_filters"
            ) as mock_build_filters,
            patch("streamlit.error") as mock_error,
        ):
            mock_search.side_effect = Exception("Database connection failed")
            mock_build_filters.return_value = {}

            tester.run_component()

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Search failed:" in error_message

            # Verify search results are cleared
            state = tester.get_session_state()
            assert state["search_results"] == []

    def test_build_search_filters_basic(self):
        """Test basic search filter building."""
        tester = StreamlitComponentTester(_build_search_filters)

        # Initialize with default filters
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        filters = tester.run_component()

        # Check required filter keys
        assert "date_from" in filters
        assert "date_to" in filters
        assert "favorites_only" in filters
        assert filters["favorites_only"] is False

    def test_build_search_filters_with_status_filter(self):
        """Test search filter building with application status."""
        tester = StreamlitComponentTester(_build_search_filters)

        # Initialize and set status filter
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Modify filters to include specific status
        state = tester.get_session_state()
        state["search_filters"]["application_status"] = "Applied"

        filters = tester.run_component()

        # Verify status filter is converted to list
        assert "application_status" in filters
        assert filters["application_status"] == ["Applied"]

    def test_build_search_filters_with_salary_range(self):
        """Test search filter building with salary range."""
        tester = StreamlitComponentTester(_build_search_filters)

        # Initialize and set salary filters
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Modify filters to include salary range
        state = tester.get_session_state()
        state["search_filters"]["salary_min"] = 80000
        state["search_filters"]["salary_max"] = 150000

        filters = tester.run_component()

        # Verify salary filters are included
        assert "salary_min" in filters
        assert "salary_max" in filters
        assert filters["salary_min"] == 80000
        assert filters["salary_max"] == 150000


class TestFilterManagement:
    """Test filter clearing and management."""

    def test_clear_all_filters_resets_state(self):
        """Test clearing all filters resets search state."""
        tester = StreamlitComponentTester(_clear_all_filters)

        # Initialize and modify state
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Set some search data
        tester.set_session_state(
            search_query="python developer",
            search_results=[{"id": 1, "title": "Test Job"}],
        )

        # Modify filters
        state = tester.get_session_state()
        state["search_filters"]["location"] = "San Francisco"
        state["search_filters"]["remote_only"] = True
        state["search_filters"]["application_status"] = "Applied"

        tester.run_component()

        # Verify everything is reset
        final_state = tester.get_session_state()
        assert final_state["search_query"] == ""
        assert final_state["search_results"] == []
        assert final_state["search_filters"]["location"] == ""
        assert final_state["search_filters"]["remote_only"] is False
        assert final_state["search_filters"]["application_status"] == "All"

    def test_trigger_search_update_with_query(self, sample_search_results):
        """Test search update trigger when query exists."""
        tester = StreamlitComponentTester(_trigger_search_update)

        # Initialize with search query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_query="python developer")

        with (
            patch(
                "src.ui.components.search_bar._perform_search"
            ) as mock_perform_search,
        ):
            tester.run_component()

            # Verify search is triggered
            mock_perform_search.assert_called_once()

    def test_trigger_search_update_without_query(self):
        """Test search update trigger when no query exists."""
        tester = StreamlitComponentTester(_trigger_search_update)

        # Initialize with empty query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch(
                "src.ui.components.search_bar._perform_search"
            ) as mock_perform_search,
        ):
            tester.run_component()

            # Verify search is not triggered
            mock_perform_search.assert_not_called()


class TestSearchInputHandling:
    """Test search input change handling and debouncing."""

    def test_handle_search_input_change_updates_query(self):
        """Test search input change handler updates query."""
        tester = StreamlitComponentTester(_handle_search_input_change)

        # Initialize state
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Set search input in session state
        tester.set_session_state(search_input="machine learning")

        with (
            patch("time.time", return_value=1000.0),
            patch(
                "src.ui.components.search_bar._perform_search"
            ) as mock_perform_search,
        ):
            tester.run_component()

            # Verify query is updated
            state = tester.get_session_state()
            assert state["search_query"] == "machine learning"

            # Verify search is triggered due to debounce delay
            mock_perform_search.assert_called_once()

    def test_handle_search_input_change_debouncing(self):
        """Test search input change respects debounce delay."""
        tester = StreamlitComponentTester(_handle_search_input_change)

        # Initialize state with recent search time
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        # Set recent search time and new input
        tester.set_session_state(
            search_input="python",
            last_search_time=999.9,  # Recent search
        )

        with (
            patch("time.time", return_value=1000.0),  # Only 0.1s later
            patch(
                "src.ui.components.search_bar._perform_search"
            ) as mock_perform_search,
        ):
            tester.run_component()

            # Verify search is not triggered due to debounce
            mock_perform_search.assert_not_called()


class TestEmptyStateRendering:
    """Test empty state display and examples."""

    def test_render_empty_state_no_query(self):
        """Test empty state rendering when no search query."""
        tester = StreamlitComponentTester(_render_empty_state)

        # Initialize with empty query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("streamlit.info") as mock_info,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            mock_button.return_value = False

            tester.run_component()

            # Verify welcome message is displayed
            mock_info.assert_called_once()
            info_message = mock_info.call_args[0][0]
            assert "Welcome to Job Search!" in info_message
            assert "FTS5 search engine" in info_message

            # Verify example searches are shown
            assert mock_button.call_count >= 4  # At least 4 example buttons

    def test_render_empty_state_with_query_no_results(self):
        """Test empty state rendering when query has no results."""
        tester = StreamlitComponentTester(_render_empty_state)

        # Initialize with query but no results
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_query="nonexistent job")

        with (
            patch("streamlit.warning") as mock_warning,
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.markdown"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.button") as mock_button,
        ):
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]
            mock_button.return_value = False

            tester.run_component()

            # Verify "no results" warning is displayed
            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "No jobs found" in warning_message

            # Verify search tips are shown
            mock_expander.assert_called_once_with("💡 Search Tips", expanded=True)


class TestSearchModal:
    """Test search modal handling for job details."""

    def test_handle_search_modal_with_valid_job(self, sample_search_results):
        """Test search modal handling with valid job ID."""
        tester = StreamlitComponentTester(_handle_search_modal)

        # Initialize with search results and modal job ID
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(
            search_results=sample_search_results, search_modal_job_id=1
        )

        with (
            patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal,
        ):
            tester.run_component()

            # Verify modal is shown with correct job
            mock_show_modal.assert_called_once()
            job_arg = mock_show_modal.call_args[0][0]
            assert job_arg.id == 1
            assert job_arg.title == "Senior Python Developer"

    def test_handle_search_modal_with_invalid_job(self, sample_search_results):
        """Test search modal handling with invalid job ID."""
        tester = StreamlitComponentTester(_handle_search_modal)

        # Initialize with search results and invalid modal job ID
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(
            search_results=sample_search_results,
            search_modal_job_id=999,  # Non-existent job ID
        )

        tester.run_component()

        # Verify modal job ID is cleared
        state = tester.get_session_state()
        assert state.get("search_modal_job_id") is None

    def test_handle_search_modal_no_job_id(self):
        """Test search modal handling when no job ID is set."""
        tester = StreamlitComponentTester(_handle_search_modal)

        # Initialize without modal job ID
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("src.ui.pages.jobs.show_job_details_modal") as mock_show_modal,
        ):
            tester.run_component()

            # Verify modal is not shown
            mock_show_modal.assert_not_called()


class TestMainSearchComponent:
    """Test main search component integration."""

    def test_render_job_search_complete_flow(self):
        """Test complete job search component rendering flow."""
        tester = StreamlitComponentTester(render_job_search)

        with (
            patch("streamlit.container") as mock_container,
            patch("streamlit.markdown") as mock_markdown,
            patch("src.ui.components.search_bar._init_search_state") as mock_init,
            patch("src.ui.components.search_bar._render_search_input") as mock_input,
            patch(
                "src.ui.components.search_bar._render_advanced_filters"
            ) as mock_filters,
            patch(
                "src.ui.components.search_bar._render_search_results"
            ) as mock_results,
            patch("src.ui.components.search_bar._handle_search_modal") as mock_modal,
        ):
            # Mock container context manager
            mock_container.return_value.__enter__ = Mock()
            mock_container.return_value.__exit__ = Mock(return_value=None)

            tester.run_component()

            # Verify all components are rendered in order
            mock_init.assert_called_once()
            mock_input.assert_called_once()
            mock_filters.assert_called_once()
            mock_results.assert_called_once()
            mock_modal.assert_called_once()

            # Verify search header is displayed
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            header_calls = [call for call in markdown_calls if "🔍 Job Search" in call]
            assert len(header_calls) == 1


class TestSearchStatusRendering:
    """Test search status and metrics display."""

    def test_render_search_status_with_query(self):
        """Test search status rendering with active query."""
        tester = StreamlitComponentTester(_render_search_status)

        # Initialize with search query and stats
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(
            search_query="python developer",
            search_stats={"query_time": 125.5, "fts_enabled": True},
            search_results=[Mock(), Mock(), Mock()],
        )

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.metric") as mock_metric,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]

            tester.run_component()

            # Verify status information is displayed
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            status_calls = [call for call in markdown_calls if "🚀 FTS5" in call]
            assert len(status_calls) == 1

            # Verify metrics are displayed
            metric_calls = mock_metric.call_args_list

            # Check query time metric
            time_metrics = [call for call in metric_calls if call[0][0] == "Query Time"]
            assert len(time_metrics) == 1
            assert "126ms" in time_metrics[0][0][1]

            # Check results count metric
            results_metrics = [call for call in metric_calls if call[0][0] == "Results"]
            assert len(results_metrics) == 1

    def test_render_search_status_no_query(self):
        """Test search status rendering when no query."""
        tester = StreamlitComponentTester(_render_search_status)

        # Initialize with empty query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session

        with (
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.metric") as mock_metric,
        ):
            tester.run_component()

            # Verify no status is rendered for empty query
            mock_columns.assert_not_called()
            mock_markdown.assert_not_called()
            mock_metric.assert_not_called()


class TestSearchEdgeCases:
    """Test edge cases and error conditions."""

    def test_search_with_special_characters(self):
        """Test search handles special characters correctly."""
        tester = StreamlitComponentTester(_perform_search)

        # Initialize with special character query
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(
            search_query='job with "quotes" & symbols!', search_limit=50
        )

        with (
            patch(
                "src.services.search_service.search_service.search_jobs"
            ) as mock_search,
            patch(
                "src.services.search_service.search_service.get_search_stats"
            ) as mock_stats,
            patch(
                "src.ui.components.search_bar._build_search_filters"
            ) as mock_build_filters,
        ):
            mock_search.return_value = []
            mock_stats.return_value = {"fts_enabled": True}
            mock_build_filters.return_value = {}

            # Should handle special characters without error
            tester.run_component()

            # Verify search service was called with special characters
            mock_search.assert_called_once()
            search_args = mock_search.call_args
            assert search_args[1]["query"] == 'job with "quotes" & symbols!'

    def test_search_results_rendering_exception(self, sample_search_results):
        """Test search results handles rendering exceptions."""
        tester = StreamlitComponentTester(_render_search_results)

        # Initialize with search results
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(search_results=sample_search_results)

        with (
            patch("src.ui.components.search_bar._render_search_status"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.markdown"),
            patch("streamlit.selectbox") as mock_selectbox,
            patch(
                "src.ui.components.search_bar._render_search_results_cards"
            ) as mock_cards,
            patch("streamlit.error") as mock_error,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_selectbox.side_effect = ["Cards", 50]
            mock_cards.side_effect = Exception("Rendering error")

            tester.run_component()

            # Verify error is handled gracefully
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Error displaying search results:" in error_message

    def test_advanced_filters_with_invalid_dates(self):
        """Test advanced filters handle invalid date inputs gracefully."""
        tester = StreamlitComponentTester(_render_advanced_filters)

        # Initialize and enable advanced filters
        init_tester = StreamlitComponentTester(_init_search_state)
        init_tester.clear_session_state()
        init_tester.run_component()
        tester.session = init_tester.session
        tester.set_session_state(show_advanced_filters=True)

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.checkbox") as mock_checkbox,
            patch("streamlit.number_input") as mock_number_input,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.date_input") as mock_date_input,
            patch("streamlit.button") as mock_button,
            patch("streamlit.markdown"),
        ):
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [Mock(), Mock()]
            mock_text_input.return_value = ""
            mock_checkbox.return_value = False
            mock_number_input.return_value = 0
            mock_selectbox.return_value = "All"
            mock_date_input.side_effect = Exception(
                "Invalid date"
            )  # Simulate date error
            mock_button.return_value = False

            # Should handle date errors gracefully
            try:
                tester.run_component()
            except Exception:
                pytest.fail(
                    "Advanced filters should handle date input errors gracefully"
                )

    def test_search_limit_boundary_values(self):
        """Test search with boundary limit values."""
        for limit_value in [1, 25, 50, 100, 1000]:
            tester = StreamlitComponentTester(_perform_search)

            # Initialize with different limit values
            init_tester = StreamlitComponentTester(_init_search_state)
            init_tester.clear_session_state()
            init_tester.run_component()
            tester.session = init_tester.session
            tester.set_session_state(
                search_query="test query", search_limit=limit_value
            )

            with (
                patch(
                    "src.services.search_service.search_service.search_jobs"
                ) as mock_search,
                patch(
                    "src.services.search_service.search_service.get_search_stats"
                ) as mock_stats,
                patch(
                    "src.ui.components.search_bar._build_search_filters"
                ) as mock_build_filters,
            ):
                mock_search.return_value = []
                mock_stats.return_value = {"fts_enabled": True}
                mock_build_filters.return_value = {}

                tester.run_component()

                # Verify limit is passed correctly
                search_args = mock_search.call_args
                assert search_args[1]["limit"] == limit_value
