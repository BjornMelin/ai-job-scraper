"""Tests for the search bar component."""

from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from src.ui.components.search_bar import (
    _build_search_filters,
    _clear_all_filters,
    _has_active_filters,
    _init_search_state,
    render_job_search,
)
from tests.ui.components.test_utils import MockSessionState


class TestSearchBarComponent:
    """Test search bar component functionality."""

    def test_init_search_state(self):
        """Test that search state is initialized properly."""
        # Mock st.session_state
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Check required keys are initialized
            assert "search_query" in st.session_state
            assert "search_results" in st.session_state
            assert "search_stats" in st.session_state
            assert "search_filters" in st.session_state
            assert "show_advanced_filters" in st.session_state

            # Check default values
            assert st.session_state["search_query"] == ""
            assert st.session_state["search_results"] == []
            assert isinstance(st.session_state["search_filters"], dict)
            assert st.session_state["show_advanced_filters"] is False

    def test_has_active_filters_false(self):
        """Test _has_active_filters returns False with default filters."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            assert _has_active_filters() is False

    def test_has_active_filters_true(self):
        """Test _has_active_filters returns True when filters are modified."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            # Modify a filter
            st.session_state.search_filters["remote_only"] = True
            assert _has_active_filters() is True

    def test_build_search_filters_basic(self):
        """Test basic search filter building."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            filters = _build_search_filters()

            # Check required keys exist
            assert "date_from" in filters
            assert "date_to" in filters
            assert "favorites_only" in filters

            # Check boolean filter defaults
            assert filters["favorites_only"] is False

    def test_build_search_filters_with_status(self):
        """Test search filter building with application status."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()
            # Set a specific status
            st.session_state.search_filters["application_status"] = "Applied"
            filters = _build_search_filters()

            # Check status filter is converted to list
            assert filters["application_status"] == ["Applied"]

    def test_clear_all_filters(self):
        """Test that clear_all_filters resets everything."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Modify some values
            st.session_state.search_query = "test query"
            st.session_state.search_filters["remote_only"] = True
            st.session_state.search_results = [{"id": 1, "title": "Test Job"}]

            # Clear filters
            _clear_all_filters()

            # Check everything is reset
            assert st.session_state.search_query == ""
            assert st.session_state.search_results == []
            assert st.session_state.search_filters["remote_only"] is False

    @patch("src.ui.components.search_bar.st.container")
    @patch("src.ui.components.search_bar.st.markdown")
    def test_render_job_search_no_crash(self, mock_markdown, mock_container):
        """Test that render_job_search doesn't crash when called."""
        with patch.object(st, "session_state", MockSessionState()):
            # Mock the container context manager
            mock_container.return_value.__enter__ = MagicMock()
            mock_container.return_value.__exit__ = MagicMock()

            # Should not raise an exception
            try:
                render_job_search()
            except Exception as e:
                pytest.fail(f"render_job_search raised an exception: {e}")

    def test_search_state_isolation(self):
        """Test that search state doesn't interfere with existing session state."""
        with patch.object(
            st, "session_state", MockSessionState({"existing_key": "existing_value"})
        ):
            _init_search_state()

            # Check existing state is preserved
            assert st.session_state["existing_key"] == "existing_value"

            # Check new search state is added
            assert "search_query" in st.session_state

    def test_multiple_init_calls_safe(self):
        """Test that multiple calls to _init_search_state are safe."""
        with patch.object(st, "session_state", MockSessionState()):
            _init_search_state()

            # Modify the state
            st.session_state["search_query"] = "modified query"

            # Call init again
            _init_search_state()

            # Original modification should be preserved
            assert st.session_state["search_query"] == "modified query"
