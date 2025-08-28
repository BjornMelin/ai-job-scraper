"""Streamlit session state initialization utilities.

This module provides a WIDGET-FIRST approach to session state management,
minimizing session state usage by leveraging native Streamlit widget keys.
Only essential cross-page state is preserved in session_state.
"""

from datetime import UTC, datetime, timedelta

import streamlit as st

from src.constants import SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN


def init_session_state() -> None:
    """Initialize MINIMAL session state with only essential cross-page values.

    This function implements the widget-first strategy, preserving only:
    1. Cross-page navigation state that cannot use widget keys
    2. User preferences that persist across page loads
    3. Authentication/session tokens (when implemented)

    All UI state now uses widget keys instead of manual session state management.
    """
    essential_defaults = {
        # ESSENTIAL: Cross-page navigation state
        "selected_tab": "all",  # Job tab selection (all/favorites/applied)
        # ESSENTIAL: Last scrape time for refresh status display
        "last_scrape": None,
        # ESSENTIAL: Modal state (unified across components)
        "modal_job_id": None,
    }

    for key, value in essential_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_filters() -> None:
    """Reset all filters by clearing widget keys.

    With widget-first approach, we clear widget keys rather than session state.
    Streamlit will reset widgets to their default values on next render.
    """
    # Clear widget keys that hold filter state
    filter_widgets = [
        "company_filter",
        "keyword_search",
        "date_from_filter",
        "date_to_filter",
        "salary_range_filter",
    ]

    for widget_key in filter_widgets:
        if widget_key in st.session_state:
            del st.session_state[widget_key]

    # Trigger a rerun to reset widgets to defaults
    st.rerun()


def get_current_filters() -> dict:
    """Get current filter values from widget keys.

    Returns filter values from widgets, falling back to defaults.
    This replaces the old session_state.filters approach.
    """
    return {
        "company": st.session_state.get("company_filter", []),
        "keyword": st.session_state.get("keyword_search", ""),
        "date_from": st.session_state.get(
            "date_from_filter", datetime.now(UTC) - timedelta(days=30)
        ),
        "date_to": st.session_state.get("date_to_filter", datetime.now(UTC)),
        "salary_min": st.session_state.get(
            "salary_range_filter", (SALARY_DEFAULT_MIN, SALARY_DEFAULT_MAX)
        )[0]
        if st.session_state.get("salary_range_filter")
        else SALARY_DEFAULT_MIN,
        "salary_max": st.session_state.get(
            "salary_range_filter", (SALARY_DEFAULT_MIN, SALARY_DEFAULT_MAX)
        )[1]
        if st.session_state.get("salary_range_filter")
        else SALARY_DEFAULT_MAX,
    }
