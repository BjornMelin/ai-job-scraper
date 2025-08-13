"""Streamlit session state initialization utilities.

This module provides a library-first approach to session state management,
replacing the custom StateManager singleton with direct st.session_state usage
for better performance and maintainability.
"""

from datetime import UTC, datetime, timedelta

import streamlit as st

from src.constants import SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN


def init_session_state() -> None:
    """Initialize session state with all required default values.

    This function replaces the StateManager singleton pattern with direct
    Streamlit session state management, following library-first principles.
    """
    defaults = {
        "filters": {
            "company": [],
            "keyword": "",
            "date_from": datetime.now(UTC) - timedelta(days=30),
            "date_to": datetime.now(UTC),
            "salary_min": SALARY_DEFAULT_MIN,
            "salary_max": SALARY_DEFAULT_MAX,
        },
        "view_mode": "Card",  # Default to attractive card view for better UX
        "sort_by": "Posted",
        "sort_asc": False,
        "last_scrape": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_filters() -> None:
    """Reset all filters to default values."""
    st.session_state.filters = {
        "company": [],
        "keyword": "",
        "date_from": datetime.now(UTC) - timedelta(days=30),
        "date_to": datetime.now(UTC),
        "salary_min": SALARY_DEFAULT_MIN,
        "salary_max": SALARY_DEFAULT_MAX,
    }


def get_search_term(tab_key: str) -> str:
    """Get search term for a specific tab."""
    search_key = f"search_{tab_key}"
    return st.session_state.get(search_key, "")
