"""Streamlit session state initialization utilities.

This module provides a library-first approach to session state management,
replacing the custom StateManager singleton with direct st.session_state usage
for better performance and maintainability.
"""

from datetime import datetime, timedelta, timezone

import streamlit as st


def init_session_state() -> None:
    """Initialize session state with all required default values.

    This function replaces the StateManager singleton pattern with direct
    Streamlit session state management, following library-first principles.
    """
    defaults = {
        "filters": {
            "company": [],
            "keyword": "",
            "date_from": datetime.now(timezone.utc) - timedelta(days=30),
            "date_to": datetime.now(timezone.utc),
            "salary_min": 0,
            "salary_max": 750000,
        },
        "view_mode": "Card",  # Default to more visual card view
        "card_page": 0,
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
        "date_from": datetime.now(timezone.utc) - timedelta(days=30),
        "date_to": datetime.now(timezone.utc),
        "salary_min": 0,
        "salary_max": 750000,
    }


def get_tab_page(tab_key: str) -> int:
    """Get page number for a specific tab."""
    page_key = f"card_page_{tab_key}"
    return st.session_state.get(page_key, 0)


def set_tab_page(tab_key: str, page: int) -> None:
    """Set page number for a specific tab."""
    page_key = f"card_page_{tab_key}"
    st.session_state[page_key] = page


def get_search_term(tab_key: str) -> str:
    """Get search term for a specific tab."""
    search_key = f"search_{tab_key}"
    return st.session_state.get(search_key, "")
