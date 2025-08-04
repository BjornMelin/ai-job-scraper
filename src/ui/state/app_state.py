"""Application state management for the AI Job Scraper UI.

This module provides a singleton StateManager class to handle centralized
state management across the Streamlit application, ensuring consistent
state access and updates throughout the UI components.
"""

from datetime import datetime, timedelta
from typing import Any

import streamlit as st


class StateManager:
    """Singleton state manager for managing application-wide state.

    This class provides centralized state management for the Streamlit application,
    including filters, view settings, pagination state, and last scrape timing.
    Uses Streamlit's session state as the underlying storage mechanism.
    """

    _instance = None

    def __new__(cls) -> "StateManager":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize state manager with default values."""
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize session state with default values if not already set."""
        defaults = {
            "filters": {
                "company": [],
                "keyword": "",
                "date_from": datetime.now() - timedelta(days=30),
                "date_to": datetime.now(),
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

    @property
    def filters(self) -> dict[str, Any]:
        """Get current filter settings."""
        return st.session_state.filters

    @filters.setter
    def filters(self, value: dict[str, Any]) -> None:
        """Set filter settings."""
        st.session_state.filters = value

    @property
    def view_mode(self) -> str:
        """Get current view mode (List or Card)."""
        return st.session_state.view_mode

    @view_mode.setter
    def view_mode(self, value: str) -> None:
        """Set view mode."""
        st.session_state.view_mode = value

    @property
    def card_page(self) -> int:
        """Get current card page number."""
        return st.session_state.get("card_page", 0)

    @card_page.setter
    def card_page(self, value: int) -> None:
        """Set card page number."""
        st.session_state.card_page = value

    @property
    def sort_by(self) -> str:
        """Get current sort field."""
        return st.session_state.sort_by

    @sort_by.setter
    def sort_by(self, value: str) -> None:
        """Set sort field."""
        st.session_state.sort_by = value

    @property
    def sort_asc(self) -> bool:
        """Get current sort direction (ascending)."""
        return st.session_state.sort_asc

    @sort_asc.setter
    def sort_asc(self, value: bool) -> None:
        """Set sort direction."""
        st.session_state.sort_asc = value

    @property
    def last_scrape(self) -> datetime | None:
        """Get timestamp of last scrape operation."""
        return st.session_state.last_scrape

    @last_scrape.setter
    def last_scrape(self, value: datetime | None) -> None:
        """Set timestamp of last scrape operation."""
        st.session_state.last_scrape = value

    def clear_filters(self) -> None:
        """Reset all filters to default values."""
        self.filters = {
            "company": [],
            "keyword": "",
            "date_from": datetime.now() - timedelta(days=30),
            "date_to": datetime.now(),
        }

    def get_tab_page(self, tab_key: str) -> int:
        """Get page number for a specific tab."""
        page_key = f"card_page_{tab_key}"
        return st.session_state.get(page_key, 0)

    def set_tab_page(self, tab_key: str, page: int) -> None:
        """Set page number for a specific tab."""
        page_key = f"card_page_{tab_key}"
        st.session_state[page_key] = page

    def get_search_term(self, tab_key: str) -> str:
        """Get search term for a specific tab."""
        search_key = f"search_{tab_key}"
        return st.session_state.get(search_key, "")
