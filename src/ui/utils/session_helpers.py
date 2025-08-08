"""Session state helper utilities for Streamlit applications.

This module provides reusable utilities for common session state operations
including initialization and feedback message display patterns.
"""

from typing import Any

import streamlit as st


def init_session_state_keys(keys: list[str], default_value: Any = None) -> None:
    """Initialize multiple session state keys with a default value.

    Args:
        keys: List of session state keys to initialize.
        default_value: Default value to set if key doesn't exist.

    Examples:
        >>> # Initialize feedback keys
        >>> init_session_state_keys(
        ...     ["add_company_error", "add_company_success"], None
        ... )

        >>> # Initialize form keys with empty strings
        >>> init_session_state_keys(["company_name", "company_url"], "")
    """
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_feedback_messages(success_keys: list[str], error_keys: list[str]) -> None:
    """Display and clear feedback messages from session state.

    This function displays success and error messages stored in session state,
    then clears them to prevent them from showing again on subsequent runs.

    Args:
        success_keys: List of session state keys containing success messages.
        error_keys: List of session state keys containing error messages.

    Examples:
        >>> # Display feedback for company operations
        >>> display_feedback_messages(
        ...     success_keys=["add_company_success", "toggle_success"],
        ...     error_keys=["add_company_error", "toggle_error"],
        ... )
    """
    # Display and clear success messages
    for key in success_keys:
        if st.session_state.get(key):
            st.success(f"✅ {st.session_state[key]}")
            st.session_state[key] = None

    # Display and clear error messages
    for key in error_keys:
        if st.session_state.get(key):
            st.error(f"❌ {st.session_state[key]}")
            st.session_state[key] = None
