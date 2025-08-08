"""Streamlit context detection utilities for page execution.

This module provides utilities to detect if code is running in a proper Streamlit
context, which is essential for preventing page functions from executing during
test imports or other non-Streamlit contexts.
"""

import logging

logger = logging.getLogger(__name__)


def is_streamlit_context() -> bool:
    """Check if we're running in a proper Streamlit context.

    This function determines whether the current execution context is within
    a running Streamlit application. This is crucial for preventing page
    functions from executing when modules are imported during testing or
    other non-Streamlit scenarios.

    Returns:
        bool: True if in Streamlit runtime context, False otherwise.

    Examples:
        >>> # In a Streamlit page file:
        >>> if is_streamlit_context():
        ...     render_page()  # Only execute when actually running in Streamlit
    """
    try:
        # Check if Streamlit's script run context exists
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        # If Streamlit is not available or the context doesn't exist
        return False
