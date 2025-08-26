"""Simplified startup helpers for application initialization.

This module provides minimal startup utilities with library-first approach:
- Simple session state initialization
- Basic logging setup
- Minimal configuration without complex performance optimization
"""

import logging

from typing import Any

# Import streamlit with fallback for non-Streamlit environments
try:
    import streamlit as st
except ImportError:
    # Create dummy streamlit for non-Streamlit environments
    class _DummyStreamlit:
        """Dummy Streamlit class for non-Streamlit environments."""

        @staticmethod
        def cache_data(**_kwargs):
            """Dummy cache decorator that passes through the function unchanged."""

            def decorator(func):
                """Inner decorator function."""
                return func

            return decorator

        class SessionState:
            """Dummy session state for non-Streamlit environments."""

            initialized = False

        session_state = SessionState()

    st = _DummyStreamlit()

logger = logging.getLogger(__name__)


def initialize_performance_optimizations() -> dict[str, Any]:
    """Initialize minimal startup configuration.

    Simplified version that just marks session as initialized.
    Uses Streamlit's native caching and no complex optimization.

    Returns:
        Dictionary with basic initialization status.
    """
    from contextlib import suppress

    with suppress(Exception):
        logger.info("Initializing simplified startup configuration...")

    # Simple initialization - just mark session as started
    if hasattr(st, "session_state"):
        st.session_state.initialized = True

    result = {
        "status": "initialized",
        "approach": "library_first_streamlit_native",
        "complex_optimization": "removed_for_simplicity",
    }

    with suppress(Exception):
        logger.info("Startup configuration initialized using library-first approach")

    return result
