"""System and environment detection utilities.

This module provides utilities for detecting the runtime environment and context,
including:
- Streamlit context detection
- Test environment detection
- Development/production environment detection

These utilities help the application adapt its behavior based on the execution context.
"""

import logging
import os
import sys

from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_streamlit_context() -> bool:
    """Check if we're running in a proper Streamlit context.

    Returns:
        bool: True if running within Streamlit's script runner context, False otherwise.

    Example:
        >>> if is_streamlit_context():
        ...     import streamlit as st
        ...
        ...     st.write("Running in Streamlit!")
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        return False


def is_test_environment() -> bool:
    """Check if we're running in a test environment.

    Detects test environments by checking for:
    - pytest module in sys.modules
    - TEST environment variable
    - Running from a test file (name contains 'test')

    Returns:
        bool: True if running in a test environment, False otherwise.

    Example:
        >>> if is_test_environment():
        ...     print("Running tests - using test configuration")
    """
    # Check if pytest is running
    if "pytest" in sys.modules:
        return True

    # Check for TEST environment variable
    if os.getenv("TEST", "").lower() in ("1", "true", "yes"):
        return True

    # Check if running from a test file
    if hasattr(sys, "argv") and sys.argv:
        script_name = Path(sys.argv[0]).name
        if "test" in script_name.lower() or script_name.startswith("pytest"):
            return True

    return False


def is_development_environment() -> bool:
    """Check if we're running in a development environment.

    Detects development environments by checking:
    - DEBUG environment variable
    - ENVIRONMENT variable set to 'dev' or 'development'
    - DEV environment variable

    Returns:
        bool: True if running in development mode, False otherwise.

    Example:
        >>> if is_development_environment():
        ...     logging.basicConfig(level=logging.DEBUG)
    """
    # Check DEBUG flag
    if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
        return True

    # Check ENVIRONMENT variable
    env = os.getenv("ENVIRONMENT", "").lower()
    if env in ("dev", "development", "debug"):
        return True

    # Check DEV flag
    return os.getenv("DEV", "").lower() in ("1", "true", "yes")


def get_runtime_context() -> dict[str, Any]:
    """Get comprehensive runtime context information.

    Returns:
        dict: Dictionary containing runtime context information including:
            - streamlit: Whether running in Streamlit context
            - test: Whether running in test environment
            - development: Whether running in development environment
            - python_version: Python version string
            - platform: Operating system platform

    Example:
        >>> context = get_runtime_context()
        >>> if context["streamlit"]:
        ...     print("Using Streamlit-specific UI components")
    """
    import platform

    return {
        "streamlit": is_streamlit_context(),
        "test": is_test_environment(),
        "development": is_development_environment(),
        "python_version": sys.version,
        "platform": platform.system(),
        "architecture": platform.machine(),
    }


# Export list for clean imports
__all__ = [
    "get_runtime_context",
    "is_development_environment",
    "is_streamlit_context",
    "is_test_environment",
]
