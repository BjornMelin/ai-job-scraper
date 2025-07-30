"""CSS utility functions for the AI Job Scraper application.

This module provides helper functions for CSS management, including
loading multiple CSS files and applying CSS classes dynamically.
"""

from pathlib import Path

import streamlit as st


def load_multiple_css(css_files: list[str]) -> None:
    """Load multiple CSS files into the Streamlit app.

    Args:
        css_files: List of CSS file paths relative to the project root.
    """
    combined_css = ""

    for css_file in css_files:
        css_path = Path(css_file)
        if css_path.exists():
            with open(css_path) as f:
                combined_css += f.read() + "\n"
        else:
            st.warning(f"CSS file not found: {css_file}")

    if combined_css:
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)


def apply_button_style(button_type: str = "primary") -> str:
    """Get CSS class name for different button types.

    Args:
        button_type: Type of button ('primary', 'success', 'danger').

    Returns:
        CSS class name to apply to the button container.
    """
    button_classes = {
        "primary": "",  # Default style
        "success": "success-button",
        "danger": "danger-button",
    }
    return button_classes.get(button_type, "")


def create_status_badge(status: str, text: str | None = None) -> str:
    """Create HTML for a status badge.

    Args:
        status: Status type ('new', 'interested', 'applied', 'rejected').
        text: Text to display in the badge. Defaults to the status.

    Returns:
        HTML string for the status badge.
    """
    if text is None:
        text = status.title()

    status_class = f"status-{status.lower()}"
    return f'<span class="status-badge {status_class}">{text}</span>'
