"""CSS loader utility for Streamlit applications.

This module provides a function to load external CSS files into Streamlit apps,
enabling better organization and maintainability of styles.
"""

from pathlib import Path

import streamlit as st


def load_css(css_file: str) -> None:
    """Load a CSS file and inject it into the Streamlit app.

    Args:
        css_file: Path to the CSS file relative to the project root.

    Raises:
        FileNotFoundError: If the CSS file doesn't exist.
    """
    css_path = Path(css_file)
    if not css_path.exists():
        msg = f"CSS file not found: {css_file}"
        raise FileNotFoundError(msg)

    with Path(css_path).open() as f:
        css_content = f.read()

    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
