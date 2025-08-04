"""Main entry point for the AI Job Scraper Streamlit application.

This module serves as the primary entry point for the web application,
handling page configuration, theme loading, and orchestrating the
sidebar and main content rendering through a modular component architecture.
"""

import streamlit as st

from src.ui.components.sidebar import render_sidebar
from src.ui.pages.jobs import render_jobs_page
from src.ui.state.app_state import StateManager
from src.ui.styles.theme import load_theme


def main() -> None:
    """Main application entry point.

    This function configures the Streamlit page, loads the application theme,
    initializes the state manager, and orchestrates the rendering of the
    sidebar and main page content.
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Job Tracker",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "AI-powered job tracker for managing your job search efficiently."
        },
    )

    # Load application theme and styles
    load_theme()

    # Initialize state manager (singleton)
    StateManager()

    # Render sidebar
    render_sidebar()

    # Render main jobs page
    render_jobs_page()


if __name__ == "__main__":
    main()
