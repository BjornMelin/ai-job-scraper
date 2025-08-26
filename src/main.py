"""Main entry point for the AI Job Scraper Streamlit application.

This module serves as the primary entry point for the web application,
handling page configuration, theme loading, and navigation using Streamlit's
built-in st.navigation() for optimal performance and maintainability.
"""

import logging

import streamlit as st

from src.db.migrations import run_migrations
from src.ui.state.session_state import init_session_state
from src.ui.styles.theme import load_theme
from src.ui.utils.database_helpers import render_database_health_widget
from src.utils.startup_helpers import initialize_performance_optimizations

logger = logging.getLogger(__name__)


def main() -> None:
    """Main application entry point.

    Configures the Streamlit page, loads theme, initializes state management,
    and sets up navigation using library-first st.navigation() approach.
    """
    # Run database migrations BEFORE any Streamlit operations
    # This ensures the database schema is up to date on every startup
    run_migrations()

    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Job Scraper",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "AI-powered job scraper for managing your job search efficiently.",
        },
    )

    # Load application theme and styles
    load_theme()

    # Initialize session state with library-first approach
    init_session_state()

    # Initialize simple startup configuration
    # Uses Streamlit's native caching with library-first approach
    initialize_performance_optimizations()

    # Define pages using st.navigation() with relative paths
    # All paths are relative to the main.py entrypoint file
    pages = [
        st.Page(
            "ui/pages/jobs.py",
            title="Jobs",
            icon="📋",
            default=True,  # Preserves default behavior from old navigation
        ),
        st.Page(
            "ui/pages/companies.py",
            title="Companies",
            icon="🏢",
        ),
        st.Page(
            "ui/pages/scraping.py",
            title="Scraping",
            icon="🔍",
        ),
        st.Page(
            "ui/pages/analytics.py",
            title="Analytics",
            icon="📊",
        ),
        st.Page(
            "ui/pages/settings.py",
            title="Settings",
            icon="⚙️",
        ),
    ]

    # Add database health monitoring to sidebar
    render_database_health_widget()

    # Streamlit handles all navigation logic automatically
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
