"""Main entry point for the AI Job Scraper Streamlit application.

This module serves as the primary entry point for the web application,
handling page configuration, theme loading, and navigation using Streamlit's
built-in st.navigation() for optimal performance and maintainability.
"""

import streamlit as st

from src.ui.state.session_state import init_session_state
from src.ui.styles.optimized_theme import load_theme
from src.ui.utils.database_utils import render_database_health_widget


def main() -> None:
    """Main application entry point.

    Configures the Streamlit page, loads theme, initializes state management,
    and sets up navigation using library-first st.navigation() approach.
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Job Scraper",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "AI-powered job scraper for managing your job search efficiently."
        },
    )

    # Load application theme and styles
    load_theme()

    # Initialize session state with library-first approach
    init_session_state()

    # Define pages with preserved functionality using st.navigation()
    pages = [
        st.Page(
            "src/ui/pages/jobs.py",
            title="Jobs",
            icon="📋",
            default=True,  # Preserves default behavior
        ),
        st.Page("src/ui/pages/companies.py", title="Companies", icon="🏢"),
        st.Page("src/ui/pages/scraping.py", title="Scraping", icon="🔍"),
        st.Page("src/ui/pages/settings.py", title="Settings", icon="⚙️"),
    ]

    # Add database health monitoring to sidebar
    render_database_health_widget()

    # Streamlit handles all navigation logic automatically
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
