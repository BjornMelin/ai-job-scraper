"""Main entry point for the AI Job Scraper Streamlit application.

This module serves as the primary entry point for the web application,
handling page configuration, theme loading, navigation, and orchestrating the
rendering of different pages through a modular component architecture.
"""

import streamlit as st

from src.ui.components.sidebar import render_sidebar
from src.ui.pages.companies import show_companies_page
from src.ui.pages.jobs import render_jobs_page
from src.ui.pages.scraping import render_scraping_page
from src.ui.pages.settings import show_settings_page
from src.ui.state.app_state import StateManager
from src.ui.styles.theme import load_theme


def main() -> None:
    """Main application entry point.

    This function configures the Streamlit page, loads the application theme,
    initializes the state manager, handles navigation, and orchestrates the
    rendering of the appropriate page content.
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

    # Initialize state manager (singleton)
    StateManager()

    # Page navigation
    _render_navigation()

    # Get selected page from session state
    selected_page = st.session_state.get("selected_page", "Jobs")

    # Render content based on selected page
    if selected_page == "Jobs":
        render_sidebar()  # Sidebar is only needed for the Jobs page
        render_jobs_page()
    elif selected_page == "Companies":
        show_companies_page()
    elif selected_page == "Scraping":
        render_scraping_page()
    elif selected_page == "Settings":
        show_settings_page()
    else:
        # Default to Jobs page
        render_sidebar()
        render_jobs_page()


def _render_navigation() -> None:
    """Render the top navigation bar for page selection.

    This creates a clean navigation interface allowing users to switch
    between different pages of the application.
    """
    # Initialize selected_page in session state if not exists
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Jobs"

    # Create navigation columns
    col1, col2, col3, col4 = st.columns(4)

    pages = [
        ("Jobs", "📋", "Browse and manage job listings"),
        ("Companies", "🏢", "Manage company sources"),
        ("Scraping", "🔍", "Monitor scraping operations"),
        ("Settings", "⚙️", "Configure application settings"),
    ]

    # Render navigation buttons
    for i, (page_name, icon, tooltip) in enumerate(pages):
        with [col1, col2, col3, col4][i]:
            # Determine button type based on selection
            button_type = (
                "primary"
                if st.session_state.selected_page == page_name
                else "secondary"
            )

            if st.button(
                f"{icon} {page_name}",
                key=f"nav_{page_name}",
                use_container_width=True,
                type=button_type,
                help=tooltip,
            ):
                st.session_state.selected_page = page_name
                st.rerun()

    # Add a visual separator
    st.markdown("---")


if __name__ == "__main__":
    main()
