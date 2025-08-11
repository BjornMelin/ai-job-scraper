"""Main entry point for the AI Job Scraper Streamlit application.

This module serves as the primary entry point for the web application,
handling page configuration, theme loading, and navigation using Streamlit's
built-in st.navigation() for optimal performance and maintainability.
"""

import logging

import streamlit as st

from alembic import command
from alembic.config import Config
from src.ui.state.session_state import init_session_state
from src.ui.styles.theme import load_theme
from src.ui.utils.database_utils import render_database_health_widget

logger = logging.getLogger(__name__)


def run_migrations() -> None:
    """Run Alembic database migrations to head revision.

    This function handles migration execution with proper error handling
    and logging. It's designed to run safely during application startup
    and is idempotent (safe to run multiple times).

    The function uses the alembic.ini configuration file and will apply
    all pending migrations to bring the database schema up to date.
    """
    try:
        logger.info("Starting database migrations...")

        # Load Alembic configuration from alembic.ini
        alembic_cfg = Config("alembic.ini")

        # Run migrations to head (latest) revision
        # This is idempotent - safe to run multiple times
        command.upgrade(alembic_cfg, "head")

        logger.info("Database migrations completed successfully")

    except Exception:
        logger.exception("Failed to run database migrations")
        # Don't raise the exception to prevent app startup failure
        # The app can still work with the current database state
        logger.warning(
            "Application will continue with current database schema. "
            "Manual migration may be required."
        )


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
            "About": "AI-powered job scraper for managing your job search efficiently."
        },
    )

    # Load application theme and styles
    load_theme()

    # Initialize session state with library-first approach
    init_session_state()

    # Define pages using st.navigation() with relative paths
    # All paths are relative to the main.py entrypoint file
    pages = [
        st.Page(
            "src/ui/pages/jobs.py",
            title="Jobs",
            icon="📋",
            default=True,  # Preserves default behavior from old navigation
        ),
        st.Page(
            "src/ui/pages/companies.py",
            title="Companies",
            icon="🏢",
        ),
        st.Page(
            "src/ui/pages/scraping.py",
            title="Scraping",
            icon="🔍",
        ),
        st.Page(
            "src/ui/pages/settings.py",
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
