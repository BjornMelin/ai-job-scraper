"""Companies management page for the AI Job Scraper application.

This module provides the Streamlit UI for managing company records, including
adding new companies and toggling their active status for scraping.
"""

import logging

import streamlit as st

from src.services.company_service import CompanyService
from src.ui.utils.session_helpers import (
    display_feedback_messages,
    init_session_state_keys,
)
from src.ui.utils.streamlit_context import is_streamlit_context

logger = logging.getLogger(__name__)


def _add_company_callback() -> None:
    """Callback function to handle adding a new company.

    This callback processes form data from session state and adds the company,
    providing proper error handling and user feedback.
    """
    try:
        company_name = st.session_state.get("company_name", "").strip()
        company_url = st.session_state.get("company_url", "").strip()

        # Validate inputs
        if not company_name:
            st.session_state.add_company_error = "Company name is required"
            return
        if not company_url:
            st.session_state.add_company_error = "Company URL is required"
            return

        # Add the company
        company = CompanyService.add_company(name=company_name, url=company_url)
        st.session_state.add_company_success = (
            f"Successfully added company: {company.name}"
        )
        logger.info("User added new company: %s", company.name)

        # Clear form inputs on success
        st.session_state.company_name = ""
        st.session_state.company_url = ""
        st.session_state.add_company_error = None

        # Trigger immediate UI refresh
        st.rerun()

    except ValueError as e:
        st.session_state.add_company_error = str(e)
        st.session_state.add_company_success = None
        logger.warning("Failed to add company due to validation: %s", e)
    except Exception:
        st.session_state.add_company_error = "Failed to add company. Please try again."
        st.session_state.add_company_success = None
        logger.exception("Failed to add company")


def _toggle_company_callback(company_id: int) -> None:
    """Callback function to toggle a company's active status.

    Args:
        company_id: Database ID of the company to toggle.
    """
    try:
        new_status = CompanyService.toggle_company_active(company_id)

        # Store feedback in session state for display after rerun
        if new_status:
            st.session_state.toggle_success = "Enabled scraping"
        else:
            st.session_state.toggle_success = "Disabled scraping"

        st.session_state.toggle_error = None
        logger.info(
            "User toggled company ID %s active status to %s", company_id, new_status
        )

    except Exception as e:
        st.session_state.toggle_error = f"Failed to update company status: {e}"
        st.session_state.toggle_success = None
        logger.exception("Failed to toggle company status for ID %s", company_id)


def _init_and_display_feedback() -> None:
    """Initialize session state and display feedback messages."""
    # Initialize all feedback keys using helper
    init_session_state_keys(
        [
            "add_company_error",
            "add_company_success",
            "toggle_error",
            "toggle_success",
        ],
        None,
    )

    # Display feedback messages using helper
    display_feedback_messages(
        success_keys=["add_company_success", "toggle_success"],
        error_keys=["add_company_error", "toggle_error"],
    )


def show_companies_page() -> None:
    """Display the companies management page.

    Provides functionality to:
    - Add new companies with name and URL using form with callback
    - View all companies with their scraping statistics
    - Toggle active status for each company using toggles with callbacks
    """
    st.title("Company Management")
    st.markdown("Manage companies for job scraping")

    # Initialize session state and display feedback
    _init_and_display_feedback()

    # Add new company section using expander with form
    with st.expander("+ Add New Company", expanded=False), st.form("add_company_form"):
        st.markdown("### Add a New Company")

        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                "Company Name",
                placeholder="e.g., TechCorp",
                help="Enter the company name (must be unique)",
                key="company_name",
            )

        with col2:
            st.text_input(
                "Careers URL",
                placeholder="e.g., https://techcorp.com/careers",
                help="Enter the company's careers page URL",
                key="company_url",
            )

        st.form_submit_button(
            "Add Company", type="primary", on_click=_add_company_callback
        )

    # Display all companies
    st.markdown("### Companies")

    try:
        companies = CompanyService.get_all_companies()

        if not companies:
            st.info("📝 No companies found. Add your first company above!")
            return

        # Display companies in a clean grid layout
        from src.ui.helpers.company_display import render_company_card

        for company in companies:
            render_company_card(company, _toggle_company_callback)

        # Show summary statistics
        st.markdown("---")
        active_count = sum(company.active for company in companies)
        total_companies = len(companies)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Companies", total_companies)

        with col2:
            st.metric("Active Companies", active_count)

        with col3:
            inactive_count = total_companies - active_count
            st.metric("Inactive Companies", inactive_count)

    except Exception:
        st.error("❌ Failed to load companies. Please refresh the page.")
        logger.exception("Failed to load companies")


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    show_companies_page()
