"""Companies management page for the AI Job Scraper application.

This module provides the Streamlit UI for managing company records, including
adding new companies and toggling their active status for scraping.
"""

import logging
import uuid

import streamlit as st

from src.services.company_service import CompanyService
from src.ui.ui_rendering import render_company_card_with_selection
from src.ui.utils import is_streamlit_context
from src.ui.utils.background_helpers import (
    get_company_progress,
    is_scraping_active,
)
from src.ui.utils.database_helpers import (
    display_feedback_messages,
    init_session_state_keys,
)

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


def _delete_company_callback(company_id: int) -> None:
    """Callback function to handle company deletion.

    This callback deletes a company and all associated jobs after confirmation.

    Args:
        company_id: Database ID of the company to delete.
    """
    try:
        # Get the toggle state from session state
        confirm_key = f"delete_confirm_{company_id}"
        if st.session_state.get(confirm_key):
            # User confirmed deletion
            if CompanyService.delete_company(company_id):
                st.session_state.delete_success = "Company deleted successfully"
                # Clear the confirmation state
                st.session_state.pop(confirm_key, None)
                # Force page rerun to refresh the list
                st.rerun()
            else:
                st.session_state.delete_error = "Company not found"

    except Exception as e:
        st.session_state.delete_error = f"Failed to delete company: {e}"
        logger.exception("Failed to delete company %s", company_id)


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
            "User toggled company ID %s active status to %s",
            company_id,
            new_status,
        )

    except Exception as e:
        st.session_state.toggle_error = f"Failed to update company status: {e}"
        st.session_state.toggle_success = None
        logger.exception("Failed to toggle company status for ID %s", company_id)


def _company_selection_callback(company_id: int) -> None:
    """Callback function to handle company selection changes.

    Args:
        company_id: Database ID of the company being selected/deselected.
    """
    if "selected_companies" not in st.session_state:
        st.session_state.selected_companies = set()

    checkbox_value = st.session_state.get(f"select_company_{company_id}", False)

    if checkbox_value:
        st.session_state.selected_companies.add(company_id)
    else:
        st.session_state.selected_companies.discard(company_id)

    # Sync selection changes with URL
    from src.ui.utils.url_state import update_url_from_company_selection

    update_url_from_company_selection()


def _select_all_callback() -> None:
    """Callback function to select all companies."""
    try:
        companies = CompanyService.get_all_companies()
        st.session_state.selected_companies = {
            company.id for company in companies if company.id
        }
        # Update individual checkboxes
        for company in companies:
            if company.id:
                st.session_state[f"select_company_{company.id}"] = True

        # Sync with URL
        from src.ui.utils.url_state import update_url_from_company_selection

        update_url_from_company_selection()
        st.rerun()
    except Exception as e:
        st.session_state.bulk_operation_error = f"Failed to select all companies: {e}"
        logger.exception("Failed to select all companies")


def _select_none_callback() -> None:
    """Callback function to clear all company selections."""
    try:
        if "selected_companies" in st.session_state:
            # Clear individual checkboxes
            for company_id in st.session_state.selected_companies:
                st.session_state[f"select_company_{company_id}"] = False
            st.session_state.selected_companies.clear()

        # Sync with URL
        from src.ui.utils.url_state import update_url_from_company_selection

        update_url_from_company_selection()
        st.rerun()
    except Exception as e:
        st.session_state.bulk_operation_error = f"Failed to clear selections: {e}"
        logger.exception("Failed to clear selections")


@st.dialog("Confirm Bulk Delete", width="medium")
def _show_bulk_delete_dialog() -> None:
    """Show bulk delete confirmation dialog using native st.dialog."""
    selected_count = len(st.session_state.get("selected_companies", set()))

    st.warning(
        f"⚠️ Are you sure you want to delete {selected_count} companies? "
        "This will also delete all associated jobs and cannot be undone.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ Confirm Delete", key="confirm_bulk_delete", type="primary"):
            _execute_bulk_delete()
    with col2:
        if st.button("❌ Cancel", key="cancel_bulk_delete"):
            st.rerun()


def _execute_bulk_delete() -> None:
    """Execute bulk deletion with proper error handling."""
    try:
        selected_ids = list(st.session_state.get("selected_companies", set()))
        if not selected_ids:
            st.session_state.bulk_operation_error = "No companies selected for deletion"
            return

        # Generate idempotency token for this operation
        operation_token = str(uuid.uuid4())

        # Check if this operation was already executed
        executed_tokens = st.session_state.get("executed_delete_tokens", set())
        if operation_token in executed_tokens:
            logger.warning(
                "Duplicate bulk delete attempt blocked by idempotency token: %s",
                operation_token,
            )
            return

        # Execute the deletion
        deleted_count = CompanyService.bulk_delete_companies(selected_ids)

        # Mark this token as executed to prevent duplicates
        executed_tokens.add(operation_token)
        st.session_state.executed_delete_tokens = executed_tokens

        # Clear state and show success
        st.session_state.bulk_operation_success = (
            f"Successfully deleted {deleted_count} companies"
        )
        st.session_state.selected_companies.clear()

        logger.info(
            "User bulk deleted %d companies with token %s",
            deleted_count,
            operation_token,
        )
        st.rerun()

    except Exception as e:
        st.session_state.bulk_operation_error = f"Failed to delete companies: {e}"
        st.session_state.bulk_operation_success = None
        logger.exception("Failed to bulk delete companies")


def _bulk_delete_callback() -> None:
    """Callback function to show bulk delete confirmation dialog."""
    selected_ids = list(st.session_state.get("selected_companies", set()))
    if not selected_ids:
        st.session_state.bulk_operation_error = "No companies selected for deletion"
        return

    # Set flag to show confirmation dialog
    st.session_state.show_bulk_delete_confirm = True
    st.rerun()


def _bulk_activate_callback() -> None:
    """Callback function to handle bulk activation of selected companies."""
    try:
        selected_ids = list(st.session_state.get("selected_companies", set()))
        if not selected_ids:
            st.session_state.bulk_operation_error = (
                "No companies selected for activation"
            )
            return

        updated_count = CompanyService.bulk_update_status(selected_ids, active=True)
        st.session_state.bulk_operation_success = (
            f"Successfully activated {updated_count} companies"
        )
        st.session_state.selected_companies.clear()
        logger.info("User bulk activated %d companies", updated_count)
        st.rerun()

    except Exception as e:
        st.session_state.bulk_operation_error = f"Failed to activate companies: {e}"
        st.session_state.bulk_operation_success = None
        logger.exception("Failed to bulk activate companies")


def _bulk_deactivate_callback() -> None:
    """Callback function to handle bulk deactivation of selected companies."""
    try:
        selected_ids = list(st.session_state.get("selected_companies", set()))
        if not selected_ids:
            st.session_state.bulk_operation_error = (
                "No companies selected for deactivation"
            )
            return

        updated_count = CompanyService.bulk_update_status(selected_ids, active=False)
        st.session_state.bulk_operation_success = (
            f"Successfully deactivated {updated_count} companies"
        )
        st.session_state.selected_companies.clear()
        logger.info("User bulk deactivated %d companies", updated_count)
        st.rerun()

    except Exception as e:
        st.session_state.bulk_operation_error = f"Failed to deactivate companies: {e}"
        st.session_state.bulk_operation_success = None
        logger.exception("Failed to bulk deactivate companies")


def _init_and_display_feedback() -> None:
    """Initialize session state and display feedback messages."""
    # Initialize all feedback keys using helper
    init_session_state_keys(
        [
            "add_company_error",
            "add_company_success",
            "toggle_error",
            "toggle_success",
            "bulk_operation_error",
            "bulk_operation_success",
            "selected_companies",
        ],
        None,
    )

    # Initialize selected_companies as empty set if not present or None
    if (
        "selected_companies" not in st.session_state
        or st.session_state.selected_companies is None
    ):
        st.session_state.selected_companies = set()

    # Display feedback messages using helper
    display_feedback_messages(
        success_keys=[
            "add_company_success",
            "toggle_success",
            "delete_success",
            "bulk_operation_success",
        ],
        error_keys=[
            "add_company_error",
            "toggle_error",
            "delete_error",
            "bulk_operation_error",
        ],
    )


def show_companies_page() -> None:
    """Display the companies management page.

    Provides functionality to:
    - Add new companies with name and URL using form with callback
    - View all companies with their scraping statistics
    - Toggle active status for each company using toggles with callbacks
    """
    # Validate URL parameters and sync company selection
    from src.ui.utils.url_state import (
        sync_company_selection_from_url,
        validate_url_params,
    )

    # Validate URL parameters and show warnings if needed
    validation_errors = validate_url_params()
    if validation_errors:
        st.warning(
            "⚠️ Some URL parameters are invalid and have been ignored: "
            + ", ".join(validation_errors.values())
        )

    sync_company_selection_from_url()

    st.title("Company Management")
    st.markdown("Manage companies for job scraping")

    # Initialize session state and display feedback
    _init_and_display_feedback()

    # Add new company section using form with clear visual hierarchy
    st.markdown("### + Add New Company")
    st.markdown("Add a new company to start tracking job opportunities")

    with st.form("add_company_form", border=True):
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
            "Add Company",
            type="primary",
            on_click=_add_company_callback,
        )

    # Display all companies
    st.markdown("### Companies")

    try:
        companies = CompanyService.get_all_companies()

        if not companies:
            st.info("📝 No companies found. Add your first company above!")
            return

        # Bulk selection and operations section
        st.markdown("#### Bulk Operations")

        # Selection controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.button(
                "Select All",
                key="select_all_btn",
                on_click=_select_all_callback,
                help="Select all companies for bulk operations",
            )
        with col2:
            st.button(
                "Select None",
                key="select_none_btn",
                on_click=_select_none_callback,
                help="Clear all company selections",
            )
        with col3:
            selected_companies = (
                st.session_state.get("selected_companies", set()) or set()
            )
            selected_count = len(selected_companies)
            total_count = len(companies)
            st.markdown(f"**{selected_count} of {total_count} companies selected**")

        # Bulk operation buttons (only show when companies are selected)
        if selected_count > 0:
            col1, col2, col3, _ = st.columns([1, 1, 1, 1])

            with col1:
                st.button(
                    "✅ Activate Selected",
                    key="bulk_activate_btn",
                    on_click=_bulk_activate_callback,
                    help=f"Activate {selected_count} selected companies",
                    type="secondary",
                )

            with col2:
                st.button(
                    "❌ Deactivate Selected",
                    key="bulk_deactivate_btn",
                    on_click=_bulk_deactivate_callback,
                    help=f"Deactivate {selected_count} selected companies",
                    type="secondary",
                )

            with col3:
                st.button(
                    "🗑️ Delete Selected",
                    key="bulk_delete_btn",
                    on_click=_bulk_delete_callback,
                    help=f"Delete {selected_count} selected companies and jobs",
                    type="secondary",
                )

        # Show bulk delete confirmation dialog
        if st.session_state.get("show_bulk_delete_confirm", False):
            selected_count = len(st.session_state.get("selected_companies", set()))
            st.warning(
                f"⚠️ Are you sure you want to delete {selected_count} companies? "
                "This will also delete all associated jobs and cannot be undone.",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Confirm Delete", key="confirm_bulk_delete"):
                    _execute_bulk_delete()
            with col2:
                if st.button("❌ Cancel", key="cancel_bulk_delete"):
                    st.session_state.show_bulk_delete_confirm = False
                    st.rerun()

        st.markdown("---")

        # Display companies with selection checkboxes
        for company in companies:
            render_company_card_with_selection(
                company,
                _toggle_company_callback,
                _delete_company_callback,
                _company_selection_callback,
            )

        # Show summary statistics
        st.markdown("---")
        _render_company_statistics(companies)

        # Show scraping progress if active
        _company_scraping_progress_fragment()

    except Exception as e:
        st.error(f"❌ Failed to load companies: {e!s}")
        logger.exception("Failed to load companies")


@st.fragment(run_every="2s")
def _company_scraping_progress_fragment():
    """Fragment for displaying real-time company scraping progress.

    This fragment auto-refreshes every 2 seconds during active scraping
    to show progress updates without affecting the main company list.
    """
    if not is_scraping_active():
        return

    st.markdown("### 🔄 Scraping Progress")

    company_progress = get_company_progress()

    if not company_progress:
        st.info("⏳ Initializing scraping...")
        return

    # Display progress for each company
    progress_cols = st.columns(min(3, len(company_progress)))

    for idx, (company_name, progress) in enumerate(company_progress.items()):
        col = progress_cols[idx % len(progress_cols)]

        with col:
            # Status indicator
            if progress.status == "Completed":
                status_icon = "✅"
                status_color = "normal"
            elif progress.status == "In Progress":
                status_icon = "🔄"
                status_color = "normal"
            elif progress.error:
                status_icon = "❌"
                status_color = "inverse"
            else:
                status_icon = "⏳"
                status_color = "normal"

            st.metric(
                f"{status_icon} {company_name}",
                f"{progress.jobs_found} jobs",
                delta=progress.status,
                delta_color=status_color,
            )

            # Show error if present
            if progress.error:
                st.error(f"Error: {progress.error}")

    # Overall progress summary
    total_companies = len(company_progress)
    completed = sum(1 for p in company_progress.values() if p.status == "Completed")

    overall_progress = completed / total_companies if total_companies > 0 else 0

    st.progress(
        overall_progress, text=f"Progress: {completed}/{total_companies} companies"
    )


def _render_company_statistics(companies):
    """Render company statistics section.

    Args:
        companies: List of company objects.
    """
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


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    show_companies_page()
