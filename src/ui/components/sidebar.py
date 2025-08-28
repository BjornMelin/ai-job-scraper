"""Sidebar component for the AI Job Scraper UI.

This module provides the sidebar functionality including search filters,
view settings, and company management features. It handles user interactions
for filtering jobs and managing company configurations.
"""

import logging

from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from src.constants import (
    SALARY_DEFAULT_MAX,
    SALARY_DEFAULT_MIN,
    SALARY_SLIDER_FORMAT,
    SALARY_SLIDER_STEP,
    SALARY_UNBOUNDED_THRESHOLD,
)

# Removed direct database import - using service layer instead
from src.ui.state.session_state import clear_filters, get_current_filters
from src.ui.utils import format_salary
from src.ui.utils.service_cache import get_company_service

logger = logging.getLogger(__name__)


def render_sidebar() -> None:
    """Render the complete sidebar with all sections.

    This function orchestrates the rendering of all sidebar components including
    search filters, view settings, and company management. It manages the
    application state and handles user interactions within the sidebar.
    """
    # Import URL state functions here to avoid circular imports
    from src.ui.utils.url_state import sync_filters_from_url

    # Sync filters from URL on sidebar render
    sync_filters_from_url()

    with st.sidebar:
        _render_search_filters()
        st.divider()
        _render_view_settings()
        st.divider()
        _render_company_management()


def _render_search_filters() -> None:
    """Render the search and filter section using WIDGET KEYS.

    This eliminates manual session state management in favor of native
    Streamlit widget key functionality for better performance.
    """
    st.markdown("### 🔍 Search & Filter")

    with st.container():
        # Get company list from cached service
        companies = _get_company_list()

        # WIDGET KEY: Company filter - no manual state management needed
        st.multiselect(
            "Filter by Company",
            options=companies,
            placeholder="All companies",
            help="Select one or more companies to filter jobs",
            key="company_filter",  # Widget handles its own state
        )

        # WIDGET KEY: Keyword search - auto-managed state
        st.text_input(
            "Search Keywords",
            placeholder="e.g., Python, Machine Learning, Remote",
            help="Full-text search with stemming (e.g., 'develop' matches 'developer')",
            key="keyword_search",  # Widget handles its own state
        )

        # WIDGET KEYS: Date range with auto-managed state
        st.markdown("**Date Range**")
        col1, col2 = st.columns(2)

        with col1:
            st.date_input(
                "From",
                value=datetime.now(UTC) - timedelta(days=30),  # Default value
                help="Show jobs posted after this date",
                key="date_from_filter",  # Widget handles its own state
            )

        with col2:
            st.date_input(
                "To",
                value=datetime.now(UTC),  # Default value
                help="Show jobs posted before this date",
                key="date_to_filter",  # Widget handles its own state
            )

        # WIDGET KEY: Salary range with auto-managed state
        st.markdown("**Salary Range**")
        salary_range = st.slider(
            "Annual Salary Range",
            min_value=0,
            max_value=SALARY_UNBOUNDED_THRESHOLD,
            value=(SALARY_DEFAULT_MIN, SALARY_DEFAULT_MAX),  # Default value
            step=SALARY_SLIDER_STEP,
            format=SALARY_SLIDER_FORMAT,
            help=(
                f"Filter jobs by annual salary range (in USD). "
                f"Set max to {format_salary(SALARY_UNBOUNDED_THRESHOLD)}+ "
                f"to include all high-value positions."
            ),
            key="salary_range_filter",  # Widget handles its own state
        )

        # Display current salary range
        _display_salary_range(salary_range)

        # Sync widget-based filters with URL
        from src.ui.utils.url_state import update_url_from_filters

        # Get current filter values from widgets and sync with URL
        current_filters = get_current_filters()
        update_url_from_filters(current_filters)

        # Clear filters button
        if st.button("Clear All Filters", use_container_width=True):
            clear_filters()
            from src.ui.utils.url_state import clear_url_params

            clear_url_params()
            st.rerun()


def _render_view_settings() -> None:
    """Render view settings using WIDGET KEYS.

    View mode selection now uses radio buttons with widget keys
    instead of manual session state management.
    """
    st.markdown("### 👁️ View Settings")

    # WIDGET KEY: View mode selection - auto-managed state
    st.radio(
        "Display Mode",
        options=["Card", "List"],
        index=0,  # Default to Card view
        horizontal=True,
        key="view_mode_selection",  # Widget handles its own state
        label_visibility="collapsed",
    )


def _render_company_management() -> None:
    """Render the company management section of the sidebar.

    This section allows users to view, edit, and add companies for job scraping.
    It includes functionality for toggling company active status and adding new
    companies.
    """
    with st.expander("🏢 Manage Companies", expanded=False):
        # Get companies from cached service layer
        company_service = get_company_service()
        companies_data = company_service.get_companies_for_management()
        comp_df = pd.DataFrame(companies_data)

        if not comp_df.empty:
            st.markdown("**Existing Companies**")
            edited_comp = st.data_editor(
                comp_df,
                column_config={
                    "Active": st.column_config.CheckboxColumn(
                        "Active",
                        width="small",
                        help="Toggle to enable/disable scraping",
                    ),
                    "URL": st.column_config.LinkColumn(
                        "URL",
                        width="large",
                        help="Company careers page URL",
                    ),
                    "Name": st.column_config.TextColumn("Company Name", width="medium"),
                    "id": st.column_config.NumberColumn(
                        "ID",
                        width="small",
                        disabled=True,
                    ),
                },
                hide_index=True,
                use_container_width=True,
                height=400,  # Streamlit 1.47+ height parameter
            )

            if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                _save_company_changes(edited_comp)

        # Add new company section
        _render_add_company_form()


def _get_company_list() -> list[str]:
    """Get list of unique company names using CACHED service layer.

    Returns:
        List of company names sorted alphabetically.
    """
    try:
        company_service = get_company_service()
        companies = company_service.get_all_companies()
        return [company.name for company in companies]

    except Exception:
        logger.exception("Failed to get company list")
        return []


def _save_company_changes(edited_comp: pd.DataFrame) -> None:
    """Save changes to company settings using CACHED service layer.

    Args:
        edited_comp: DataFrame containing edited company data.
    """
    try:
        company_service = get_company_service()
        for _, row in edited_comp.iterrows():
            company_service.update_company_active_status(row["id"], row["Active"])
        st.success("✅ Company settings saved!")

    except Exception:
        logger.exception("Save companies failed")
        st.error("❌ Save failed. Please try again.")


def _render_add_company_form() -> None:
    """Render form for adding new companies using service layer."""
    st.markdown("**Add New Company**")

    with st.form("add_company_form", clear_on_submit=True):
        new_name = st.text_input(
            "Company Name",
            placeholder="e.g., OpenAI",
            help="Enter the company name",
        )
        new_url = st.text_input(
            "Careers Page URL",
            placeholder="e.g., https://openai.com/careers",
            help="Enter the URL of the company's careers page",
        )

        if st.form_submit_button(
            "+ Add Company",
            use_container_width=True,
            type="primary",
        ):
            _handle_add_company(new_name, new_url)


def _display_salary_range(salary_range: tuple[int, int]) -> None:
    """Display selected salary range with improved formatting and high-value indicators.

    Args:
        salary_range: Tuple containing (min_salary, max_salary) values.
    """
    min_val, max_val = salary_range
    is_filtered = min_val > 0 or max_val < SALARY_UNBOUNDED_THRESHOLD
    is_high_value = max_val >= SALARY_UNBOUNDED_THRESHOLD

    if not is_filtered:
        st.caption("💰 Showing all salary ranges")
        return

    # Always show selected range (clamp to threshold for display)
    top = SALARY_UNBOUNDED_THRESHOLD if is_high_value else max_val
    st.caption(f"💰 Selected: {format_salary(min_val)} - {format_salary(top)}")

    if is_high_value:
        st.caption(f"💡 Including all positions above {format_salary(max_val)}")


def _handle_add_company(name: str, url: str) -> None:
    """Handle adding a new company using CACHED service layer.

    Args:
        name: Company name.
        url: Company careers page URL.
    """
    if not name or not url:
        st.error("Please fill in both fields")
        return

    if not url.startswith(("http://", "https://")):
        st.error("URL must start with http:// or https://")
        return

    try:
        company_service = get_company_service()
        company_service.add_company(name, url)
        st.success(f"✅ Added {name} successfully!")
        st.rerun()

    except ValueError as e:
        st.error(f"❌ {e}")
    except Exception:
        logger.exception("Add company failed")
        st.error("❌ Failed to add company. Please try again.")
