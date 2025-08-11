"""Sidebar component for the AI Job Scraper UI.

This module provides the sidebar functionality including search filters,
view settings, and company management features. It handles user interactions
for filtering jobs and managing company configurations.
"""

import logging

import pandas as pd
import streamlit as st


# Removed direct database import - using service layer instead
from src.services.company_service import CompanyService
from src.ui.state.session_state import clear_filters

logger = logging.getLogger(__name__)


def render_sidebar() -> None:
    """Render the complete sidebar with all sections.

    This function orchestrates the rendering of all sidebar components including
    search filters, view settings, and company management. It manages the
    application state and handles user interactions within the sidebar.
    """
    with st.sidebar:
        _render_search_filters()
        st.divider()
        _render_view_settings()
        st.divider()
        _render_company_management()


def _render_search_filters() -> None:
    """Render the search and filter section of the sidebar."""
    st.markdown("### 🔍 Search & Filter")

    with st.container():
        # Get company list from database
        companies = _get_company_list()

        # Company filter with better default
        selected_companies = st.multiselect(
            "Filter by Company",
            options=companies,
            default=st.session_state.filters["company"] or None,
            placeholder="All companies",
            help="Select one or more companies to filter jobs",
        )

        # Update filters in state manager
        current_filters = st.session_state.filters.copy()
        current_filters["company"] = selected_companies
        st.session_state.filters = current_filters

        # Keyword search with placeholder
        keyword_value = st.text_input(
            "Search Keywords",
            value=st.session_state.filters["keyword"],
            placeholder="e.g., Python, Machine Learning, Remote",
            help="Search in job titles and descriptions",
        )

        # Update keyword in filters
        current_filters = st.session_state.filters.copy()
        current_filters["keyword"] = keyword_value
        st.session_state.filters = current_filters

        # Date range with column layout
        st.markdown("**Date Range**")
        col1, col2 = st.columns(2)

        with col1:
            date_from = st.date_input(
                "From",
                value=st.session_state.filters["date_from"],
                help="Show jobs posted after this date",
            )

        with col2:
            date_to = st.date_input(
                "To",
                value=st.session_state.filters["date_to"],
                help="Show jobs posted before this date",
            )

        # Update date filters
        current_filters = st.session_state.filters.copy()
        current_filters["date_from"] = date_from
        current_filters["date_to"] = date_to
        st.session_state.filters = current_filters

        # Salary range filter
        st.markdown("**Salary Range**")
        current_salary_min = st.session_state.filters.get("salary_min", 0)
        current_salary_max = st.session_state.filters.get("salary_max", 300000)

        salary_range = st.slider(
            "Annual Salary Range",
            min_value=0,
            max_value=300000,
            value=(current_salary_min, current_salary_max),
            step=10000,
            format="$%dk",
            help="Filter jobs by annual salary range (in USD)",
        )

        # Update salary filters
        current_filters = st.session_state.filters.copy()
        current_filters["salary_min"] = salary_range[0]
        current_filters["salary_max"] = salary_range[1]
        st.session_state.filters = current_filters

        # Display formatted salary range
        if salary_range[0] > 0 or salary_range[1] < 300000:
            st.caption(f"Selected range: ${salary_range[0]:,} - ${salary_range[1]:,}")

        # Clear filters button
        if st.button("Clear All Filters", use_container_width=True):
            clear_filters()
            st.rerun()


def _render_view_settings() -> None:
    """Render the view settings section of the sidebar."""
    st.markdown("### 👁️ View Settings")

    view_col1, view_col2 = st.columns(2)

    with view_col1:
        if st.button(
            "📋 List View",
            use_container_width=True,
            type="secondary" if st.session_state.view_mode == "Card" else "primary",
        ):
            st.session_state.view_mode = "List"
            st.rerun()

    with view_col2:
        if st.button(
            "🎴 Card View",
            use_container_width=True,
            type="secondary" if st.session_state.view_mode == "List" else "primary",
        ):
            st.session_state.view_mode = "Card"
            st.rerun()


def _render_company_management() -> None:
    """Render the company management section of the sidebar.

    This section allows users to view, edit, and add companies for job scraping.
    It includes functionality for toggling company active status and adding new
    companies.
    """
    with st.expander("🏢 Manage Companies", expanded=False):
        # Get companies from service layer instead of direct DB access
        companies_data = CompanyService.get_companies_for_management()
        comp_df = pd.DataFrame(companies_data)

        if not comp_df.empty:
            st.markdown("**Existing Companies**")
            edited_comp = st.data_editor(
                comp_df,
                column_config={
                    "Active": st.column_config.CheckboxColumn(
                        "Active", help="Toggle to enable/disable scraping"
                    ),
                    "URL": st.column_config.LinkColumn(
                        "URL", help="Company careers page URL"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

            if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                _save_company_changes(edited_comp)

        # Add new company section
        _render_add_company_form()


def _get_company_list() -> list[str]:
    """Get list of unique company names using service layer.

    Returns:
        List of company names sorted alphabetically.
    """
    try:
        companies = CompanyService.get_all_companies()
        return [company.name for company in companies]

    except Exception:
        logger.exception("Failed to get company list")
        return []


def _save_company_changes(edited_comp: pd.DataFrame) -> None:
    """Save changes to company settings using service layer.

    Args:
        edited_comp: DataFrame containing edited company data.
    """
    try:
        for _, row in edited_comp.iterrows():
            CompanyService.update_company_active_status(row["id"], row["Active"])
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
            "+ Add Company", use_container_width=True, type="primary"
        ):
            _handle_add_company(new_name, new_url)


def _handle_add_company(name: str, url: str) -> None:
    """Handle adding a new company using service layer.

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
        CompanyService.add_company(name, url)
        st.success(f"✅ Added {name} successfully!")
        st.rerun()

    except ValueError as e:
        st.error(f"❌ {e}")
    except Exception:
        logger.exception("Add company failed")
        st.error("❌ Failed to add company. Please try again.")
