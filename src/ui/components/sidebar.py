"""Sidebar component for the AI Job Scraper UI.

This module provides the sidebar functionality including search filters,
view settings, and company management features. It handles user interactions
for filtering jobs and managing company configurations.
"""

import logging

import pandas as pd
import streamlit as st

from src.constants import (
    SALARY_DEFAULT_MAX,
    SALARY_SLIDER_FORMAT,
    SALARY_SLIDER_STEP,
    SALARY_UNBOUNDED_THRESHOLD,
)

# Removed direct database import - using service layer instead
from src.services.company_service import CompanyService
from src.ui.state.session_state import clear_filters
from src.ui.utils.ui_helpers import format_salary

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

        # Company filter with popover help
        col1, col2 = st.columns([4, 1])
        with col1:
            selected_companies = st.multiselect(
                "Filter by Company",
                options=companies,
                default=st.session_state.filters["company"] or None,
                placeholder="All companies",
            )
        with col2:
            with st.popover("ℹ️", help="Company filter help"):
                st.markdown(
                    "**Company Filter**\n\n"
                    "• Select one or more companies to filter jobs\n"
                    "• Leave empty to see jobs from all companies\n"
                    "• Use Ctrl+Click to select multiple companies"
                )

        # Update filters in state manager
        current_filters = st.session_state.filters.copy()
        current_filters["company"] = selected_companies
        st.session_state.filters = current_filters

        # Keyword search with popover help
        col1, col2 = st.columns([4, 1])
        with col1:
            keyword_value = st.text_input(
                "Search Keywords",
                value=st.session_state.filters["keyword"],
                placeholder="e.g., Python, Machine Learning, Remote",
            )
        with col2:
            with st.popover("ℹ️", help="Keyword search help"):
                st.markdown(
                    "**Keyword Search**\n\n"
                    "• Searches job titles and descriptions\n"
                    "• Use multiple keywords separated by commas\n"
                    "• Search is case-insensitive\n"
                    "• Examples: 'Python', 'Remote, Senior', 'ML Engineer'"
                )

        # Update keyword in filters
        current_filters = st.session_state.filters.copy()
        current_filters["keyword"] = keyword_value
        st.session_state.filters = current_filters

        # Date range with popover help
        col_header, col_help = st.columns([4, 1])
        with col_header:
            st.markdown("**Date Range**")
        with col_help:
            with st.popover("ℹ️", help="Date range filter help"):
                st.markdown(
                    "**Date Range Filter**\n\n"
                    "• Filter jobs by their posting date\n"
                    "• Leave 'From' empty to see all older jobs\n"
                    "• Leave 'To' empty to include jobs through today\n"
                    "• Use this to find recent opportunities or historical data"
                )

        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(
                "From",
                value=st.session_state.filters["date_from"],
            )
        with col2:
            date_to = st.date_input(
                "To",
                value=st.session_state.filters["date_to"],
            )

        # Update date filters using single update call
        st.session_state.filters.update(
            {
                "date_from": date_from,
                "date_to": date_to,
            }
        )

        # Salary range filter with high-value support
        current_salary_min = st.session_state.filters.get("salary_min", 0)
        current_salary_max = st.session_state.filters.get(
            "salary_max", SALARY_DEFAULT_MAX
        )

        # Salary range header with popover help
        col_header, col_help = st.columns([4, 1])
        with col_header:
            st.markdown("**Salary Range**")
        with col_help:
            with st.popover("ℹ️", help="Salary range filter help"):
                st.markdown(
                    "**Salary Range Filter**\n\n"
                    f"• Filter jobs by annual salary (USD)\n"
                    f"• Set max to {format_salary(SALARY_UNBOUNDED_THRESHOLD)}+ for high-value positions\n"
                    "• Only jobs with salary data will be shown\n"
                    "• Use this to find opportunities in your target range"
                )

        salary_range = st.slider(
            "Annual Salary Range",
            min_value=0,
            max_value=SALARY_UNBOUNDED_THRESHOLD,
            value=(current_salary_min, current_salary_max),
            step=SALARY_SLIDER_STEP,
            format=SALARY_SLIDER_FORMAT,
            label_visibility="collapsed",  # Hide label since we have header above
        )

        # Update salary filters using single update call
        st.session_state.filters.update(
            {
                "salary_min": salary_range[0],
                "salary_max": salary_range[1],
            }
        )

        # Display formatted salary range with improved formatting
        _display_salary_range(salary_range)

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
            # Show companies with individual toggles for better UX
            st.markdown("**Active Companies**")
            for _, company in comp_df.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{company['Name']}**")
                    st.caption(company["URL"])
                with col2:
                    # Use st.toggle for better UX than checkbox
                    is_active = st.toggle(
                        "Active",
                        value=company["Active"],
                        key=f"toggle_company_{company['id']}",
                        help="Enable/disable scraping for this company",
                    )
                    # Update company status if changed
                    if is_active != company["Active"]:
                        try:
                            CompanyService.update_company_active_status(
                                company["id"], is_active
                            )
                            st.success(
                                f"{'Enabled' if is_active else 'Disabled'} {company['Name']}"
                            )
                            st.rerun()
                        except Exception:
                            st.error(f"Failed to update {company['Name']}")
                with col3:
                    # Add quick link to careers page
                    st.link_button("🔗", company["URL"], help="Visit careers page")

            # Toggles save immediately, no save button needed

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
    # Header with popover help
    col_header, col_help = st.columns([4, 1])
    with col_header:
        st.markdown("**Add New Company**")
    with col_help:
        with st.popover("ℹ️", help="Add company help"):
            st.markdown(
                "**Adding Companies**\n\n"
                "• Company names must be unique\n"
                "• Use the main careers page URL\n"
                "• URL must start with http:// or https://\n"
                "• Added companies are active by default"
            )

    with st.form("add_company_form", clear_on_submit=True):
        new_name = st.text_input(
            "Company Name",
            placeholder="e.g., OpenAI",
        )
        new_url = st.text_input(
            "Careers Page URL",
            placeholder="e.g., https://openai.com/careers",
        )

        if st.form_submit_button(
            "+ Add Company", use_container_width=True, type="primary"
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
