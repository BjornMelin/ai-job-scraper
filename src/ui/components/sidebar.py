"""Sidebar component for the AI Job Scraper UI.

This module provides the sidebar functionality including search filters,
view settings, and company management features. It handles user interactions
for filtering jobs and managing company configurations.
"""

import logging

import pandas as pd
import streamlit as st

from sqlalchemy.orm import Session
from sqlmodel import select
from src.database import db_session
from src.models import CompanySQL
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
    with (
        st.expander("🏢 Manage Companies", expanded=False),
        db_session() as session,
    ):
        # Create DataFrame of existing companies using modern SQLModel syntax
        companies = session.exec(select(CompanySQL)).all()
        comp_df = pd.DataFrame(
            [
                {"id": c.id, "Name": c.name, "URL": c.url, "Active": c.active}
                for c in companies
            ]
        )

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
                _save_company_changes(session, edited_comp)

        # Add new company section
        _render_add_company_form(session)


def _get_company_list() -> list[str]:
    """Get list of unique company names from database.

    Returns:
        List of company names sorted alphabetically.
    """
    try:
        with db_session() as session:
            # Get unique company names using modern SQLModel syntax
            companies = session.exec(
                select(CompanySQL.name).distinct().order_by(CompanySQL.name)
            ).all()
            return list(companies)

    except Exception:
        logger.exception("Failed to get company list")
        return []


def _save_company_changes(session: Session, edited_comp: pd.DataFrame) -> None:
    """Save changes to company settings.

    Args:
        session: Database session.
        edited_comp: DataFrame containing edited company data.
    """
    try:
        for _, row in edited_comp.iterrows():
            comp = session.exec(select(CompanySQL).filter_by(id=row["id"])).first()
            if comp:
                comp.active = row["Active"]
        # Note: session.commit() handled by db_session context manager
        st.success("✅ Company settings saved!")

    except Exception:
        logger.exception("Save companies failed")
        st.error("❌ Save failed. Please try again.")


def _render_add_company_form(session: Session) -> None:
    """Render form for adding new companies.

    Args:
        session: Database session for adding new companies.
    """
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
            _handle_add_company(session, new_name, new_url)


def _handle_add_company(session: Session, name: str, url: str) -> None:
    """Handle adding a new company to the database.

    Args:
        session: Database session.
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
        session.add(CompanySQL(name=name, url=url, active=True))
        # Note: session.commit() handled by db_session context manager
        st.success(f"✅ Added {name} successfully!")
        st.rerun()

    except Exception:
        logger.exception("Add company failed")
        st.error("❌ Failed to add company. Name might already exist.")
