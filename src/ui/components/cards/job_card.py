"""Job card component for displaying individual job postings.

This module provides the job card rendering functionality with interactive
controls for status updates, favorites, and notes. It handles the visual
display and user interactions for individual job items in the card view.
"""

import html
import logging

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.models import JobSQL
from src.services.job_service import JobService
from src.ui.state.session_state import init_session_state

logger = logging.getLogger(__name__)


def render_job_card(job: JobSQL) -> None:
    """Render an individual job card with interactive controls.

    This function creates a visually appealing job card with job details,
    status badge, favorite toggle, and view details functionality as specified
    in the requirements.

    Args:
        job: JobSQL object containing job information.
    """
    # Use st.container with border as required
    with st.container(border=True):
        # Format posted date for display
        time_str = _format_posted_date(job.posted_date)

        # Job title and company
        st.markdown(f"### {html.escape(job.title)}")
        st.markdown(
            f"**{html.escape(job.company)}** • {html.escape(job.location)} • {time_str}"
        )

        # Job description preview
        description_preview = (
            job.description[:200] + "..."
            if len(job.description) > 200
            else job.description
        )
        st.markdown(description_preview)

        # Status badge and favorite indicator
        col1, col2 = st.columns([2, 1])
        with col1:
            status_class = f"status-{job.application_status.lower()}"
            status_html = (
                f'<span class="status-badge {status_class}">'
                f"{html.escape(job.application_status)}</span>"
            )
            st.markdown(status_html, unsafe_allow_html=True)
        with col2:
            if job.favorite:
                st.markdown("⭐")

        # Interactive controls row
        col1, col2, col3 = st.columns(3)

        with col1:
            # Status selectbox with on_change callback
            status_options = ["New", "Interested", "Applied", "Rejected"]
            current_index = (
                status_options.index(job.application_status)
                if job.application_status in status_options
                else 0
            )

            st.selectbox(
                "Status",
                status_options,
                index=current_index,
                key=f"status_{job.id}",
                on_change=_handle_status_change,
                args=(job.id,),
            )

        with col2:
            # Favorite toggle button with heart icons
            favorite_icon = "❤️" if job.favorite else "🤍"
            if st.button(
                favorite_icon,
                key=f"favorite_{job.id}",
                help="Toggle favorite",
                on_click=_handle_favorite_toggle,
                args=(job.id,),
            ):
                pass  # onClick is handled by the on_click parameter

        with col3:
            # View Details button
            if st.button(
                "View Details",
                key=f"details_{job.id}",
                on_click=_handle_view_details,
                args=(job.id,),
            ):
                pass  # onClick is handled by the on_click parameter


def _format_posted_date(posted_date: Any) -> str:
    """Format the posted date for display.

    Args:
        posted_date: The posted date value (can be string, datetime, or None).

    Returns:
        Formatted time string (e.g., "Today", "2 days ago").
    """
    if pd.notna(posted_date):
        if isinstance(posted_date, str):
            try:
                posted_date = datetime.strptime(posted_date, "%Y-%m-%d").replace(
                    tzinfo=datetime.timezone.utc
                )
            except ValueError:
                return ""

        days_ago = (datetime.now(datetime.timezone.utc) - posted_date).days

        if days_ago == 0:
            return "Today"
        if days_ago == 1:
            return "Yesterday"
        return f"{days_ago} days ago"

    return ""


def _handle_status_change(job_id: int) -> None:
    """Handle status change callback.

    Args:
        job_id: Database ID of the job to update.
    """
    try:
        if new_status := st.session_state.get(f"status_{job_id}"):
            JobService.update_job_status(job_id, new_status)
            st.rerun()
    except Exception:
        logger.exception("Failed to update job status")
        st.error("Failed to update job status")


def _handle_favorite_toggle(job_id: int) -> None:
    """Handle favorite toggle callback.

    Args:
        job_id: Database ID of the job to toggle.
    """
    try:
        JobService.toggle_favorite(job_id)
        st.rerun()
    except Exception:
        logger.exception("Failed to toggle favorite")
        st.error("Failed to toggle favorite")


def _handle_view_details(job_id: int) -> None:
    """Handle view details button click.

    Args:
        job_id: Database ID of the job to view details for.
    """
    st.session_state.expanded_job_id = job_id


def render_job_details_expander(job: JobSQL) -> None:
    """Render job details expander if this job is selected.

    This function should be called after render_job_card to check if the
    job details should be expanded based on session state.

    Args:
        job: JobSQL object to potentially show details for.
    """
    if st.session_state.get("expanded_job_id") == job.id:
        with st.expander("Details", expanded=True):
            # Display job description
            st.markdown("**Job Description:**")
            st.markdown(job.description)

            # Notes text area with save button to prevent excessive database writes
            notes_key = f"notes_{job.id}"
            notes_value = st.text_area(
                "Notes",
                value=job.notes or "",
                key=notes_key,
                help="Add your personal notes about this job",
            )

            # Save button to update notes only when explicitly requested
            if st.button("Save Notes", key=f"save_notes_{job.id}"):
                _handle_notes_save(job.id, notes_value)


def _handle_notes_save(job_id: int, notes: str) -> None:
    """Handle notes save button click.

    This function updates notes only when the save button is clicked,
    preventing excessive database writes on every keystroke.

    Args:
        job_id: Database ID of the job to update notes for.
        notes: New notes content to save.
    """
    try:
        JobService.update_notes(job_id, notes)
        logger.info("Updated notes for job %s", job_id)
        st.success("Notes saved successfully!")
        st.rerun()
    except Exception:
        logger.exception("Failed to update notes")
        st.error("Failed to update notes")


# Legacy function for backward compatibility with existing grid rendering
def _render_card_controls(job_data: pd.Series, tab_key: str, page_num: int) -> None:
    """Legacy function for backward compatibility with existing grid rendering."""
    # This function is kept for compatibility with the existing grid rendering system


def render_jobs_list(jobs: list[JobSQL]) -> None:
    """Render a list of job cards with details expanders.

    This is the main function for rendering jobs according to T1.1 requirements.

    Args:
        jobs: List of JobSQL objects to render.
    """
    if not jobs:
        st.info("No jobs to display.")
        return

    for job in jobs:
        # Render the job card
        render_job_card(job)

        # Render the details expander if this job is selected
        render_job_details_expander(job)

        # Add some spacing between cards
        st.markdown("---")


def render_job_cards_grid(jobs_df: pd.DataFrame, tab_key: str) -> None:
    """Render a grid of job cards with pagination and sorting.

    Args:
        jobs_df: DataFrame containing job data to display.
        tab_key: Unique identifier for the current tab.
    """
    if jobs_df.empty:
        return

    init_session_state()

    # Sorting controls
    _render_sorting_controls(tab_key)

    # Apply sorting to DataFrame
    sorted_df = _apply_sorting(jobs_df)

    # Pagination controls
    page_num = _render_pagination_controls(sorted_df, tab_key)

    # Get paginated data
    paginated_df = _get_paginated_data(sorted_df, page_num)

    # Render cards in grid
    _render_cards_grid(paginated_df, tab_key, page_num)


def _render_sorting_controls(tab_key: str) -> None:
    """Render sorting controls for the job cards.

    Args:
        tab_key: Tab key for unique widget keys.
    """
    sort_options = {"Posted": "Posted", "Title": "Title", "Company": "Company"}

    col1, col2 = st.columns(2)

    with col1:
        selected_sort = st.selectbox(
            "Sort By",
            list(sort_options.values()),
            index=list(sort_options.values()).index(st.session_state.sort_by),
            key=f"sort_by_{tab_key}",
        )
        st.session_state.sort_by = selected_sort

    with col2:
        sort_asc = st.checkbox(
            "Ascending",
            st.session_state.sort_asc,
            key=f"sort_asc_{tab_key}",
        )
        st.session_state.sort_asc = sort_asc


def _apply_sorting(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sorting to the DataFrame.

    Args:
        df: DataFrame to sort.

    Returns:
        Sorted DataFrame.
    """
    sort_options = {"Posted": "Posted", "Title": "Title", "Company": "Company"}
    sort_key = next(
        (k for k, v in sort_options.items() if v == st.session_state.sort_by),
        "Posted",
    )

    return df.sort_values(by=sort_key, ascending=st.session_state.sort_asc)


def _render_pagination_controls(df: pd.DataFrame, tab_key: str) -> int:
    """Render pagination controls and return current page.

    Args:
        df: DataFrame for pagination calculation.
        tab_key: Tab key for state management.

    Returns:
        Current page number.
    """
    cards_per_page = 9
    total_pages = (len(df) + cards_per_page - 1) // cards_per_page

    page_key = f"{tab_key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    current_page = st.session_state[page_key]
    current_page = max(0, min(current_page, total_pages - 1))

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Previous Page", key=f"prev_{tab_key}") and current_page > 0:
            st.session_state[page_key] = current_page - 1
            st.rerun()

    with col2:
        st.write(f"Page {current_page + 1} of {total_pages}")

    with col3:
        if (
            st.button("Next Page", key=f"next_{tab_key}")
            and current_page < total_pages - 1
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    return current_page


def _get_paginated_data(df: pd.DataFrame, page_num: int) -> pd.DataFrame:
    """Get paginated subset of DataFrame.

    Args:
        df: Full DataFrame.
        page_num: Current page number.

    Returns:
        Paginated DataFrame subset.
    """
    cards_per_page = 9
    start = page_num * cards_per_page
    end = start + cards_per_page
    return df.iloc[start:end]


def _render_cards_grid(df: pd.DataFrame, tab_key: str, page_num: int) -> None:
    """Render the actual grid of job cards.

    Args:
        df: DataFrame with job data to render.
        tab_key: Tab key for widget keys.
        page_num: Page number for widget keys.
    """
    num_cols = 3
    cols = st.columns(num_cols)

    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % num_cols]:
            render_job_card(row, tab_key, page_num)
