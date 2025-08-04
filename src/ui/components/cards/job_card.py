"""Job card component for displaying individual job postings.

This module provides the job card rendering functionality with interactive
controls for status updates, favorites, and notes. It handles the visual
display and user interactions for individual job items in the card view.
"""

import html
import logging

from datetime import datetime

import pandas as pd
import streamlit as st

from src.database import SessionLocal
from src.models import JobSQL
from src.ui.state.app_state import StateManager

logger = logging.getLogger(__name__)


def render_job_card(job_data: pd.Series, tab_key: str, page_num: int) -> None:
    """Render an individual job card with interactive controls.

    This function creates a visually appealing job card with job details,
    status badge, favorite toggle, and editable fields for status and notes.

    Args:
        job_data: Pandas Series containing job information.
        tab_key: Unique identifier for the current tab.
        page_num: Current page number for unique widget keys.
    """
    # Format posted date for display
    time_str = _format_posted_date(job_data["Posted"])

    # Determine status badge CSS class
    status_class = f"status-{job_data['Status'].lower()}"

    # Render the main card HTML
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{html.escape(str(job_data["Title"]))}</div>
            <div class="card-meta">
                <strong>{html.escape(str(job_data["Company"]))}</strong> • 
                {html.escape(str(job_data["Location"]))} • 
                {time_str}
            </div>
            <div class="card-desc">{
            html.escape(str(job_data["Description"])[:200])
        }...</div>
            <div class="card-footer">
                <span class="status-badge {status_class}">{
            html.escape(str(job_data["Status"]))
        }</span>
                {
            "<span style='color: #f59e0b; font-size: 1.2em;'>⭐</span>"
            if job_data["Favorite"]
            else ""
        }
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render interactive controls
    _render_card_controls(job_data, tab_key, page_num)


def _format_posted_date(posted_date: any) -> str:
    """Format the posted date for display.

    Args:
        posted_date: The posted date value (can be string, datetime, or None).

    Returns:
        Formatted time string (e.g., "Today", "2 days ago").
    """
    if pd.notna(posted_date):
        if isinstance(posted_date, str):
            try:
                posted_date = datetime.strptime(posted_date, "%Y-%m-%d")
            except ValueError:
                return ""

        days_ago = (datetime.now() - posted_date).days

        if days_ago == 0:
            return "Today"
        elif days_ago == 1:
            return "Yesterday"
        else:
            return f"{days_ago} days ago"

    return ""


def _render_card_controls(job_data: pd.Series, tab_key: str, page_num: int) -> None:
    """Render interactive controls for the job card.

    Args:
        job_data: Pandas Series containing job information.
        tab_key: Unique identifier for the current tab.
        page_num: Current page number for unique widget keys.
    """
    job_id = job_data["id"]

    # Apply button
    apply_link = job_data["Link"] if job_data["Link"] else "#"
    st.link_button("Apply", apply_link)

    # Toggle favorite button
    if st.button(
        "Toggle Favorite",
        key=f"fav_{job_id}_{tab_key}_{page_num}",
    ):
        _toggle_job_favorite(job_id)

    # Status selectbox
    status_options = ["New", "Interested", "Applied", "Rejected"]
    current_status_index = status_options.index(job_data["Status"])

    st.selectbox(
        "Status",
        status_options,
        index=current_status_index,
        key=f"status_{job_id}_{tab_key}_{page_num}",
        on_change=_update_job_status,
        args=(job_id, tab_key, page_num),
    )

    # Notes text area
    st.text_area(
        "Notes",
        job_data["Notes"],
        key=f"notes_{job_id}_{tab_key}_{page_num}",
        on_change=_update_job_notes,
        args=(job_id, tab_key, page_num),
    )


def _toggle_job_favorite(job_id: int) -> None:
    """Toggle the favorite status of a job.

    Args:
        job_id: Database ID of the job to toggle.
    """
    session = SessionLocal()

    try:
        job = session.query(JobSQL).filter_by(id=job_id).first()
        if job:
            job.favorite = not job.favorite
            session.commit()
            st.rerun()

    except Exception as e:
        logger.error(f"Toggle favorite failed for job {job_id}: {e}")
        st.error("Failed to update favorite status")

    finally:
        session.close()


def _update_job_status(job_id: int, tab_key: str, page_num: int) -> None:
    """Update job application status in database.

    Callback function for Streamlit status selectbox changes.

    Args:
        job_id: Database ID of the job to update.
        tab_key: Tab identifier for session state management.
        page_num: Page number for session state key.
    """
    session = SessionLocal()

    try:
        job = session.query(JobSQL).filter_by(id=job_id).first()
        if job:
            # Get the new status from session state
            status_key = f"status_{job_id}_{tab_key}_{page_num}"
            new_status = st.session_state.get(status_key)

            if new_status:
                job.application_status = new_status
                session.commit()
                st.rerun()

    except Exception as e:
        logger.error(f"Update status failed for job {job_id}: {e}")
        st.error("Failed to update job status")

    finally:
        session.close()


def _update_job_notes(job_id: int, tab_key: str, page_num: int) -> None:
    """Update job notes in database.

    Callback function for Streamlit text area changes.

    Args:
        job_id: Database ID of the job to update.
        tab_key: Tab identifier for session state management.
        page_num: Page number for session state key.
    """
    session = SessionLocal()

    try:
        job = session.query(JobSQL).filter_by(id=job_id).first()
        if job:
            # Get the new notes from session state
            notes_key = f"notes_{job_id}_{tab_key}_{page_num}"
            new_notes = st.session_state.get(notes_key, "")

            job.notes = new_notes
            session.commit()
            st.rerun()

    except Exception as e:
        logger.error(f"Update notes failed for job {job_id}: {e}")
        st.error("Failed to update job notes")

    finally:
        session.close()


def render_job_cards_grid(jobs_df: pd.DataFrame, tab_key: str) -> None:
    """Render a grid of job cards with pagination and sorting.

    Args:
        jobs_df: DataFrame containing job data to display.
        tab_key: Unique identifier for the current tab.
    """
    if jobs_df.empty:
        return

    from src.ui.state.app_state import StateManager

    state_manager = StateManager()

    # Sorting controls
    _render_sorting_controls(state_manager, tab_key)

    # Apply sorting to DataFrame
    sorted_df = _apply_sorting(jobs_df, state_manager)

    # Pagination controls
    page_num = _render_pagination_controls(sorted_df, tab_key, state_manager)

    # Get paginated data
    paginated_df = _get_paginated_data(sorted_df, page_num)

    # Render cards in grid
    _render_cards_grid(paginated_df, tab_key, page_num)


def _render_sorting_controls(state_manager: StateManager, tab_key: str) -> None:
    """Render sorting controls for the job cards.

    Args:
        state_manager: State manager for accessing sort settings.
        tab_key: Tab key for unique widget keys.
    """
    sort_options = {"Posted": "Posted", "Title": "Title", "Company": "Company"}

    col1, col2 = st.columns(2)

    with col1:
        selected_sort = st.selectbox(
            "Sort By",
            list(sort_options.values()),
            index=list(sort_options.values()).index(state_manager.sort_by),
            key=f"sort_by_{tab_key}",
        )
        state_manager.sort_by = selected_sort

    with col2:
        sort_asc = st.checkbox(
            "Ascending",
            state_manager.sort_asc,
            key=f"sort_asc_{tab_key}",
        )
        state_manager.sort_asc = sort_asc


def _apply_sorting(df: pd.DataFrame, state_manager: StateManager) -> pd.DataFrame:
    """Apply sorting to the DataFrame.

    Args:
        df: DataFrame to sort.
        state_manager: State manager with sort settings.

    Returns:
        Sorted DataFrame.
    """
    sort_options = {"Posted": "Posted", "Title": "Title", "Company": "Company"}
    sort_key = next(
        (k for k, v in sort_options.items() if v == state_manager.sort_by),
        "Posted",
    )

    return df.sort_values(by=sort_key, ascending=state_manager.sort_asc)


def _render_pagination_controls(
    df: pd.DataFrame, tab_key: str, state_manager: StateManager
) -> int:
    """Render pagination controls and return current page.

    Args:
        df: DataFrame for pagination calculation.
        tab_key: Tab key for state management.
        state_manager: State manager for page state.

    Returns:
        Current page number.
    """
    cards_per_page = 9
    total_pages = (len(df) + cards_per_page - 1) // cards_per_page

    current_page = state_manager.get_tab_page(tab_key)
    current_page = max(0, min(current_page, total_pages - 1))

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Previous Page", key=f"prev_{tab_key}") and current_page > 0:
            state_manager.set_tab_page(tab_key, current_page - 1)
            st.rerun()

    with col2:
        st.write(f"Page {current_page + 1} of {total_pages}")

    with col3:
        if (
            st.button("Next Page", key=f"next_{tab_key}")
            and current_page < total_pages - 1
        ):
            state_manager.set_tab_page(tab_key, current_page + 1)
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
