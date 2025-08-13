"""Job card component for displaying individual job postings.

This module provides the job card rendering functionality with interactive
controls for status updates, favorites, and notes. It handles the visual
display and user interactions for individual job items in the card view.
"""

import html
import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.schemas import Job

import pandas as pd
import streamlit as st

from src.constants import APPLICATION_STATUSES
from src.services.job_service import JobService
from src.ui.styles.styles import apply_job_grid_styles

logger = logging.getLogger(__name__)


def render_job_card(job: "Job") -> None:
    """Render an individual job card with interactive controls.

    This function creates a visually appealing job card with job details,
    status badge, favorite toggle, and view details functionality as specified
    in the requirements.

    Args:
        job: Job DTO object containing job information.
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
            status_options = APPLICATION_STATUSES
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


def _format_posted_date(posted_date: "Any") -> str:
    """Format the posted date for display with robust timezone handling.

    This function handles both timezone-naive and timezone-aware datetime objects,
    normalizing them to UTC for consistent calculation. It follows modern Python
    datetime best practices using the zoneinfo module.

    Args:
        posted_date: The posted date value (can be string, datetime, or None).

    Returns:
        Formatted time string (e.g., "Today", "2 days ago").
    """
    if pd.notna(posted_date):
        # Handle string input - parse and assume UTC if no timezone info
        if isinstance(posted_date, str):
            try:
                posted_date = datetime.strptime(posted_date, "%Y-%m-%d").replace(
                    tzinfo=UTC
                )
            except ValueError:
                logger.warning("Failed to parse posted_date string: %s", posted_date)
                return ""

        # Handle datetime objects - ensure timezone awareness
        elif isinstance(posted_date, datetime):
            if posted_date.tzinfo is None:
                # Naive datetime - assume UTC (common for jobspy data)
                posted_date = posted_date.replace(tzinfo=UTC)
                logger.debug("Converted naive datetime to UTC: %s", posted_date)
            # Already timezone-aware - use as-is

        else:
            logger.warning("Unexpected posted_date type: %s", type(posted_date))
            return ""

        # Calculate difference using timezone-aware datetimes
        try:
            now_utc = datetime.now(UTC)
            time_diff = now_utc - posted_date
            days_ago = time_diff.days

            if days_ago == 0:
                return "Today"
            if days_ago == 1:
                return "Yesterday"
            if days_ago > 0:
                return f"{days_ago} days ago"
            # Future date
            return f"In {abs(days_ago)} days"
        except Exception:
            logger.exception("Error calculating time difference for posted_date")
            return ""

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
    st.session_state.view_job_id = job_id


def render_jobs_grid(jobs: list["Job"], num_columns: int = 3) -> None:
    """Render jobs in a responsive grid layout.

    This function creates a responsive grid of job cards using st.columns
    and renders each job as a card with interactive controls.

    Args:
        jobs: List of Job DTO objects to render in grid.
        num_columns: Number of columns for the grid layout (default: 3).
    """
    if not jobs:
        st.info("No jobs to display.")
        return

    # Apply centralized CSS styles for job grid
    apply_job_grid_styles()

    # Create responsive grid using st.columns
    for i in range(0, len(jobs), num_columns):
        # Create columns with equal width and medium gap for better spacing
        cols = st.columns(num_columns, gap="medium")

        # Render jobs for this row
        row_jobs = jobs[i : i + num_columns]
        for j, job in enumerate(row_jobs):
            with cols[j], st.container():
                # Wrap each card in a container for better height management
                render_job_card(job)

        # Add spacing between rows, but only if there are more jobs
        if i + num_columns < len(jobs):
            st.markdown('<div class="job-card-grid"></div>', unsafe_allow_html=True)


def render_jobs_list(jobs: list["Job"]) -> None:
    """Render a list of job cards.

    This is the main function for rendering jobs according to T1.1 requirements.

    Args:
        jobs: List of Job DTO objects to render.
    """
    if not jobs:
        st.info("No jobs to display.")
        return

    for job in jobs:
        # Render the job card
        render_job_card(job)

        # Add some spacing between cards
        st.markdown("---")
