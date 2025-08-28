"""Simplified job card using native Streamlit components only."""

import html

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import Job

import streamlit as st

from src.constants import APPLICATION_STATUSES
from src.services.job_service import JobService


def render_job_card(job: "Job") -> None:
    """Render job card with interactive controls."""
    with st.container(border=True):
        # Simple date formatting
        try:
            if isinstance(job.posted_date, str):
                days_ago = (
                    datetime.now(UTC)
                    - datetime.strptime(job.posted_date, "%Y-%m-%d").replace(tzinfo=UTC)
                ).days
            elif isinstance(job.posted_date, datetime):
                pd = (
                    job.posted_date.replace(tzinfo=UTC)
                    if job.posted_date.tzinfo is None
                    else job.posted_date
                )
                days_ago = (datetime.now(UTC) - pd).days
            else:
                days_ago = -1
            time_str = (
                "Today"
                if days_ago == 0
                else "Yesterday"
                if days_ago == 1
                else f"{days_ago} days ago"
                if days_ago > 0
                else ""
            )
        except Exception:
            time_str = ""

        st.markdown(f"### {html.escape(job.title)}")
        st.markdown(
            f"**{html.escape(job.company)}** • {html.escape(job.location)} • {time_str}"
        )
        st.markdown(
            job.description[:200] + "..."
            if len(job.description) > 200
            else job.description
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            status = html.escape(job.application_status)
            st.markdown(
                f'<span class="status-badge">{status}</span>', unsafe_allow_html=True
            )
        with col2:
            if job.favorite:
                st.markdown("⭐")

        col1, col2, col3 = st.columns(3)
        with col1:
            idx = (
                APPLICATION_STATUSES.index(job.application_status)
                if job.application_status in APPLICATION_STATUSES
                else 0
            )
            new_status = st.selectbox(
                "Status", APPLICATION_STATUSES, index=idx, key=f"status_{job.id}"
            )
            if new_status != job.application_status:
                try:
                    JobService.update_job_status(job.id, new_status)
                    st.rerun()
                except Exception:
                    st.error("Failed to update status")
        with col2:
            if st.button("❤️" if job.favorite else "🤍", key=f"favorite_{job.id}"):
                try:
                    JobService.toggle_favorite(job.id)
                    st.rerun()
                except Exception:
                    st.error("Failed to toggle favorite")
        with col3:
            if st.button("View Details", key=f"details_{job.id}"):
                st.session_state.view_job_id = job.id


def render_jobs_grid(jobs: list["Job"], num_columns: int = 3) -> None:
    """Render jobs in grid layout."""
    if not jobs:
        st.info("No jobs to display.")
        return
    for i in range(0, len(jobs), num_columns):
        cols = st.columns(num_columns)
        for j, job in enumerate(jobs[i : i + num_columns]):
            with cols[j]:
                render_job_card(job)
