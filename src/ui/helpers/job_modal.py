"""Job details modal UI helpers.

Focused helper functions for rendering job modal components.
"""

import streamlit as st

from src.schemas import Job


def render_job_header(job: Job) -> None:
    """Render job modal header with title and company info."""
    st.markdown(f"### {job.title}")
    st.markdown(f"**{job.company}** • {job.location}")


def render_job_status(job: Job) -> None:
    """Render job status and posted date."""
    col1, col2 = st.columns(2)
    with col1:
        if job.posted_date:
            st.markdown(f"**Posted:** {job.posted_date}")
    with col2:
        status_colors = {
            "New": "🔵",
            "Interested": "🟡",
            "Applied": "🟢",
            "Rejected": "🔴",
        }
        icon = status_colors.get(job.application_status, "⚪")
        st.markdown(f"**Status:** {icon} {job.application_status}")


def render_job_description(job: Job) -> None:
    """Render job description section."""
    st.markdown("---")
    st.markdown("### Job Description")
    st.markdown(job.description)


def render_notes_section(job: Job) -> str:
    """Render notes section and return the notes value.

    Returns:
        str: Current notes value from text area.
    """
    st.markdown("---")
    st.markdown("### Notes")
    return st.text_area(
        "Your notes about this position",
        value=job.notes or "",
        key=f"modal_notes_{job.id}",
        help="Add your personal notes about this job",
        height=150,
    )


def render_action_buttons(job: Job, notes_value: str) -> None:
    """Render modal action buttons."""
    from src.ui.pages.jobs import _save_job_notes

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Save Notes", type="primary", use_container_width=True):
            _save_job_notes(job.id, notes_value)

    with col2:
        if job.link:
            st.link_button(
                "Apply Now", job.link, use_container_width=True, type="secondary"
            )

    with col3:
        if st.button("Close", use_container_width=True):
            st.session_state.view_job_id = None
            st.rerun()
