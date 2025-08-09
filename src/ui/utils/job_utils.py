"""Utility functions for job-related UI operations."""

import logging

import streamlit as st

from src.services.job_service import JobService

logger = logging.getLogger(__name__)


def save_job_notes(job_id: int, notes: str) -> None:
    """Save job notes and show feedback.

    Args:
        job_id: Database ID of the job to update notes for.
        notes: New notes content to save.
    """
    try:
        JobService.update_notes(job_id, notes)
        logger.info("Updated notes for job %s", job_id)
        st.success("Notes saved successfully!")
        # Don't rerun here to avoid closing the modal
    except Exception:
        logger.exception("Failed to update notes")
        st.error("Failed to update notes")
