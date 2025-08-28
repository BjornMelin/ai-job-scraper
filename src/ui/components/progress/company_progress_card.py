"""Minimal company progress card using native Streamlit components.

Simple progress visualization using st.progress(), st.status(), and st.container()
without fragments or manual session state management.

Example usage:
    render_company_progress_card(company_progress)
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.ui.utils.background_helpers import CompanyProgress


def render_company_progress_card(company_progress: "CompanyProgress") -> None:
    """Minimal company progress using native Streamlit components.

    Args:
        company_progress: CompanyProgress object with company status info.
    """
    # Status emoji mapping
    status_emoji = {
        "Pending": "⏳",
        "Scraping": "🔄",
        "Completed": "✅",
        "Error": "❌",
    }

    # Calculate progress value
    if company_progress.status == "Completed":
        progress_value = 1.0
    elif company_progress.status == "Scraping":
        # Estimate progress based on elapsed time (max 90% until completion)
        if company_progress.start_time:
            elapsed = datetime.now(UTC) - company_progress.start_time
            progress_value = min(0.9, elapsed.total_seconds() / 120.0)
        else:
            progress_value = 0.1
    else:  # Pending or Error
        progress_value = 0.0

    # Native Streamlit components only
    with st.container(border=True, key=f"progress_{company_progress.name}"):
        emoji = status_emoji.get(company_progress.status, "⚪")

        # Simple header
        st.markdown(f"**{emoji} {company_progress.name}** - {company_progress.status}")

        # Native progress bar
        st.progress(progress_value, text=f"Jobs found: {company_progress.jobs_found}")

        # Show error if present
        if company_progress.error:
            st.error(f"Error: {company_progress.error}")
