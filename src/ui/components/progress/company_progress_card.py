"""Company progress card component for real-time scraping dashboard.

This module provides a reusable progress card component that displays individual
company scraping progress with metrics and visual indicators.

Key features:
- Professional card layout with border
- Real-time progress bar
- Calculated metrics (jobs found, scraping speed)
- Status-based styling and icons
- Responsive design for grid layouts

Example usage:
    from src.ui.components.progress.company_progress_card import CompanyProgressCard

    # Create and render progress card
    card = CompanyProgressCard()
    card.render(company_progress=progress_data)
"""

import logging

from datetime import datetime, timezone

import streamlit as st

from src.ui.utils.background_helpers import CompanyProgress
from src.ui.utils.ui_helpers import (
    calculate_scraping_speed,
    format_duration,
    format_jobs_count,
    format_timestamp,
)

logger = logging.getLogger(__name__)


class CompanyProgressCard:
    """Reusable component for displaying company scraping progress.

    This component renders a professional progress card showing real-time
    scraping status, metrics, and progress indicators for individual companies.
    """

    def __init__(self):
        """Initialize the company progress card component."""
        self.status_config = {
            "Pending": {
                "emoji": "⏳",
                "color": "#6c757d",
                "bg_color": "#f8f9fa",
                "border_color": "#dee2e6",
            },
            "Scraping": {
                "emoji": "🔄",
                "color": "#007bff",
                "bg_color": "#e3f2fd",
                "border_color": "#2196f3",
            },
            "Completed": {
                "emoji": "✅",
                "color": "#28a745",
                "bg_color": "#d4edda",
                "border_color": "#28a745",
            },
            "Error": {
                "emoji": "❌",
                "color": "#dc3545",
                "bg_color": "#f8d7da",
                "border_color": "#dc3545",
            },
        }

    def render(self, company_progress: CompanyProgress) -> None:
        """Render the company progress card.

        Args:
            company_progress: CompanyProgress object with company status info.
        """
        try:
            # Get status configuration
            status_info = self.status_config.get(
                company_progress.status, self.status_config["Pending"]
            )

            # Create bordered container for the card
            with st.container(border=True):
                self._render_card_header(company_progress, status_info)
                self._render_progress_bar(company_progress)
                self._render_metrics(company_progress)
                self._render_timing_info(company_progress)

                # Show error message if present
                if company_progress.error and company_progress.status == "Error":
                    st.error(f"Error: {company_progress.error}")

        except Exception:
            logger.exception("Error rendering company progress card")
            st.error(f"Error displaying progress for {company_progress.name}")

    def _render_card_header(
        self, company_progress: CompanyProgress, status_info: dict
    ) -> None:
        """Render the card header with company name and status.

        Args:
            company_progress: Company progress data.
            status_info: Status styling configuration.
        """
        # Company name and status in columns
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(
                f"**{status_info['emoji']} {company_progress.name}**",
                help=f"Status: {company_progress.status}",
            )

        with col2:
            # Status badge
            st.markdown(
                f"""
                <div style='text-align: right; padding: 2px 8px;
                           background-color: {status_info["bg_color"]};
                           border: 1px solid {status_info["border_color"]};
                           border-radius: 12px; font-size: 12px;
                           color: {status_info["color"]};'>
                    <strong>{company_progress.status.upper()}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_progress_bar(self, company_progress: CompanyProgress) -> None:
        """Render the progress bar for the company.

        Args:
            company_progress: Company progress data.
        """
        # Calculate progress percentage
        if company_progress.status == "Completed":
            progress = 1.0
            progress_text = "Completed"
        elif company_progress.status == "Scraping":
            # For active scraping, show animated progress
            # Since we don't have granular progress data, use time-based estimation
            if company_progress.start_time:
                elapsed = datetime.now(timezone.utc) - company_progress.start_time
                # Estimate progress based on elapsed time (max 90% until completion)
                estimated_progress = min(
                    0.9, elapsed.total_seconds() / 120.0
                )  # 2 min estimate
                progress = estimated_progress
                progress_text = f"Scraping... ({int(progress * 100)}%)"
            else:
                progress = 0.1  # Show some progress for active scraping
                progress_text = "Scraping..."
        elif company_progress.status == "Error":
            progress = 0.0
            progress_text = "Failed"
        else:  # Pending
            progress = 0.0
            progress_text = "Waiting to start"

        # Render progress bar with text
        st.progress(progress, text=progress_text)

    def _render_metrics(self, company_progress: CompanyProgress) -> None:
        """Render metrics section with jobs found and scraping speed.

        Args:
            company_progress: Company progress data.
        """
        col1, col2 = st.columns(2)

        with col1:
            # Jobs found metric
            jobs_display = format_jobs_count(company_progress.jobs_found)

            # Calculate delta for jobs (would need previous value for real delta)
            st.metric(
                label="Jobs Found",
                value=company_progress.jobs_found,
                help=f"Total {jobs_display} discovered",
            )

        with col2:
            # Scraping speed metric
            speed = calculate_scraping_speed(
                company_progress.jobs_found,
                company_progress.start_time,
                company_progress.end_time,
            )

            speed_display = f"{speed} /min" if speed > 0 else "N/A"

            st.metric(label="Speed", value=speed_display, help="Jobs per minute")

    def _render_timing_info(self, company_progress: CompanyProgress) -> None:
        """Render timing information section.

        Args:
            company_progress: Company progress data.
        """
        # Create timing info display
        timing_parts = []

        if company_progress.start_time:
            start_str = format_timestamp(company_progress.start_time)
            timing_parts.append(f"Started: {start_str}")

            if company_progress.end_time:
                end_str = format_timestamp(company_progress.end_time)
                duration = company_progress.end_time - company_progress.start_time
                duration_str = format_duration(duration.total_seconds())
                timing_parts.extend(
                    (f"Completed: {end_str}", f"Duration: {duration_str}")
                )
            elif company_progress.status == "Scraping":
                elapsed = datetime.now(timezone.utc) - company_progress.start_time
                elapsed_str = format_duration(elapsed.total_seconds())
                timing_parts.append(f"Elapsed: {elapsed_str}")

        if timing_parts:
            timing_text = " | ".join(timing_parts)
            st.caption(timing_text)


def render_company_progress_card(company_progress: CompanyProgress) -> None:
    """Convenience function to render a company progress card.

    Args:
        company_progress: CompanyProgress object with company status info.

    """
    card = CompanyProgressCard()
    card.render(company_progress)
