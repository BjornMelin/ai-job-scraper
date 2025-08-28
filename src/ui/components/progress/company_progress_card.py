"""Fragment-based company progress card for real-time scraping dashboard.

This module provides reusable progress components with st.fragment() architecture for:
- Fragment-isolated progress updates without full page reruns
- Real-time background task monitoring
- Auto-refreshing progress bars and metrics
- Fragment-scoped error handling and recovery

Key features:
- Fragment-based card layout with isolated updates
- Real-time progress bars with auto-refresh (2s intervals)
- Live calculated metrics (jobs found, scraping speed)
- Fragment-scoped status-based styling and icons
- Responsive design optimized for fragment performance
- Background task coordination with fragment isolation

Example usage:
    # Fragment-based progress (auto-refreshing)
    render_company_progress_fragment(progress_data, enable_auto_refresh=True)

    # Traditional progress card (backward compatibility)
    card = CompanyProgressCard()
    card.render(company_progress=progress_data)

This implementation follows Stream C fragment architecture specifications for
background task monitoring and coordination simplification.
"""

import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import streamlit as st

from src.ui.utils import (
    calculate_scraping_speed,
    format_duration,
    format_jobs_count,
    format_timestamp,
)

if TYPE_CHECKING:
    from src.ui.utils.background_helpers import CompanyProgress

logger = logging.getLogger(__name__)


class CompanyProgressCard:
    """Reusable component for displaying company scraping progress.

    This component renders a progress card showing real-time scraping status,
    metrics, and progress indicators for individual companies.
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

    def render(self, company_progress: "CompanyProgress") -> None:
        """Render the company progress card.

        Args:
            company_progress: CompanyProgress object with company status info.
        """
        try:
            # Get status configuration
            status_info = self.status_config.get(
                company_progress.status,
                self.status_config["Pending"],
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
        self,
        company_progress: "CompanyProgress",
        status_info: dict,
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

    def _render_progress_bar(self, company_progress: "CompanyProgress") -> None:
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
                elapsed = datetime.now(UTC) - company_progress.start_time
                # Estimate progress based on elapsed time (max 90% until completion)
                estimated_progress = min(
                    0.9,
                    elapsed.total_seconds() / 120.0,
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

    def _render_metrics(self, company_progress: "CompanyProgress") -> None:
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

    def _render_timing_info(self, company_progress: "CompanyProgress") -> None:
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
                    (f"Completed: {end_str}", f"Duration: {duration_str}"),
                )
            elif company_progress.status == "Scraping":
                elapsed = datetime.now(UTC) - company_progress.start_time
                elapsed_str = format_duration(elapsed.total_seconds())
                timing_parts.append(f"Elapsed: {elapsed_str}")

        if timing_parts:
            timing_text = " | ".join(timing_parts)
            st.caption(timing_text)


def render_company_progress_card(company_progress: "CompanyProgress") -> None:
    """Convenience function to render a company progress card.

    Args:
        company_progress: CompanyProgress object with company status info.

    """
    card = CompanyProgressCard()
    card.render(company_progress)


# ========== FRAGMENT-BASED PROGRESS COMPONENTS ==========
# Enhanced progress monitoring with st.fragment() architecture for Stream C


@st.fragment(run_every="1s")
def render_company_progress_fragment(
    company_progress: "CompanyProgress", enable_auto_refresh: bool = False
) -> None:
    """Fragment-based company progress card with real-time updates.

    This fragment isolates progress updates to prevent full page reruns while
    providing real-time background task monitoring.

    Args:
        company_progress: CompanyProgress object with company status info.
        enable_auto_refresh: Whether to enable auto-refresh for real-time updates.
    """
    try:
        # Performance optimization: early exit for inactive states
        if not company_progress:
            st.info("⚡ No progress data available")
            return

        # Skip updates for completed tasks unless auto-refresh is enabled
        if company_progress.status == "Completed" and not enable_auto_refresh:
            # Render static completed state to save CPU
            _render_completed_static_card(company_progress)
            return

        # Get latest progress data if auto-refresh is enabled
        if enable_auto_refresh:
            # Refresh progress data from session state or background helpers
            try:
                from src.ui.utils.background_helpers import get_company_progress

                latest_progress = get_company_progress()

                # Update company progress if we have newer data
                if company_progress.company_name in latest_progress:
                    company_progress = latest_progress[company_progress.company_name]
            except Exception as e:
                logger.warning("Failed to refresh progress data: %s", e)

        # Use bordered container with unique key for fragment isolation
        with st.container(
            border=True,
            key=f"company_progress_fragment_{company_progress.company_name}",
        ):
            # Real-time company header with live status
            _render_fragment_company_header(company_progress)

            # Live progress bar with auto-updating percentage
            _render_fragment_progress_bar(company_progress)

            # Real-time metrics with live updates
            _render_fragment_metrics(company_progress)

            # Live timing information
            _render_fragment_timing(company_progress)

    except Exception as e:
        logger.exception(
            "Fragment error for company %s",
            getattr(company_progress, "company_name", "unknown"),
        )
        st.error(f"⚠️ Progress update error: {str(e)[:100]}...")


@st.fragment(run_every="2s")
def render_scraping_overview_fragment() -> None:
    """Fragment for real-time scraping overview and system status.

    This fragment provides a live overview of all background tasks and
    system status without triggering full page reruns.
    """
    try:
        from src.ui.utils.background_helpers import (
            get_company_progress,
            is_scraping_active,
        )

        # Get real-time system status
        is_active = is_scraping_active()
        company_progress = get_company_progress()

        with st.container():
            col_header, col_status, col_indicator = st.columns([3, 2, 1])

            with col_header:
                st.markdown("### 🎯 System Overview")
            with col_status:
                status_color = "🟢" if is_active else "⚪"
                st.markdown(
                    f"**Status:** {status_color} {'ACTIVE' if is_active else 'IDLE'}"
                )
            with col_indicator:
                st.caption("🔄 Live")

            if is_active and company_progress:
                # Live overall progress metrics
                total_companies = len(company_progress)
                active_companies = sum(
                    1 for cp in company_progress.values() if cp.status == "Scraping"
                )
                completed_companies = sum(
                    1 for cp in company_progress.values() if cp.status == "Completed"
                )
                total_jobs = sum(cp.jobs_found for cp in company_progress.values())

                # Real-time metrics display
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Jobs", f"{total_jobs:,}")
                col2.metric("Companies", f"{completed_companies}/{total_companies}")
                col3.metric("Active", active_companies)

                # Live progress percentage
                progress_pct = (
                    completed_companies / total_companies if total_companies > 0 else 0
                )
                col4.metric("Progress", f"{progress_pct:.0%}")

                # System-wide progress bar
                st.progress(
                    progress_pct,
                    text=f"Overall: {completed_companies}/{total_companies} completed",
                )

            else:
                st.info("🔄 No active scraping operations")

        # Show last update timestamp
        st.caption(f"Last updated: {datetime.now(UTC).strftime('%H:%M:%S')}")

    except Exception as e:
        logger.exception("Error in scraping overview fragment")
        st.error(f"⚠️ System overview temporarily unavailable: {str(e)[:100]}...")


@st.fragment(run_every="3s")
def render_task_notifications_fragment() -> None:
    """Fragment for real-time task notifications and alerts.

    This fragment monitors background tasks and displays notifications
    for important events without disrupting the main UI.
    """
    try:
        from src.ui.utils.background_helpers import get_company_progress

        # Check for task status changes since last fragment run
        if "last_task_notification_check" not in st.session_state:
            st.session_state.last_task_notification_check = datetime.now(UTC)
            st.session_state.notified_completions = set()
            return

        company_progress = get_company_progress()
        current_time = datetime.now(UTC)

        # Check for newly completed tasks
        new_completions = []
        for company_name, progress in company_progress.items():
            if (
                progress.status == "Completed"
                and company_name not in st.session_state.notified_completions
            ):
                new_completions.append(progress)
                st.session_state.notified_completions.add(company_name)

        # Display toast notifications for completions
        for progress in new_completions:
            st.toast(
                f"✅ {progress.company_name} completed! "
                f"Found {progress.jobs_found} jobs",
                icon="✅",
            )

        # Check for errors
        error_tasks = [
            progress
            for progress in company_progress.values()
            if progress.status == "Error"
        ]

        if error_tasks:
            with st.container():
                st.error(
                    f"⚠️ {len(error_tasks)} task(s) failed. Check logs for details.",
                    icon="⚠️",
                )

        # Update last check time
        st.session_state.last_task_notification_check = current_time

    except Exception:
        logger.exception("Error in task notifications fragment")
        # Silently fail to avoid disrupting notifications


def _render_fragment_company_header(company_progress: "CompanyProgress") -> None:
    """Render company header with live status for fragments."""
    status_config = {
        "Completed": {"icon": "✅", "color": "green"},
        "Scraping": {"icon": "🔄", "color": "blue"},
        "Error": {"icon": "❌", "color": "red"},
        "Pending": {"icon": "⏳", "color": "gray"},
    }

    config = status_config.get(company_progress.status, status_config["Pending"])

    # Live header with real-time status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {config['icon']} {company_progress.company_name}")
    with col2:
        st.markdown(f"**{company_progress.status}**")


def _render_fragment_progress_bar(company_progress: "CompanyProgress") -> None:
    """Render live progress bar for fragments."""
    # Calculate real-time progress
    if company_progress.status == "Completed":
        progress = 1.0
        progress_text = "✅ Completed"
    elif company_progress.status == "Scraping":
        if company_progress.start_time:
            elapsed = datetime.now(UTC) - company_progress.start_time
            # Real-time progress estimation
            estimated_progress = min(0.95, elapsed.total_seconds() / 120.0)
            progress = estimated_progress
            progress_text = f"🔄 Scraping... ({int(progress * 100)}%)"
        else:
            progress = 0.1
            progress_text = "🔄 Starting..."
    elif company_progress.status == "Error":
        progress = 0.0
        progress_text = "❌ Failed"
    else:
        progress = 0.0
        progress_text = "⏳ Pending"

    # Live progress bar
    st.progress(progress, text=progress_text)


def _render_fragment_metrics(company_progress: "CompanyProgress") -> None:
    """Render live metrics for fragments."""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jobs Found", company_progress.jobs_found)

    with col2:
        # Live scraping speed calculation
        if (
            company_progress.status == "Scraping"
            and company_progress.start_time
            and company_progress.jobs_found > 0
        ):
            elapsed = (datetime.now(UTC) - company_progress.start_time).total_seconds()
            speed = company_progress.jobs_found / (elapsed / 60) if elapsed > 0 else 0
            st.metric("Speed", f"{speed:.1f}/min")
        else:
            st.metric("Speed", "N/A")


def _render_fragment_timing(company_progress: "CompanyProgress") -> None:
    """Render live timing information for fragments."""


def _render_completed_static_card(company_progress: "CompanyProgress") -> None:
    """Render optimized static card for completed tasks to save CPU.

    Args:
        company_progress: Completed company progress data.
    """
    with st.container(
        border=True,
        key=f"completed_card_{company_progress.company_name}",
    ):
        # Static completed header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### ✅ {company_progress.company_name}")
        with col2:
            st.markdown("**Completed**")

        # Static progress bar
        st.progress(1.0, text="✅ Completed")

        # Static metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jobs Found", company_progress.jobs_found)
        with col2:
            if company_progress.start_time and company_progress.end_time:
                duration = (
                    company_progress.end_time - company_progress.start_time
                ).total_seconds()
                speed = (
                    company_progress.jobs_found / (duration / 60) if duration > 0 else 0
                )
                st.metric("Speed", f"{speed:.1f}/min")
            else:
                st.metric("Speed", "N/A")

        # Static timing
        if company_progress.start_time and company_progress.end_time:
            duration = (
                company_progress.end_time - company_progress.start_time
            ).total_seconds()
            st.caption(f"⏱️ Duration: {duration:.0f}s")
