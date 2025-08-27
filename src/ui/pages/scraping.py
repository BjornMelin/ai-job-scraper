"""Scraping page component for the AI Job Scraper UI.

This module provides the scraping dashboard with real-time progress monitoring,
background task management, and user controls for starting and stopping scraping
operations.

Key improvements from library-first-optimization branch:
- Replaced manual refresh buttons with auto-refreshing fragments
- Simplified status displays using native st.success/st.info
- Removed complex st.status usage in favor of cleaner UX
- Added throttled auto-refresh for real-time updates
"""

import logging

from datetime import UTC, datetime

import streamlit as st

from src.services.job_service import JobService
from src.ui.components.progress.company_progress_card import (
    render_company_progress_card,
)
from src.ui.utils import calculate_eta, is_streamlit_context
from src.ui.utils.background_helpers import (
    get_company_progress,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
    throttled_rerun,
)

logger = logging.getLogger(__name__)


def render_scraping_page() -> None:
    """Render the complete scraping page with controls and progress display.

    This function orchestrates the rendering of the scraping dashboard including
    the header, control buttons, and real-time progress monitoring.
    """
    # Initialize auto-refresh tracking in session state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = 0.0

    # Page header
    st.markdown("# 🔍 Job Scraping Dashboard")
    st.markdown(
        "Monitor and control job scraping operations with real-time progress tracking",
    )

    # Control buttons section
    _render_control_buttons()

    # Show progress dashboard only if scraping is active
    if is_scraping_active():
        _render_progress_dashboard()
        _render_native_progress_section()
        _handle_auto_refresh()

    # Recent activity summary
    _render_activity_summary()


def _render_control_buttons() -> None:
    """Render the main control buttons: Start, Stop, and Reset."""
    st.markdown("---")
    st.markdown("### 🎛️ Scraping Controls")

    # Get current state
    is_scraping = is_scraping_active()

    # Get active companies
    try:
        active_companies = JobService.get_active_companies()
    except Exception:
        logger.exception("Failed to get active companies")
        active_companies = []
        st.error(
            "⚠️ Failed to load company configuration. "
            "Please check the database connection.",
        )

    # Status indicator
    status_text = "🟢 ACTIVE" if is_scraping else "⚪️ IDLE"
    st.markdown(f"**Scraping Status:** {status_text}")

    # Create three columns for the main control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        # START button - calls start_background_scraping()
        start_disabled = is_scraping or not active_companies
        if st.button(
            "🚀 Start",
            disabled=start_disabled,
            use_container_width=True,
            type="primary",
            help="Begin scraping jobs from all active company sources"
            if active_companies
            else "No active companies configured",
        ):
            try:
                start_background_scraping()
                # Use native toast notification instead of st.success
                from src.ui.components.native_progress import show_progress_toast

                show_progress_toast(
                    f"🚀 Scraping initiated! Monitoring {len(active_companies)} companies.",
                    icon="🚀",
                )
                st.balloons()  # Celebratory feedback
                st.rerun()
            except Exception as e:
                logger.exception("Failed to start scraping")
                st.error(f"❌ Failed to start scraping: {e!s}")
                st.exception(e)  # Show detailed error for debugging

    with col2:
        # STOP button
        if st.button(
            "⏹️ Stop",
            disabled=not is_scraping,
            use_container_width=True,
            type="secondary",
            help="Stop the current scraping operation",
        ):
            try:
                stopped_count = stop_all_scraping()
                if stopped_count > 0:
                    st.warning(
                        f"⚠️ Scraping stopped. {stopped_count} task(s) cancelled.",
                    )
                    st.rerun()
                else:
                    st.info("No active scraping tasks found to stop")
            except Exception:
                logger.exception("Error stopping scraping")
                st.error("❌ Error stopping scraping")

    with col3:
        # RESET button
        if st.button(
            "🔄 Reset",
            disabled=is_scraping,
            use_container_width=True,
            help="Clear progress data and reset dashboard",
        ):
            try:
                # Clear progress data from session state
                progress_keys = [
                    "task_progress",
                    "company_progress",
                    "scraping_results",
                ]
                cleared_count = 0
                for key in progress_keys:
                    if key in st.session_state:
                        if hasattr(st.session_state[key], "clear"):
                            st.session_state[key].clear()
                        cleared_count += 1

                st.success(
                    f"✨ Progress data reset successfully! "
                    f"Cleared {cleared_count} data stores.",
                )
                st.rerun()
            except Exception:
                logger.exception("Error resetting progress")
                st.error("❌ Error resetting progress")

    # Show current status
    current_status = st.session_state.get(
        "scraping_status",
        "Ready to start scraping...",
    )
    if is_scraping:
        st.info(f"🔄 {current_status}")
    else:
        st.success(f"✅ {current_status}")

    # Show company status
    st.markdown(f"**Active Companies:** {len(active_companies)} configured")
    if active_companies:
        companies_text = ", ".join(active_companies[:3])  # Show first 3
        if len(active_companies) > 3:
            companies_text += f" and {len(active_companies) - 3} more..."
        st.caption(companies_text)


@st.fragment(run_every=1)  # Auto-refresh every 1 second for responsive updates
def _render_progress_dashboard() -> None:
    """Render the real-time progress dashboard with auto-updating fragments."""
    if not is_scraping_active():
        return  # Don't render if not scraping

    st.markdown("---")

    # Header with real-time indicator
    col_header, col_indicator = st.columns([4, 1])
    with col_header:
        st.markdown("### 📊 Real-time Progress Dashboard")
    with col_indicator:
        st.markdown("🔄 **Auto-updating**")

    # Get progress data with performance optimization
    company_progress = get_company_progress()

    # Early exit if no progress data to reduce CPU usage
    if not company_progress:
        st.info("⚡ Waiting for scraping tasks to start...")
        return

    logger.debug("Fragment update: %d companies in progress", len(company_progress))

    # Calculate overall metrics
    total_jobs_found = sum(company.jobs_found for company in company_progress.values())
    completed_companies = sum(
        1 for company in company_progress.values() if company.status == "Completed"
    )
    total_companies = len(company_progress)
    active_companies = sum(
        1 for company in company_progress.values() if company.status == "Scraping"
    )

    # Overall metrics using st.metric as required
    eta: str
    if total_companies > 0 and completed_companies > 0:
        # Get start time from first company or task progress
        start_time = None
        for company in company_progress.values():
            if company.start_time:
                start_time = company.start_time
                break

        if start_time:
            time_elapsed = (datetime.now(UTC) - start_time).total_seconds()
            eta = calculate_eta(total_companies, completed_companies, time_elapsed)
        else:
            eta = "Calculating..."
    else:
        eta = "N/A"

    _render_metrics(
        [
            (
                "Total Jobs Found",
                total_jobs_found,
                "Total jobs discovered across all companies",
            ),
            ("ETA", eta, "Estimated time to completion"),
            (
                "Active Companies",
                f"{active_companies}/{total_companies}",
                "Companies currently being scraped",
            ),
        ],
    )

    # Overall progress bar
    if total_companies > 0:
        progress_pct = completed_companies / total_companies
        st.progress(
            progress_pct,
            text=f"Overall Progress: {completed_companies}/{total_companies} "
            f"companies completed",
        )

    # Company progress grid using st.columns as required
    if company_progress:
        st.markdown("#### 🏢 Company Progress")
        _render_company_grid(list(company_progress.values()))


def _render_company_grid(companies: list) -> None:
    """Render company progress using responsive st.columns grid layout."""
    if not companies:
        st.info("No company progress data available")
        return

    # Use 2 columns for responsive grid
    cols_per_row = 2

    # Process companies in groups for the grid
    for i in range(0, len(companies), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")

        for j in range(cols_per_row):
            if i + j < len(companies):
                with cols[j]:
                    render_company_progress_card(companies[i + j])


def _render_activity_summary() -> None:
    """Render recent activity summary section."""
    st.markdown("---")
    st.markdown("### 📈 Recent Activity")

    # Get latest results from session state
    results = st.session_state.get("scraping_results", {})
    company_progress = get_company_progress()

    # Metrics
    last_run_jobs = sum(results.values()) if results else 0
    last_run_jobs_display = last_run_jobs if last_run_jobs > 0 else "N/A"

    # Find most recent start time
    last_run_time = "Never"
    if company_progress:
        latest_start = max(
            (c.start_time for c in company_progress.values() if c.start_time),
            default=None,
        )
        if latest_start:
            last_run_time = latest_start.strftime("%H:%M:%S")

    # Calculate duration from company progress
    duration_text = "N/A"
    if company_progress:
        completed_companies = [
            c
            for c in company_progress.values()
            if c.status == "Completed" and c.start_time and c.end_time
        ]
        if completed_companies:
            avg_duration = sum(
                (c.end_time - c.start_time).total_seconds() for c in completed_companies
            ) / len(completed_companies)
            duration_text = f"{avg_duration:.1f}s"

    _render_metrics(
        [
            ("Last Run Jobs", last_run_jobs_display, ""),
            ("Last Run Time", last_run_time, ""),
            ("Avg Duration", duration_text, ""),
        ],
    )


def _handle_auto_refresh() -> None:
    """Handle automatic page refresh while scraping is active.

    Implements throttled refresh every ~2 seconds to provide real-time updates
    without excessive refresh calls that could cause UI flicker.
    """
    try:
        throttled_rerun(should_rerun=is_scraping_active())
    except Exception:
        logger.exception("Error in auto-refresh handler")


def _render_metrics(items: list[tuple[str, object, str]]) -> None:
    """Render a row of metrics using a concise helper to reduce boilerplate."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items, strict=False):
        with col:
            st.metric(label=label, value=value, help=help_text)


@st.fragment(run_every=2)  # Auto-refresh native progress every 2 seconds
def _render_native_progress_section() -> None:
    """Render native progress tracking section using st.status() and st.progress()."""
    # Check for active workflows
    if "native_progress" in st.session_state and st.session_state.native_progress:
        st.markdown("---")
        st.markdown("### 🔄 Live Progress Tracking (Native)")

        # Show progress for each active workflow
        for workflow_id, progress_data in st.session_state.native_progress.items():
            if progress_data.get("is_active", True):
                percentage = progress_data.get("current_percentage", 0.0)
                message = progress_data.get("current_message", "Processing...")
                phase = progress_data.get("current_phase", "processing")

                # Create a native progress display
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Workflow:** {workflow_id[:8]}...")
                        st.progress(
                            percentage / 100.0, text=f"{message} ({percentage:.1f}%)"
                        )

                    with col2:
                        # Status indicator
                        if percentage >= 100.0:
                            st.success("✅ Complete")
                        elif "error" in phase.lower():
                            st.error("❌ Error")
                        else:
                            st.info(f"🔄 {phase.title()}")

                # Show ETA if available
                start_time = progress_data.get("start_time")
                if start_time and percentage > 0:
                    elapsed = (datetime.now(UTC) - start_time).total_seconds()
                    if elapsed > 0:
                        estimated_total = elapsed / (percentage / 100.0)
                        remaining = max(0, estimated_total - elapsed)
                        st.caption(f"⏱️ ETA: {remaining:.1f}s remaining")

    else:
        # No active workflows - show placeholder
        with st.container():
            st.info(
                "🔍 No active workflows to display. Start a scraping operation to see live progress!"
            )


def _show_native_completion_toast() -> None:
    """Show native completion toast when workflow finishes."""
    # Dynamic import to avoid circular imports
    from src.ui.components.native_progress import show_progress_toast

    show_progress_toast("🎉 Scraping completed successfully!", icon="🎉")


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    render_scraping_page()
