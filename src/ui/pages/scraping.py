"""Scraping page component for the AI Job Scraper UI.

This module provides the scraping dashboard with real-time progress monitoring,
background task management, and user controls for starting and stopping scraping
operations.
"""

import logging

from datetime import datetime, timezone

import streamlit as st

from src.services.job_service import JobService
from src.ui.components.progress.company_progress_card import (
    render_company_progress_card,
)
from src.ui.utils.background_helpers import (
    get_company_progress,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)
from src.ui.utils.ui_helpers import calculate_eta, is_streamlit_context

logger = logging.getLogger(__name__)


def render_scraping_page() -> None:
    """Render the complete scraping page with controls and progress display.

    This function orchestrates the rendering of the scraping dashboard including
    the header, control buttons, and real-time progress monitoring.
    """
    # Initialize session state for manual refresh control
    if "manual_refresh_count" not in st.session_state:
        st.session_state.manual_refresh_count = 0

    # Page header
    st.markdown("# 🔍 Job Scraping Dashboard")
    st.markdown(
        "Monitor and control job scraping operations with real-time progress tracking"
    )

    # Status indicators section
    _render_status_indicators()

    # Control buttons section
    _render_control_buttons()

    # Manual refresh control section
    if is_scraping_active():
        st.markdown("---")
        col_refresh, col_info = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh Progress", key="refresh_progress_btn"):
                st.session_state.manual_refresh_count += 1
                st.rerun()
        with col_info:
            st.caption("Click to update progress data manually")

    # Show progress dashboard only if scraping is active
    if is_scraping_active():
        _render_progress_dashboard()

    # Recent activity summary with manual refresh
    _render_activity_summary()

    # Show manual refresh information
    _render_manual_refresh_info()


def _render_status_indicators() -> None:
    """Render status indicators with manual refresh capability."""
    # Get current state
    is_scraping = is_scraping_active()

    # Status indicator with manual refresh timestamp
    status_text = "🟢 ACTIVE" if is_scraping else "⚪️ IDLE"
    current_time = datetime.now(timezone.utc).strftime("%H:%M:%S")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Scraping Status:** {status_text}")
    with col2:
        st.caption(f"Last check: {current_time}")
    with col3:
        if st.button(
            "🔄 Refresh Status", key="refresh_status_btn", help="Update status manually"
        ):
            st.rerun()

    # Show current status with st.status()
    current_status = st.session_state.get(
        "scraping_status", "Ready to start scraping..."
    )

    # Use st.status() for better visual progress indication
    if is_scraping:
        with st.status("Scraping in progress...", expanded=True, state="running"):
            st.write(current_status)
            # Add progress context if available
            company_progress = get_company_progress()
            if company_progress:
                completed = sum(
                    1 for c in company_progress.values() if c.status == "Completed"
                )
                total = len(company_progress)
                st.write(f"Progress: {completed}/{total} companies completed")
    else:
        with st.status("Scraping idle", expanded=False, state="complete"):
            st.write(current_status)


def _render_control_buttons() -> None:
    """Render the main control buttons: Start, Stop, and Reset."""
    st.markdown("---")
    st.markdown("### 🎛️ Scraping Controls")

    # Render status indicators with manual refresh
    _render_status_indicators()

    # Get active companies (static configuration data)
    try:
        active_companies = JobService.get_active_companies()
    except Exception:
        logger.exception("Failed to get active companies")
        active_companies = []
        st.error(
            "⚠️ Failed to load company configuration. "
            "Please check the database connection."
        )

    # Get current scraping state (refreshed manually)
    is_scraping = is_scraping_active()

    # Use horizontal flex container for control buttons
    with st.container():
        control_cols = st.columns(3, gap="large")

        with control_cols[0]:
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
                    # Use st.status for better feedback
                    with st.status("Starting scraping...", state="running"):
                        st.write(f"Monitoring {len(active_companies)} companies")
                        st.write("Progress will appear below")
                    st.balloons()  # Celebratory feedback
                    st.rerun()
                except Exception as e:
                    logger.exception("Failed to start scraping")
                    st.error(f"❌ Failed to start scraping: {e!s}")
                    st.exception(e)  # Show detailed error for debugging

        with control_cols[1]:
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
                        with st.status("Stopping scraping...", state="complete"):
                            st.write(f"{stopped_count} task(s) cancelled")
                        st.rerun()
                    else:
                        st.info("No active scraping tasks found to stop")
                except Exception:
                    logger.exception("Error stopping scraping")
                    st.error("❌ Error stopping scraping")

        with control_cols[2]:
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

                    with st.status("Resetting progress data...", state="complete"):
                        st.write(f"Cleared {cleared_count} data stores")
                        st.write("Dashboard is now ready for new scraping session")
                    st.rerun()
                except Exception:
                    logger.exception("Error resetting progress")
                    st.error("❌ Error resetting progress")

    # Show company status (static configuration data)
    st.markdown(f"**Active Companies:** {len(active_companies)} configured")
    if active_companies:
        companies_text = ", ".join(active_companies[:3])  # Show first 3
        if len(active_companies) > 3:
            companies_text += f" and {len(active_companies) - 3} more..."
        st.caption(companies_text)


def _render_progress_dashboard() -> None:
    """Render the progress dashboard with manual refresh capability."""
    if not is_scraping_active():
        return  # Don't render if not scraping

    st.markdown("---")

    # Header with manual refresh indicator
    col_header, col_timestamp, col_refresh = st.columns([3, 1, 1])
    with col_header:
        st.markdown("### 📊 Progress Dashboard")
    with col_timestamp:
        current_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
        st.caption(f"Updated: {current_time}")
    with col_refresh:
        if st.button(
            "🔄 Update", key="refresh_dashboard_btn", help="Refresh dashboard data"
        ):
            st.rerun()

    # Get progress data
    company_progress = get_company_progress()
    logger.info("Dashboard update: %d companies in progress", len(company_progress))

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
            time_elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
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
        ]
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
    """Render recent activity summary section with manual refresh capability."""
    st.markdown("---")

    # Header with manual refresh capability
    col_header, col_timestamp, col_refresh = st.columns([2, 1, 1])
    with col_header:
        st.markdown("### 📈 Recent Activity")
    with col_timestamp:
        update_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
        st.caption(f"Updated: {update_time}")
    with col_refresh:
        if st.button(
            "🔄 Refresh", key="refresh_activity_btn", help="Update activity summary"
        ):
            st.rerun()

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
        ]
    )


def _render_manual_refresh_info() -> None:
    """Render information about manual refresh functionality."""
    with st.expander("💡 About Manual Refresh", expanded=False):
        st.markdown("""
        **Manual Refresh Mode**: This dashboard updates when you click refresh
        buttons or interact with controls.

        - **🔄 Refresh Status**: Update scraping status indicator
        - **🔄 Refresh Progress**: Update progress data during active scraping
        - **🔄 Update Dashboard**: Refresh dashboard metrics and company progress
        - **🔄 Refresh Activity**: Update recent activity summary

        This approach provides better performance and reduces CPU usage
        compared to auto-refresh.
        """)


def _render_metrics(items: list[tuple[str, object, str]]) -> None:
    """Render a row of metrics using a concise helper to reduce boilerplate."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items, strict=False):
        with col:
            st.metric(label=label, value=value, help=help_text)


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    render_scraping_page()
