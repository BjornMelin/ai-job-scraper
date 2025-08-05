"""Scraping page component for the AI Job Scraper UI.

This module provides the scraping dashboard with real-time progress monitoring,
background task management, and user controls for starting and stopping scraping
operations.
"""

import logging
import re
import time

from datetime import datetime, timezone

import streamlit as st

from src.services.job_service import JobService
from src.ui.components.progress.company_progress_card import (
    render_company_progress_card,
)
from src.ui.utils.background_tasks import (
    CompanyProgress,
    get_task_manager,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)
from src.ui.utils.formatters import calculate_eta, format_jobs_count

logger = logging.getLogger(__name__)


def render_scraping_page() -> None:
    """Render the complete scraping page with controls and progress display.

    This function orchestrates the rendering of the scraping dashboard including
    the header, control buttons, and real-time progress monitoring.
    """
    # Initialize session state - handled automatically by get_task_manager()
    get_task_manager()

    # Render page header
    _render_page_header()

    # Render control buttons
    _render_control_section()

    # Render progress section if scraping is active
    if st.session_state.get("scraping_active", False):
        _render_progress_section()

    # Render recent results section
    _render_recent_results_section()


def _render_page_header() -> None:
    """Render the page header with title and description."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            """
            <h1 style='margin-bottom: 0;'>🔍 Job Scraping Dashboard</h1>
            <p style='color: var(--text-muted); margin-top: 0;'>
                Monitor and control job scraping operations with real-time progress
                tracking
            </p>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        # Display current time
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 20px;'>
                <small style='color: var(--text-muted);'>
                    Current time: {datetime.now(timezone.utc).strftime("%H:%M:%S")}
                </small>
            </div>
        """,
            unsafe_allow_html=True,
        )


def _render_control_section() -> None:
    """Render the control section with start/stop buttons and status."""
    st.markdown("---")
    st.markdown("### 🎛️ Scraping Controls")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    # Get current scraping status
    is_scraping = is_scraping_active()

    # Get active companies from database
    try:
        active_companies = JobService.get_active_companies()
    except Exception:
        logger.exception("Failed to get active companies")
        active_companies = []
        st.error(
            "⚠️ Failed to load company configuration. Please check the database "
            "connection."
        )

    with col1:
        # Start Scraping button
        # Disable start button if no active companies
        start_disabled = is_scraping or not active_companies

        if st.button(
            "🚀 Start Scraping",
            disabled=start_disabled,
            use_container_width=True,
            type="primary",
            help="Begin scraping jobs from all active company sources"
            if active_companies
            else "No active companies configured",
        ):
            try:
                task_id = start_background_scraping()
                company_count = len(active_companies)
                st.success(
                    f"✅ Scraping started successfully! Monitoring {company_count} "
                    f"active companies. Task ID: {task_id[:8]}..."
                )
                st.rerun()  # Refresh to show progress section
            except Exception:
                logger.exception("Failed to start scraping")
                st.error("❌ Failed to start scraping")

    with col2:
        # Stop Scraping button
        if st.button(
            "⏹️ Stop Scraping",
            disabled=not is_scraping,
            use_container_width=True,
            type="secondary",
            help="Stop the current scraping operation",
        ):
            try:
                stopped_count = stop_all_scraping()
                if stopped_count > 0:
                    st.warning(
                        f"⚠️ Scraping stopped by user. {stopped_count} task(s) "
                        "cancelled."
                    )
                    st.rerun()
                else:
                    st.info("ℹ️ No active scraping tasks found to stop")  # noqa: RUF001
            except Exception:
                logger.exception("Error stopping scraping")
                st.error("❌ Error stopping scraping")

    with col3:
        # Reset Progress button
        if st.button(
            "🔄 Reset Progress",
            disabled=is_scraping,
            use_container_width=True,
            help="Clear progress data and reset dashboard",
        ):
            try:
                # Clear progress data from session state
                progress_count = 0
                if "task_progress" in st.session_state:
                    progress_count = len(st.session_state.task_progress)
                    st.session_state.task_progress.clear()

                # Clear background tasks data
                if "background_tasks" in st.session_state:
                    st.session_state.background_tasks.clear()

                st.success(
                    f"✨ Progress data reset successfully! Cleared {progress_count} "
                    "task records."
                )
                st.rerun()
            except Exception:
                logger.exception("Error resetting progress")
                st.error("❌ Error resetting progress")

    with col4:
        # Status indicator
        if is_scraping:
            st.markdown(
                """
                <div style='text-align: center; padding: 10px;
                           background-color: #d4edda; border-radius: 5px;
                           border: 1px solid #c3e6cb;'>
                    <strong style='color: #155724;'>🟢 ACTIVE</strong>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='text-align: center; padding: 10px;
                           background-color: #f8f9fa; border-radius: 5px;
                           border: 1px solid #dee2e6;'>
                    <strong style='color: #6c757d;'>⚪ IDLE</strong>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Display active companies info
    st.markdown(f"**Active Companies:** {len(active_companies)} sources configured")
    if active_companies:
        companies_text = ", ".join(active_companies[:5])  # Show first 5
        if len(active_companies) > 5:
            companies_text += f" and {len(active_companies) - 5} more..."
        st.markdown(f"*{companies_text}*")
    else:
        st.warning(
            "⚠️ No active companies configured. Please configure companies in "
            "the database."
        )


def _render_progress_section() -> None:
    """Render the real-time progress monitoring section with actual company data."""
    st.markdown("---")
    st.markdown("### 📊 Real-time Progress Dashboard")

    # Get progress data from session state
    task_progress = st.session_state.get("task_progress", {})

    # Import function here to avoid import issues
    from src.ui.utils.background_tasks import get_company_progress

    company_progress = get_company_progress()

    # Create real progress data with actual company tracking
    progress_data = _get_real_progress_data(task_progress, company_progress)

    # Render overall metrics at the top
    _render_overall_metrics(progress_data)

    # Overall progress bar
    st.markdown("**Overall Progress**")
    st.progress(
        progress_data.overall_progress / 100.0, text=progress_data.current_stage
    )

    # Error handling
    if progress_data.has_error:
        st.error(f"❌ Error: {progress_data.error_message}")

    # Success message
    if progress_data.is_complete and not progress_data.has_error:
        total_jobs_str = format_jobs_count(progress_data.total_jobs_found)
        st.success(f"✅ Scraping completed! Found {total_jobs_str}")

    # Enhanced company-specific progress with card grid
    _render_company_progress_grid(progress_data)

    # Auto-refresh for real-time updates with throttling to prevent excessive reruns
    if not progress_data.is_complete and not progress_data.has_error:
        # Only rerun if enough time has passed since last update (~2 seconds)
        current_time = time.time()
        last_rerun_time = st.session_state.get("last_rerun_time", 0)

        if current_time - last_rerun_time >= 2.0:  # 2 second throttle
            st.session_state.last_rerun_time = current_time
            st.rerun()


class RealProgressData:
    """Real progress data using actual company tracking."""

    def __init__(self, task_progress: dict, company_progress: dict):
        # Get overall progress from task progress if available
        if task_progress:
            latest_progress = max(task_progress.values(), key=lambda x: x.timestamp)
            self.overall_progress = latest_progress.progress
            self.current_stage = latest_progress.message or "Running..."
            self.has_error = "Error:" in self.current_stage
            self.error_message = self.current_stage if self.has_error else ""
            self.is_complete = self.overall_progress >= 100.0
            self.start_time = latest_progress.timestamp

            # Extract job count from message if available
            self.total_jobs_found = (
                int(job_match[1])
                if (job_match := re.search(r"Found (\d+)", self.current_stage))
                else 0
            )
        else:
            self.overall_progress = 0.0
            self.current_stage = "No active tasks"
            self.has_error = False
            self.error_message = ""
            self.is_complete = True
            self.total_jobs_found = 0
            self.start_time = None

        # Use real company progress data
        self.companies = list(company_progress.values()) if company_progress else []

        # Calculate total jobs from real company data if available
        if self.companies:
            self.total_jobs_found = sum(
                company.jobs_found for company in self.companies
            )


def _get_real_progress_data(task_progress: dict, company_progress: dict):
    """Create real progress data with actual company tracking."""
    return RealProgressData(task_progress, company_progress)


def _render_overall_metrics(progress_data):
    """Render overall metrics section with ETA and total jobs."""
    col1, col2, col3 = st.columns(3)

    with col1:
        # Total Jobs Found metric
        st.metric(
            label="Total Jobs Found",
            value=progress_data.total_jobs_found,
            help="Total jobs discovered across all companies",
        )

    with col2:
        # Calculate and display ETA
        if hasattr(progress_data, "companies") and progress_data.companies:
            total_companies = len(progress_data.companies)
            completed_companies = sum(
                c.status == "Completed" for c in progress_data.companies
            )

            if progress_data.start_time:
                time_elapsed = (
                    datetime.now(timezone.utc) - progress_data.start_time
                ).total_seconds()
                eta = calculate_eta(total_companies, completed_companies, time_elapsed)
            else:
                eta = "Calculating..."
        else:
            eta = "N/A"

        st.metric(label="ETA", value=eta, help="Estimated time to completion")

    with col3:
        # Active Companies
        if hasattr(progress_data, "companies"):
            active_count = sum(c.status == "Scraping" for c in progress_data.companies)
            total_count = len(progress_data.companies)
            companies_text = f"{active_count}/{total_count}"
        else:
            companies_text = "0/0"

        st.metric(
            label="Active Companies",
            value=companies_text,
            help="Companies currently being scraped",
        )


def _render_company_progress_grid(progress_data):
    """Render company progress using responsive card grid layout."""
    if not hasattr(progress_data, "companies") or not progress_data.companies:
        st.info("No company progress data available")
        return

    st.markdown("---")
    st.markdown("#### 🏢 Company Progress")

    companies = progress_data.companies

    # Create responsive grid layout - 2 columns on most screens, 1 on mobile
    # Use 2 columns for better utilization of horizontal space
    cols_per_row = 2

    # Process companies in groups for the grid
    for i in range(0, len(companies), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")

        for j in range(cols_per_row):
            with cols[j]:
                if i + j < len(companies):
                    render_company_progress_card(companies[i + j])
                else:
                    # Empty column for the last row if odd number of companies
                    st.empty()

    # Summary statistics
    completed = sum(c.status == "Completed" for c in companies)
    active = sum(c.status == "Scraping" for c in companies)
    total_companies = len(companies)

    st.markdown("---")
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Completed", completed, help="Companies finished scraping")
    with summary_col2:
        st.metric("Active", active, help="Companies currently scraping")
    with summary_col3:
        completion_pct = (
            round((completed / total_companies) * 100, 1) if total_companies > 0 else 0
        )
        st.metric(
            "Completion", f"{completion_pct}%", help="Overall completion percentage"
        )


def _render_company_status(company_progress: CompanyProgress) -> None:
    """Render status information for a single company.

    Args:
        company_progress: CompanyProgress object with company status info.
    """
    # Status emoji mapping
    status_emoji = {"Pending": "⏳", "Scraping": "🔄", "Completed": "✅", "Error": "❌"}

    # Get timing information
    timing_info = ""
    if company_progress.start_time:
        if company_progress.end_time:
            duration = company_progress.end_time - company_progress.start_time
            timing_info = f" ({duration.total_seconds():.1f}s)"
        else:
            elapsed = datetime.now(timezone.utc) - company_progress.start_time
            timing_info = f" ({elapsed.total_seconds():.1f}s elapsed)"

    # Construct status text
    emoji = status_emoji.get(company_progress.status, "❓")
    status_text = f"{emoji} {company_progress.name}: {company_progress.status}"

    if company_progress.jobs_found > 0:
        status_text += f" - {company_progress.jobs_found} jobs found"

    status_text += timing_info

    # Display with appropriate styling
    if company_progress.status == "Error":
        st.text(status_text)
        if company_progress.error:
            st.caption(f"   Error: {company_progress.error}")
    else:
        st.text(status_text)


def _render_recent_results_section() -> None:
    """Render section showing recent scraping results and statistics."""
    st.markdown("---")
    st.markdown("### 📈 Recent Activity")

    # Get progress data from session state
    task_progress = st.session_state.get("task_progress", {})

    # Import and get company progress data
    from src.ui.utils.background_tasks import get_company_progress

    company_progress = get_company_progress()

    # Create real progress data object
    progress_data = _get_real_progress_data(task_progress, company_progress)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Last Run Jobs",
            progress_data.total_jobs_found
            if progress_data.total_jobs_found > 0
            else "N/A",
            delta=None,
        )

    with col2:
        if progress_data.start_time:
            last_run = progress_data.start_time.strftime("%H:%M:%S")
        else:
            last_run = "Never"
        st.metric("Last Run Time", last_run)

    with col3:
        if progress_data.start_time and progress_data.end_time:
            duration = progress_data.end_time - progress_data.start_time
            duration_text = f"{duration.total_seconds():.1f}s"
        elif progress_data.start_time:
            duration_text = "Running..."
        else:
            duration_text = "N/A"
        st.metric("Duration", duration_text)

    # Quick tips
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info(
        """
        - **Start Scraping** to begin collecting jobs from all active company sources
        - **Real-time progress** shows the current status for each company being scraped
        - **Stop Scraping** to halt the operation at any time (may take a moment to \
          respond)
        - **Reset Progress** to clear the dashboard and start fresh
        """
    )

    # Debug information (only in development)
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.markdown("---")
        st.markdown("### 🔧 Debug Information")
        st.json(
            {
                "scraping_active": st.session_state.get("scraping_active", False),
                "task_id": st.session_state.get("task_id"),
                "progress_data": {
                    "overall_progress": progress_data.overall_progress,
                    "current_stage": progress_data.current_stage,
                    "total_jobs_found": progress_data.total_jobs_found,
                    "is_complete": progress_data.is_complete,
                    "has_error": progress_data.has_error,
                    "companies_count": len(progress_data.companies)
                    if progress_data.companies
                    else 0,
                },
            }
        )


# Execute page when loaded by st.navigation()
render_scraping_page()
