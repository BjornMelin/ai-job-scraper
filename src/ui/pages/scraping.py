"""Scraping page component for the AI Job Scraper UI.

This module provides the scraping dashboard with real-time progress monitoring,
background task management, and user controls for starting and stopping scraping
operations.
"""

import logging

from datetime import datetime

import streamlit as st

from src.services.job_service import JobService
from src.ui.utils.background_tasks import (
    CompanyProgress,
    get_task_manager,
    is_scraping_active,
    start_background_scraping,
    stop_all_scraping,
)

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
                    Current time: {datetime.now().strftime("%H:%M:%S")}
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
    except Exception as e:
        logger.error(f"Failed to get active companies: {e}")
        active_companies = []
        st.error(
            "⚠️ Failed to load company configuration. Please check the database "
            "connection."
        )

    with col1:
        # Start Scraping button
        # Disable start button if no active companies
        start_disabled = is_scraping or len(active_companies) == 0

        if st.button(
            "🚀 Start Scraping",
            disabled=start_disabled,
            use_container_width=True,
            type="primary",
            help="Begin scraping jobs from all active company sources"
            if len(active_companies) > 0
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
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Failed to start scraping: {error_msg}")
                logger.error(f"Failed to start scraping: {error_msg}", exc_info=True)

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
                    st.info("ℹ️ No active scraping tasks found to stop")
            except Exception as e:
                st.error(f"❌ Error stopping scraping: {str(e)}")
                logger.error(f"Error stopping scraping: {e}", exc_info=True)

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
            except Exception as e:
                st.error(f"❌ Error resetting progress: {str(e)}")
                logger.error(f"Error resetting progress: {e}", exc_info=True)

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
    """Render the enhanced progress monitoring section with card-based layout."""
    from src.ui.utils.formatters import (
        format_jobs_count,
    )

    st.markdown("---")
    st.markdown("### 📊 Real-time Progress Dashboard")

    # Get progress data from session state
    task_progress = st.session_state.get("task_progress", {})

    # Create enhanced progress data with better company tracking
    progress_data = _get_enhanced_progress_data(task_progress)

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

    # Auto-refresh for real-time updates
    if not progress_data.is_complete and not progress_data.has_error:
        st.rerun()


def _get_enhanced_progress_data(task_progress: dict):
    """Create enhanced progress data with better company tracking."""

    class EnhancedProgressData:
        def __init__(self):
            if task_progress:
                latest_progress = max(task_progress.values(), key=lambda x: x.timestamp)
                self.overall_progress = latest_progress.progress
                self.current_stage = latest_progress.message or "Running..."
                self.has_error = "Error:" in self.current_stage
                self.error_message = self.current_stage if self.has_error else ""
                self.is_complete = self.overall_progress >= 100.0

                # Extract job count from message if available
                import re

                job_match = re.search(r"Found (\d+)", self.current_stage)
                self.total_jobs_found = int(job_match.group(1)) if job_match else 0

                # Create sample company data for demonstration
                # In a real implementation, this would come from session state
                self.companies = _create_sample_company_data(latest_progress)
                self.start_time = latest_progress.timestamp
            else:
                self.overall_progress = 0.0
                self.current_stage = "No active tasks"
                self.has_error = False
                self.error_message = ""
                self.is_complete = True
                self.total_jobs_found = 0
                self.companies = []
                self.start_time = None

    return EnhancedProgressData()


def _create_sample_company_data(latest_progress):
    """Create sample company progress data for demonstration."""
    import random

    from datetime import timedelta

    # Sample companies (in a real implementation, this would come from database)
    sample_companies = ["TechCorp", "InnovateCo", "StartupXYZ", "MegaCorp", "DevStudio"]
    companies = []

    base_time = latest_progress.timestamp
    progress_pct = latest_progress.progress / 100.0

    for i, company_name in enumerate(sample_companies):
        # Simulate different stages based on overall progress
        if progress_pct > (i + 1) / len(sample_companies):
            status = "Completed"
            start_time = base_time - timedelta(minutes=5 - i)
            end_time = base_time - timedelta(minutes=2 - i)
            jobs_found = random.randint(10, 50)
        elif progress_pct > i / len(sample_companies):
            status = "Scraping"
            start_time = base_time - timedelta(minutes=3 - i)
            end_time = None
            jobs_found = random.randint(5, 25)
        else:
            status = "Pending"
            start_time = None
            end_time = None
            jobs_found = 0

        companies.append(
            CompanyProgress(
                name=company_name,
                status=status,
                jobs_found=jobs_found,
                start_time=start_time,
                end_time=end_time,
            )
        )

    return companies


def _render_overall_metrics(progress_data):
    """Render overall metrics section with ETA and total jobs."""
    from src.ui.utils.formatters import calculate_eta

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
                1 for c in progress_data.companies if c.status == "Completed"
            )

            if progress_data.start_time:
                time_elapsed = (
                    datetime.now() - progress_data.start_time
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
            active_count = sum(
                1 for c in progress_data.companies if c.status == "Scraping"
            )
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
    from src.ui.components.progress.company_progress_card import (
        render_company_progress_card,
    )

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
            if i + j < len(companies):
                with cols[j]:
                    render_company_progress_card(companies[i + j])
            else:
                # Empty column for the last row if odd number of companies
                with cols[j]:
                    st.empty()

    # Summary statistics
    completed = sum(1 for c in companies if c.status == "Completed")
    active = sum(1 for c in companies if c.status == "Scraping")
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
            elapsed = datetime.now() - company_progress.start_time
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

    # Create a simple progress data object (reusing the same logic)
    class ProgressData:
        def __init__(self):
            if task_progress:
                latest_progress = max(task_progress.values(), key=lambda x: x.timestamp)
                self.overall_progress = latest_progress.progress
                self.current_stage = latest_progress.message or "Running..."
                self.has_error = False
                self.error_message = ""
                self.is_complete = self.overall_progress >= 100.0
                self.total_jobs_found = 0
            else:
                self.overall_progress = 0.0
                self.current_stage = "No active tasks"
                self.has_error = False
                self.error_message = ""
                self.is_complete = True
                self.total_jobs_found = 0

    progress_data = ProgressData()

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
        elif progress_data.start_time and not progress_data.end_time:
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
