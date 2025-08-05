"""Scraping page component for the AI Job Scraper UI.

This module provides the scraping dashboard with real-time progress monitoring,
background task management, and user controls for starting and stopping scraping
operations.
"""

import logging
import time

from datetime import datetime

import streamlit as st

from src.ui.utils.background_tasks import CompanyProgress, StreamlitTaskManager

logger = logging.getLogger(__name__)


def render_scraping_page() -> None:
    """Render the complete scraping page with controls and progress display.

    This function orchestrates the rendering of the scraping dashboard including
    the header, control buttons, and real-time progress monitoring.
    """
    # Initialize session state
    StreamlitTaskManager.initialize_session_state()

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
    is_scraping = StreamlitTaskManager.is_scraping_active()
    active_companies = StreamlitTaskManager.get_active_companies()

    with col1:
        # Start Scraping button
        if st.button(
            "🚀 Start Scraping",
            disabled=is_scraping,
            use_container_width=True,
            type="primary",
            help="Begin scraping jobs from all active company sources",
        ):
            try:
                task_id = StreamlitTaskManager.start_background_scraping()
                st.success(f"✅ Scraping started! Task ID: {task_id}")
                st.rerun()  # Refresh to show progress section
            except Exception as e:
                st.error(f"❌ Failed to start scraping: {e}")
                logger.error(f"Failed to start scraping: {e}")

    with col2:
        # Stop Scraping button
        if st.button(
            "⏹️ Stop Scraping",
            disabled=not is_scraping,
            use_container_width=True,
            type="secondary",
            help="Stop the current scraping operation",
        ):
            if StreamlitTaskManager.stop_scraping():
                st.warning("⚠️ Scraping stopped by user")
                st.rerun()
            else:
                st.info("ℹ️ No active scraping to stop")

    with col3:
        # Reset Progress button
        if st.button(
            "🔄 Reset Progress",
            disabled=is_scraping,
            use_container_width=True,
            help="Clear progress data and reset dashboard",
        ):
            StreamlitTaskManager.reset_progress()
            st.info("✨ Progress data reset")
            st.rerun()

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
    """Render the progress monitoring section during active scraping."""
    st.markdown("---")
    st.markdown("### 📊 Real-time Progress")

    # Get progress data
    progress_data = StreamlitTaskManager.get_progress_data()

    # Overall progress bar
    st.markdown("**Overall Progress**")
    st.progress(progress_data.overall_progress / 100.0)

    # Current stage
    stage_col1, stage_col2 = st.columns([3, 1])
    with stage_col1:
        st.markdown(f"**Current Stage:** {progress_data.current_stage}")
    with stage_col2:
        st.markdown(f"**Progress:** {progress_data.overall_progress:.1f}%")

    # Error handling
    if progress_data.has_error:
        st.error(f"❌ Error: {progress_data.error_message}")

    # Success message
    if progress_data.is_complete and not progress_data.has_error:
        st.success(
            f"✅ Scraping completed! Found {progress_data.total_jobs_found} jobs"
        )

    # Company-specific progress
    if progress_data.companies:
        st.markdown("---")
        st.markdown("#### 🏢 Company Progress")

        # Create columns for company status display
        companies = list(progress_data.companies.values())

        # Display company statuses
        for company_progress in companies:
            _render_company_status(company_progress)

        # Summary statistics
        if companies:
            completed = sum(1 for c in companies if c.status == "Completed")
            total_companies = len(companies)

            st.markdown(
                f"**Company Progress:** {completed}/{total_companies} completed"
            )

    # Auto-refresh for real-time updates
    if not progress_data.is_complete and not progress_data.has_error:
        # Use a placeholder to update automatically
        time.sleep(1)
        st.rerun()


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

    progress_data = StreamlitTaskManager.get_progress_data()

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
