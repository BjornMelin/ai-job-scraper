"""Jobs page component for the AI Job Scraper UI.

This module provides the main jobs page functionality including job display,
filtering, search, and management features. It handles both list and card views
with tab-based organization for different job categories.
"""

import asyncio
import logging

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.schemas import Job
from src.scraper import scrape_all
from src.services.company_service import CompanyService
from src.services.job_service import JobService
from src.ui.components.sidebar import render_sidebar
from src.ui.utils.ui_helpers import is_streamlit_context

logger = logging.getLogger(__name__)


@st.dialog("Job Details", width="large")
def show_job_details_modal(job: Job) -> None:
    """Show job details in a modal dialog.

    Args:
        job: Job DTO object to display details for.
    """
    from src.ui.helpers.job_modal import (
        render_action_buttons,
        render_job_description,
        render_job_header,
        render_job_status,
        render_notes_section,
    )

    render_job_header(job)
    render_job_status(job)
    notes_value = render_notes_section(job)
    render_job_description(job)
    render_action_buttons(job, notes_value)


def _handle_job_details_modal(jobs: list[Job]) -> None:
    """Handle showing the job details modal when a job is selected.

    Args:
        jobs: List of available jobs to find the selected job.
    """
    if view_job_id := st.session_state.get("view_job_id"):
        if selected_job := next((job for job in jobs if job.id == view_job_id), None):
            show_job_details_modal(selected_job)
        else:
            # Job not found in current filtered list, clear the selection
            st.session_state.view_job_id = None


def _execute_scraping_safely() -> dict[str, int]:
    """Execute scraping with comprehensive error handling and logging.

    This function provides a robust wrapper around the scraping functionality,
    ensuring proper asyncio lifecycle management and detailed error reporting.
    It uses modern Python async patterns for reliable execution.

    Returns:
        dict[str, int]: Synchronization statistics from SmartSyncEngine containing:
            - 'inserted': Number of new jobs added to database
            - 'updated': Number of existing jobs updated
            - 'archived': Number of stale jobs archived (preserved user data)
            - 'deleted': Number of stale jobs deleted (no user data)
            - 'skipped': Number of jobs skipped (no changes detected)

    Raises:
        Exception: If scraping execution fails with detailed error context.
    """
    try:
        logger.debug("SCRAPING_EXECUTION: Starting asyncio.run(scrape_all())")

        # Use simple asyncio.run() - handles event loop lifecycle automatically
        # This is the recommended pattern for running async code from sync context
        sync_stats = asyncio.run(scrape_all())

        logger.debug(
            "SCRAPING_EXECUTION_SUCCESS: asyncio.run completed, stats type: %s",
            type(sync_stats).__name__,
        )

    except Exception:
        logger.exception("SCRAPING_EXECUTION_FAILURE: Scraping execution failed")
        # Re-raise with context preserved for upstream handling
        raise
    else:
        return sync_stats


def render_jobs_page() -> None:
    """Render the complete jobs page with all functionality.

    This function orchestrates the rendering of the jobs page including
    the header, action bar, job tabs, and statistics dashboard.
    """
    # Render sidebar for Jobs page (moved from main.py for st.navigation compatibility)
    render_sidebar()

    # Render page header
    _render_page_header()

    # Render action bar
    _render_action_bar()

    # Get filtered jobs data
    jobs = _get_filtered_jobs()

    if not jobs:
        st.info(
            "🔍 No jobs found. Try adjusting your filters or refreshing the job list."
        )
        return

    # Render job tabs
    _render_job_tabs(jobs)

    # Render statistics dashboard
    _render_statistics_dashboard(jobs)

    # Show job details modal if a job is selected
    _handle_job_details_modal(jobs)


def _render_page_header() -> None:
    """Render the page header with title and last updated time."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            """
            <h1 style='margin-bottom: 0;'>AI Job Tracker</h1>
            <p style='color: var(--text-muted); margin-top: 0;'>
                Track and manage your job applications efficiently
            </p>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 20px;'>
                <small style='color: var(--text-muted);'>
                    Last updated: {
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            }
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_action_bar() -> None:
    """Render the action bar with refresh button and status info using containers."""
    # Use horizontal flex container for better responsive layout
    with st.container():
        # Use better proportions for action bar elements
        action_cols = st.columns([3, 2, 2], gap="medium")

        with action_cols[0]:
            if st.button(
                "🔄 Refresh Jobs",
                use_container_width=True,
                type="primary",
                help="Scrape latest job postings from all active companies",
            ):
                _handle_refresh_jobs()

        with action_cols[1]:
            _render_last_refresh_status()

        with action_cols[2]:
            _render_active_sources_metric()


def _handle_refresh_jobs() -> None:
    """Handle the job refresh operation with comprehensive logging.

    This function orchestrates the job refresh workflow with detailed logging
    for monitoring, debugging, and user feedback. It follows modern Python
    logging best practices with structured log messages.
    """
    refresh_start_time = datetime.now(timezone.utc)
    logger.info(
        "REFRESH_JOBS_START: Starting job refresh workflow at %s",
        refresh_start_time.isoformat(),
    )

    # Use st.status for better progress visualization
    with st.status("🔍 Searching for new jobs...", expanded=True, state="running"):
        try:
            # Log active companies count before scraping
            try:
                active_count = CompanyService.get_active_companies_count()
                st.write(f"Found {active_count} active companies to scrape")
                logger.info(
                    "REFRESH_JOBS_SOURCES: Found %d active companies for scraping",
                    active_count,
                )
            except Exception:
                st.write("Warning: Could not determine company count")
                logger.warning(
                    "REFRESH_JOBS_SOURCES_ERROR: Failed to get active companies count"
                )

            # Execute scraping and get sync stats
            st.write("Starting job scraping...")
            logger.info("REFRESH_JOBS_SCRAPING: Starting scraping execution")
            scraping_start = datetime.now(timezone.utc)

            sync_stats = _execute_scraping_safely()
            st.write("Scraping completed successfully!")

            scraping_duration = (
                datetime.now(timezone.utc) - scraping_start
            ).total_seconds()
            logger.info(
                "REFRESH_JOBS_SCRAPING_COMPLETE: Scraping completed in %.2f seconds",
                scraping_duration,
            )

            # Update session state with timezone-aware datetime
            st.session_state.last_scrape = datetime.now(timezone.utc)

            # Defensive validation: ensure we got a dict with sync stats
            if not isinstance(sync_stats, dict):
                logger.error(
                    "REFRESH_JOBS_ERROR: Expected sync_stats dict, got %s: %s",
                    type(sync_stats).__name__,
                    sync_stats,
                )
                st.error("❌ Scrape completed but returned unexpected data format")
                return

            # Log detailed sync statistics
            inserted = sync_stats.get("inserted", 0)
            updated = sync_stats.get("updated", 0)
            archived = sync_stats.get("archived", 0)
            deleted = sync_stats.get("deleted", 0)
            skipped = sync_stats.get("skipped", 0)

            total_processed = inserted + updated
            total_duration = (
                datetime.now(timezone.utc) - refresh_start_time
            ).total_seconds()

            logger.info(
                "REFRESH_JOBS_SYNC_STATS: inserted=%d, updated=%d, archived=%d, "
                "deleted=%d, skipped=%d, total_processed=%d, duration=%.2fs",
                inserted,
                updated,
                archived,
                deleted,
                skipped,
                total_processed,
                total_duration,
            )

            # Display results using st.status for better UX
            with st.status(
                f"Refresh completed in {total_duration:.1f}s",
                expanded=True,
                state="complete",
            ):
                st.write(f"✅ Processed **{total_processed}** jobs")
                st.write(f"• Inserted: **{inserted}** new jobs")
                st.write(f"• Updated: **{updated}** existing jobs")
                st.write(f"• Archived: **{archived}** stale jobs")
                if skipped > 0:
                    st.write(f"• Skipped: **{skipped}** unchanged jobs")

            logger.info(
                "REFRESH_JOBS_COMPLETE: Job refresh workflow completed "
                "successfully in %.2fs",
                total_duration,
            )
            st.rerun()

        except Exception:
            total_duration = (
                datetime.now(timezone.utc) - refresh_start_time
            ).total_seconds()
            logger.exception(
                "REFRESH_JOBS_FAILURE: Job refresh workflow failed after %.2fs",
                total_duration,
            )
            # Use st.status for error indication
            with st.status(
                f"Scraping failed after {total_duration:.1f}s",
                expanded=True,
                state="error",
            ):
                st.write("❌ Check logs for detailed error information")
                st.write(
                    "Try again in a few moments or contact support if the issue "
                    "persists"
                )


def _render_last_refresh_status() -> None:
    """Render the last refresh status information."""
    if hasattr(st.session_state, "last_scrape") and st.session_state.last_scrape:
        time_diff = datetime.now(timezone.utc) - st.session_state.last_scrape

        if time_diff.total_seconds() < 3600:
            minutes = int(time_diff.total_seconds() / 60)
            st.info(
                f"Last refreshed: {minutes} minute{'s' if minutes != 1 else ''} ago"
            )
        else:
            hours = int(time_diff.total_seconds() / 3600)
            st.info(f"Last refreshed: {hours} hour{'s' if hours != 1 else ''} ago")
    else:
        st.info("No recent refresh")


def _render_active_sources_metric() -> None:
    """Render the active sources metric using service layer."""
    try:
        active_companies = CompanyService.get_active_companies_count()
        st.metric("Active Sources", active_companies)
    except Exception:
        logger.exception("Failed to get active sources count")
        st.metric("Active Sources", 0)


def _get_filtered_jobs() -> list[Job]:
    """Get jobs filtered by current filter settings.

    Returns:
        List of filtered Job DTO objects.
    """
    try:
        # Convert session state filters to JobService format
        filters = {
            "text_search": st.session_state.filters.get("keyword", ""),
            "company": st.session_state.filters.get("company", []),
            "application_status": [],  # We'll handle status filtering in tabs
            "date_from": st.session_state.filters.get("date_from"),
            "date_to": st.session_state.filters.get("date_to"),
            "favorites_only": False,
            "include_archived": False,
        }

        return JobService.get_filtered_jobs(filters)

    except Exception:
        logger.exception("Job query failed")
        return []


def _construct_job_filters(favorites_only: bool = False, **kwargs) -> dict:
    """Shared utility to construct job filters for database queries.

    Args:
        favorites_only (bool): Whether to filter for favorite jobs.
        **kwargs: Additional filter parameters.

    Returns:
        dict: Filter parameters for JobService.
    """
    filters = {
        "text_search": st.session_state.filters.get("keyword", ""),
        "company": st.session_state.filters.get("company", []),
        "application_status": kwargs.get("application_status", []),
        "date_from": st.session_state.filters.get("date_from"),
        "date_to": st.session_state.filters.get("date_to"),
        "favorites_only": favorites_only,
        "include_archived": False,
    }
    # Allow overriding with additional kwargs
    filters.update(kwargs)
    return filters


def _get_favorites_jobs() -> list[Job]:
    """Get favorite jobs filtered by current filter settings using database query.

    Returns:
        List of filtered favorite Job DTO objects.
    """
    try:
        # Use shared filter construction utility
        filters = _construct_job_filters(favorites_only=True)

        return JobService.get_filtered_jobs(filters)

    except Exception:
        logger.exception("Favorites job query failed")
        return []


def _get_applied_jobs() -> list[Job]:
    """Get applied jobs filtered by current filter settings using database query.

    Returns:
        List of filtered applied Job DTO objects.
    """
    try:
        # Use shared filter construction utility with applied status filter
        filters = _construct_job_filters(application_status=["Applied"])

        return JobService.get_filtered_jobs(filters)

    except Exception:
        logger.exception("Applied job query failed")
        return []


def _render_job_tabs(jobs: list[Job]) -> None:
    """Render the job tabs using native st.dataframe with built-in features.

    Args:
        jobs: List of all jobs for calculating tab counts.
    """
    # Get jobs for each tab category
    all_jobs = jobs
    favorites_jobs = _get_favorites_jobs()
    applied_jobs = _get_applied_jobs()

    # Create tabs with counts
    total_all = len(all_jobs)
    total_favorites = len(favorites_jobs)
    total_applied = len(applied_jobs)

    tab1, tab2, tab3 = st.tabs(
        [
            f"All Jobs 📋 ({total_all:,})",
            f"Favorites ⭐ ({total_favorites:,})",
            f"Applied ✅ ({total_applied:,})",
        ]
    )

    # Render each tab with native dataframe
    with tab1:
        if total_all == 0:
            st.info(
                "🔍 No jobs found. Try adjusting your filters or refreshing the "
                "job list."
            )
        else:
            _render_job_display(all_jobs, "all")

    with tab2:
        if total_favorites == 0:
            st.info(
                "💡 No favorite jobs yet. Star jobs you're interested in "
                "to see them here!"
            )
        else:
            _render_job_display(favorites_jobs, "favorites")

    with tab3:
        if total_applied == 0:
            st.info(
                "🚀 No applications yet. Update job status to 'Applied' "
                "to track them here!"
            )
        else:
            _render_job_display(applied_jobs, "applied")


def _render_job_display(jobs: list[Job], tab_key: str) -> None:
    """Render job display with view mode selection (Card or List).

    Args:
        jobs: List of jobs to display.
        tab_key: Unique key for the tab.
    """
    if not jobs:
        return

    # Apply per-tab search to jobs list
    filtered_jobs = _apply_tab_search_to_jobs(jobs, tab_key)

    # Use helper for view mode selection and rendering
    from src.ui.helpers.view_mode import apply_view_mode, select_view_mode

    view_mode, grid_columns = select_view_mode(tab_key)
    apply_view_mode(filtered_jobs, view_mode, grid_columns)


def _jobs_to_dataframe(jobs: list[Job]) -> pd.DataFrame:
    """Convert job objects to pandas DataFrame optimized for st.dataframe display.

    Args:
        jobs: List of job objects.

    Returns:
        DataFrame with job data formatted for native Streamlit display.
    """
    if not jobs:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "id": j.id,
                "Company": j.company,
                "Title": j.title,
                "Location": j.location or "Remote",
                "Posted": j.posted_date,
                "Last Seen": j.last_seen,
                "Favorite": j.favorite,
                "Status": j.application_status or "New",
                "Notes": j.notes or "",
                "Link": j.link,
                # Keep description for job details modal but hide from main view
                "Description": j.description,
            }
            for j in jobs
        ]
    )


def _apply_tab_search_to_jobs(jobs: list[Job], tab_key: str) -> list[Job]:
    """Apply per-tab search filtering to Job DTO objects.

    Args:
        jobs: List of Job DTO objects to filter.
        tab_key: Tab key for search state.

    Returns:
        Filtered list of Job DTO objects.
    """
    # Per-tab search with visual feedback
    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_key = f"search_{tab_key}"
        search_term = st.text_input(
            "🔍 Search in this tab",
            key=search_key,
            placeholder="Search by job title, description, or company...",
            help="Search is case-insensitive and searches across title, "
            "description, and company",
        )

    # Apply search filter if search term exists
    if search_term:
        search_term_lower = search_term.lower()
        filtered_jobs = [
            job
            for job in jobs
            if (
                search_term_lower in job.title.lower()
                or search_term_lower in job.description.lower()
                or search_term_lower in job.company.lower()
            )
        ]

        with search_col2:
            st.metric(
                "Results",
                len(filtered_jobs),
                delta=f"-{len(jobs) - len(filtered_jobs)}"
                if len(filtered_jobs) < len(jobs)
                else None,
            )

        return filtered_jobs

    return jobs


def _render_statistics_dashboard(jobs: list[Job]) -> None:
    """Render the statistics dashboard.

    Args:
        jobs: List of all jobs for statistics calculation.
    """
    st.markdown("---")
    st.markdown("### 📊 Dashboard")

    # Calculate statistics
    total_jobs = len(jobs)
    favorites = sum(j.favorite for j in jobs)
    applied = sum(j.application_status == "Applied" for j in jobs)
    interested = sum(j.application_status == "Interested" for j in jobs)
    new_jobs = sum(j.application_status == "New" for j in jobs)
    rejected = sum(j.application_status == "Rejected" for j in jobs)

    # Render metric cards
    _render_metric_cards(
        total_jobs=total_jobs,
        new_jobs=new_jobs,
        interested=interested,
        applied=applied,
        favorites=favorites,
        rejected=rejected,
    )

    # Render progress visualization
    if total_jobs > 0:
        _render_progress_visualization(
            total_jobs, new_jobs, interested, applied, rejected
        )


def _render_metric_cards(
    *,
    total_jobs: int,
    new_jobs: int,
    interested: int,
    applied: int,
    favorites: int,
    rejected: int,
) -> None:
    """Render the metric cards section using enhanced st.metric components.

    Args:
        total_jobs: Total number of jobs.
        new_jobs: Number of new jobs.
        interested: Number of interested jobs.
        applied: Number of applied jobs.
        favorites: Number of favorite jobs.
        rejected: Number of rejected jobs.
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Jobs", total_jobs)

    with col2:
        # Calculate percentage of new jobs
        new_percentage = (new_jobs / total_jobs * 100) if total_jobs > 0 else 0
        st.metric(
            "New Jobs",
            new_jobs,
            delta=f"{new_percentage:.1f}%" if new_percentage > 0 else None,
        )

    with col3:
        # Calculate percentage of interested jobs
        interested_percentage = (interested / total_jobs * 100) if total_jobs > 0 else 0
        st.metric(
            "Interested",
            interested,
            delta=f"{interested_percentage:.1f}%"
            if interested_percentage > 0
            else None,
            delta_color="normal",
        )

    with col4:
        # Calculate percentage of applied jobs
        applied_percentage = (applied / total_jobs * 100) if total_jobs > 0 else 0
        st.metric(
            "Applied",
            applied,
            delta=f"{applied_percentage:.1f}%" if applied_percentage > 0 else None,
            delta_color="normal",
        )

    with col5:
        # Calculate percentage of favorites
        favorites_percentage = (favorites / total_jobs * 100) if total_jobs > 0 else 0
        st.metric(
            "Favorites",
            favorites,
            delta=f"{favorites_percentage:.1f}%" if favorites_percentage > 0 else None,
            delta_color="normal",
        )

    with col6:
        # Calculate percentage of rejected jobs
        rejected_percentage = (rejected / total_jobs * 100) if total_jobs > 0 else 0
        st.metric(
            "Rejected",
            rejected,
            delta=f"{rejected_percentage:.1f}%" if rejected_percentage > 0 else None,
            delta_color="inverse",
        )


def _render_progress_visualization(
    total_jobs: int,
    new_jobs: int,
    interested: int,
    applied: int,
    rejected: int,
) -> None:
    """Render the progress visualization section.

    Args:
        total_jobs: Total number of jobs.
        new_jobs: Number of new jobs.
        interested: Number of interested jobs.
        applied: Number of applied jobs.
        rejected: Number of rejected jobs.
    """
    st.markdown("### 📈 Application Progress")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Create progress data
        progress_data = {
            "Status": ["New", "Interested", "Applied", "Rejected"],
            "Count": [new_jobs, interested, applied, rejected],
            "Percentage": [
                (new_jobs / total_jobs) * 100,
                (interested / total_jobs) * 100,
                (applied / total_jobs) * 100,
                (rejected / total_jobs) * 100,
            ],
        }

        # Display progress bars
        for status, count, pct in zip(
            progress_data["Status"],
            progress_data["Count"],
            progress_data["Percentage"],
            strict=False,
        ):
            st.markdown(f"**{status}** - {count} jobs ({pct:.1f}%)")
            st.progress(pct / 100)

    with col2:
        # Application rate metric using st.metric
        application_rate = (applied / total_jobs) * 100 if total_jobs > 0 else 0
        # Calculate delta compared to a benchmark (e.g., 20% target application rate)
        target_rate = 20.0
        rate_delta = application_rate - target_rate
        st.metric(
            "Application Rate",
            f"{application_rate:.1f}%",
            delta=f"{rate_delta:+.1f}%" if abs(rate_delta) >= 0.1 else None,
            delta_color="normal" if rate_delta >= 0 else "inverse",
        )


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    render_jobs_page()
