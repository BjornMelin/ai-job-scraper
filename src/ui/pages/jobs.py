"""Jobs page component for the AI Job Scraper UI.

This module provides the main jobs page functionality including job display,
filtering, search, and management features. It handles both list and card views
with tab-based organization for different job categories.
"""

import asyncio
import logging

from datetime import datetime

import pandas as pd
import streamlit as st

from src.database import SessionLocal
from src.models import CompanySQL, JobSQL
from src.scraper import scrape_all, update_db
from src.ui.components.cards.job_card import render_job_cards_grid
from src.ui.state.app_state import StateManager

logger = logging.getLogger(__name__)


def _run_async_scraping_task() -> str:
    """Create and manage async scraping task properly.

    Returns:
        Task ID for tracking the scraping operation.
    """
    task_id = f"scraping_{datetime.now().timestamp()}"

    # Initialize task tracking in session state
    if "active_tasks" not in st.session_state:
        st.session_state.active_tasks = {}

    return task_id


def _execute_scraping_safely():
    """Execute scraping with proper event loop management.

    Returns:
        DataFrame with scraped job data.
    """
    # Proper event loop handling for Streamlit (2025 pattern)
    try:
        loop = asyncio.get_running_loop()
        logger.info("Using existing event loop")
    except RuntimeError:
        # No event loop running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new event loop")

    try:
        # Run the async scraping function
        return loop.run_until_complete(scrape_all())
    except Exception as e:
        logger.error(f"Scraping execution failed: {e}")
        raise
    finally:
        # Clean up if we created the loop
        if not loop.is_running():
            try:
                loop.close()
            except Exception as cleanup_error:
                logger.warning(f"Loop cleanup warning: {cleanup_error}")


def render_jobs_page() -> None:
    """Render the complete jobs page with all functionality.

    This function orchestrates the rendering of the jobs page including
    the header, action bar, job tabs, and statistics dashboard.
    """
    state_manager = StateManager()

    # Render page header
    _render_page_header()

    # Render action bar
    _render_action_bar(state_manager)

    # Get filtered jobs data
    jobs = _get_filtered_jobs(state_manager)

    if not jobs:
        st.info(
            "🔍 No jobs found. Try adjusting your filters or refreshing the job list."
        )
        return

    # Render job tabs
    _render_job_tabs(jobs, state_manager)

    # Render statistics dashboard
    _render_statistics_dashboard(jobs)


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
                    Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                </small>
            </div>
        """,
            unsafe_allow_html=True,
        )


def _render_action_bar(state_manager: StateManager) -> None:
    """Render the action bar with refresh button and status info.

    Args:
        state_manager: State manager for accessing last scrape time.
    """
    main_container = st.container()

    with main_container:
        action_col1, action_col2, action_col3 = st.columns([2, 2, 1])

        with action_col1:
            if st.button(
                "🔄 Refresh Jobs",
                use_container_width=True,
                type="primary",
                help="Scrape latest job postings from all active companies",
            ):
                _handle_refresh_jobs(state_manager)

        with action_col2:
            _render_last_refresh_status(state_manager)

        with action_col3:
            _render_active_sources_metric()


def _handle_refresh_jobs(state_manager: StateManager) -> None:
    """Handle the job refresh operation.

    Args:
        state_manager: State manager to update last scrape time.
    """
    with st.spinner("🔍 Searching for new jobs..."):
        try:
            # Proper async task management for Streamlit
            task_id = _run_async_scraping_task()

            # Store task info in session state for tracking
            if "scraping_task" not in st.session_state:
                st.session_state.scraping_task = None

            st.session_state.scraping_task = task_id

            # Execute scraping with proper event loop handling
            jobs_df = _execute_scraping_safely()
            update_db(jobs_df)
            state_manager.last_scrape = datetime.now()

            st.success(f"✅ Success! Found {len(jobs_df)} jobs from active companies.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Scrape failed: {e!s}")
            logger.error(f"UI scrape failed: {e}")


def _render_last_refresh_status(state_manager: StateManager) -> None:
    """Render the last refresh status information.

    Args:
        state_manager: State manager for accessing last scrape time.
    """
    if state_manager.last_scrape:
        time_diff = datetime.now() - state_manager.last_scrape

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
    """Render the active sources metric."""
    session = SessionLocal()

    try:
        active_companies = session.query(CompanySQL).filter_by(active=True).count()
        st.metric("Active Sources", active_companies)
    finally:
        session.close()


def _get_filtered_jobs(state_manager: StateManager) -> list[JobSQL]:
    """Get jobs filtered by current filter settings.

    Args:
        state_manager: State manager with current filter settings.

    Returns:
        List of filtered job objects.
    """
    session = SessionLocal()

    try:
        query = session.query(JobSQL)

        # Apply company filter
        if (
            "All" not in state_manager.filters["company"]
            and state_manager.filters["company"]
        ):
            company_ids = (
                session.query(CompanySQL.id)
                .filter(CompanySQL.name.in_(state_manager.filters["company"]))
                .subquery()
            )
            query = query.filter(JobSQL.company_id.in_(company_ids))

        # Apply keyword filter
        if state_manager.filters["keyword"]:
            query = query.filter(
                JobSQL.title.ilike(f"%{state_manager.filters['keyword']}%")
            )

        # Apply date filters
        if state_manager.filters["date_from"]:
            query = query.filter(
                JobSQL.posted_date
                >= datetime.combine(
                    state_manager.filters["date_from"], datetime.min.time()
                )
            )

        if state_manager.filters["date_to"]:
            query = query.filter(
                JobSQL.posted_date
                <= datetime.combine(
                    state_manager.filters["date_to"], datetime.max.time()
                )
            )

        # Load all jobs with their company relationships
        all_jobs = query.all()

        # Ensure company relationships are loaded
        for job in all_jobs:
            if job.company_id and not job.company_relation:
                job.company_relation = (
                    session.query(CompanySQL).filter_by(id=job.company_id).first()
                )

        return all_jobs

    except Exception as e:
        logger.error(f"Job query failed: {e}")
        return []

    finally:
        session.close()


def _render_job_tabs(jobs: list[JobSQL], state_manager: StateManager) -> None:
    """Render the job tabs with filtered content.

    Args:
        jobs: List of all jobs to organize into tabs.
        state_manager: State manager for view mode settings.
    """
    # Calculate tab counts
    favorites_count = sum(1 for j in jobs if j.favorite)
    applied_count = sum(1 for j in jobs if j.status == "Applied")

    # Create tabs with counts
    tab1, tab2, tab3 = st.tabs(
        [
            f"All Jobs 📋 ({len(jobs)})",
            f"Favorites ⭐ ({favorites_count})",
            f"Applied ✅ ({applied_count})",
        ]
    )

    # Render each tab
    with tab1:
        _render_job_display(jobs, "all", state_manager)

    with tab2:
        favorites = [j for j in jobs if j.favorite]
        if not favorites:
            st.info(
                "💡 No favorite jobs yet. Star jobs you're interested in "
                "to see them here!"
            )
        else:
            _render_job_display(favorites, "favorites", state_manager)

    with tab3:
        applied = [j for j in jobs if j.status == "Applied"]
        if not applied:
            st.info(
                "🚀 No applications yet. Update job status to 'Applied' "
                "to track them here!"
            )
        else:
            _render_job_display(applied, "applied", state_manager)


def _render_job_display(
    jobs: list[JobSQL], tab_key: str, state_manager: StateManager
) -> None:
    """Render job display for a specific tab.

    Args:
        jobs: List of jobs to display.
        tab_key: Unique key for the tab.
        state_manager: State manager for view mode settings.
    """
    if not jobs:
        return

    # Convert jobs to DataFrame
    df = _jobs_to_dataframe(jobs)

    # Apply per-tab search
    df = _apply_tab_search(df, tab_key)

    # Render based on view mode
    if state_manager.view_mode == "List":
        _render_list_view(df, tab_key)
    else:
        render_job_cards_grid(df, tab_key)


def _jobs_to_dataframe(jobs: list[JobSQL]) -> pd.DataFrame:
    """Convert job objects to pandas DataFrame.

    Args:
        jobs: List of job objects.

    Returns:
        DataFrame with job data.
    """
    return pd.DataFrame(
        [
            {
                "id": j.id,
                "Company": j.company,
                "Title": j.title,
                "Location": j.location,
                "Posted": j.posted_date,
                "Last Seen": j.last_seen,
                "Favorite": j.favorite,
                "Status": j.status,
                "Notes": j.notes,
                "Link": j.link,
                "Description": j.description,
            }
            for j in jobs
        ]
    )


def _apply_tab_search(df: pd.DataFrame, tab_key: str) -> pd.DataFrame:
    """Apply per-tab search filtering.

    Args:
        df: DataFrame to filter.
        tab_key: Tab key for search state.

    Returns:
        Filtered DataFrame.
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
        filtered_df = df[
            df["Title"].str.contains(search_term, case=False, na=False)
            | df["Description"].str.contains(search_term, case=False, na=False)
            | df["Company"].str.contains(search_term, case=False, na=False)
        ]

        with search_col2:
            st.metric(
                "Results",
                len(filtered_df),
                delta=f"-{len(df) - len(filtered_df)}"
                if len(filtered_df) < len(df)
                else None,
            )

        return filtered_df

    return df


def _render_list_view(df: pd.DataFrame, tab_key: str) -> None:
    """Render the list view for jobs.

    Args:
        df: DataFrame with job data.
        tab_key: Tab key for unique widget keys.
    """
    edited_df = st.data_editor(
        df.drop(columns=["Description"]),
        column_config={
            "Link": st.column_config.LinkColumn("Link", display_text="Apply"),
            "Favorite": st.column_config.CheckboxColumn("Favorite ⭐"),
            "Status": st.column_config.SelectboxColumn(
                "Status 🔄", options=["New", "Interested", "Applied", "Rejected"]
            ),
            "Notes": st.column_config.TextColumn("Notes 📝"),
        },
        hide_index=False,
        use_container_width=True,
    )

    # Save changes button
    if st.button("Save Changes", key=f"save_{tab_key}"):
        _save_list_view_changes(edited_df)

    # Export CSV button
    csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export CSV 📥",
        csv,
        "jobs.csv",
        "text/csv",
        key=f"export_{tab_key}",
    )


def _save_list_view_changes(edited_df: pd.DataFrame) -> None:
    """Save changes from list view editing.

    Args:
        edited_df: DataFrame with edited job data.
    """
    session = SessionLocal()

    try:
        for _, row in edited_df.iterrows():
            job = session.query(JobSQL).filter_by(id=row["id"]).first()
            if job:
                job.favorite = row["Favorite"]
                job.application_status = row["Status"]
                job.notes = row["Notes"]

        session.commit()
        st.success("Saved!")

    except Exception as e:
        st.error("Save failed.")
        logger.error(f"Save failed: {e}")

    finally:
        session.close()


def _render_statistics_dashboard(jobs: list[JobSQL]) -> None:
    """Render the statistics dashboard.

    Args:
        jobs: List of all jobs for statistics calculation.
    """
    st.markdown("---")
    st.markdown("### 📊 Dashboard")

    # Calculate statistics
    total_jobs = len(jobs)
    favorites = sum(1 for j in jobs if j.favorite)
    applied = sum(1 for j in jobs if j.status == "Applied")
    interested = sum(1 for j in jobs if j.status == "Interested")
    new_jobs = sum(1 for j in jobs if j.status == "New")
    rejected = sum(1 for j in jobs if j.status == "Rejected")

    # Render metric cards
    _render_metric_cards(total_jobs, new_jobs, interested, applied, favorites, rejected)

    # Render progress visualization
    if total_jobs > 0:
        _render_progress_visualization(
            total_jobs, new_jobs, interested, applied, rejected
        )


def _render_metric_cards(
    total_jobs: int,
    new_jobs: int,
    interested: int,
    applied: int,
    favorites: int,
    rejected: int,
) -> None:
    """Render the metric cards section.

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
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_jobs}</div>
                <div class="metric-label">Total Jobs</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--primary-color);">
                    {new_jobs}
                </div>
                <div class="metric-label">New</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--warning-color);">
                    {interested}
                </div>
                <div class="metric-label">Interested</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--success-color);">
                    {applied}
                </div>
                <div class="metric-label">Applied</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #f59e0b;">{favorites}</div>
                <div class="metric-label">Favorites</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--danger-color);">
                    {rejected}
                </div>
                <div class="metric-label">Rejected</div>
            </div>
        """,
            unsafe_allow_html=True,
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
        # Application rate metric
        application_rate = (applied / total_jobs) * 100 if total_jobs > 0 else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{application_rate:.1f}%</div>
                <div class="metric-label">Application Rate</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
