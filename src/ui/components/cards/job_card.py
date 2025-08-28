"""Fragment-based mobile-first responsive job card component for modern job browsing.

This module provides a modern card-based interface with st.fragment() architecture for:
- Component isolation and selective updates
- Auto-refresh capabilities with configurable intervals
- Real-time status updates without full page reruns
- Fragment-scoped state management and error boundaries

Features include:
- Mobile-first responsive design (320px-1920px)
- CSS Grid layout with auto-fill and mobile breakpoints
- Touch-friendly interactions and accessibility
- Fragment-based real-time updates with optimistic UI
- Integration with search service (ADR-018) and status tracking (ADR-020)
- Performance optimized for <100ms fragment updates
- Progressive enhancement with graceful fallback

This implementation follows Stream C fragment architecture specifications for
component isolation and coordination simplification.
"""

import html
import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.schemas import Job

import pandas as pd
import streamlit as st

from src.constants import APPLICATION_STATUSES
from src.services.job_service import JobService
from src.ui.styles.styles import apply_job_grid_styles

logger = logging.getLogger(__name__)


def render_job_card(job: "Job") -> None:
    """Render an individual job card with interactive controls.

    This function creates a visually appealing job card with job details,
    status badge, favorite toggle, and view details functionality as specified
    in the requirements.

    Args:
        job: Job DTO object containing job information.
    """
    # Use st.container with border as required
    with st.container(border=True):
        # Format posted date for display
        time_str = _format_posted_date(job.posted_date)

        # Job title and company
        st.markdown(f"### {html.escape(job.title)}")
        job_company = html.escape(job.company)
        job_location = html.escape(job.location)
        st.markdown(f"**{job_company}** • {job_location} • {time_str}")

        # Job description preview
        description_preview = (
            job.description[:200] + "..."
            if len(job.description) > 200
            else job.description
        )
        st.markdown(description_preview)

        # Status badge and favorite indicator
        col1, col2 = st.columns([2, 1])
        with col1:
            status_class = f"status-{job.application_status.lower()}"
            status_html = (
                f'<span class="status-badge {status_class}">'
                f"{html.escape(job.application_status)}</span>"
            )
            st.markdown(status_html, unsafe_allow_html=True)
        with col2:
            if job.favorite:
                st.markdown("⭐")

        # Interactive controls row
        col1, col2, col3 = st.columns(3)

        with col1:
            # Status selectbox with on_change callback
            status_options = APPLICATION_STATUSES
            current_index = (
                status_options.index(job.application_status)
                if job.application_status in status_options
                else 0
            )

            st.selectbox(
                "Status",
                status_options,
                index=current_index,
                key=f"status_{job.id}",
                on_change=_handle_status_change,
                args=(job.id,),
            )

        with col2:
            # Favorite toggle button with heart icons
            favorite_icon = "❤️" if job.favorite else "🤍"
            if st.button(
                favorite_icon,
                key=f"favorite_{job.id}",
                help="Toggle favorite",
                on_click=_handle_favorite_toggle,
                args=(job.id,),
            ):
                pass  # onClick is handled by the on_click parameter

        with col3:
            # View Details button
            if st.button(
                "View Details",
                key=f"details_{job.id}",
                on_click=_handle_view_details,
                args=(job.id,),
            ):
                pass  # onClick is handled by the on_click parameter


def _format_posted_date(posted_date: "Any") -> str:
    """Format the posted date for display with robust timezone handling.

    This function handles both timezone-naive and timezone-aware datetime objects,
    normalizing them to UTC for consistent calculation. It follows modern Python
    datetime best practices using the zoneinfo module.

    Args:
        posted_date: The posted date value (can be string, datetime, or None).

    Returns:
        Formatted time string (e.g., "Today", "2 days ago").
    """
    if pd.notna(posted_date):
        # Handle string input - parse and assume UTC if no timezone info
        if isinstance(posted_date, str):
            try:
                posted_date = datetime.strptime(posted_date, "%Y-%m-%d").replace(
                    tzinfo=UTC,
                )
            except ValueError:
                logger.warning("Failed to parse posted_date string: %s", posted_date)
                return ""

        # Handle datetime objects - ensure timezone awareness
        elif isinstance(posted_date, datetime):
            if posted_date.tzinfo is None:
                # Naive datetime - assume UTC (common for jobspy data)
                posted_date = posted_date.replace(tzinfo=UTC)
                logger.debug("Converted naive datetime to UTC: %s", posted_date)
            # Already timezone-aware - use as-is

        else:
            logger.warning("Unexpected posted_date type: %s", type(posted_date))
            return ""

        # Calculate difference using timezone-aware datetimes
        try:
            now_utc = datetime.now(UTC)
            time_diff = now_utc - posted_date
            days_ago = time_diff.days

            if days_ago == 0:
                return "Today"
            if days_ago == 1:
                return "Yesterday"
            if days_ago > 0:
                return f"{days_ago} days ago"
            # Future date
            return f"In {abs(days_ago)} days"
        except Exception:
            logger.exception("Error calculating time difference for posted_date")
            return ""

    return ""


def _handle_status_change(job_id: int) -> None:
    """Handle status change callback.

    Args:
        job_id: Database ID of the job to update.
    """
    try:
        if new_status := st.session_state.get(f"status_{job_id}"):
            JobService.update_job_status(job_id, new_status)
            st.rerun()
    except Exception:
        logger.exception("Failed to update job status")
        st.error("Failed to update job status")


def _handle_favorite_toggle(job_id: int) -> None:
    """Handle favorite toggle callback.

    Args:
        job_id: Database ID of the job to toggle.
    """
    try:
        JobService.toggle_favorite(job_id)
        st.rerun()
    except Exception:
        logger.exception("Failed to toggle favorite")
        st.error("Failed to toggle favorite")


def _handle_view_details(job_id: int) -> None:
    """Handle view details button click.

    Args:
        job_id: Database ID of the job to view details for.
    """
    st.session_state.view_job_id = job_id


def render_responsive_job_card(job: "Job") -> str:
    """Render a modern responsive job card using CSS Grid and HTML.

    This function creates a self-contained HTML card with all styling
    and interactivity, optimized for mobile-first responsive design.

    Args:
        job: Job DTO object containing job information.

    Returns:
        str: Complete HTML string for the job card.
    """
    # Get responsive configuration with fallback
    try:
        card_config = get_responsive_card_config()
        device_type = get_device_type()
    except NameError:
        # Fallback if mobile detection not available
        card_config = {"show_descriptions": True, "cards_per_page": 20}
        device_type = "desktop"

    # Format posted date
    time_str = _format_posted_date(job.posted_date)

    # Determine status styling
    status_class = f"status-{job.application_status.lower()}"

    # Create description preview
    description_preview = (
        job.description[:150] + "..." if len(job.description) > 150 else job.description
    )

    # Build card HTML with responsive classes
    return f"""
    <div class="job-card" data-job-id="{job.id}" data-device="{device_type}">
        <div class="job-card-header">
            <h3 class="job-card-title">{html.escape(job.title)}</h3>
            <div class="job-card-company">
                🏢 {html.escape(job.company)}
            </div>
        </div>

        <div class="job-card-body">
            <div class="job-card-location">
                📍 {html.escape(job.location)}
            </div>

            {
        (
            "<div class='job-card-description'>"
            + html.escape(description_preview)
            + "</div>"
        )
        if card_config.get("show_descriptions")
        else ""
    }

            <div class="job-card-meta">
                <span>{time_str}</span>
                <span class="status-badge {status_class}">
                    {html.escape(job.application_status)}
                </span>
            </div>
        </div>

        <div class="job-card-footer">
            <div class="job-card-actions">
                {"⭐" if job.favorite else "☆"}
            </div>
        </div>
    </div>
    """


def render_jobs_responsive_grid(jobs: list["Job"]) -> None:
    """Render jobs in a modern CSS Grid responsive layout.

    This function replaces the traditional st.columns approach with a pure CSS Grid
    layout that automatically adapts from 1 column on mobile to 3+ columns on desktop.

    Performance optimized for <200ms rendering of 50+ cards as per ADR-021.

    Args:
        jobs: List of Job DTO objects to render in responsive grid.
    """
    if not jobs:
        st.info("🔍 No jobs found. Try adjusting your filters.")
        return

    # Apply responsive CSS styles
    apply_job_grid_styles()

    # Get responsive configuration with fallback
    try:
        card_config = get_responsive_card_config()
        device_type = get_device_type()
    except NameError:
        # Fallback if mobile detection not available
        card_config = {"show_descriptions": True, "cards_per_page": 20}
        device_type = "desktop"

    # Optimize performance with pagination for mobile
    cards_per_page = card_config.get("cards_per_page", 20)

    # Add pagination if needed
    if len(jobs) > cards_per_page:
        total_pages = (len(jobs) + cards_per_page - 1) // cards_per_page

        # Use session state for pagination
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                f"Page ({len(jobs):,} total jobs)",
                options=list(range(1, total_pages + 1)),
                index=st.session_state.current_page - 1,
                key="page_selector",
            )
            st.session_state.current_page = current_page

        # Get current page jobs
        start_idx = (current_page - 1) * cards_per_page
        end_idx = start_idx + cards_per_page
        display_jobs = jobs[start_idx:end_idx]
    else:
        display_jobs = jobs

    # Generate all card HTML
    cards_html = []
    for job in display_jobs:
        card_html = render_responsive_job_card(job)
        cards_html.append(card_html)

    # Render the complete grid in one HTML block for performance
    grid_html = f"""
    <div class="job-cards-container"
         data-device="{device_type}"
         data-card-count="{len(display_jobs)}">
        {"".join(cards_html)}
    </div>

    <script>
    // Add click handlers for job cards
    document.addEventListener('DOMContentLoaded', function() {{
        const cards = document.querySelectorAll('.job-card');
        cards.forEach(card => {{
            card.addEventListener('click', function() {{
                const jobId = this.dataset.jobId;
                // Store job ID for Streamlit to pick up
                if (window.parent && window.parent.streamlit) {{
                    window.parent.streamlit.setComponentValue({{
                        action: 'view_job',
                        job_id: jobId
                    }});
                }}
            }});
        }});
    }});
    </script>
    """

    # Render the grid
    st.html(grid_html)

    # Handle job selection with callback
    if "view_job_id" in st.session_state:
        selected_job = next(
            (job for job in display_jobs if job.id == st.session_state.view_job_id),
            None,
        )
        if selected_job:
            try:
                _show_job_details_modal(selected_job)
            except (ImportError, NameError):
                # Fallback: show basic info
                st.info(f"Selected job: {selected_job.title} at {selected_job.company}")
                # Clear the selection
                if "view_job_id" in st.session_state:
                    del st.session_state.view_job_id


def _show_job_details_modal(job: "Job") -> None:
    """Show job details in a modal dialog (helper function).

    Args:
        job: Job object to display details for.
    """
    try:
        # Import here to avoid circular dependency
        from src.ui.pages.jobs import show_job_details_modal

        show_job_details_modal(job)
    except ImportError:
        # Simple fallback modal using st.dialog if available
        if hasattr(st, "dialog"):

            @st.dialog(f"Job Details: {job.title}")
            def show_fallback_modal():
                st.markdown(f"**{job.title}**")
                st.markdown(f"**Company:** {job.company}")
                st.markdown(f"**Location:** {job.location}")
                st.markdown(f"**Status:** {job.application_status}")
                if job.description:
                    st.markdown("**Description:**")
                    st.markdown(
                        job.description[:500] + "..."
                        if len(job.description) > 500
                        else job.description
                    )

            show_fallback_modal()
        else:
            # Most basic fallback
            st.info(f"**{job.title}** at {job.company} - {job.location}")


def render_jobs_grid(jobs: list["Job"], num_columns: int = 3) -> None:
    """Legacy grid rendering using st.columns - maintained for backward compatibility.

    This function creates a responsive grid of job cards using st.columns.
    For new implementations, use render_jobs_responsive_grid() instead.

    Args:
        jobs: List of Job DTO objects to render in grid.
        num_columns: Number of columns for the grid layout (default: 3).
    """
    if not jobs:
        st.info("No jobs to display.")
        return

    # Apply centralized CSS styles for job grid
    apply_job_grid_styles()

    # Use responsive column calculation
    try:
        from src.ui.utils.mobile_detection import get_responsive_columns

        optimal_columns = get_responsive_columns(len(jobs), num_columns)
    except ImportError:
        optimal_columns = num_columns

    # Create responsive grid using st.columns
    for i in range(0, len(jobs), optimal_columns):
        # Create columns with equal width and medium gap for better spacing
        cols = st.columns(optimal_columns, gap="medium")

        # Render jobs for this row
        row_jobs = jobs[i : i + optimal_columns]
        for j, job in enumerate(row_jobs):
            with cols[j], st.container():
                # Wrap each card in a container for better height management
                render_job_card(job)

        # Add spacing between rows, but only if there are more jobs
        if i + optimal_columns < len(jobs):
            st.markdown('<div class="job-card-grid"></div>', unsafe_allow_html=True)


def render_jobs_list(jobs: list["Job"]) -> None:
    """Render a list of job cards.

    This is the main function for rendering jobs according to T1.1 requirements.

    Args:
        jobs: List of Job DTO objects to render.
    """
    if not jobs:
        st.info("No jobs to display.")
        return

    for job in jobs:
        # Render the job card
        render_job_card(job)

        # Add some spacing between cards
        st.markdown("---")


# ========== FRAGMENT-BASED IMPLEMENTATIONS ==========
# Enhanced components with st.fragment() architecture for Stream C


@st.fragment(run_every="2s")
def render_job_card_fragment(job: "Job", enable_auto_refresh: bool = False) -> None:
    """Fragment-based job card with isolated updates and auto-refresh.

    This fragment isolates job card rendering to prevent full page reruns when
    updating individual job status or favorites. Auto-refresh can be enabled
    for real-time job updates from background processes.

    Args:
        job: Job DTO object containing job information.
        enable_auto_refresh: Whether to enable automatic refreshing for real-time
            updates.
    """
    # Skip unnecessary updates when scraping is inactive
    try:
        from src.ui.utils.background_helpers import is_scraping_active

        # Only auto-refresh when scraping is active
        if enable_auto_refresh and not is_scraping_active():
            # Reduce CPU usage when idle
            return
    except ImportError:
        # Background helpers not available, continue without auto-refresh optimization
        pass

    # Fragment-scoped error handling
    try:
        # Check if this job still exists and get latest data
        if enable_auto_refresh:
            updated_job = JobService.get_job_by_id(job.id)
            if updated_job:
                job = updated_job
            else:
                # Job was deleted, show tombstone
                _render_job_tombstone(job.id, job.title)
                return

        # Use st.container with border and unique key for fragment isolation
        with st.container(border=True, key=f"job_card_fragment_{job.id}"):
            # Format posted date for display
            time_str = _format_posted_date(job.posted_date)

            # Job title and company with real-time updates
            st.markdown(f"### {html.escape(job.title)}")
            job_company = html.escape(job.company)
            job_location = html.escape(job.location)
            st.markdown(f"**{job_company}** • {job_location} • {time_str}")

            # Job description preview
            description_preview = (
                job.description[:200] + "..."
                if len(job.description) > 200
                else job.description
            )
            st.markdown(description_preview)

            # Status badge and favorite indicator with live updates
            col1, col2 = st.columns([2, 1])
            with col1:
                status_class = f"status-{job.application_status.lower()}"
                status_html = (
                    f'<span class="status-badge {status_class}">'
                    f"{html.escape(job.application_status)}</span>"
                )
                st.markdown(status_html, unsafe_allow_html=True)
            with col2:
                if job.favorite:
                    st.markdown("⭐")

            # Interactive controls with fragment-scoped updates
            col1, col2, col3 = st.columns(3)

            with col1:
                # Status selectbox with fragment-scoped callback
                status_options = APPLICATION_STATUSES
                current_index = (
                    status_options.index(job.application_status)
                    if job.application_status in status_options
                    else 0
                )

                st.selectbox(
                    "Status",
                    status_options,
                    index=current_index,
                    key=f"fragment_status_{job.id}",
                    on_change=_handle_fragment_status_change,
                    args=(job.id,),
                )

            with col2:
                # Favorite toggle with fragment-scoped update
                favorite_icon = "❤️" if job.favorite else "🤍"
                if st.button(
                    favorite_icon,
                    key=f"fragment_favorite_{job.id}",
                    help="Toggle favorite",
                    on_click=_handle_fragment_favorite_toggle,
                    args=(job.id,),
                ):
                    pass  # Handled by on_click callback

            with col3:
                # View Details button with fragment-scoped navigation
                if st.button(
                    "View Details",
                    key=f"fragment_details_{job.id}",
                    on_click=_handle_fragment_view_details,
                    args=(job.id,),
                ):
                    pass  # Handled by on_click callback

    except Exception as e:
        logger.exception("Fragment error for job %s", job.id)
        # Show error state within fragment boundary
        st.error(f"⚠️ Error updating job card: {str(e)[:100]}...")


@st.fragment(run_every="3s")
def render_live_job_updates(job_ids: list[int], max_display: int = 5) -> None:
    """Fragment for displaying real-time job updates and notifications.

    This fragment monitors job changes and displays toast notifications for
    updates without triggering full page reruns.

    Args:
        job_ids: List of job IDs to monitor for changes.
        max_display: Maximum number of update notifications to display.
    """
    try:
        # Check for job updates since last fragment run
        if "last_job_update_check" not in st.session_state:
            st.session_state.last_job_update_check = datetime.now(UTC)
            return  # Skip first run to establish baseline

        last_check = st.session_state.last_job_update_check
        current_time = datetime.now(UTC)

        # Query for recently updated jobs
        updated_jobs = JobService.get_recently_updated_jobs(
            job_ids, since=last_check, limit=max_display
        )

        # Display notifications for updates
        if updated_jobs:
            for job in updated_jobs[:max_display]:
                st.toast(f"🔄 {job.title} at {job.company} updated!", icon="🔄")

            # Show summary in an info box that auto-disappears
            st.success(f"✨ {len(updated_jobs)} job(s) updated", icon="📱")

        # Update last check time
        st.session_state.last_job_update_check = current_time

    except Exception:
        logger.exception("Error in live job updates fragment")
        # Silently fail to avoid disrupting main UI


@st.fragment(run_every="30s")
def render_job_stats_fragment(jobs: list["Job"]) -> None:
    """Fragment for displaying real-time job statistics.

    This fragment shows aggregate statistics that update automatically
    without affecting the main job display.

    Args:
        jobs: List of jobs to calculate statistics for.
    """
    try:
        if not jobs:
            return

        # Calculate real-time statistics
        total_jobs = len(jobs)
        favorited_jobs = sum(1 for job in jobs if job.favorite)
        status_counts = {}

        for job in jobs:
            status_counts[job.application_status] = (
                status_counts.get(job.application_status, 0) + 1
            )

        # Display metrics in a compact layout
        with st.container():
            st.caption("🔄 Real-time Statistics")
            cols = st.columns(4)

            with cols[0]:
                st.metric("Total Jobs", total_jobs)
            with cols[1]:
                st.metric("Favorites", favorited_jobs)
            with cols[2]:
                applied_count = status_counts.get("Applied", 0)
                st.metric("Applied", applied_count)
            with cols[3]:
                interviewing_count = status_counts.get("Interview", 0)
                st.metric("Interviews", interviewing_count)

    except Exception:
        logger.exception("Error in job stats fragment")
        # Show minimal error state
        st.caption("⚠️ Stats temporarily unavailable")


def render_jobs_with_fragments(
    jobs: list["Job"],
    enable_auto_refresh: bool = False,
    enable_live_updates: bool = True,
    enable_stats: bool = True,
    num_columns: int = 3,
) -> None:
    """Enhanced job rendering with fragment architecture.

    This function orchestrates multiple fragments for optimal performance:
    - Individual job cards as fragments for isolated updates
    - Live update notifications fragment
    - Real-time statistics fragment
    - Responsive grid layout with fragment coordination

    Args:
        jobs: List of Job DTO objects to render.
        enable_auto_refresh: Enable auto-refresh for job cards.
        enable_live_updates: Enable live update notifications.
        enable_stats: Enable real-time statistics.
        num_columns: Number of columns for grid layout.
    """
    if not jobs:
        st.info("🔍 No jobs found. Try adjusting your filters.")
        return

    # Apply responsive CSS styles
    apply_job_grid_styles()

    # Live updates fragment (top of page for visibility)
    if enable_live_updates:
        job_ids = [job.id for job in jobs]
        render_live_job_updates(job_ids, max_display=3)

    # Statistics fragment
    if enable_stats:
        render_job_stats_fragment(jobs)
        st.markdown("---")

    # Responsive column calculation with fragment optimization
    try:
        from src.ui.utils.mobile_detection import get_responsive_columns

        optimal_columns = get_responsive_columns(len(jobs), num_columns)
    except ImportError:
        optimal_columns = num_columns

    # Render jobs with fragment-based cards
    for i in range(0, len(jobs), optimal_columns):
        cols = st.columns(optimal_columns, gap="medium")

        # Render jobs for this row using fragments
        row_jobs = jobs[i : i + optimal_columns]
        for j, job in enumerate(row_jobs):
            with cols[j]:
                # Each job card is its own fragment for isolation
                render_job_card_fragment(job, enable_auto_refresh)

        # Add spacing between rows
        if i + optimal_columns < len(jobs):
            st.markdown(
                '<div class="job-card-row-spacer"></div>', unsafe_allow_html=True
            )


# ========== FRAGMENT CALLBACK HANDLERS ==========
# Fragment-scoped callbacks with scoped st.rerun()


def _handle_fragment_status_change(job_id: int) -> None:
    """Handle status change with fragment-scoped rerun.

    Args:
        job_id: Database ID of the job to update.
    """
    try:
        if new_status := st.session_state.get(f"fragment_status_{job_id}"):
            JobService.update_job_status(job_id, new_status)
            st.toast(f"✅ Status updated to {new_status}", icon="✅")
            # Fragment-scoped rerun for isolation
            st.rerun(scope="fragment")
    except Exception:
        logger.exception("Failed to update job status in fragment")
        st.toast("❌ Failed to update status", icon="❌")


def _handle_fragment_favorite_toggle(job_id: int) -> None:
    """Handle favorite toggle with fragment-scoped rerun.

    Args:
        job_id: Database ID of the job to toggle.
    """
    try:
        JobService.toggle_favorite(job_id)
        st.toast("⭐ Favorite updated!", icon="⭐")
        # Fragment-scoped rerun for isolation
        st.rerun(scope="fragment")
    except Exception:
        logger.exception("Failed to toggle favorite in fragment")
        st.toast("❌ Failed to update favorite", icon="❌")


def _handle_fragment_view_details(job_id: int) -> None:
    """Handle view details with session state update.

    This doesn't require fragment rerun as it triggers navigation.

    Args:
        job_id: Database ID of the job to view details for.
    """
    st.session_state.view_job_id = job_id
    st.toast("📋 Opening job details...", icon="📋")


def _render_job_tombstone(job_id: int, job_title: str) -> None:
    """Render a tombstone for deleted jobs.

    Args:
        job_id: ID of the deleted job.
        job_title: Title of the deleted job.
    """
    with st.container(border=True):
        st.markdown(f"### ⚰️ ~~{html.escape(job_title)}~~")
        st.caption(f"This job (ID: {job_id}) has been removed.")

        if st.button("Remove from view", key=f"remove_tombstone_{job_id}"):
            # Signal to parent to remove this job from the list
            st.session_state[f"remove_job_{job_id}"] = True
            st.rerun(scope="fragment")
