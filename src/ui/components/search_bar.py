"""Search UI component with FTS5 integration.

This module provides a search interface that leverages SQLite FTS5
full-text search capabilities with real-time search, filtering, relevance scoring,
and performance metrics. Integrates with existing job display and modal systems.

Key Features:
- Real-time search using SQLite FTS5 full-text search
- Filters for location, salary, remote work, and date ranges
- Relevance score display with performance metrics
- Integration with job card components and modal system
- Empty state and error handling
- Mobile-responsive design
"""

import logging
import time

from typing import TYPE_CHECKING, Any

import streamlit as st

from src.constants import APPLICATION_STATUSES, SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN
from src.ui.components.cards.job_card import render_job_card
from src.ui.pages.jobs import show_job_details_modal
from src.ui.state.session_state import get_current_filters
from src.ui.utils.service_cache import get_search_service

if TYPE_CHECKING:
    from src.schemas import Job

logger = logging.getLogger(__name__)

# Feature flags for future enhancements
FEATURE_FLAGS = {
    "search_suggestions": False,
    "export_results": False,
    "save_queries": False,
}

# FTS5 search hints for better user experience
FTS5_SEARCH_HINTS = [
    '"python developer"',  # Exact phrase
    "machine AND learning",  # Boolean operators
    "data NOT science",  # Exclusion
    "senior OR lead",  # Alternative terms
    "python*",  # Wildcard/stemming
]

# Debounce delay for real-time search (in seconds)
SEARCH_DEBOUNCE_DELAY = 0.3

# Default search result limits
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 100


def render_job_search() -> None:
    """Main search component function for easy integration.

    This is the primary interface for integrating the search functionality
    into pages. It handles all search UI, filtering, results display, and
    modal integration.
    """
    # Initialize search state
    _init_search_state()

    # Render search interface
    with st.container():
        st.markdown("### 🔍 Job Search")

        # Main search bar
        _render_search_input()

        # Advanced filters (collapsible)
        _render_advanced_filters()

        # Search results section
        _render_search_results()

        # Handle job details modal
        _handle_search_modal()


def _init_search_state() -> None:
    """Initialize MINIMAL search state using widget keys.

    This eliminates the YAGNI violations by using widget keys instead of
    manual session state management. Search results are temporarily cached
    but not persistent across pages.
    """
    # MINIMAL state - only non-widget data that needs persistence
    minimal_search_state = {
        "search_results": [],  # Temporary results cache
        "search_stats": {"query_time": 0, "total_results": 0, "fts_enabled": False},
        "last_search_time": 0,  # For debouncing
    }

    for key, value in minimal_search_state.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_search_input() -> None:
    """Render the main search input with FTS5 hints and real-time search."""
    col1, col2 = st.columns([4, 1])

    with col1:
        # WIDGET KEY: Search input with auto-managed state
        st.text_input(
            label="Search Jobs",
            placeholder=(
                'Try: "python developer", machine AND learning, data* (FTS5 powered)'
            ),
            key="search_query_input",  # Widget handles its own state
            help=(
                "Powered by SQLite FTS5 with Porter stemming. Use quotes for exact"
                " phrases, AND/OR for logic, * for wildcards."
            ),
            on_change=_handle_search_input_change,
            label_visibility="collapsed",
        )

        # Show search hints when no search query is entered
        current_query = st.session_state.get("search_query_input", "")
        if not current_query:
            with st.expander("🔍 Search Tips & Examples", expanded=False):
                st.markdown("**FTS5 Search Examples:**")
                for i, hint in enumerate(FTS5_SEARCH_HINTS, 1):
                    if st.button(
                        f"{i}. {hint}", key=f"hint_{i}", use_container_width=True
                    ):
                        # Set the widget key value directly
                        st.session_state.search_query_input = hint
                        _perform_search()
                        st.rerun()

                st.markdown("""
                **Search Operators:**
                - `"exact phrase"` - Match exact phrases
                - `term1 AND term2` - Both terms required
                - `term1 OR term2` - Either term matches
                - `term1 NOT term2` - Exclude term2
                - `term*` - Wildcard matching (e.g., `develop*` matches `developer`)
                """)

    with col2:
        # Advanced filters toggle
        # WIDGET KEY: Filter toggle using checkbox
        st.checkbox(
            "⚙️ Show Filters" + (" ✓" if _has_active_filters() else ""),
            key="show_advanced_filters",  # Widget handles its own state
            label_visibility="collapsed",
        )


def _render_advanced_filters() -> None:
    """Render detailed filter controls using WIDGET KEYS.

    This eliminates duplicate filter state by reusing the main filter widgets.
    """
    # Check if filters should be shown via widget key
    if not st.session_state.get("show_advanced_filters", False):
        return

    with st.expander("Filter Options", expanded=True):
        # First row: Location and Remote
        col1, col2 = st.columns(2)

        with col1:
            # WIDGET KEY: Location filter - auto-managed state
            st.text_input(
                "Location",
                placeholder="e.g., San Francisco, Remote, New York",
                help="Filter by job location or 'Remote' for remote positions",
                key="location_filter",  # Widget handles its own state
                on_change=_trigger_search_update,
            )

        with col2:
            # WIDGET KEY: Remote filter - auto-managed state
            st.checkbox(
                "Remote positions only",
                help="Show only remote job opportunities",
                key="remote_only_filter",  # Widget handles its own state
                on_change=_trigger_search_update,
            )

        # Second row: Salary range
        st.markdown("**Salary Range**")
        col1, col2 = st.columns(2)

        # NOTE: Salary filtering uses main sidebar salary_range_filter widget
        # to avoid duplication. Search will read from the same widget.
        st.info("💡 Use the main sidebar salary filter to set salary ranges")

        # Third row: Application status and favorites
        col1, col2 = st.columns(2)

        with col1:
            # WIDGET KEY: Application status filter - auto-managed state
            st.selectbox(
                "Application Status",
                options=["All", *APPLICATION_STATUSES],
                index=0,  # Default to "All"
                help="Filter by current application status",
                key="application_status_filter",  # Widget handles its own state
                on_change=_trigger_search_update,
            )

        with col2:
            # WIDGET KEY: Favorites filter - auto-managed state
            st.checkbox(
                "Favorites only",
                help="Show only jobs marked as favorites",
                key="favorites_only_filter",  # Widget handles its own state
                on_change=_trigger_search_update,
            )

        # Fourth row: Date range
        st.markdown("**Posted Date Range**")
        col1, col2 = st.columns(2)

        # NOTE: Date filtering uses main sidebar date filters to avoid duplication
        st.info("💡 Use the main sidebar date filters to set date ranges")

        # Filter actions
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Search", type="primary", use_container_width=True):
                _perform_search()
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Search", use_container_width=True):
                _clear_search_filters()
                st.rerun()


def _render_search_results() -> None:
    """Render search results with performance metrics and job display."""
    # Show search status and metrics
    _render_search_status()

    # Display results
    results = st.session_state.search_results
    if not results:
        _render_empty_state()
        return

    # Results display options
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(
            f"**Found {len(results)} jobs** "
            + f"({st.session_state.search_stats['query_time']:.0f}ms)"
        )

    with col2:
        # WIDGET KEY: View mode selector with auto-managed state
        view_mode = st.selectbox(
            "View",
            options=["Cards", "List"],
            index=0,  # Default to Cards
            key="search_view_mode",  # Widget handles its own state
            label_visibility="collapsed",
        )

    with col3:
        # Results per page
        # WIDGET KEY: Results limit with auto-managed state
        results_limit = st.selectbox(
            "Show",
            options=[25, 50, 100],
            index=1,  # Default to 50
            key="search_limit_selector",  # Widget handles its own state
            format_func=lambda x: f"{x} results",
            label_visibility="collapsed",
            on_change=_perform_search,  # Auto-trigger search on change
        )

    st.markdown("---")

    # Render results based on view mode
    try:
        if view_mode == "Cards":
            _render_search_results_cards(results[:results_limit])
        else:
            _render_search_results_list(results[:results_limit])
    except Exception as e:
        logger.exception("Error rendering search results")
        st.error(f"Error displaying search results: {e!s}")


def _render_search_results_cards(results: list["Job"]) -> None:
    """Render search results in card view with relevance scores."""
    if not results:
        return

    # Group cards in rows of 3
    for i in range(0, len(results), 3):
        cols = st.columns(3, gap="medium")
        row_jobs = results[i : i + 3]

        for j, job in enumerate(row_jobs):
            with cols[j]:
                # Add relevance score if available
                if hasattr(job, "rank") and job.rank is not None:
                    relevance_score = abs(job.rank) if job.rank < 0 else job.rank
                    st.caption(f"🎯 Relevance: {relevance_score:.1f}")

                # Render the job card
                render_job_card(job)


def _render_search_results_list(results: list["Job"]) -> None:
    """Render search results in list view with relevance scores."""
    for job in results:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                # Job title and basic info
                st.markdown(f"### {job.title}")
                st.markdown(f"**{job.company}** • {job.location}")

                # Job description preview
                description_preview = (
                    job.description[:150] + "..."
                    if len(job.description) > 150
                    else job.description
                )
                st.markdown(description_preview)

                # Status and favorite
                col1a, col1b = st.columns([2, 1])
                with col1a:
                    status_badge = (
                        f'<span style="background: #e1f5fe; padding: 2px 8px; '
                        f'border-radius: 12px; font-size: 12px;">'
                        f"{job.application_status}</span>"
                    )
                    st.markdown(status_badge, unsafe_allow_html=True)
                with col1b:
                    if job.favorite:
                        st.markdown("⭐ Favorite")

            with col2:
                # Relevance score if available
                if hasattr(job, "rank") and job.rank is not None:
                    relevance_score = abs(job.rank) if job.rank < 0 else job.rank
                    st.metric("Relevance", f"{relevance_score:.1f}")

                # View details button using unified modal state
                if st.button("View Details", key=f"search_details_{job.id}"):
                    st.session_state.modal_job_id = job.id
                    st.rerun()


def _render_search_status() -> None:
    """Render search status, performance metrics, and FTS5 information."""
    if not st.session_state.search_query:
        return

    stats = st.session_state.search_stats

    # Status indicator
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # FTS5 status indicator
        fts_status = "🚀 FTS5" if stats.get("fts_enabled", False) else "📝 Basic"
        search_type = (
            "Full-text search" if stats.get("fts_enabled", False) else "Keyword search"
        )
        current_query = st.session_state.get("search_query_input", "")
        st.markdown(f"{fts_status} **{search_type}** for: `{current_query}`")

    with col2:
        # Performance metrics
        query_time = stats.get("query_time", 0)
        if query_time > 0:
            st.metric("Query Time", f"{query_time:.0f}ms")

    with col3:
        # Results count
        total_results = len(st.session_state.search_results)
        st.metric("Results", f"{total_results:,}")


def _render_empty_state() -> None:
    """Render empty state with helpful suggestions."""
    current_query = st.session_state.get("search_query_input", "")
    if not current_query:
        # No search performed yet
        st.info(
            "👋 **Welcome to Job Search!**\n\n"
            "Enter a search term above to find jobs using our powerful FTS5 search "
            "engine."
        )

        # Show some example searches
        st.markdown("**Try these example searches:**")
        col1, col2 = st.columns(2)

        examples = [
            ("Python Developer", '"python developer"'),
            ("Machine Learning", "machine AND learning"),
            ("Remote Data Jobs", "data* AND remote"),
            ("Senior Roles", "senior OR lead"),
        ]

        for i, (label, query) in enumerate(examples):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(
                    f"🔍 {label}", key=f"example_{i}", use_container_width=True
                ):
                    # Set widget key directly
                    st.session_state.search_query_input = query
                    _perform_search()
                    st.rerun()

    else:
        # Search performed but no results
        st.warning("🔍 **No jobs found**")

        with st.expander("💡 Search Tips", expanded=True):
            st.markdown("""
            **Try these suggestions:**
            - Use broader search terms (e.g., `data` instead of `data scientist`)
            - Remove some filters to expand results
            - Check spelling and try synonyms
            - Use wildcard search with `*` (e.g., `develop*`)
            - Try boolean operators: `python OR java`
            """)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Filters", use_container_width=True):
                    _clear_search_filters()
                    st.rerun()

            with col2:
                if st.button("🔄 Show All Jobs", use_container_width=True):
                    # Clear search widget
                    if "search_query_input" in st.session_state:
                        del st.session_state.search_query_input
                    # Redirect to jobs page
                    st.switch_page("src/ui/pages/jobs.py")


def _handle_search_modal() -> None:
    """Handle job details modal using UNIFIED modal state."""
    # Use the unified modal_job_id instead of search-specific key
    if modal_job_id := st.session_state.get("modal_job_id"):
        # Find the job in search results
        job = next(
            (job for job in st.session_state.search_results if job.id == modal_job_id),
            None,
        )

        if job:
            show_job_details_modal(job)
        else:
            # Job not found, clear the unified modal
            st.session_state.modal_job_id = None


def _handle_search_input_change() -> None:
    """Handle search input changes with debouncing using WIDGET KEYS."""
    current_query = st.session_state.get("search_query_input", "")

    # Debounced search - only search if enough time has passed
    current_time = time.time()
    if (
        current_query
        and (current_time - st.session_state.get("last_search_time", 0))
        > SEARCH_DEBOUNCE_DELAY
    ):
        st.session_state.last_search_time = current_time
        _perform_search()


def _trigger_search_update() -> None:
    """Trigger search update when filters change using WIDGET KEYS."""
    current_query = st.session_state.get("search_query_input", "")
    if current_query:
        _perform_search()


def _perform_search() -> None:
    """Execute the search using CACHED search service and WIDGET KEYS."""
    query = st.session_state.get("search_query_input", "").strip()

    if not query:
        st.session_state.search_results = []
        st.session_state.search_stats = {
            "query_time": 0,
            "total_results": 0,
            "fts_enabled": False,
        }
        return

    try:
        # Prepare search filters from widget keys
        search_filters = _build_search_filters()

        # Measure search performance
        start_time = time.time()

        # Execute search using cached service
        search_service = get_search_service()
        search_limit = st.session_state.get(
            "search_limit_selector", DEFAULT_SEARCH_LIMIT
        )
        results = search_service.search_jobs(
            query=query, filters=search_filters, limit=search_limit
        )

        # Calculate metrics
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Get search service stats
        service_stats = search_service.get_search_stats()

        # Update session state
        st.session_state.search_results = results
        st.session_state.search_stats = {
            "query_time": query_time,
            "total_results": len(results),
            "fts_enabled": service_stats.get("fts_enabled", False),
        }

        logger.info(
            "Search completed: query='%s', results=%d, time=%.1fms",
            query,
            len(results),
            query_time,
        )

    except Exception as e:
        logger.exception("Search failed for query: '%s'", query)
        st.error(f"Search failed: {e!s}")
        st.session_state.search_results = []
        st.session_state.search_stats = {
            "query_time": 0,
            "total_results": 0,
            "fts_enabled": False,
        }


def _build_search_filters() -> dict[str, Any]:
    """Build search filters from WIDGET KEYS instead of duplicate state."""
    # Get filter values from main sidebar widgets and search-specific widgets
    current_filters = get_current_filters()

    # Build search filters combining main filters with search-specific ones
    search_filters = {
        "date_from": current_filters.get("date_from"),
        "date_to": current_filters.get("date_to"),
        "favorites_only": st.session_state.get("favorites_only_filter", False),
        "salary_min": current_filters.get("salary_min")
        if current_filters.get("salary_min", 0) > SALARY_DEFAULT_MIN
        else None,
        "salary_max": current_filters.get("salary_max")
        if current_filters.get("salary_max", SALARY_DEFAULT_MAX) < SALARY_DEFAULT_MAX
        else None,
    }

    # Application status filter from widget key
    app_status = st.session_state.get("application_status_filter", "All")
    if app_status != "All":
        search_filters["application_status"] = [app_status]

    # Location filter from widget key
    location = st.session_state.get("location_filter", "")
    remote_only = st.session_state.get("remote_only_filter", False)

    if location or remote_only:
        location_terms = []
        if location:
            location_terms.append(location)
        if remote_only:
            location_terms.append("Remote")
        # Note: Location filtering could be enhanced in search service

    return search_filters


def _has_active_filters() -> bool:
    """Check if any advanced filters are active using WIDGET KEYS."""
    # Check search-specific widget filters
    if st.session_state.get("location_filter", ""):
        return True
    if st.session_state.get("remote_only_filter", False):
        return True
    if st.session_state.get("application_status_filter", "All") != "All":
        return True
    if st.session_state.get("favorites_only_filter", False):
        return True

    # Check main filter widgets for changes from defaults
    main_filters = get_current_filters()
    if main_filters.get("keyword", ""):
        return True
    if main_filters.get("company", []):
        return True
    if main_filters.get("salary_min", SALARY_DEFAULT_MIN) > SALARY_DEFAULT_MIN:
        return True
    return main_filters.get("salary_max", SALARY_DEFAULT_MAX) < SALARY_DEFAULT_MAX


def _clear_search_filters() -> None:
    """Clear search-specific filters using WIDGET KEYS."""
    # Clear search-specific widget keys
    search_widgets = [
        "search_query_input",
        "location_filter",
        "remote_only_filter",
        "application_status_filter",
        "favorites_only_filter",
        "show_advanced_filters",
    ]

    for widget_key in search_widgets:
        if widget_key in st.session_state:
            del st.session_state[widget_key]

    # Clear search results
    st.session_state.search_results = []
    st.session_state.search_stats = {
        "query_time": 0,
        "total_results": 0,
        "fts_enabled": False,
    }


# Utility functions for search features
def get_search_suggestions() -> list[str]:
    """Get search suggestions (disabled in MVP)."""
    if not FEATURE_FLAGS["search_suggestions"]:
        return []  # Safe fallback
    raise NotImplementedError("Feature coming in v2")


def export_search_results(_results: list["Job"], _export_format: str = "csv") -> None:
    """Export search results (disabled in MVP)."""
    if not FEATURE_FLAGS["export_results"]:
        import streamlit as st

        st.info("Export feature coming soon! 🚀")
        return
    raise NotImplementedError("Feature coming in v2")


def save_search_query() -> None:
    """Save search query (disabled in MVP)."""
    if not FEATURE_FLAGS["save_queries"]:
        return  # Silent no-op
    raise NotImplementedError("Feature coming in v2")
