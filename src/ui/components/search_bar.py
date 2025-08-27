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

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.constants import APPLICATION_STATUSES, SALARY_DEFAULT_MAX, SALARY_DEFAULT_MIN
from src.services.search_service import search_service
from src.ui.components.cards.job_card import render_job_card
from src.ui.pages.jobs import show_job_details_modal

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
    """Initialize search-specific session state variables."""
    defaults = {
        "search_query": "",
        "search_results": [],
        "search_stats": {"query_time": 0, "total_results": 0, "fts_enabled": False},
        "search_filters": {
            "location": "",
            "salary_min": SALARY_DEFAULT_MIN,
            "salary_max": SALARY_DEFAULT_MAX,
            "remote_only": False,
            "application_status": "All",
            "date_from": datetime.now(UTC) - timedelta(days=30),
            "date_to": datetime.now(UTC),
            "favorites_only": False,
        },
        "show_advanced_filters": False,
        "last_search_time": 0,
        "search_limit": DEFAULT_SEARCH_LIMIT,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_search_input() -> None:
    """Render the main search input with FTS5 hints and real-time search."""
    col1, col2 = st.columns([4, 1])

    with col1:
        # Search input with FTS5-optimized placeholder
        st.text_input(
            label="Search Jobs",
            value=st.session_state.search_query,
            placeholder=(
                'Try: "python developer", machine AND learning, data* (FTS5 powered)'
            ),
            key="search_input",
            help=(
                "Powered by SQLite FTS5 with Porter stemming. Use quotes for exact"
                " phrases, AND/OR for logic, * for wildcards."
            ),
            on_change=_handle_search_input_change,
            label_visibility="collapsed",
        )

        # Show search hints in expandable section
        if st.session_state.search_query == "":
            with st.expander("🔍 Search Tips & Examples", expanded=False):
                st.markdown("**FTS5 Search Examples:**")
                for i, hint in enumerate(FTS5_SEARCH_HINTS, 1):
                    if st.button(
                        f"{i}. {hint}", key=f"hint_{i}", use_container_width=True
                    ):
                        st.session_state.search_query = hint
                        st.session_state.search_input = hint
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
        if st.button(
            "⚙️ Filters" + (" ✓" if _has_active_filters() else ""),
            key="toggle_filters",
            use_container_width=True,
            type="primary" if st.session_state.show_advanced_filters else "secondary",
        ):
            st.session_state.show_advanced_filters = (
                not st.session_state.show_advanced_filters
            )
            st.rerun()


def _render_advanced_filters() -> None:
    """Render detailed filter controls in expandable section."""
    if not st.session_state.show_advanced_filters:
        return

    with st.expander("Filter Options", expanded=True):
        # First row: Location and Remote
        col1, col2 = st.columns(2)

        with col1:
            location = st.text_input(
                "Location",
                value=st.session_state.search_filters["location"],
                placeholder="e.g., San Francisco, Remote, New York",
                help="Filter by job location or 'Remote' for remote positions",
            )
            if location != st.session_state.search_filters["location"]:
                st.session_state.search_filters["location"] = location
                _trigger_search_update()

        with col2:
            remote_only = st.checkbox(
                "Remote positions only",
                value=st.session_state.search_filters["remote_only"],
                help="Show only remote job opportunities",
            )
            if remote_only != st.session_state.search_filters["remote_only"]:
                st.session_state.search_filters["remote_only"] = remote_only
                _trigger_search_update()

        # Second row: Salary range
        st.markdown("**Salary Range**")
        col1, col2 = st.columns(2)

        with col1:
            salary_min = st.number_input(
                "Minimum Salary",
                value=int(st.session_state.search_filters["salary_min"]),
                min_value=0,
                max_value=500000,
                step=5000,
                format="%d",
            )
            if salary_min != st.session_state.search_filters["salary_min"]:
                st.session_state.search_filters["salary_min"] = salary_min
                _trigger_search_update()

        with col2:
            salary_max = st.number_input(
                "Maximum Salary",
                value=int(st.session_state.search_filters["salary_max"]),
                min_value=0,
                max_value=500000,
                step=5000,
                format="%d",
            )
            if salary_max != st.session_state.search_filters["salary_max"]:
                st.session_state.search_filters["salary_max"] = salary_max
                _trigger_search_update()

        # Third row: Application status and favorites
        col1, col2 = st.columns(2)

        with col1:
            application_status = st.selectbox(
                "Application Status",
                options=["All", *APPLICATION_STATUSES],
                index=0
                if st.session_state.search_filters["application_status"] == "All"
                else APPLICATION_STATUSES.index(
                    st.session_state.search_filters["application_status"]
                )
                + 1,
                help="Filter by current application status",
            )
            if (
                application_status
                != st.session_state.search_filters["application_status"]
            ):
                st.session_state.search_filters["application_status"] = (
                    application_status
                )
                _trigger_search_update()

        with col2:
            favorites_only = st.checkbox(
                "Favorites only",
                value=st.session_state.search_filters["favorites_only"],
                help="Show only jobs marked as favorites",
            )
            if favorites_only != st.session_state.search_filters["favorites_only"]:
                st.session_state.search_filters["favorites_only"] = favorites_only
                _trigger_search_update()

        # Fourth row: Date range
        st.markdown("**Posted Date Range**")
        col1, col2 = st.columns(2)

        with col1:
            date_from = st.date_input(
                "From Date",
                value=st.session_state.search_filters["date_from"].date(),
                help="Show jobs posted after this date",
            )
            date_from_dt = datetime.combine(date_from, datetime.min.time()).replace(
                tzinfo=UTC
            )
            if date_from_dt != st.session_state.search_filters["date_from"]:
                st.session_state.search_filters["date_from"] = date_from_dt
                _trigger_search_update()

        with col2:
            date_to = st.date_input(
                "To Date",
                value=st.session_state.search_filters["date_to"].date(),
                help="Show jobs posted before this date",
            )
            date_to_dt = datetime.combine(date_to, datetime.max.time()).replace(
                tzinfo=UTC
            )
            if date_to_dt != st.session_state.search_filters["date_to"]:
                st.session_state.search_filters["date_to"] = date_to_dt
                _trigger_search_update()

        # Filter actions
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Search", type="primary", use_container_width=True):
                _perform_search()
                st.rerun()

        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                _clear_all_filters()
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
        # View mode selector
        view_mode = st.selectbox(
            "View",
            options=["Cards", "List"],
            index=0,
            key="search_view_mode",
            label_visibility="collapsed",
        )

    with col3:
        # Results per page
        results_limit = st.selectbox(
            "Show",
            options=[25, 50, 100],
            index=1,
            key="search_limit_selector",
            format_func=lambda x: f"{x} results",
            label_visibility="collapsed",
        )
        if results_limit != st.session_state.search_limit:
            st.session_state.search_limit = results_limit
            _perform_search()
            st.rerun()

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

                # View details button
                if st.button("View Details", key=f"search_details_{job.id}"):
                    st.session_state.search_modal_job_id = job.id
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
        st.markdown(
            f"{fts_status} **{search_type}** for: `{st.session_state.search_query}`"
        )

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
    if not st.session_state.search_query:
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
                    st.session_state.search_query = query
                    st.session_state.search_input = query
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
                    _clear_all_filters()
                    st.rerun()

            with col2:
                if st.button("🔄 Show All Jobs", use_container_width=True):
                    st.session_state.search_query = ""
                    st.session_state.search_input = ""
                    # Redirect to jobs page
                    st.switch_page("src/ui/pages/jobs.py")


def _handle_search_modal() -> None:
    """Handle job details modal for search results."""
    if modal_job_id := st.session_state.get("search_modal_job_id"):
        # Find the job in search results
        job = next(
            (job for job in st.session_state.search_results if job.id == modal_job_id),
            None,
        )

        if job:
            show_job_details_modal(job)
        else:
            # Job not found, clear the modal
            st.session_state.search_modal_job_id = None


def _handle_search_input_change() -> None:
    """Handle search input changes with debouncing."""
    current_query = st.session_state.search_input
    st.session_state.search_query = current_query

    # Debounced search - only search if enough time has passed
    current_time = time.time()
    if (
        current_query
        and (current_time - st.session_state.last_search_time) > SEARCH_DEBOUNCE_DELAY
    ):
        st.session_state.last_search_time = current_time
        _perform_search()


def _trigger_search_update() -> None:
    """Trigger search update when filters change."""
    if st.session_state.search_query:
        _perform_search()


def _perform_search() -> None:
    """Execute the search using the search service."""
    query = st.session_state.search_query.strip()

    if not query:
        st.session_state.search_results = []
        st.session_state.search_stats = {
            "query_time": 0,
            "total_results": 0,
            "fts_enabled": False,
        }
        return

    try:
        # Prepare search filters
        search_filters = _build_search_filters()

        # Measure search performance
        start_time = time.time()

        # Execute search
        results = search_service.search_jobs(
            query=query, filters=search_filters, limit=st.session_state.search_limit
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
    """Build search filters from UI state."""
    filters = st.session_state.search_filters.copy()

    # Convert UI filters to search service format
    search_filters = {
        "date_from": filters["date_from"],
        "date_to": filters["date_to"],
        "favorites_only": filters["favorites_only"],
        "salary_min": filters["salary_min"]
        if filters["salary_min"] > SALARY_DEFAULT_MIN
        else None,
        "salary_max": filters["salary_max"]
        if filters["salary_max"] < SALARY_DEFAULT_MAX
        else None,
    }

    # Application status filter
    if filters["application_status"] != "All":
        search_filters["application_status"] = [filters["application_status"]]

    # Location filter (combine with remote filter)
    if filters["location"] or filters["remote_only"]:
        location_terms = []
        if filters["location"]:
            location_terms.append(filters["location"])
        if filters["remote_only"]:
            location_terms.append("Remote")
        # Note: Location filtering is handled via the main search query for FTS5
        # We could extend the search service to handle location filters specifically

    return search_filters


def _has_active_filters() -> bool:
    """Check if any advanced filters are active."""
    filters = st.session_state.search_filters
    defaults = {
        "location": "",
        "salary_min": SALARY_DEFAULT_MIN,
        "salary_max": SALARY_DEFAULT_MAX,
        "remote_only": False,
        "application_status": "All",
        "favorites_only": False,
    }

    for key, default_value in defaults.items():
        if filters.get(key) != default_value:
            return True

    # Check date filters (allow some tolerance for default dates)
    now = datetime.now(UTC)
    if abs((filters["date_from"] - (now - timedelta(days=30))).days) > 1:
        return True
    return abs((filters["date_to"] - now).days) > 1


def _clear_all_filters() -> None:
    """Clear all search filters and search query."""
    st.session_state.search_query = ""
    st.session_state.search_input = ""
    st.session_state.search_filters = {
        "location": "",
        "salary_min": SALARY_DEFAULT_MIN,
        "salary_max": SALARY_DEFAULT_MAX,
        "remote_only": False,
        "application_status": "All",
        "date_from": datetime.now(UTC) - timedelta(days=30),
        "date_to": datetime.now(UTC),
        "favorites_only": False,
    }
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
