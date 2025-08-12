"""Pagination component for job listings.

This component provides a reusable pagination interface with:
- Previous/Next navigation
- Page number display
- Total count information
- Load more functionality for incremental loading
- Performance optimizations with state management
"""

import logging

from collections.abc import Callable
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def render_pagination_controls(
    pagination_info: dict[str, Any],
    on_page_change: Callable[[int], None] | None = None,
    key_prefix: str = "",
) -> int:
    """Render pagination controls with navigation buttons.

    Args:
        pagination_info: Dictionary containing pagination metadata
        on_page_change: Callback function when page changes
        key_prefix: Unique prefix for widget keys

    Returns:
        New page number if changed, current page otherwise
    """
    current_page = pagination_info.get("page", 1)
    total_pages = pagination_info.get("total_pages", 1)
    total_count = pagination_info.get("total_count", 0)
    has_previous = pagination_info.get("has_previous", False)
    has_next = pagination_info.get("has_next", False)

    if total_pages <= 1:
        return current_page

    # Create pagination layout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    new_page = current_page

    with col1:
        if st.button(
            "⏮️ First",
            disabled=not has_previous,
            key=f"{key_prefix}_first",
            help="Go to first page",
        ):
            new_page = 1

    with col2:
        if st.button(
            "◀️ Previous",
            disabled=not has_previous,
            key=f"{key_prefix}_prev",
            help="Go to previous page",
        ):
            new_page = max(1, current_page - 1)

    with col3:
        # Page info display
        st.markdown(
            f"""
            <div style='text-align: center; padding: 8px;'>
                <strong>Page {current_page} of {total_pages}</strong><br>
                <small>{total_count:,} total jobs</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        if st.button(
            "Next ▶️",
            disabled=not has_next,
            key=f"{key_prefix}_next",
            help="Go to next page",
        ):
            new_page = min(total_pages, current_page + 1)

    with col5:
        if st.button(
            "Last ⏭️",
            disabled=not has_next,
            key=f"{key_prefix}_last",
            help="Go to last page",
        ):
            new_page = total_pages

    # Call callback if page changed
    if new_page != current_page and on_page_change:
        on_page_change(new_page)

    return new_page


def render_load_more_button(
    pagination_info: dict[str, Any],
    current_jobs_count: int,
    on_load_more: Callable[[], None] | None = None,
    key_prefix: str = "",
) -> bool:
    """Render a 'Load More' button for incremental loading.

    Args:
        pagination_info: Dictionary containing pagination metadata
        current_jobs_count: Number of jobs currently displayed
        on_load_more: Callback function when load more is clicked
        key_prefix: Unique prefix for widget keys

    Returns:
        True if load more was clicked, False otherwise
    """
    has_next = pagination_info.get("has_next", False)
    total_count = pagination_info.get("total_count", 0)

    if not has_next:
        return False

    remaining_count = max(0, total_count - current_jobs_count)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            f"📄 Load More ({remaining_count:,} remaining)",
            key=f"{key_prefix}_load_more",
            use_container_width=True,
            help="Load the next batch of jobs",
        ):
            if on_load_more:
                on_load_more()
            return True

    return False


def render_page_size_selector(
    current_page_size: int = 50,
    options: list[int] = [25, 50, 100, 200],
    key_prefix: str = "",
) -> int:
    """Render page size selector dropdown.

    Args:
        current_page_size: Currently selected page size
        options: List of page size options
        key_prefix: Unique prefix for widget keys

    Returns:
        Selected page size
    """
    return st.selectbox(
        "Jobs per page:",
        options=options,
        index=options.index(current_page_size) if current_page_size in options else 1,
        key=f"{key_prefix}_page_size",
        help="Choose how many jobs to display per page",
    )


def render_pagination_info(pagination_info: dict[str, Any]) -> None:
    """Render pagination information summary.

    Args:
        pagination_info: Dictionary containing pagination metadata
    """
    current_page = pagination_info.get("page", 1)
    page_size = pagination_info.get("page_size", 50)
    total_count = pagination_info.get("total_count", 0)
    total_pages = pagination_info.get("total_pages", 1)

    # Calculate range of items being shown
    start_item = ((current_page - 1) * page_size) + 1
    end_item = min(current_page * page_size, total_count)

    if total_count == 0:
        st.info("No jobs found matching your criteria.")
        return

    # Display range information
    st.markdown(
        f"""
        <div style='text-align: center; padding: 10px; background-color: var(--background-color); 
                    border-radius: 5px; margin: 10px 0;'>
            <strong>Showing jobs {start_item:,} - {end_item:,} of {total_count:,}</strong>
            <br>
            <small>Page {current_page} of {total_pages}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_pagination_session_key(tab_key: str, filter_hash: str = "") -> str:
    """Generate unique session key for pagination state.

    Args:
        tab_key: Tab identifier
        filter_hash: Hash of current filters

    Returns:
        Unique session key for pagination
    """
    return f"pagination_{tab_key}_{filter_hash}"


def initialize_pagination_state(tab_key: str, default_page_size: int = 50) -> None:
    """Initialize pagination state in session.

    Args:
        tab_key: Tab identifier
        default_page_size: Default number of items per page
    """
    page_key = f"{tab_key}_current_page"
    size_key = f"{tab_key}_page_size"

    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    if size_key not in st.session_state:
        st.session_state[size_key] = default_page_size


def get_pagination_state(tab_key: str) -> tuple[int, int]:
    """Get current pagination state.

    Args:
        tab_key: Tab identifier

    Returns:
        Tuple of (current_page, page_size)
    """
    page_key = f"{tab_key}_current_page"
    size_key = f"{tab_key}_page_size"

    return (st.session_state.get(page_key, 1), st.session_state.get(size_key, 50))


def set_pagination_state(
    tab_key: str, page: int, page_size: int | None = None
) -> None:
    """Set pagination state in session.

    Args:
        tab_key: Tab identifier
        page: Page number to set
        page_size: Page size to set (optional)
    """
    page_key = f"{tab_key}_current_page"
    size_key = f"{tab_key}_page_size"

    st.session_state[page_key] = max(1, page)

    if page_size is not None:
        st.session_state[size_key] = page_size


def reset_pagination_to_first_page(tab_key: str) -> None:
    """Reset pagination to first page (useful when filters change).

    Args:
        tab_key: Tab identifier
    """
    page_key = f"{tab_key}_current_page"
    if page_key in st.session_state:
        st.session_state[page_key] = 1
