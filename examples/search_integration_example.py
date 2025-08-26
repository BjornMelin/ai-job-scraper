"""Search Integration Example - FTS5 Full-Text Search Demo.

This example demonstrates how to integrate the advanced search component into
Streamlit pages. It showcases the library-first SQLite FTS5 implementation with
Porter stemming and BM25 ranking.

Usage:
    uv run streamlit run examples/search_integration_example.py

Features Demonstrated:
- SQLite FTS5 full-text search with Porter stemming
- BM25 relevance ranking for best matches first
- Real-time search with debouncing
- Advanced filtering (location, salary, status, date range)
- Search syntax examples (exact phrases, Boolean operators, wildcards)
- Performance metrics and search statistics
- Integration with existing job card components

Technical Implementation:
- Uses src.services.search_service.JobSearchService (sqlite-utils + FTS5)
- Integrates src.ui.components.search_bar.render_job_search()
- Library-first architecture using SQLite FTS5 with minimal custom code
"""

import streamlit as st

from src.ui.components.search_bar import render_job_search
from src.ui.state.session_state import init_session_state


def main():
    """Example main function showing search component integration."""
    # Initialize session state (this should be done early in your app)
    init_session_state()

    # Page configuration
    st.set_page_config(
        page_title="Job Search Example",
        page_icon="🔍",
        layout="wide",
    )

    # Page header
    st.title("🔍 Advanced Job Search")
    st.markdown(
        "Powered by SQLite FTS5 full-text search with Porter stemming and "
        "BM25 relevance ranking."
    )

    # Main search component
    render_job_search()

    # Additional information
    with st.expander("About This Search", expanded=False):
        st.markdown("""
        **Features:**
        - 🚀 **FTS5 Full-Text Search**: Advanced SQLite search with Porter stemming
        - 🎯 **Relevance Ranking**: BM25 algorithm for best matches first
        - ⚡ **Real-time Search**: Debounced search as you type
        - 🎛️ **Advanced Filters**: Location, salary, status, date range, and more
        - 📊 **Performance Metrics**: Query timing and result counts
        - 📱 **Mobile Responsive**: Works great on all devices

        **Search Syntax:**
        - `"exact phrase"` - Match exact phrases
        - `term1 AND term2` - Both terms required
        - `term1 OR term2` - Either term matches
        - `term1 NOT term2` - Exclude term2
        - `term*` - Wildcard matching (e.g., `develop*` matches `developer`)

        **Integration:**
        ```python
        from src.ui.components.search_bar import render_job_search

        # Simply call this function in your Streamlit page
        render_job_search()
        ```
        """)


if __name__ == "__main__":
    main()
