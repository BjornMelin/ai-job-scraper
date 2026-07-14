"""Streamlit entry point for Job Tracker."""

import streamlit as st

from src.ui.design import apply_design


def main() -> None:
    """Configure the app shell and run the selected page."""
    st.set_page_config(
        page_title="Job Tracker",
        page_icon=":material/work:",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "A focused, local-first workspace for finding and tracking jobs.",
        },
    )
    apply_design()

    jobs_page = st.Page(
        "ui/pages/jobs.py",
        title="Jobs",
        icon=":material/work:",
        url_path="jobs",
    )
    pages = (
        jobs_page,
        st.Page(
            "ui/pages/searches.py",
            title="Searches",
            icon=":material/search:",
            url_path="searches",
        ),
        st.Page(
            "ui/pages/insights.py",
            title="Insights",
            icon=":material/insights:",
            url_path="insights",
        ),
    )
    home_page = st.Page(
        lambda: st.switch_page(jobs_page),
        title="Jobs",
        default=True,
        visibility="hidden",
    )
    page = st.navigation([home_page, *pages], position="hidden")

    with st.container(
        key="primary-navigation",
        horizontal=True,
        vertical_alignment="center",
        gap="xsmall",
    ):
        for destination in pages:
            st.page_link(destination)

    page.run()


if __name__ == "__main__":
    main()
