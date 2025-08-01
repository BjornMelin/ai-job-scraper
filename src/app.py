"""Streamlit dashboard for the AI Job Scraper.

This module provides an interactive UI for browsing jobs, applying filters,
toggling favorites and company status, adding notes, and triggering scrapes.
"""

import datetime
import os

from typing import Any

import pandas as pd
import sqlmodel
import streamlit as st

from .config import Settings
from .database import engine
from .models import CompanySQL, JobSQL
from .scraper import scrape_all

settings = Settings()


def load_css() -> None:
    """Load custom CSS for theming the dashboard."""
    css_path = os.path.join(os.path.dirname(__file__), "../static/css/main.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_jobs(favorite_only: bool = False) -> list[JobSQL]:
    """Load jobs from the database, optionally filtering for favorites.

    Args:
        favorite_only: If True, only load favorited jobs.

    Returns:
        List of JobSQL instances.
    """
    with sqlmodel.Session(engine) as session:
        query = sqlmodel.select(JobSQL).order_by(JobSQL.posted_date.desc())
        if favorite_only:
            query = query.where(JobSQL.favorite == True)  # noqa: E712
        return session.exec(query).all()


def load_companies() -> list[CompanySQL]:
    """Load all companies from the database.

    Returns:
        List of CompanySQL instances.
    """
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(CompanySQL).order_by(CompanySQL.name)).all()


def filter_jobs(
    jobs: list[JobSQL],
    company: str | None,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
    min_salary: int | None,
    max_salary: int | None,
) -> list[JobSQL]:
    """Filter jobs based on user-selected criteria.

    Args:
        jobs: List of jobs to filter.
        company: Selected company name or None for all.
        start_date: Start date for posted_date filter.
        end_date: End date for posted_date filter.
        min_salary: Minimum salary filter.
        max_salary: Maximum salary filter.

    Returns:
        Filtered list of JobSQL instances.
    """
    filtered = jobs
    if company and company != "All":
        filtered = [j for j in filtered if j.company == company]
    if start_date:
        filtered = [
            j for j in filtered if j.posted_date and j.posted_date.date() >= start_date
        ]
    if end_date:
        filtered = [
            j for j in filtered if j.posted_date and j.posted_date.date() <= end_date
        ]
    if min_salary is not None:
        filtered = [
            j for j in filtered if j.salary[0] is not None and j.salary[0] >= min_salary
        ]
    if max_salary is not None:
        filtered = [
            j for j in filtered if j.salary[1] is not None and j.salary[1] <= max_salary
        ]
    return filtered


def update_jobs_from_df(original_jobs: list[JobSQL], edited_df: pd.DataFrame) -> None:
    """Update job favorites and notes in the DB based on edited dataframe.

    Args:
        original_jobs: Original list of jobs for comparison.
        edited_df: Edited dataframe from st.data_editor.
    """
    job_map = {j.id: j for j in original_jobs if j.id is not None}
    with sqlmodel.Session(engine) as session:
        for _, row in edited_df.iterrows():
            job_id = row["id"]
            if job_id in job_map:
                job = session.get(JobSQL, job_id)
                if job:
                    job.favorite = row["favorite"]
                    job.notes = row["notes"]
                    session.add(job)
        session.commit()


def update_companies_from_df(
    original_companies: list[CompanySQL], edited_df: pd.DataFrame
) -> None:
    """Update company active status in the DB based on edited dataframe.

    Args:
        original_companies: Original list of companies for comparison.
        edited_df: Edited dataframe from st.data_editor.
    """
    company_map = {c.id: c for c in original_companies if c.id is not None}
    with sqlmodel.Session(engine) as session:
        for _, row in edited_df.iterrows():
            company_id = row["id"]
            if company_id in company_map:
                company = session.get(CompanySQL, company_id)
                if company:
                    company.active = row["active"]
                    session.add(company)
        session.commit()


def display_paginated_editor(
    data: list[Any],
    key: str,
    column_config: dict[str, Any],
    num_rows: str = "dynamic",
    page_size: int = 10,
) -> pd.DataFrame:
    """Display a paginated data editor.

    Args:
        data: List of data items.
        key: Unique key for session state.
        column_config: Configuration for data editor columns.
        num_rows: Mode for number of rows.
        page_size: Number of items per page.

    Returns:
        Edited dataframe for the current page.
    """
    if f"{key}_page" not in st.session_state:
        st.session_state[f"{key}_page"] = 0

    start = st.session_state[f"{key}_page"] * page_size
    end = start + page_size
    page_data = data[start:end]

    df = pd.DataFrame([item.model_dump() for item in page_data])
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        num_rows=num_rows,
        use_container_width=True,
        hide_index=False,
        key=f"{key}_editor",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous", disabled=start == 0, key=f"{key}_prev"):
            st.session_state[f"{key}_page"] -= 1
    with col3:
        if st.button("Next", disabled=end >= len(data), key=f"{key}_next"):
            st.session_state[f"{key}_page"] += 1

    return edited_df


def main() -> None:
    """Run the Streamlit dashboard."""
    load_css()
    st.title("AI Job Scraper Dashboard")

    if st.button("Scrape Now"):
        with st.spinner("Scraping jobs..."):
            scrape_all()
        st.success("Scraping completed!")
        st.rerun()

    tab_all, tab_fav, tab_comp = st.tabs(["All Jobs", "Favorites", "Companies"])

    companies = load_companies()
    company_names = ["All"] + [c.name for c in companies]

    with tab_all:
        jobs = load_jobs()
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_company = st.selectbox("Company", company_names)
        with col2:
            start_date = st.date_input("Start Date")
        with col3:
            end_date = st.date_input("End Date")
        col4, col5 = st.columns(2)
        with col4:
            min_salary = st.number_input("Min Salary", min_value=0)
        with col5:
            max_salary = st.number_input("Max Salary", min_value=0)

        filtered_jobs = filter_jobs(
            jobs,
            selected_company,
            start_date,
            end_date,
            min_salary if min_salary > 0 else None,
            max_salary if max_salary > 0 else None,
        )

        st.subheader("Jobs")
        column_config = {
            "id": None,
            "company": st.column_config.TextColumn("Company", disabled=True),
            "title": st.column_config.TextColumn("Title", disabled=True),
            "description": st.column_config.TextColumn("Description", disabled=True),
            "link": st.column_config.LinkColumn("Link", disabled=True),
            "location": st.column_config.TextColumn("Location", disabled=True),
            "posted_date": st.column_config.DateColumn("Posted Date", disabled=True),
            "salary": st.column_config.TextColumn("Salary", disabled=True),
            "favorite": st.column_config.CheckboxColumn("Favorite"),
            "notes": st.column_config.TextColumn("Notes"),
        }
        edited_df = display_paginated_editor(filtered_jobs, "all_jobs", column_config)
        if st.button("Save Changes", key="save_all"):
            update_jobs_from_df(filtered_jobs, edited_df)
            st.success("Changes saved!")
            st.rerun()

    with tab_fav:
        fav_jobs = load_jobs(favorite_only=True)
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_company_fav = st.selectbox(
                "Company", company_names, key="fav_comp"
            )
        with col2:
            start_date_fav = st.date_input("Start Date", key="fav_start")
        with col3:
            end_date_fav = st.date_input("End Date", key="fav_end")
        col4, col5 = st.columns(2)
        with col4:
            min_salary_fav = st.number_input("Min Salary", min_value=0, key="fav_min")
        with col5:
            max_salary_fav = st.number_input("Max Salary", min_value=0, key="fav_max")

        filtered_fav = filter_jobs(
            fav_jobs,
            selected_company_fav,
            start_date_fav,
            end_date_fav,
            min_salary_fav if min_salary_fav > 0 else None,
            max_salary_fav if max_salary_fav > 0 else None,
        )

        st.subheader("Favorite Jobs")
        edited_df_fav = display_paginated_editor(
            filtered_fav, "fav_jobs", column_config
        )
        if st.button("Save Changes", key="save_fav"):
            update_jobs_from_df(filtered_fav, edited_df_fav)
            st.success("Changes saved!")
            st.rerun()

    with tab_comp:
        st.subheader("Manage Companies")
        column_config_comp = {
            "id": None,
            "name": st.column_config.TextColumn("Name", disabled=True),
            "url": st.column_config.LinkColumn("URL", disabled=True),
            "active": st.column_config.CheckboxColumn("Active"),
        }
        edited_df_comp = display_paginated_editor(
            companies, "companies", column_config_comp, num_rows="fixed"
        )
        if st.button("Save Changes", key="save_comp"):
            update_companies_from_df(companies, edited_df_comp)
            st.success("Changes saved!")
            st.rerun()


if __name__ == "__main__":
    main()
