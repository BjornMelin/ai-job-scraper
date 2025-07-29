"""Streamlit web application for the AI Job Scraper.

This module provides an interactive dashboard for managing job scraping,
viewing and filtering job postings, tracking applications, and managing
company configurations. Features include tabs for different job views,
search and filtering capabilities, inline editing, and CSV export.
"""

import asyncio
import logging
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy.orm import sessionmaker

from models import CompanySQL, JobSQL
from scraper import engine, scrape_all, update_db

logger = logging.getLogger(__name__)

Session = sessionmaker(bind=engine)

st.set_page_config(page_title="AI Job Tracker", layout="wide")

# CSS (with mobile fixes)
TECH_CSS = """
<style>
    [data-testid="stAppViewContainer"] { 
        background: linear-gradient(to bottom right, #0a192f, #1e3a8a); 
        color: #e2e8f0; 
    }
    [data-testid="stSidebar"] { 
        background-color: #1e293b; 
    }
    .stButton > button { 
        background-color: #3b82f6; 
        color: white; 
        border: none; 
        border-radius: 5px; 
        padding: 8px 16px; 
    }
    .stButton > button:hover { 
        background-color: #2563eb; 
    }
    .card { 
        background-color: #334155; 
        border-radius: 10px; 
        padding: 16px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
        margin-bottom: 16px; 
        overflow: hidden; 
        word-break: break-word; 
        width: 100%; 
    }
    .card:hover { 
        box-shadow: 0 6px 8px rgba(59, 130, 246, 0.5); 
    }
    .card-title { 
        color: #93c5fd; 
        font-size: 1.2em; 
        margin-bottom: 8px; 
    }
    .card-desc { 
        color: #cbd5e1; 
        font-size: 0.9em; 
    }
    /* Mobile */
    @media (max-width: 768px) {
        .card { 
            margin: 8px 0; 
        }
        [data-testid="column"] { 
            width: 100% !important; 
            flex: 1 1 100% !important; 
            min-width: 100% !important; 
        }
    }
</style>
"""
st.markdown(TECH_CSS, unsafe_allow_html=True)


def update_status(job_id: int, tab_key: str) -> None:
    """Update job application status in database.

    Callback function for Streamlit status selectbox changes.
    Updates the job's status field and triggers UI refresh.

    Args:
        job_id (int): Database ID of the job to update.
        tab_key (str): Tab identifier for session state management.
    """
    try:
        session = Session()
        job = session.query(JobSQL).filter_by(id=job_id).first()
        job.status = st.session_state[
            f"status_{job_id}_{tab_key}_{st.session_state[f'card_page_{tab_key}']}"
        ]
        session.commit()
    except Exception as e:
        logger.error(f"Update status failed: {e}")
    finally:
        session.close()
    st.experimental_rerun()


def update_notes(job_id: int, tab_key: str) -> None:
    """Update job notes in database.

    Callback function for Streamlit text area changes.
    Updates the job's notes field and triggers UI refresh.

    Args:
        job_id (int): Database ID of the job to update.
        tab_key (str): Tab identifier for session state management.
    """
    try:
        session = Session()
        job = session.query(JobSQL).filter_by(id=job_id).first()
        job.notes = st.session_state[
            f"notes_{job_id}_{tab_key}_{st.session_state[f'card_page_{tab_key}']}"
        ]
        session.commit()
    except Exception as e:
        logger.error(f"Update notes failed: {e}")
    finally:
        session.close()
    st.experimental_rerun()


def display_jobs(jobs: list[JobSQL], tab_key: str) -> None:
    """Display jobs in selected view format with search and filtering.

    Provides both list view (editable table) and card view (visual grid)
    with search functionality, sorting, pagination, and inline editing
    capabilities for job status, favorites, and notes.

    Args:
        jobs (list[JobSQL]): List of job objects from database query.
        tab_key (str): Unique identifier for the current tab to maintain
            separate UI state across different job views.
    """
    if not jobs:
        st.info("No jobs in this section.")
        return

    df = pd.DataFrame(
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

    # Per-tab search
    search_key = f"search_{tab_key}"
    search_term = st.text_input("Search in this tab", key=search_key)
    if search_term:
        df = df[
            df["Title"].str.contains(search_term, case=False)
            | df["Description"].str.contains(search_term, case=False)
        ]

    if st.session_state.view_mode == "List":
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

        if st.button("Save Changes", key=f"save_{tab_key}"):
            try:
                session = Session()
                for idx, row in edited_df.iterrows():
                    job = session.query(JobSQL).filter_by(id=row["id"]).first()
                    if job:
                        job.favorite = row["Favorite"]
                        job.status = row["Status"]
                        job.notes = row["Notes"]
                session.commit()
                st.success("Saved!")
            except Exception as e:
                st.error("Save failed.")
                logger.error(f"Save failed: {e}")
            finally:
                session.close()

        csv = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV 📥", csv, "jobs.csv", "text/csv", key=f"export_{tab_key}"
        )

    else:  # Card View
        # Sorting
        sort_options = {"Posted": "Posted", "Title": "Title", "Company": "Company"}
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sort_by = st.selectbox(
                "Sort By",
                list(sort_options.values()),
                index=list(sort_options.values()).index(st.session_state.sort_by),
                key=f"sort_by_{tab_key}",
            )
        with col2:
            st.session_state.sort_asc = st.checkbox(
                "Ascending", st.session_state.sort_asc, key=f"sort_asc_{tab_key}"
            )

        sort_key = [
            k for k, v in sort_options.items() if v == st.session_state.sort_by
        ][0]
        sorted_df = df.sort_values(by=sort_key, ascending=st.session_state.sort_asc)

        # Pagination
        cards_per_page = 9
        total_pages = (len(sorted_df) + cards_per_page - 1) // cards_per_page
        page_key = f"card_page_{tab_key}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 0
        st.session_state[page_key] = max(
            0, min(st.session_state[page_key], total_pages - 1)
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if (
                st.button("Previous Page", key=f"prev_{tab_key}")
                and st.session_state[page_key] > 0
            ):
                st.session_state[page_key] -= 1
                st.experimental_rerun()
        with col2:
            st.write(f"Page {st.session_state[page_key] + 1} of {total_pages}")
        with col3:
            if (
                st.button("Next Page", key=f"next_{tab_key}")
                and st.session_state[page_key] < total_pages - 1
            ):
                st.session_state[page_key] += 1
                st.experimental_rerun()

        start = st.session_state[page_key] * cards_per_page
        end = start + cards_per_page
        paginated_df = sorted_df.iloc[start:end]

        # Grid
        num_cols = 3
        cols = st.columns(num_cols)
        for i, row in paginated_df.iterrows():
            with cols[i % num_cols]:
                st.markdown(
                    f"""
                <div class="card">
                    <div class="card-title">{row["Company"]}: {row["Title"]}</div>
                    <div class="card-desc">{row["Description"][:150]}...</div>
                    <p>Location: {row["Location"]} | Posted: {row["Posted"]}</p>
                    <p>Status: {row["Status"]} | Favorite: {"⭐" if row["Favorite"] else ""}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.link_button("Apply", row["Link"] if row["Link"] else "#")
                if st.button(
                    "Toggle Favorite",
                    key=f"fav_{row['id']}_{tab_key}_{st.session_state[page_key]}",
                ):
                    try:
                        session = Session()
                        job = session.query(JobSQL).filter_by(id=row["id"]).first()
                        job.favorite = not job.favorite
                        session.commit()
                    except Exception as e:
                        logger.error(f"Toggle favorite failed: {e}")
                    finally:
                        session.close()
                    st.experimental_rerun()
                status_options = ["New", "Interested", "Applied", "Rejected"]
                st.selectbox(
                    "Status",
                    status_options,
                    index=status_options.index(row["Status"]),
                    key=f"status_{row['id']}_{tab_key}_{st.session_state[page_key]}",
                    on_change=update_status,
                    args=(row["id"], tab_key),
                )
                st.text_area(
                    "Notes",
                    row["Notes"],
                    key=f"notes_{row['id']}_{tab_key}_{st.session_state[page_key]}",
                    on_change=update_notes,
                    args=(row["id"], tab_key),
                )


st.title("AI Job Tracker 🚀")

# Session state
if "filters" not in st.session_state:
    st.session_state.filters = {
        "company": [],
        "keyword": "",
        "date_from": None,
        "date_to": None,
    }
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "List"
if "card_page" not in st.session_state:
    st.session_state.card_page = 0
if "sort_by" not in st.session_state:
    st.session_state.sort_by = "Posted"
if "sort_asc" not in st.session_state:
    st.session_state.sort_asc = False

# Sidebar
with st.sidebar:
    st.header("Global Filters 🔍")
    companies = ["All"] + sorted(
        set([j.company for j in Session().query(JobSQL.company).distinct()])
    )
    st.session_state.filters["company"] = st.multiselect(
        "Companies", companies, default=st.session_state.filters["company"]
    )
    st.session_state.filters["keyword"] = st.text_input(
        "Keyword Search", st.session_state.filters["keyword"]
    )
    st.session_state.filters["date_from"] = st.date_input(
        "Posted From", value=st.session_state.filters["date_from"]
    )
    st.session_state.filters["date_to"] = st.date_input(
        "Posted To", value=st.session_state.filters["date_to"]
    )

    st.header("View Mode 👁️")
    st.session_state.view_mode = st.radio("Select View", ["List", "Card"])

    st.header("Manage Companies 🏢")
    session = Session()
    comp_df = pd.DataFrame(
        [
            {"id": c.id, "Name": c.name, "URL": c.url, "Active": c.active}
            for c in session.query(CompanySQL).all()
        ]
    )
    edited_comp = st.data_editor(
        comp_df, column_config={"Active": st.column_config.CheckboxColumn("Active")}
    )
    if st.button("Save Companies"):
        try:
            for idx, row in edited_comp.iterrows():
                comp = session.query(CompanySQL).filter_by(id=row["id"]).first()
                if comp:
                    comp.active = row["Active"]
            new_name = st.text_input("Add New Company Name")
            new_url = st.text_input("Add New URL")
            if new_name and new_url:
                session.add(CompanySQL(name=new_name, url=new_url))
            session.commit()
            st.success("Saved!")
        except Exception as e:
            logger.error(f"Save companies failed: {e}")
            st.error("Save failed.")
        finally:
            session.close()

# Rescrape
if st.button("Rescrape Jobs"):
    with st.spinner("Scraping..."):
        try:
            jobs_df = asyncio.run(scrape_all())
            update_db(jobs_df)
            st.success("Updated!")
        except Exception as e:
            st.error("Scrape failed.")
            logger.error(f"UI scrape failed: {e}")

# Query jobs
session = Session()
try:
    query = session.query(JobSQL)
    if (
        "All" not in st.session_state.filters["company"]
        and st.session_state.filters["company"]
    ):
        query = query.filter(JobSQL.company.in_(st.session_state.filters["company"]))
    if st.session_state.filters["keyword"]:
        query = query.filter(
            JobSQL.title.ilike(f"%{st.session_state.filters['keyword']}%")
        )
    if st.session_state.filters["date_from"]:
        query = query.filter(
            JobSQL.posted_date
            >= datetime.combine(
                st.session_state.filters["date_from"], datetime.min.time()
            )
        )
    if st.session_state.filters["date_to"]:
        query = query.filter(
            JobSQL.posted_date
            <= datetime.combine(
                st.session_state.filters["date_to"], datetime.max.time()
            )
        )

    all_jobs = query.all()
except Exception as e:
    logger.error(f"Job query failed: {e}")
    all_jobs = []
finally:
    session.close()

if not all_jobs:
    st.info("No jobs.")
else:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["All Jobs 📋", "Favorites ⭐", "Applied ✅"])

    with tab1:
        display_jobs(all_jobs, "all")

    with tab2:
        favorites = [j for j in all_jobs if j.favorite]
        display_jobs(favorites, "favorites")

    with tab3:
        applied = [j for j in all_jobs if j.status == "Applied"]
        display_jobs(applied, "applied")

# Stats
st.header("Stats 📈")
st.write(f"Total Jobs: {len(all_jobs)}")
st.write(f"Favorites: {sum(1 for j in all_jobs if j.favorite)}")
st.write(f"Applied: {sum(1 for j in all_jobs if j.status == 'Applied')}")
