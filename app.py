"""Streamlit web application for the AI Job Scraper.

This module provides an interactive dashboard for managing job scraping,
viewing and filtering job postings, tracking applications, and managing
company configurations. Features include tabs for different job views,
search and filtering capabilities, inline editing, and CSV export.
"""

import asyncio
import html
import logging

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.database import SessionLocal
from src.models import CompanySQL, JobSQL
from scraper import scrape_all, update_db
from utils.css_loader import load_css

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Job Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered job tracker for managing your job search efficiently."
    },
)

# Load external CSS
load_css("static/css/main.css")


def update_status(job_id: int, tab_key: str) -> None:
    """Update job application status in database.

    Callback function for Streamlit status selectbox changes.
    Updates the job's status field and triggers UI refresh.

    Args:
        job_id (int): Database ID of the job to update.
        tab_key (str): Tab identifier for session state management.

    """
    try:
        session = SessionLocal()
        job = session.query(JobSQL).filter_by(id=job_id).first()
        job.status = st.session_state[
            f"status_{job_id}_{tab_key}_{st.session_state[f'card_page_{tab_key}']}"
        ]
        session.commit()
    except Exception as e:
        logger.error(f"Update status failed: {e}")
    finally:
        session.close()
    st.rerun()


def update_notes(job_id: int, tab_key: str) -> None:
    """Update job notes in database.

    Callback function for Streamlit text area changes.
    Updates the job's notes field and triggers UI refresh.

    Args:
        job_id (int): Database ID of the job to update.
        tab_key (str): Tab identifier for session state management.

    """
    try:
        session = SessionLocal()
        job = session.query(JobSQL).filter_by(id=job_id).first()
        job.notes = st.session_state[
            f"notes_{job_id}_{tab_key}_{st.session_state[f'card_page_{tab_key}']}"
        ]
        session.commit()
    except Exception as e:
        logger.error(f"Update notes failed: {e}")
    finally:
        session.close()
    st.rerun()


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

    # Apply search filter
    if search_term:
        df = df[
            df["Title"].str.contains(search_term, case=False, na=False)
            | df["Description"].str.contains(search_term, case=False, na=False)
            | df["Company"].str.contains(search_term, case=False, na=False)
        ]

        with search_col2:
            st.metric(
                "Results",
                len(df),
                delta=f"-{len(jobs) - len(df)}" if len(df) < len(jobs) else None,
            )

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
                session = SessionLocal()
                for _, row in edited_df.iterrows():
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

        sort_key = next(
            (k for k, v in sort_options.items() if v == st.session_state.sort_by),
            "Posted",  # Default fallback if not found
        )
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
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state[page_key] + 1} of {total_pages}")
        with col3:
            if (
                st.button("Next Page", key=f"next_{tab_key}")
                and st.session_state[page_key] < total_pages - 1
            ):
                st.session_state[page_key] += 1
                st.rerun()

        start = st.session_state[page_key] * cards_per_page
        end = start + cards_per_page
        paginated_df = sorted_df.iloc[start:end]

        # Grid
        num_cols = 3
        cols = st.columns(num_cols)
        for i, row in paginated_df.iterrows():
            with cols[i % num_cols]:
                # Format posted date
                posted_date = row["Posted"]
                if pd.notna(posted_date):
                    if isinstance(posted_date, str):
                        posted_date = datetime.strptime(posted_date, "%Y-%m-%d")
                    days_ago = (datetime.now() - posted_date).days
                    if days_ago == 0:
                        time_str = "Today"
                    elif days_ago == 1:
                        time_str = "Yesterday"
                    else:
                        time_str = f"{days_ago} days ago"
                else:
                    time_str = ""

                # Status badge color
                status_class = f"status-{row['Status'].lower()}"

                st.markdown(
                    f"""
                <div class="card">
                    <div class="card-title">{html.escape(str(row["Title"]))}</div>
                    <div class="card-meta">
                        <strong>{html.escape(str(row["Company"]))}</strong> • 
                        {html.escape(str(row["Location"]))} • 
                        {time_str}
                    </div>
                    <div class="card-desc">{
                        html.escape(str(row["Description"])[:200])
                    }...</div>
                    <div class="card-footer">
                        <span class="status-badge {status_class}">{
                        html.escape(str(row["Status"]))
                    }</span>
                        {
                        "<span style='color: #f59e0b; font-size: 1.2em;'>⭐</span>"
                        if row["Favorite"]
                        else ""
                    }
                    </div>
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
                        session = SessionLocal()
                        job = session.query(JobSQL).filter_by(id=row["id"]).first()
                        job.favorite = not job.favorite
                        session.commit()
                    except Exception as e:
                        logger.error(f"Toggle favorite failed: {e}")
                    finally:
                        session.close()
                    st.rerun()
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


# Header with improved styling
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

# Session state initialization with better defaults
if "filters" not in st.session_state:
    st.session_state.filters = {
        "company": [],
        "keyword": "",
        "date_from": datetime.now() - timedelta(days=30),  # Default to last 30 days
        "date_to": datetime.now(),
    }
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Card"  # Default to more visual card view
if "card_page" not in st.session_state:
    st.session_state.card_page = 0
if "sort_by" not in st.session_state:
    st.session_state.sort_by = "Posted"
if "sort_asc" not in st.session_state:
    st.session_state.sort_asc = False
if "last_scrape" not in st.session_state:
    st.session_state.last_scrape = None

# Sidebar with improved organization
with st.sidebar:
    # Search and Filter Section
    st.markdown("### 🔍 Search & Filter")
    with st.container():
        # Get company list
        session = SessionLocal()
        try:
            companies = sorted(
                {j.company for j in session.query(JobSQL.company).distinct()}
            )
        except Exception:
            companies = []
        finally:
            session.close()

        # Company filter with better default
        selected_companies = st.multiselect(
            "Filter by Company",
            options=companies,
            default=st.session_state.filters["company"]
            if st.session_state.filters["company"]
            else None,
            placeholder="All companies",
            help="Select one or more companies to filter jobs",
        )
        st.session_state.filters["company"] = selected_companies

        # Keyword search with placeholder
        st.session_state.filters["keyword"] = st.text_input(
            "Search Keywords",
            value=st.session_state.filters["keyword"],
            placeholder="e.g., Python, Machine Learning, Remote",
            help="Search in job titles and descriptions",
        )

        # Date range with column layout
        st.markdown("**Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.filters["date_from"] = st.date_input(
                "From",
                value=st.session_state.filters["date_from"],
                help="Show jobs posted after this date",
            )
        with col2:
            st.session_state.filters["date_to"] = st.date_input(
                "To",
                value=st.session_state.filters["date_to"],
                help="Show jobs posted before this date",
            )

        # Clear filters button
        if st.button("Clear All Filters", use_container_width=True):
            st.session_state.filters = {
                "company": [],
                "keyword": "",
                "date_from": datetime.now() - timedelta(days=30),
                "date_to": datetime.now(),
            }
            st.rerun()

    st.divider()

    # View Settings Section
    st.markdown("### 👁️ View Settings")
    view_col1, view_col2 = st.columns(2)
    with view_col1:
        if st.button(
            "📋 List View",
            use_container_width=True,
            type="secondary" if st.session_state.view_mode == "Card" else "primary",
        ):
            st.session_state.view_mode = "List"
            st.rerun()
    with view_col2:
        if st.button(
            "🎴 Card View",
            use_container_width=True,
            type="secondary" if st.session_state.view_mode == "List" else "primary",
        ):
            st.session_state.view_mode = "Card"
            st.rerun()

    st.divider()

    # Company Management Section
    with st.expander("🏢 Manage Companies", expanded=False):
        session = SessionLocal()
        try:
            comp_df = pd.DataFrame(
                [
                    {"id": c.id, "Name": c.name, "URL": c.url, "Active": c.active}
                    for c in session.query(CompanySQL).all()
                ]
            )

            if not comp_df.empty:
                st.markdown("**Existing Companies**")
                edited_comp = st.data_editor(
                    comp_df,
                    column_config={
                        "Active": st.column_config.CheckboxColumn(
                            "Active", help="Toggle to enable/disable scraping"
                        ),
                        "URL": st.column_config.LinkColumn(
                            "URL", help="Company careers page URL"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                if st.button(
                    "💾 Save Changes", use_container_width=True, type="primary"
                ):
                    try:
                        for _, row in edited_comp.iterrows():
                            comp = (
                                session.query(CompanySQL)
                                .filter_by(id=row["id"])
                                .first()
                            )
                            if comp:
                                comp.active = row["Active"]
                        session.commit()
                        st.success("✅ Company settings saved!")
                    except Exception as e:
                        logger.error(f"Save companies failed: {e}")
                        st.error("❌ Save failed. Please try again.")

            # Add new company section
            st.markdown("**Add New Company**")
            with st.form("add_company_form", clear_on_submit=True):
                new_name = st.text_input(
                    "Company Name",
                    placeholder="e.g., OpenAI",
                    help="Enter the company name",
                )
                new_url = st.text_input(
                    "Careers Page URL",
                    placeholder="e.g., https://openai.com/careers",
                    help="Enter the URL of the company's careers page",
                )

                if st.form_submit_button(
                    "+ Add Company", use_container_width=True, type="primary"
                ):
                    if new_name and new_url:
                        if not new_url.startswith(("http://", "https://")):
                            st.error("URL must start with http:// or https://")
                        else:
                            try:
                                session.add(
                                    CompanySQL(name=new_name, url=new_url, active=True)
                                )
                                session.commit()
                                st.success(f"✅ Added {new_name} successfully!")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Add company failed: {e}")
                                st.error(
                                    "❌ Failed to add company. "
                                    "Name might already exist."
                                )
                    else:
                        st.error("Please fill in both fields")
        finally:
            session.close()

# Main content area
main_container = st.container()

# Action bar with improved styling
with main_container:
    action_col1, action_col2, action_col3 = st.columns([2, 2, 1])

    with action_col1:
        if st.button(
            "🔄 Refresh Jobs",
            use_container_width=True,
            type="primary",
            help="Scrape latest job postings from all active companies",
        ):
            with st.spinner("🔍 Searching for new jobs..."):
                try:
                    # Create new event loop for Streamlit environment
                    import nest_asyncio

                    nest_asyncio.apply()

                    # Try to get or create new event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            raise RuntimeError("Event loop is closed")
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    jobs_df = loop.run_until_complete(scrape_all())
                    update_db(jobs_df)
                    st.session_state.last_scrape = datetime.now()
                    st.success(
                        f"✅ Success! Found {len(jobs_df)} jobs from active companies."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Scrape failed: {e!s}")
                    logger.error(f"UI scrape failed: {e}")

    with action_col2:
        if st.session_state.last_scrape:
            time_diff = datetime.now() - st.session_state.last_scrape
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

    with action_col3:
        # Quick stats
        session = SessionLocal()
        try:
            active_companies = session.query(CompanySQL).filter_by(active=True).count()
            st.metric("Active Sources", active_companies)
        finally:
            session.close()

# Query jobs
session = SessionLocal()
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
    st.info("🔍 No jobs found. Try adjusting your filters or refreshing the job list.")
else:
    # Enhanced tabs with counts
    favorites_count = sum(1 for j in all_jobs if j.favorite)
    applied_count = sum(1 for j in all_jobs if j.status == "Applied")

    tab1, tab2, tab3 = st.tabs(
        [
            f"All Jobs 📋 ({len(all_jobs)})",
            f"Favorites ⭐ ({favorites_count})",
            f"Applied ✅ ({applied_count})",
        ]
    )

    with tab1:
        display_jobs(all_jobs, "all")

    with tab2:
        favorites = [j for j in all_jobs if j.favorite]
        if not favorites:
            st.info(
                "💡 No favorite jobs yet. Star jobs you're interested in "
                "to see them here!"
            )
        else:
            display_jobs(favorites, "favorites")

    with tab3:
        applied = [j for j in all_jobs if j.status == "Applied"]
        if not applied:
            st.info(
                "🚀 No applications yet. Update job status to 'Applied' "
                "to track them here!"
            )
        else:
            display_jobs(applied, "applied")

# Enhanced Statistics Dashboard
st.markdown("---")
st.markdown("### 📊 Dashboard")

# Calculate statistics
total_jobs = len(all_jobs)
favorites = sum(1 for j in all_jobs if j.favorite)
applied = sum(1 for j in all_jobs if j.status == "Applied")
interested = sum(1 for j in all_jobs if j.status == "Interested")
new_jobs = sum(1 for j in all_jobs if j.status == "New")
rejected = sum(1 for j in all_jobs if j.status == "Rejected")

# Create metric cards with improved styling
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

# Add progress bars for visual representation
if total_jobs > 0:
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
        for i, (status, count, pct) in enumerate(
            zip(
                progress_data["Status"],
                progress_data["Count"],
                progress_data["Percentage"],
                strict=False,
            )
        ):
            color = ["primary", "warning", "success", "danger"][i]
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
