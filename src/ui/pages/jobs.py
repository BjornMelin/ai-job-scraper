"""Jobs workspace."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import streamlit as st

from src.models.job_models import ApplicationStage
from src.services.company_service import company_service
from src.services.job_service import JobService
from src.services.search_service import search_service
from src.ui.design import WORKFLOW_STAGES, empty_state, page_intro, relative_time

if TYPE_CHECKING:
    from src.schemas import Job

logger = logging.getLogger(__name__)


def _job_filters(
    stage: ApplicationStage,
    companies: list[str],
    starred_only: bool,
) -> dict[str, object]:
    return {
        "application_status": [stage],
        "company": companies,
        "favorites_only": starred_only,
    }


def _count_jobs(query: str, filters: dict[str, object]) -> int:
    if query.strip():
        return search_service.count_jobs(query.strip(), filters)
    return JobService.count_filtered_jobs(filters)


def _load_jobs(
    query: str,
    filters: dict[str, object],
    *,
    limit: int,
    offset: int,
) -> list[Job]:
    if query.strip():
        return search_service.search_jobs(
            query.strip(),
            filters,
            limit=limit,
            offset=offset,
        )
    return JobService.get_filtered_jobs(filters, limit=limit, offset=offset)


def _save_job(job: Job, stage: ApplicationStage, starred: bool, notes: str) -> None:
    if job.id is None:
        raise ValueError("Cannot update a job without an ID")
    JobService.bulk_update_jobs(
        [
            {
                "id": job.id,
                "application_status": stage,
                "favorite": starred,
                "notes": notes,
            }
        ]
    )


def _render_job(job: Job) -> None:
    with st.container(border=True, key=f"job-card-{job.id}"):
        st.subheader(job.title, anchor=False)
        st.markdown(f"**{job.company}** · {job.location or 'Location not listed'}")
        details = [job.salary_range_display]
        if job.posted_date:
            details.append(f"posted {relative_time(job.posted_date)}")
        st.caption(" · ".join(details))

        with st.expander("Review and update"):
            with st.form(f"job-form-{job.id}"):
                stage = st.selectbox(
                    "Stage",
                    WORKFLOW_STAGES,
                    index=WORKFLOW_STAGES.index(job.application_status),
                    format_func=lambda value: value.value,
                )
                starred = st.checkbox("Starred", value=job.favorite)
                notes = st.text_area(
                    "Notes",
                    value=job.notes,
                    placeholder="Record context for your next decision.",
                )
                if st.form_submit_button("Save changes", type="primary"):
                    try:
                        _save_job(job, stage, starred, notes)
                    except Exception:
                        logger.exception("Could not update job %s", job.id)
                        st.error("This job could not be updated. Try again.")
                    else:
                        st.session_state["jobs-notice"] = "Job updated."
                        st.rerun()

            if job.link.startswith("legacy://recovered-job/"):
                st.caption(
                    "Original posting unavailable; this legacy record was preserved "
                    "during migration."
                )
            else:
                st.markdown(f"[Open original posting]({job.link})")
            if job.description:
                st.markdown("#### Description")
                st.markdown(job.description)


def render_jobs_page() -> None:
    """Render filters, stage counts, and job review controls."""
    page_intro(
        "Working set",
        "Jobs",
        "Move each opportunity through one clear workflow, from first review to a final decision.",
    )
    if notice := st.session_state.pop("jobs-notice", None):
        st.success(notice)

    try:
        counts = JobService.get_job_counts_by_status()
        companies = company_service.get_all_companies()
    except Exception:
        logger.exception("Could not load job filters")
        st.error("Jobs could not be loaded. Check the database and try again.")
        return

    stage = st.radio(
        "Workflow stage",
        WORKFLOW_STAGES,
        index=0,
        format_func=lambda value: f"{value.value}  {counts.get(value, 0)}",
        key="jobs-stage",
        horizontal=True,
    )

    filter_left, filter_right = st.columns([3, 2], gap="medium")
    with filter_left:
        query = st.text_input(
            "Search jobs",
            placeholder="Role, company, location, or keyword",
            key="jobs-query",
        )
    with filter_right:
        selected_companies = st.multiselect(
            "Companies",
            [company.name for company in companies],
            placeholder="All companies",
            key="jobs-companies",
        )
    starred_only = st.checkbox(
        "Starred only",
        key="jobs-starred",
        help="Show only jobs you have marked for quick reference.",
    )

    filters = _job_filters(stage, selected_companies, starred_only)
    try:
        total_jobs = _count_jobs(query, filters)
    except Exception:
        logger.exception("Could not query jobs")
        st.error(
            "The current job view could not be loaded. Adjust the filters or retry."
        )
        return

    if total_jobs == 0:
        empty_state(
            "Nothing here yet",
            "Run a saved search to collect jobs, or choose another workflow stage.",
        )
        return

    pagination_left, pagination_right = st.columns([1, 1], gap="medium")
    with pagination_left:
        jobs_per_page = st.selectbox(
            "Jobs per page",
            (10, 25, 50),
            index=1,
            key="jobs-per-page",
        )
    page_count = math.ceil(total_jobs / jobs_per_page)
    with pagination_right:
        page = st.selectbox(
            "Page",
            range(1, page_count + 1),
            format_func=lambda value: f"{value} of {page_count}",
            key="jobs-page",
        )

    start = (page - 1) * jobs_per_page
    end = min(start + jobs_per_page, total_jobs)
    try:
        jobs = _load_jobs(
            query,
            filters,
            limit=jobs_per_page,
            offset=start,
        )
    except Exception:
        logger.exception("Could not load the requested job page")
        st.error("The requested job page could not be loaded. Try again.")
        return

    st.caption(f"Showing {start + 1} to {end} of {total_jobs} jobs")
    for job in jobs:
        _render_job(job)


render_jobs_page()
