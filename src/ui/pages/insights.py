"""Read-only job-search insights."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.services.analytics_service import AnalyticsService
from src.services.company_service import company_service
from src.services.job_service import JobService
from src.services.saved_search_service import saved_search_service
from src.ui.design import WORKFLOW_STAGES, empty_state, page_intro, relative_time

logger = logging.getLogger(__name__)


def _money(value: float | int) -> str:
    return "—" if not value else f"${value:,.0f}"


def render_insights_page() -> None:
    """Render stage, trend, salary, and company evidence."""
    page_intro(
        "Evidence",
        "Insights",
        "See where the search is moving and which companies are producing useful opportunities.",
    )

    analytics = AnalyticsService()
    try:
        counts = JobService.get_job_counts_by_status()
        trends = analytics.get_job_trends(days=30)
        salaries = analytics.get_salary_analytics(days=90)
        companies = company_service.get_all_companies()
        searches = saved_search_service.list()
    except Exception:
        logger.exception("Could not load insights")
        st.error("Insights could not be calculated. Check the database and retry.")
        return

    total_jobs = sum(counts.values())
    top_metrics = st.columns(3, gap="medium")
    top_metrics[0].metric("Tracked jobs", total_jobs, border=True)
    top_metrics[1].metric("Companies", len(companies), border=True)
    top_metrics[2].metric("Saved searches", len(searches), border=True)

    st.subheader("Workflow", anchor=False)
    stage_metrics = st.columns(len(WORKFLOW_STAGES), gap="small")
    for column, stage in zip(stage_metrics, WORKFLOW_STAGES, strict=True):
        column.metric(stage.value, counts.get(stage, 0), border=True)

    st.subheader("New listings", anchor=False)
    if trends["status"] == "error":
        st.error("Listing trends are temporarily unavailable.")
    elif trends["trends"]:
        trend_frame = pd.DataFrame(trends["trends"]).set_index("date")
        st.bar_chart(trend_frame["job_count"], color="#176b5b")
        st.caption("Active jobs by posted date over the past 30 days.")
    else:
        empty_state(
            "No trend yet",
            "Run a saved search to build a history of job listings.",
        )

    st.subheader("Compensation", anchor=False)
    if salaries["status"] == "error":
        st.error("Salary insights are temporarily unavailable.")
    else:
        salary = salaries["salary_data"]
        salary_metrics = st.columns(3, gap="medium")
        salary_metrics[0].metric(
            "Jobs with salary data",
            salary["total_jobs_with_salary"],
            border=True,
        )
        salary_metrics[1].metric(
            "Average minimum",
            _money(salary["avg_min_salary"]),
            border=True,
        )
        salary_metrics[2].metric(
            "Average maximum",
            _money(salary["avg_max_salary"]),
            border=True,
        )

    st.subheader("Company facets", anchor=False)
    if not companies:
        empty_state(
            "No companies yet",
            "Companies appear here automatically when a saved search finds jobs.",
        )
        return

    company_frame = pd.DataFrame(
        [
            {
                "Company": company.name,
                "Active jobs": company.active_jobs,
                "All jobs": company.total_jobs,
                "Latest listing": relative_time(company.last_job_posted),
                "Website": company.url,
            }
            for company in companies
        ]
    )
    st.dataframe(
        company_frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Website": st.column_config.LinkColumn("Website", display_text="Open"),
        },
    )
    st.caption("Companies are derived from collected jobs and cannot be edited here.")


render_insights_page()
