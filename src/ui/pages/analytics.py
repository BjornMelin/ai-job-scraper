"""Analytics dashboard page for the AI Job Scraper application.

This module provides analytics visualization including cost monitoring,
job market trends, and company hiring analysis using the library-first refactored
analytics services with DuckDB sqlite_scanner and SQLModel cost tracking.
"""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.services.analytics_service import AnalyticsService
from src.services.cost_monitor import CostMonitor
from src.ui.utils import is_streamlit_context

logger = logging.getLogger(__name__)


def render_analytics_page() -> None:
    """Analytics dashboard with cost monitoring and job trends.

    This function renders the complete analytics dashboard including:
    - Cost monitoring with budget alerts
    - Job market trends over time
    - Company hiring analysis
    - Salary analytics
    - Service status information
    """
    st.title("📊 Analytics Dashboard")

    # Initialize services in session state for performance
    if "analytics_service" not in st.session_state:
        st.session_state.analytics_service = AnalyticsService()
    if "cost_monitor" not in st.session_state:
        st.session_state.cost_monitor = CostMonitor()

    analytics = st.session_state.analytics_service
    cost_monitor = st.session_state.cost_monitor

    # Cost tracking section
    _render_cost_monitoring_section(cost_monitor)

    # Job trends section
    _render_job_trends_section(analytics)

    # Company analytics section
    _render_company_analytics_section(analytics)

    # Salary analytics section
    _render_salary_analytics_section(analytics)

    # Analytics status (expandable)
    _render_analytics_status_section(analytics)


def _render_cost_monitoring_section(cost_monitor: CostMonitor) -> None:
    """Render the cost monitoring section with budget tracking.

    Args:
        cost_monitor: Cost monitoring service instance.
    """
    st.subheader("💰 Cost Monitoring")

    try:
        summary = cost_monitor.get_monthly_summary()

        # Display cost metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Monthly Spend", f"${summary['total_cost']:.2f}")
        col2.metric("Remaining", f"${summary['remaining']:.2f}")
        col3.metric("Usage", f"{summary['utilization_percent']:.0f}%")
        col4.metric("Budget", f"${summary['monthly_budget']:.0f}")

        # Show cost alerts
        alerts = cost_monitor.get_cost_alerts()
        for alert in alerts:
            if alert["type"] == "error":
                st.error(alert["message"])
            elif alert["type"] == "warning":
                st.warning(alert["message"])

        # Service cost breakdown chart
        if summary["costs_by_service"]:
            st.subheader("📈 Cost Breakdown by Service")

            # Create pie chart for cost distribution
            fig_pie = px.pie(
                values=list(summary["costs_by_service"].values()),
                names=list(summary["costs_by_service"].keys()),
                title=f"Service Costs - {summary['month_year']}",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        logger.exception("Failed to render cost monitoring section")
        st.error(f"Cost monitoring unavailable: {e}")


def _render_job_trends_section(analytics: AnalyticsService) -> None:
    """Render the job market trends section.

    Args:
        analytics: Analytics service instance.
    """
    st.subheader("📈 Job Market Trends")

    # Time range selector
    time_range = st.selectbox(
        "Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )
    days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
    days = days_map[time_range]

    try:
        trends_data = analytics.get_job_trends(days)

        if trends_data["status"] == "success" and trends_data["trends"]:
            st.info(f"🚀 Analytics powered by {trends_data['method']}")

            # Convert to DataFrame for plotting
            trends_df = pd.DataFrame(trends_data["trends"])

            # Create trends line chart
            fig_trends = px.line(
                trends_df,
                x="date",
                y="job_count",
                title=f"Job Postings - {time_range}",
                labels={"date": "Date", "job_count": "Jobs Posted"},
                markers=True,
            )
            st.plotly_chart(fig_trends, use_container_width=True)

            # Summary metrics
            col1, col2 = st.columns(2)
            col1.metric("Total Jobs", f"{trends_data['total_jobs']:,}")
            col2.metric("Daily Average", f"{trends_data['total_jobs'] / days:.0f}")

        else:
            st.error(
                f"Job trends unavailable: {trends_data.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.exception("Failed to render job trends section")
        st.error(f"Job trends unavailable: {e}")


def _render_company_analytics_section(analytics: AnalyticsService) -> None:
    """Render the company hiring analysis section.

    Args:
        analytics: Analytics service instance.
    """
    st.subheader("🏢 Company Hiring Analysis")

    try:
        company_data = analytics.get_company_analytics()

        if company_data["status"] == "success" and company_data["companies"]:
            st.info(f"🚀 Company analytics via {company_data['method']}")

            # Convert to DataFrame
            df_companies = pd.DataFrame(company_data["companies"])

            # Display as interactive dataframe
            st.dataframe(df_companies, use_container_width=True)

            # Top companies chart
            if len(df_companies) > 0:
                top_10 = df_companies.head(10)
                fig_companies = px.bar(
                    top_10,
                    x="total_jobs",
                    y="company",
                    orientation="h",
                    title="Top 10 Companies by Job Count",
                    labels={"total_jobs": "Number of Jobs", "company": "Company"},
                )
                fig_companies.update_layout(height=500)
                st.plotly_chart(fig_companies, use_container_width=True)

        else:
            error_msg = company_data.get("error", "Unknown error")
            st.error(f"Company analytics unavailable: {error_msg}")

    except Exception as e:
        logger.exception("Failed to render company analytics section")
        st.error(f"Company analytics unavailable: {e}")


def _render_salary_analytics_section(analytics: AnalyticsService) -> None:
    """Render the salary analytics section.

    Args:
        analytics: Analytics service instance.
    """
    st.subheader("💰 Salary Analytics")

    # Time range selector for salary data
    salary_days = st.selectbox(
        "Salary Analysis Period",
        [30, 60, 90, 180],
        index=2,  # Default to 90 days
        format_func=lambda x: f"Last {x} Days",
    )

    try:
        salary_data = analytics.get_salary_analytics(days=salary_days)

        if salary_data["status"] == "success" and salary_data["salary_data"]:
            st.info(f"🚀 Salary analytics via {salary_data['method']}")

            data = salary_data["salary_data"]

            # Display salary metrics
            if data["total_jobs_with_salary"] > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Jobs with Salary", f"{data['total_jobs_with_salary']:,}")
                col2.metric("Avg Min Salary", f"${data['avg_min_salary']:,.0f}")
                col3.metric("Avg Max Salary", f"${data['avg_max_salary']:,.0f}")
                col4.metric(
                    "Salary Range",
                    f"${data['min_salary']:,.0f} - ${data['max_salary']:,.0f}",
                )

                # Salary insights
                avg_midpoint = (data["avg_min_salary"] + data["avg_max_salary"]) / 2
                st.metric("Average Salary Midpoint", f"${avg_midpoint:,.0f}")

            else:
                st.info("No salary data available for the selected time period")

        else:
            error_msg = salary_data.get("error", "Unknown error")
            st.error(f"Salary analytics unavailable: {error_msg}")

    except Exception as e:
        logger.exception("Failed to render salary analytics section")
        st.error(f"Salary analytics unavailable: {e}")


def _render_analytics_status_section(analytics: AnalyticsService) -> None:
    """Render the analytics service status section.

    Args:
        analytics: Analytics service instance.
    """
    with st.expander("🔧 Analytics Status"):
        try:
            status = analytics.get_status_report()
            st.json(status)
        except Exception as e:
            logger.exception("Failed to get analytics status")
            st.error(f"Status unavailable: {e}")


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    render_analytics_page()
