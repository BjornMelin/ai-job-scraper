"""Fragment-based analytics dashboard with real-time updates and auto-refresh.

This module provides analytics visualization with st.fragment() architecture for:
- Real-time cost monitoring with auto-refresh
- Live job market trends without full page reruns
- Auto-updating company hiring analytics
- Fragment-isolated performance optimization

Features include:
- Auto-refresh cost monitoring (30s intervals)
- Real-time job trends (10s intervals)
- Live company analytics (60s intervals)
- Fragment-scoped error handling and recovery
- Performance optimized with selective updates
- Integration with DuckDB sqlite_scanner and SQLModel cost tracking

This implementation follows Stream C fragment architecture specifications for
component isolation and coordination simplification.
"""

import logging

from datetime import UTC, datetime

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


# ========== FRAGMENT-BASED ANALYTICS COMPONENTS ==========
# Enhanced analytics with st.fragment() architecture for Stream C


@st.fragment(run_every="30s")
def render_cost_monitoring_fragment() -> None:
    """Fragment for real-time cost monitoring with auto-refresh.

    This fragment provides live cost tracking and budget alerts that update
    automatically without triggering full page reruns.
    """
    try:
        # Initialize cost monitor in session state for performance
        if "cost_monitor_fragment" not in st.session_state:
            st.session_state.cost_monitor_fragment = CostMonitor()

        cost_monitor = st.session_state.cost_monitor_fragment

        with st.container():
            col_header, col_indicator = st.columns([4, 1])
            with col_header:
                st.subheader("💰 Live Cost Monitor")
            with col_indicator:
                st.caption("🔄 Auto-updating")

            # Get real-time summary
            summary = cost_monitor.get_monthly_summary()

            # Display cost metrics with live updates
            col1, col2, col3, col4 = st.columns(4)

            # Calculate usage percentage for color coding
            usage_pct = summary["utilization_percent"]
            if usage_pct > 90:
                pass  # Red
            elif usage_pct > 75:
                pass  # Orange-ish

            col1.metric("Monthly Spend", f"${summary['total_cost']:.2f}")
            col2.metric("Remaining", f"${summary['remaining']:.2f}")
            col3.metric("Usage", f"{usage_pct:.0f}%")
            col4.metric("Budget", f"${summary['monthly_budget']:.0f}")

            # Show real-time cost alerts
            alerts = cost_monitor.get_cost_alerts()
            for alert in alerts:
                if alert["type"] == "error":
                    st.error(alert["message"])
                elif alert["type"] == "warning":
                    st.warning(alert["message"])

            # Service cost breakdown with auto-update
            if summary["costs_by_service"]:
                with st.expander("📊 Service Breakdown", expanded=False):
                    # Create real-time pie chart
                    fig_pie = px.pie(
                        values=list(summary["costs_by_service"].values()),
                        names=list(summary["costs_by_service"].keys()),
                        title=f"Real-time Costs - {summary['month_year']}",
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        logger.exception("Error in cost monitoring fragment")
        st.error(f"⚠️ Cost monitor temporarily unavailable: {str(e)[:100]}...")


@st.fragment(run_every="10s")
def render_job_trends_fragment(time_range: str = "Last 7 Days") -> None:
    """Fragment for real-time job market trends with auto-refresh.

    Args:
        time_range: Time range for trends analysis.
    """
    try:
        # Initialize analytics service in fragment state
        if "analytics_fragment" not in st.session_state:
            st.session_state.analytics_fragment = AnalyticsService()

        analytics = st.session_state.analytics_fragment

        with st.container():
            col_header, col_indicator = st.columns([4, 1])
            with col_header:
                st.subheader("📈 Live Job Trends")
            with col_indicator:
                st.caption("🔄 Auto-updating")

            # Get time range mapping
            days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
            days = days_map.get(time_range, 7)

            # Get real-time trends data
            trends_data = analytics.get_job_trends(days)

            if trends_data["status"] == "success" and trends_data["trends"]:
                # Real-time success indicator
                st.success(f"🚀 Live data via {trends_data['method']}", icon="📡")

                # Convert to DataFrame for live plotting
                trends_df = pd.DataFrame(trends_data["trends"])

                # Create live trends chart
                fig_trends = px.line(
                    trends_df,
                    x="date",
                    y="job_count",
                    title=f"Live Job Trends - {time_range}",
                    labels={"date": "Date", "job_count": "Jobs Posted"},
                    markers=True,
                )
                fig_trends.update_layout(height=400)
                st.plotly_chart(fig_trends, use_container_width=True)

                # Live summary metrics
                col1, col2 = st.columns(2)
                col1.metric("Total Jobs", f"{trends_data['total_jobs']:,}")
                col2.metric("Daily Avg", f"{trends_data['total_jobs'] / days:.0f}")

                # Show last update time
                st.caption(f"Last updated: {datetime.now(UTC).strftime('%H:%M:%S')}")

            else:
                st.warning(
                    f"📊 Trends data loading... "
                    f"{trends_data.get('error', 'Retrying...')}"
                )

    except Exception as e:
        logger.exception("Error in job trends fragment")
        st.error(f"⚠️ Trends temporarily unavailable: {str(e)[:100]}...")


@st.fragment(run_every="60s")
def render_company_analytics_fragment() -> None:
    """Fragment for real-time company hiring analysis with auto-refresh."""
    try:
        # Initialize analytics service in fragment state
        if "analytics_fragment" not in st.session_state:
            st.session_state.analytics_fragment = AnalyticsService()

        analytics = st.session_state.analytics_fragment

        with st.container():
            col_header, col_indicator = st.columns([4, 1])
            with col_header:
                st.subheader("🏢 Live Company Analytics")
            with col_indicator:
                st.caption("🔄 Auto-updating")

            # Get real-time company data
            company_data = analytics.get_company_analytics()

            if company_data["status"] == "success" and company_data["companies"]:
                st.info(f"🚀 Live company data via {company_data['method']}")

                # Convert to DataFrame for live analysis
                df_companies = pd.DataFrame(company_data["companies"])

                # Live metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Active Companies", len(df_companies))
                col2.metric("Total Jobs", df_companies["total_jobs"].sum())
                col3.metric(
                    "Avg Jobs/Company", f"{df_companies['total_jobs'].mean():.1f}"
                )

                # Top companies live chart
                if len(df_companies) > 0:
                    top_10 = df_companies.head(10)
                    fig_companies = px.bar(
                        top_10,
                        x="total_jobs",
                        y="company",
                        orientation="h",
                        title="Top 10 Companies (Live)",
                        labels={"total_jobs": "Jobs", "company": "Company"},
                    )
                    fig_companies.update_layout(height=400)
                    st.plotly_chart(fig_companies, use_container_width=True)

                    # Interactive company table with live updates
                    with st.expander("📋 Full Company Data", expanded=False):
                        st.dataframe(df_companies, use_container_width=True)

                # Show last update time
                st.caption(f"Last updated: {datetime.now(UTC).strftime('%H:%M:%S')}")

            else:
                st.warning(
                    f"🏢 Company data loading... {company_data.get('error', 'Retrying...')}"
                )

    except Exception as e:
        logger.exception("Error in company analytics fragment")
        st.error(f"⚠️ Company analytics temporarily unavailable: {str(e)[:100]}...")


@st.fragment(run_every="45s")
def render_salary_analytics_fragment(salary_days: int = 90) -> None:
    """Fragment for real-time salary analytics with auto-refresh.

    Args:
        salary_days: Days of salary data to analyze.
    """
    try:
        # Initialize analytics service in fragment state
        if "analytics_fragment" not in st.session_state:
            st.session_state.analytics_fragment = AnalyticsService()

        analytics = st.session_state.analytics_fragment

        with st.container():
            col_header, col_indicator = st.columns([4, 1])
            with col_header:
                st.subheader("💰 Live Salary Analytics")
            with col_indicator:
                st.caption("🔄 Auto-updating")

            # Get real-time salary data
            salary_data = analytics.get_salary_analytics(days=salary_days)

            if salary_data["status"] == "success" and salary_data["salary_data"]:
                st.success(f"🚀 Live salary data via {salary_data['method']}")

                data = salary_data["salary_data"]

                # Live salary metrics
                if data["total_jobs_with_salary"] > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Jobs w/ Salary", f"{data['total_jobs_with_salary']:,}")
                    col2.metric("Avg Min", f"${data['avg_min_salary']:,.0f}")
                    col3.metric("Avg Max", f"${data['avg_max_salary']:,.0f}")
                    col4.metric(
                        "Range",
                        f"${data['min_salary']:,.0f} - ${data['max_salary']:,.0f}",
                    )

                    # Live salary insights
                    avg_midpoint = (data["avg_min_salary"] + data["avg_max_salary"]) / 2
                    st.metric("💡 Avg Salary Midpoint", f"${avg_midpoint:,.0f}")

                else:
                    st.info("📊 No salary data available for selected period")

                # Show last update time
                st.caption(f"Last updated: {datetime.now(UTC).strftime('%H:%M:%S')}")

            else:
                st.warning(
                    f"💰 Salary data loading... {salary_data.get('error', 'Retrying...')}"
                )

    except Exception as e:
        logger.exception("Error in salary analytics fragment")
        st.error(f"⚠️ Salary analytics temporarily unavailable: {str(e)[:100]}...")


def render_analytics_page_with_fragments() -> None:
    """Enhanced analytics dashboard with fragment architecture.

    This function orchestrates multiple auto-refreshing fragments for optimal
    performance and real-time updates without full page reruns.
    """
    st.title("📊 Live Analytics Dashboard")
    st.caption("🚀 Real-time updates with fragment architecture")

    # Fragment control panel
    with st.expander("⚙️ Fragment Controls", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            enable_cost_monitor = st.checkbox("Cost Monitor (30s)", value=True)
            enable_job_trends = st.checkbox("Job Trends (10s)", value=True)
        with col2:
            enable_company_analytics = st.checkbox(
                "Company Analytics (60s)", value=True
            )
            enable_salary_analytics = st.checkbox("Salary Analytics (45s)", value=True)

        # Time range controls for fragments
        time_range = st.selectbox(
            "Trends Time Range",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
            index=0,
        )

        salary_days = st.selectbox(
            "Salary Analysis Period",
            [30, 60, 90, 180],
            index=2,
            format_func=lambda x: f"Last {x} Days",
        )

    # Render enabled fragments
    if enable_cost_monitor:
        render_cost_monitoring_fragment()
        st.markdown("---")

    if enable_job_trends:
        render_job_trends_fragment(time_range)
        st.markdown("---")

    if enable_company_analytics:
        render_company_analytics_fragment()
        st.markdown("---")

    if enable_salary_analytics:
        render_salary_analytics_fragment(salary_days)
        st.markdown("---")

    # Analytics status section (static)
    with st.expander("🔧 Analytics Status"):
        try:
            # Use singleton analytics service for status
            if "analytics_service" not in st.session_state:
                st.session_state.analytics_service = AnalyticsService()

            status = st.session_state.analytics_service.get_status_report()
            st.json(status)
        except Exception as e:
            logger.exception("Failed to get analytics status")
            st.error(f"Status unavailable: {e}")


# Execute page when loaded by st.navigation()
# Only run when in a proper Streamlit context (not during test imports)
if is_streamlit_context():
    # Check if fragments are enabled in session state
    use_fragments = st.session_state.get("use_fragment_analytics", True)

    if use_fragments:
        render_analytics_page_with_fragments()
    else:
        render_analytics_page()
