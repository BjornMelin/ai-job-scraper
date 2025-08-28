"""Comprehensive tests for the Analytics page UI component.

This module tests the Analytics page functionality including:
- Service initialization and caching in session state
- Cost monitoring display with metrics and charts
- Job market trends visualization
- Company hiring analysis with interactive charts
- Salary analytics and insights
- Error handling and service failure scenarios
- Interactive components (selectboxes, expandable sections)
- Data processing and visualization
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tests.utils.streamlit_utils import StreamlitComponentTester

from src.ui.pages.analytics import (
    _render_analytics_status_section,
    _render_company_analytics_section,
    _render_cost_monitoring_section,
    _render_job_trends_section,
    _render_salary_analytics_section,
    render_analytics_page,
)


@pytest.fixture
def sample_cost_summary():
    """Create sample cost monitoring summary data."""
    return {
        "total_cost": 45.67,
        "remaining": 54.33,
        "utilization_percent": 45.67,
        "monthly_budget": 100.00,
        "month_year": "January 2024",
        "costs_by_service": {
            "OpenAI API": 25.50,
            "Scraping Service": 15.00,
            "Database": 5.17,
        },
    }


@pytest.fixture
def sample_cost_alerts():
    """Create sample cost alerts data."""
    return [
        {"type": "warning", "message": "Monthly budget is 50% utilized"},
        {"type": "error", "message": "Cost threshold exceeded for OpenAI API"},
    ]


@pytest.fixture
def sample_job_trends_data():
    """Create sample job trends data."""
    return {
        "status": "success",
        "method": "DuckDB Analytics",
        "total_jobs": 150,
        "trends": [
            {"date": "2024-01-01", "job_count": 10},
            {"date": "2024-01-02", "job_count": 15},
            {"date": "2024-01-03", "job_count": 12},
            {"date": "2024-01-04", "job_count": 18},
            {"date": "2024-01-05", "job_count": 22},
        ],
    }


@pytest.fixture
def sample_company_analytics_data():
    """Create sample company analytics data."""
    return {
        "status": "success",
        "method": "DuckDB Analytics",
        "companies": [
            {
                "company": "Tech Corp",
                "total_jobs": 45,
                "active_jobs": 40,
                "avg_salary": 120000,
            },
            {
                "company": "AI Startup",
                "total_jobs": 32,
                "active_jobs": 28,
                "avg_salary": 110000,
            },
            {
                "company": "Scale Inc",
                "total_jobs": 28,
                "active_jobs": 25,
                "avg_salary": 105000,
            },
            {
                "company": "Data Co",
                "total_jobs": 22,
                "active_jobs": 20,
                "avg_salary": 95000,
            },
        ],
    }


@pytest.fixture
def sample_salary_analytics_data():
    """Create sample salary analytics data."""
    return {
        "status": "success",
        "method": "DuckDB Analytics",
        "salary_data": {
            "total_jobs_with_salary": 85,
            "avg_min_salary": 80000,
            "avg_max_salary": 120000,
            "min_salary": 60000,
            "max_salary": 180000,
        },
    }


@pytest.fixture
def sample_analytics_status():
    """Create sample analytics status data."""
    return {
        "duckdb_status": "connected",
        "sqlite_scanner_available": True,
        "total_records_processed": 1500,
        "last_analysis_time": "2024-01-15T10:30:00Z",
        "performance_metrics": {"query_time_ms": 125, "memory_usage_mb": 45},
    }


class TestAnalyticsPageInitialization:
    """Test analytics page initialization and service setup."""

    def test_render_analytics_page_initializes_services(self):
        """Test main page function initializes services in session state."""
        tester = StreamlitComponentTester(render_analytics_page)

        with (
            patch(
                "src.ui.pages.analytics._render_cost_monitoring_section"
            ) as mock_cost,
            patch("src.ui.pages.analytics._render_job_trends_section") as mock_trends,
            patch(
                "src.ui.pages.analytics._render_company_analytics_section"
            ) as mock_company,
            patch(
                "src.ui.pages.analytics._render_salary_analytics_section"
            ) as mock_salary,
            patch(
                "src.ui.pages.analytics._render_analytics_status_section"
            ) as mock_status,
            patch("streamlit.title") as mock_title,
            patch("src.services.analytics_service.AnalyticsService"),
            patch("src.services.cost_monitor.CostMonitor"),
        ):
            tester.run_component()

            # Verify page title
            mock_title.assert_called_once_with("📊 Analytics Dashboard")

            # Verify services are initialized
            state = tester.get_session_state()
            assert "analytics_service" in state
            assert "cost_monitor" in state

            # Verify all sections are rendered
            mock_cost.assert_called_once()
            mock_trends.assert_called_once()
            mock_company.assert_called_once()
            mock_salary.assert_called_once()
            mock_status.assert_called_once()

    def test_render_analytics_page_reuses_cached_services(self):
        """Test page reuses services from session state instead of creating new ones."""
        tester = StreamlitComponentTester(render_analytics_page)

        # Pre-populate session state with mock services
        mock_analytics = Mock()
        mock_cost_mon = Mock()
        tester.set_session_state(
            analytics_service=mock_analytics, cost_monitor=mock_cost_mon
        )

        with (
            patch("src.ui.pages.analytics._render_cost_monitoring_section"),
            patch("src.ui.pages.analytics._render_job_trends_section"),
            patch("src.ui.pages.analytics._render_company_analytics_section"),
            patch("src.ui.pages.analytics._render_salary_analytics_section"),
            patch("src.ui.pages.analytics._render_analytics_status_section"),
            patch("streamlit.title"),
            patch(
                "src.services.analytics_service.AnalyticsService"
            ) as mock_analytics_service,
            patch("src.services.cost_monitor.CostMonitor") as mock_cost_monitor,
        ):
            tester.run_component()

            # Verify services are not created again
            mock_analytics_service.assert_not_called()
            mock_cost_monitor.assert_not_called()

            # Verify cached services are used
            state = tester.get_session_state()
            assert state["analytics_service"] is mock_analytics
            assert state["cost_monitor"] is mock_cost_mon


class TestCostMonitoringSection:
    """Test cost monitoring section functionality."""

    def test_render_cost_monitoring_section_successful(
        self, sample_cost_summary, sample_cost_alerts
    ):
        """Test successful cost monitoring section rendering."""
        tester = StreamlitComponentTester(_render_cost_monitoring_section)

        # Create mock cost monitor
        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.return_value = sample_cost_summary
        mock_cost_monitor.get_cost_alerts.return_value = sample_cost_alerts

        with (
            patch("streamlit.subheader") as mock_subheader,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.warning") as mock_warning,
            patch("streamlit.error") as mock_error,
            patch("plotly.express.pie") as mock_pie,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            # Mock columns
            mock_col1, mock_col2, mock_col3, mock_col4 = Mock(), Mock(), Mock(), Mock()
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]

            # Mock plotly chart
            mock_fig = Mock()
            mock_pie.return_value = mock_fig

            tester.run_component(mock_cost_monitor)

            # Verify section header
            mock_subheader.assert_called_with("💰 Cost Monitoring")

            # Verify metrics are displayed
            mock_col1.metric.assert_called_with("Monthly Spend", "$45.67")
            mock_col2.metric.assert_called_with("Remaining", "$54.33")
            mock_col3.metric.assert_called_with("Usage", "46%")
            mock_col4.metric.assert_called_with("Budget", "$100")

            # Verify alerts are displayed
            mock_warning.assert_called_once_with("Monthly budget is 50% utilized")
            mock_error.assert_called_once_with("Cost threshold exceeded for OpenAI API")

            # Verify chart is created and displayed
            mock_pie.assert_called_once()
            mock_plotly_chart.assert_called_once_with(
                mock_fig, use_container_width=True
            )

    def test_render_cost_monitoring_section_no_service_costs(self, sample_cost_summary):
        """Test cost monitoring with no service cost breakdown."""
        tester = StreamlitComponentTester(_render_cost_monitoring_section)

        # Remove service costs
        summary_no_services = sample_cost_summary.copy()
        summary_no_services["costs_by_service"] = {}

        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.return_value = summary_no_services
        mock_cost_monitor.get_cost_alerts.return_value = []

        with (
            patch("streamlit.subheader"),
            patch("streamlit.columns") as mock_columns,
            patch("plotly.express.pie") as mock_pie,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

            tester.run_component(mock_cost_monitor)

            # Verify chart functions are not called
            mock_pie.assert_not_called()
            mock_plotly_chart.assert_not_called()

    def test_render_cost_monitoring_section_service_exception(self):
        """Test cost monitoring handles service exceptions gracefully."""
        tester = StreamlitComponentTester(_render_cost_monitoring_section)

        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.side_effect = Exception(
            "Service unavailable"
        )

        with patch("streamlit.subheader"), patch("streamlit.error") as mock_error:
            tester.run_component(mock_cost_monitor)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Cost monitoring unavailable" in error_message
            assert "Service unavailable" in error_message


class TestJobTrendsSection:
    """Test job trends section functionality."""

    def test_render_job_trends_section_successful(self, sample_job_trends_data):
        """Test successful job trends section rendering."""
        tester = StreamlitComponentTester(_render_job_trends_section)

        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = sample_job_trends_data

        with (
            patch("streamlit.subheader") as mock_subheader,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.info") as mock_info,
            patch("streamlit.columns") as mock_columns,
            patch("pandas.DataFrame") as mock_dataframe,
            patch("plotly.express.line") as mock_line_chart,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            # Mock selectbox selection
            mock_selectbox.return_value = "Last 7 Days"

            # Mock columns
            mock_col1, mock_col2 = Mock(), Mock()
            mock_columns.return_value = [mock_col1, mock_col2]

            # Mock DataFrame and chart
            mock_df = Mock()
            mock_dataframe.return_value = mock_df
            mock_fig = Mock()
            mock_line_chart.return_value = mock_fig

            tester.run_component(mock_analytics)

            # Verify section header and time selector
            mock_subheader.assert_called_with("📈 Job Market Trends")
            mock_selectbox.assert_called_once_with(
                "Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )

            # Verify analytics service is called with correct parameters
            mock_analytics.get_job_trends.assert_called_once_with(7)

            # Verify success info is displayed
            mock_info.assert_called_once_with(
                "🚀 Analytics powered by DuckDB Analytics"
            )

            # Verify DataFrame is created with trends data
            mock_dataframe.assert_called_once_with(sample_job_trends_data["trends"])

            # Verify chart is created and displayed
            mock_line_chart.assert_called_once()
            mock_plotly_chart.assert_called_once_with(
                mock_fig, use_container_width=True
            )

            # Verify metrics are displayed
            mock_col1.metric.assert_called_with("Total Jobs", "150")
            mock_col2.metric.assert_called_with("Daily Average", "21")

    def test_render_job_trends_section_different_time_ranges(
        self, sample_job_trends_data
    ):
        """Test job trends with different time range selections."""
        tester = StreamlitComponentTester(_render_job_trends_section)

        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = sample_job_trends_data

        test_cases = [("Last 7 Days", 7), ("Last 30 Days", 30), ("Last 90 Days", 90)]

        for selection, expected_days in test_cases:
            with (
                patch("streamlit.selectbox", return_value=selection),
                patch("streamlit.subheader"),
                patch("streamlit.info"),
                patch("streamlit.columns") as mock_columns,
                patch("pandas.DataFrame"),
                patch("plotly.express.line"),
                patch("streamlit.plotly_chart"),
            ):
                mock_columns.return_value = [Mock(), Mock()]

                tester.run_component(mock_analytics)

                # Verify correct days parameter is passed
                mock_analytics.get_job_trends.assert_called_with(expected_days)
                mock_analytics.reset_mock()

    def test_render_job_trends_section_no_data(self):
        """Test job trends section when no data is available."""
        tester = StreamlitComponentTester(_render_job_trends_section)

        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = {
            "status": "error",
            "error": "No data available",
        }

        with (
            patch("streamlit.subheader"),
            patch("streamlit.selectbox", return_value="Last 7 Days"),
            patch("streamlit.error") as mock_error,
        ):
            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Job trends unavailable" in error_message
            assert "No data available" in error_message

    def test_render_job_trends_section_service_exception(self):
        """Test job trends handles service exceptions."""
        tester = StreamlitComponentTester(_render_job_trends_section)

        mock_analytics = Mock()
        mock_analytics.get_job_trends.side_effect = Exception(
            "Database connection failed"
        )

        with (
            patch("streamlit.subheader"),
            patch("streamlit.selectbox", return_value="Last 7 Days"),
            patch("streamlit.error") as mock_error,
        ):
            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Job trends unavailable" in error_message
            assert "Database connection failed" in error_message


class TestCompanyAnalyticsSection:
    """Test company analytics section functionality."""

    def test_render_company_analytics_section_successful(
        self, sample_company_analytics_data
    ):
        """Test successful company analytics section rendering."""
        tester = StreamlitComponentTester(_render_company_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_company_analytics.return_value = (
            sample_company_analytics_data
        )

        with (
            patch("streamlit.subheader") as mock_subheader,
            patch("streamlit.info") as mock_info,
            patch("pandas.DataFrame") as mock_dataframe,
            patch("streamlit.dataframe") as mock_st_dataframe,
            patch("plotly.express.bar") as mock_bar_chart,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            # Mock DataFrame
            mock_df = pd.DataFrame(sample_company_analytics_data["companies"])
            mock_dataframe.return_value = mock_df

            # Mock chart
            mock_fig = Mock()
            mock_bar_chart.return_value = mock_fig

            tester.run_component(mock_analytics)

            # Verify section header
            mock_subheader.assert_called_with("🏢 Company Hiring Analysis")

            # Verify analytics method is displayed
            mock_info.assert_called_once_with(
                "🚀 Company analytics via DuckDB Analytics"
            )

            # Verify DataFrame is created and displayed
            mock_dataframe.assert_called_once_with(
                sample_company_analytics_data["companies"]
            )
            mock_st_dataframe.assert_called_once_with(mock_df, use_container_width=True)

            # Verify bar chart is created for top companies
            mock_bar_chart.assert_called_once()
            chart_call = mock_bar_chart.call_args
            assert chart_call[1]["x"] == "total_jobs"
            assert chart_call[1]["y"] == "company"
            assert chart_call[1]["orientation"] == "h"
            assert "Top 10 Companies" in chart_call[1]["title"]

            # Verify chart is displayed
            mock_plotly_chart.assert_called_once_with(
                mock_fig, use_container_width=True
            )

    def test_render_company_analytics_section_empty_data(self):
        """Test company analytics with empty data."""
        tester = StreamlitComponentTester(_render_company_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_company_analytics.return_value = {
            "status": "success",
            "method": "DuckDB Analytics",
            "companies": [],
        }

        with (
            patch("streamlit.subheader"),
            patch("streamlit.info"),
            patch("pandas.DataFrame") as mock_dataframe,
            patch("streamlit.dataframe"),
            patch("plotly.express.bar") as mock_bar_chart,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            # Mock empty DataFrame
            mock_df = pd.DataFrame([])
            mock_dataframe.return_value = mock_df

            tester.run_component(mock_analytics)

            # Verify DataFrame is still displayed (empty)
            mock_dataframe.assert_called_once_with([])

            # Verify chart functions are not called for empty data
            mock_bar_chart.assert_not_called()
            mock_plotly_chart.assert_not_called()

    def test_render_company_analytics_section_error_response(self):
        """Test company analytics with error response."""
        tester = StreamlitComponentTester(_render_company_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_company_analytics.return_value = {
            "status": "error",
            "error": "Database query failed",
        }

        with patch("streamlit.subheader"), patch("streamlit.error") as mock_error:
            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Company analytics unavailable" in error_message
            assert "Database query failed" in error_message

    def test_render_company_analytics_section_service_exception(self):
        """Test company analytics handles service exceptions."""
        tester = StreamlitComponentTester(_render_company_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_company_analytics.side_effect = Exception("Service error")

        with patch("streamlit.subheader"), patch("streamlit.error") as mock_error:
            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Company analytics unavailable" in error_message
            assert "Service error" in error_message


class TestSalaryAnalyticsSection:
    """Test salary analytics section functionality."""

    def test_render_salary_analytics_section_successful(
        self, sample_salary_analytics_data
    ):
        """Test successful salary analytics section rendering."""
        tester = StreamlitComponentTester(_render_salary_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_salary_analytics.return_value = sample_salary_analytics_data

        with (
            patch("streamlit.subheader") as mock_subheader,
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.info") as mock_info,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.metric") as mock_metric,
        ):
            # Mock selectbox selection
            mock_selectbox.return_value = 90

            # Mock columns
            mock_col1, mock_col2, mock_col3, mock_col4 = Mock(), Mock(), Mock(), Mock()
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]

            tester.run_component(mock_analytics)

            # Verify section header
            mock_subheader.assert_called_with("💰 Salary Analytics")

            # Verify time selector
            mock_selectbox.assert_called_once()
            selectbox_call = mock_selectbox.call_args
            assert selectbox_call[0][1] == [30, 60, 90, 180]
            assert selectbox_call[1]["index"] == 2  # Default to 90 days

            # Verify analytics service is called
            mock_analytics.get_salary_analytics.assert_called_once_with(days=90)

            # Verify success info is displayed
            mock_info.assert_called_once_with(
                "🚀 Salary analytics via DuckDB Analytics"
            )

            # Verify salary metrics are displayed
            mock_col1.metric.assert_called_with("Jobs with Salary", "85")
            mock_col2.metric.assert_called_with("Avg Min Salary", "$80,000")
            mock_col3.metric.assert_called_with("Avg Max Salary", "$120,000")
            mock_col4.metric.assert_called_with("Salary Range", "$60,000 - $180,000")

            # Verify average midpoint is calculated and displayed
            mock_metric.assert_called_once_with("Average Salary Midpoint", "$100,000")

    def test_render_salary_analytics_section_no_salary_data(self):
        """Test salary analytics when no salary data is available."""
        tester = StreamlitComponentTester(_render_salary_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_salary_analytics.return_value = {
            "status": "success",
            "method": "DuckDB Analytics",
            "salary_data": {
                "total_jobs_with_salary": 0,
                "avg_min_salary": 0,
                "avg_max_salary": 0,
                "min_salary": 0,
                "max_salary": 0,
            },
        }

        with (
            patch("streamlit.subheader"),
            patch("streamlit.selectbox", return_value=90),
            patch("streamlit.info") as mock_info,
            patch("streamlit.columns"),
            patch("streamlit.metric"),
        ):
            tester.run_component(mock_analytics)

            # Verify info message for no data
            assert mock_info.call_count == 2  # Success message + no data message
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            assert any("No salary data available" in call for call in info_calls)

    def test_render_salary_analytics_section_different_time_periods(
        self, sample_salary_analytics_data
    ):
        """Test salary analytics with different time period selections."""
        tester = StreamlitComponentTester(_render_salary_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_salary_analytics.return_value = sample_salary_analytics_data

        test_periods = [30, 60, 90, 180]

        for period in test_periods:
            with (
                patch("streamlit.selectbox", return_value=period),
                patch("streamlit.subheader"),
                patch("streamlit.info"),
                patch("streamlit.columns") as mock_columns,
                patch("streamlit.metric"),
            ):
                mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

                tester.run_component(mock_analytics)

                # Verify correct period is passed to analytics service
                mock_analytics.get_salary_analytics.assert_called_with(days=period)
                mock_analytics.reset_mock()

    def test_render_salary_analytics_section_error_response(self):
        """Test salary analytics with error response."""
        tester = StreamlitComponentTester(_render_salary_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_salary_analytics.return_value = {
            "status": "error",
            "error": "Insufficient salary data",
        }

        with (
            patch("streamlit.subheader"),
            patch("streamlit.selectbox", return_value=90),
            patch("streamlit.error") as mock_error,
        ):
            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Salary analytics unavailable" in error_message
            assert "Insufficient salary data" in error_message

    def test_render_salary_analytics_section_service_exception(self):
        """Test salary analytics handles service exceptions."""
        tester = StreamlitComponentTester(_render_salary_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_salary_analytics.side_effect = Exception(
            "Analytics service failed"
        )

        with (
            patch("streamlit.subheader"),
            patch("streamlit.selectbox", return_value=90),
            patch("streamlit.error") as mock_error,
        ):
            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Salary analytics unavailable" in error_message
            assert "Analytics service failed" in error_message


class TestAnalyticsStatusSection:
    """Test analytics status section functionality."""

    def test_render_analytics_status_section_successful(self, sample_analytics_status):
        """Test successful analytics status section rendering."""
        tester = StreamlitComponentTester(_render_analytics_status_section)

        mock_analytics = Mock()
        mock_analytics.get_status_report.return_value = sample_analytics_status

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.json") as mock_json,
        ):
            # Mock expander context manager
            mock_expander_context = Mock()
            mock_expander.return_value.__enter__ = Mock(
                return_value=mock_expander_context
            )
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            tester.run_component(mock_analytics)

            # Verify expander is created
            mock_expander.assert_called_once_with("🔧 Analytics Status")

            # Verify status is displayed as JSON
            mock_json.assert_called_once_with(sample_analytics_status)

    def test_render_analytics_status_section_service_exception(self):
        """Test analytics status handles service exceptions."""
        tester = StreamlitComponentTester(_render_analytics_status_section)

        mock_analytics = Mock()
        mock_analytics.get_status_report.side_effect = Exception(
            "Status service failed"
        )

        with (
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.error") as mock_error,
        ):
            # Mock expander context manager
            mock_expander_context = Mock()
            mock_expander.return_value.__enter__ = Mock(
                return_value=mock_expander_context
            )
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            tester.run_component(mock_analytics)

            # Verify error is displayed
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Status unavailable" in error_message
            assert "Status service failed" in error_message


class TestInteractiveComponents:
    """Test interactive components in the analytics page."""

    def test_time_range_selector_job_trends(self, sample_job_trends_data):
        """Test time range selector in job trends affects data query."""
        tester = StreamlitComponentTester(_render_job_trends_section)

        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = sample_job_trends_data

        # Test different selections
        time_selections = [
            ("Last 7 Days", 7),
            ("Last 30 Days", 30),
            ("Last 90 Days", 90),
        ]

        for selection, expected_days in time_selections:
            with (
                patch("streamlit.selectbox", return_value=selection) as mock_selectbox,
                patch("streamlit.subheader"),
                patch("streamlit.info"),
                patch("streamlit.columns") as mock_columns,
                patch("pandas.DataFrame"),
                patch("plotly.express.line"),
                patch("streamlit.plotly_chart"),
            ):
                mock_columns.return_value = [Mock(), Mock()]

                tester.run_component(mock_analytics)

                # Verify selectbox has correct options
                selectbox_call = mock_selectbox.call_args
                assert selectbox_call[0][1] == [
                    "Last 7 Days",
                    "Last 30 Days",
                    "Last 90 Days",
                ]

                # Verify analytics service is called with correct parameters
                mock_analytics.get_job_trends.assert_called_with(expected_days)
                mock_analytics.reset_mock()

    def test_salary_period_selector_affects_query(self, sample_salary_analytics_data):
        """Test salary period selector affects analytics query."""
        tester = StreamlitComponentTester(_render_salary_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_salary_analytics.return_value = sample_salary_analytics_data

        # Test different period selections
        periods = [30, 60, 90, 180]

        for period in periods:
            with (
                patch("streamlit.selectbox", return_value=period) as mock_selectbox,
                patch("streamlit.subheader"),
                patch("streamlit.info"),
                patch("streamlit.columns") as mock_columns,
                patch("streamlit.metric"),
            ):
                mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

                tester.run_component(mock_analytics)

                # Verify selectbox has correct options and format function
                selectbox_call = mock_selectbox.call_args
                assert selectbox_call[0][1] == [30, 60, 90, 180]
                assert selectbox_call[1]["index"] == 2  # Default to 90 days

                # Test format function
                format_func = selectbox_call[1]["format_func"]
                assert format_func(30) == "Last 30 Days"

                # Verify analytics service is called with correct period
                mock_analytics.get_salary_analytics.assert_called_with(days=period)
                mock_analytics.reset_mock()

    def test_expandable_status_section(self, sample_analytics_status):
        """Test expandable status section functionality."""
        tester = StreamlitComponentTester(_render_analytics_status_section)

        mock_analytics = Mock()
        mock_analytics.get_status_report.return_value = sample_analytics_status

        with patch("streamlit.expander") as mock_expander, patch("streamlit.json"):
            # Mock expander context manager
            mock_expander_context = Mock()
            mock_expander.return_value.__enter__ = Mock(
                return_value=mock_expander_context
            )
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            tester.run_component(mock_analytics)

            # Verify expander is created with correct title
            mock_expander.assert_called_once_with("🔧 Analytics Status")


class TestDataProcessingAndVisualization:
    """Test data processing and visualization functionality."""

    def test_cost_breakdown_pie_chart_creation(self, sample_cost_summary):
        """Test pie chart creation for cost breakdown."""
        tester = StreamlitComponentTester(_render_cost_monitoring_section)

        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.return_value = sample_cost_summary
        mock_cost_monitor.get_cost_alerts.return_value = []

        with (
            patch("streamlit.subheader"),
            patch("streamlit.columns") as mock_columns,
            patch("plotly.express.pie") as mock_pie,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            mock_fig = Mock()
            mock_pie.return_value = mock_fig

            tester.run_component(mock_cost_monitor)

            # Verify pie chart is created with correct data
            pie_call = mock_pie.call_args
            assert pie_call[1]["values"] == [25.50, 15.00, 5.17]
            assert pie_call[1]["names"] == [
                "OpenAI API",
                "Scraping Service",
                "Database",
            ]
            assert "January 2024" in pie_call[1]["title"]

            # Verify chart is displayed
            mock_plotly_chart.assert_called_once_with(
                mock_fig, use_container_width=True
            )

    def test_job_trends_line_chart_creation(self, sample_job_trends_data):
        """Test line chart creation for job trends."""
        tester = StreamlitComponentTester(_render_job_trends_section)

        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = sample_job_trends_data

        with (
            patch("streamlit.selectbox", return_value="Last 7 Days"),
            patch("streamlit.subheader"),
            patch("streamlit.info"),
            patch("streamlit.columns") as mock_columns,
            patch("pandas.DataFrame") as mock_dataframe,
            patch("plotly.express.line") as mock_line,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            mock_columns.return_value = [Mock(), Mock()]
            mock_df = Mock()
            mock_dataframe.return_value = mock_df
            mock_fig = Mock()
            mock_line.return_value = mock_fig

            tester.run_component(mock_analytics)

            # Verify DataFrame is created with trends data
            mock_dataframe.assert_called_once_with(sample_job_trends_data["trends"])

            # Verify line chart is created with correct parameters
            line_call = mock_line.call_args
            assert line_call[0][0] is mock_df  # DataFrame
            assert line_call[1]["x"] == "date"
            assert line_call[1]["y"] == "job_count"
            assert "Last 7 Days" in line_call[1]["title"]
            assert line_call[1]["markers"] is True

            # Verify chart is displayed
            mock_plotly_chart.assert_called_once_with(
                mock_fig, use_container_width=True
            )

    def test_company_analytics_bar_chart_creation(self, sample_company_analytics_data):
        """Test bar chart creation for company analytics."""
        tester = StreamlitComponentTester(_render_company_analytics_section)

        mock_analytics = Mock()
        mock_analytics.get_company_analytics.return_value = (
            sample_company_analytics_data
        )

        with (
            patch("streamlit.subheader"),
            patch("streamlit.info"),
            patch("pandas.DataFrame") as mock_dataframe,
            patch("streamlit.dataframe"),
            patch("plotly.express.bar") as mock_bar,
            patch("streamlit.plotly_chart") as mock_plotly_chart,
        ):
            # Create real DataFrame to test head() method
            mock_df = pd.DataFrame(sample_company_analytics_data["companies"])
            mock_dataframe.return_value = mock_df
            mock_fig = Mock()
            mock_bar.return_value = mock_fig

            tester.run_component(mock_analytics)

            # Verify bar chart is created with correct parameters
            bar_call = mock_bar.call_args
            # The top_10 should be the first 10 companies (all 4 in our sample)
            assert len(bar_call[0][0]) == 4  # All companies since we have < 10
            assert bar_call[1]["x"] == "total_jobs"
            assert bar_call[1]["y"] == "company"
            assert bar_call[1]["orientation"] == "h"
            assert "Top 10 Companies" in bar_call[1]["title"]

            # Verify chart layout and display
            mock_fig.update_layout.assert_called_once_with(height=500)
            mock_plotly_chart.assert_called_once_with(
                mock_fig, use_container_width=True
            )


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""

    def test_all_sections_handle_service_unavailable(self):
        """Test all sections handle service unavailable gracefully."""
        # Test each section with service unavailable
        sections_and_services = [
            (_render_cost_monitoring_section, "get_monthly_summary"),
            (_render_job_trends_section, "get_job_trends"),
            (_render_company_analytics_section, "get_company_analytics"),
            (_render_salary_analytics_section, "get_salary_analytics"),
            (_render_analytics_status_section, "get_status_report"),
        ]

        for section_func, service_method in sections_and_services:
            tester = StreamlitComponentTester(section_func)

            mock_service = Mock()
            getattr(mock_service, service_method).side_effect = Exception(
                "Service unavailable"
            )

            with (
                patch("streamlit.subheader"),
                patch("streamlit.selectbox", return_value="Last 7 Days"),
                patch("streamlit.error") as mock_error,
                patch("streamlit.expander"),
            ):
                # Handle expander context manager for status section
                if section_func == _render_analytics_status_section:
                    with patch("streamlit.expander") as mock_expander:
                        mock_expander.return_value.__enter__ = Mock()
                        mock_expander.return_value.__exit__ = Mock(return_value=None)

                        tester.run_component(mock_service)
                else:
                    tester.run_component(mock_service)

                # Verify error is displayed
                mock_error.assert_called_once()
                error_message = mock_error.call_args[0][0]
                assert "unavailable" in error_message
                assert "Service unavailable" in error_message

    def test_empty_data_scenarios(self):
        """Test handling of empty data scenarios across sections."""
        # Test job trends with empty trends
        job_trends_tester = StreamlitComponentTester(_render_job_trends_section)
        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = {
            "status": "success",
            "method": "DuckDB Analytics",
            "total_jobs": 0,
            "trends": [],
        }

        with (
            patch("streamlit.selectbox", return_value="Last 7 Days"),
            patch("streamlit.subheader"),
            patch("streamlit.info"),
            patch("streamlit.columns") as mock_columns,
            patch("pandas.DataFrame"),
            patch("plotly.express.line"),
            patch("streamlit.plotly_chart"),
        ):
            mock_columns.return_value = [Mock(), Mock()]

            # Should not raise exception
            job_trends_tester.run_component(mock_analytics)

    def test_malformed_response_handling(self):
        """Test handling of malformed service responses."""
        tester = StreamlitComponentTester(_render_cost_monitoring_section)

        mock_cost_monitor = Mock()
        # Return malformed response missing required keys
        mock_cost_monitor.get_monthly_summary.return_value = {"invalid": "response"}
        mock_cost_monitor.get_cost_alerts.return_value = []

        with patch("streamlit.subheader"), patch("streamlit.error") as mock_error:
            # Should handle gracefully and show error
            tester.run_component(mock_cost_monitor)

            # Verify error is displayed (KeyError should be caught)
            mock_error.assert_called_once()


# Integration tests combining multiple sections
class TestAnalyticsPageIntegration:
    """Integration tests for analytics page components working together."""

    def test_full_page_render_with_all_data(
        self,
        sample_cost_summary,
        sample_job_trends_data,
        sample_company_analytics_data,
        sample_salary_analytics_data,
        sample_analytics_status,
    ):
        """Test complete page renders correctly with all data available."""
        tester = StreamlitComponentTester(render_analytics_page)

        # Create mock services with all data
        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = sample_job_trends_data
        mock_analytics.get_company_analytics.return_value = (
            sample_company_analytics_data
        )
        mock_analytics.get_salary_analytics.return_value = sample_salary_analytics_data
        mock_analytics.get_status_report.return_value = sample_analytics_status

        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.return_value = sample_cost_summary
        mock_cost_monitor.get_cost_alerts.return_value = []

        # Pre-populate session state with services
        tester.set_session_state(
            analytics_service=mock_analytics, cost_monitor=mock_cost_monitor
        )

        with (
            patch("streamlit.title"),
            patch("streamlit.subheader"),
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.info"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.metric"),
            patch("pandas.DataFrame"),
            patch("streamlit.dataframe"),
            patch("plotly.express.pie"),
            patch("plotly.express.line"),
            patch("plotly.express.bar"),
            patch("streamlit.plotly_chart"),
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.json"),
        ):
            # Mock UI components
            mock_selectbox.side_effect = ["Last 7 Days", 90]  # Two selectors
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

            # Mock expander context manager
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            # Should render without errors
            tester.run_component()

            # Verify all service methods are called
            mock_analytics.get_job_trends.assert_called_once()
            mock_analytics.get_company_analytics.assert_called_once()
            mock_analytics.get_salary_analytics.assert_called_once()
            mock_analytics.get_status_report.assert_called_once()
            mock_cost_monitor.get_monthly_summary.assert_called_once()
            mock_cost_monitor.get_cost_alerts.assert_called_once()

    def test_partial_service_failures_dont_break_page(self):
        """Test page renders correctly when some services fail but others work."""
        tester = StreamlitComponentTester(render_analytics_page)

        # Create mock services with mixed success/failure
        mock_analytics = Mock()
        mock_analytics.get_job_trends.return_value = {
            "status": "success",
            "method": "Test",
            "total_jobs": 100,
            "trends": [],
        }
        mock_analytics.get_company_analytics.side_effect = Exception(
            "Company service failed"
        )
        mock_analytics.get_salary_analytics.return_value = {
            "status": "error",
            "error": "No salary data",
        }
        mock_analytics.get_status_report.return_value = {"status": "ok"}

        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.side_effect = Exception(
            "Cost monitor failed"
        )

        tester.set_session_state(
            analytics_service=mock_analytics, cost_monitor=mock_cost_monitor
        )

        with (
            patch("streamlit.title"),
            patch("streamlit.subheader"),
            patch("streamlit.selectbox") as mock_selectbox,
            patch("streamlit.info"),
            patch("streamlit.error") as mock_error,
            patch("streamlit.columns") as mock_columns,
            patch("pandas.DataFrame"),
            patch("plotly.express.line"),
            patch("streamlit.plotly_chart"),
            patch("streamlit.expander") as mock_expander,
            patch("streamlit.json"),
        ):
            mock_selectbox.side_effect = ["Last 7 Days", 90]
            mock_columns.return_value = [Mock(), Mock()]
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock(return_value=None)

            # Should render without crashing
            tester.run_component()

            # Verify error messages are displayed for failed services
            assert (
                mock_error.call_count >= 2
            )  # At least cost monitor and company analytics errors

    def test_session_state_service_caching(self):
        """Test services are properly cached in session state across page renders."""
        tester = StreamlitComponentTester(render_analytics_page)

        # First render should create services
        with (
            patch(
                "src.services.analytics_service.AnalyticsService"
            ) as mock_analytics_class,
            patch("src.services.cost_monitor.CostMonitor") as mock_cost_monitor_class,
            patch("src.ui.pages.analytics._render_cost_monitoring_section"),
            patch("src.ui.pages.analytics._render_job_trends_section"),
            patch("src.ui.pages.analytics._render_company_analytics_section"),
            patch("src.ui.pages.analytics._render_salary_analytics_section"),
            patch("src.ui.pages.analytics._render_analytics_status_section"),
            patch("streamlit.title"),
        ):
            mock_analytics_instance = Mock()
            mock_cost_monitor_instance = Mock()
            mock_analytics_class.return_value = mock_analytics_instance
            mock_cost_monitor_class.return_value = mock_cost_monitor_instance

            # First render
            tester.run_component()

            # Verify services are created
            mock_analytics_class.assert_called_once()
            mock_cost_monitor_class.assert_called_once()

            # Verify services are cached in session state
            state = tester.get_session_state()
            assert state["analytics_service"] is mock_analytics_instance
            assert state["cost_monitor"] is mock_cost_monitor_instance

            # Second render should reuse cached services
            tester.run_component()

            # Services should not be created again
            assert mock_analytics_class.call_count == 1
            assert mock_cost_monitor_class.call_count == 1
