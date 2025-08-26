"""UI tests for the analytics dashboard page.

This test suite validates the analytics dashboard UI components and their integration
with the refactored AnalyticsService and CostMonitor services.

Test coverage includes:
- Analytics dashboard page rendering
- Service initialization in Streamlit session state
- Cost monitoring dashboard sections
- Job trends visualization components
- Company analytics display
- Salary analytics rendering
- Error handling in UI components
- Dashboard performance
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.models import CompanySQL, JobSQL
from src.services.analytics_service import DUCKDB_AVAILABLE, AnalyticsService
from src.services.cost_monitor import CostMonitor

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for UI testing."""
    with patch("src.ui.pages.analytics.st") as mock_st:
        # Mock common Streamlit components
        mock_st.title = Mock()
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.info = Mock()
        mock_st.success = Mock()
        mock_st.warning = Mock()
        mock_st.error = Mock()
        mock_st.write = Mock()
        mock_st.metric = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.container = Mock(return_value=Mock())
        mock_st.expander = Mock(return_value=Mock())
        mock_st.tabs = Mock(return_value=[Mock(), Mock(), Mock()])

        # Mock session state
        mock_st.session_state = {}

        # Mock plotly components
        mock_st.plotly_chart = Mock()
        mock_st.dataframe = Mock()

        yield mock_st


@pytest.fixture
def test_analytics_data(tmp_path):
    """Create test data for analytics dashboard testing."""
    # Create test databases
    jobs_db = tmp_path / "test_analytics_ui.db"
    costs_db = tmp_path / "test_costs_ui.db"

    # Create jobs database with test data
    engine = create_engine(
        f"sqlite:///{jobs_db}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    base_date = datetime.now(UTC)
    companies_data = [
        {"id": 1, "name": "UI Test Corp", "url": "https://uitest.com", "active": True},
        {
            "id": 2,
            "name": "Dashboard Inc",
            "url": "https://dashboard.com",
            "active": True,
        },
    ]

    jobs_data = [
        {
            "id": 1,
            "company_id": 1,
            "title": "UI Engineer",
            "description": "Frontend development",
            "link": "https://uitest.com/job1",
            "location": "Remote",
            "posted_date": base_date - timedelta(days=1),
            "salary": [100000, 130000],
            "archived": False,
            "content_hash": "ui_hash1",
        },
        {
            "id": 2,
            "company_id": 2,
            "title": "Dashboard Developer",
            "description": "Analytics dashboard development",
            "link": "https://dashboard.com/job1",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=2),
            "salary": [120000, 150000],
            "archived": False,
            "content_hash": "ui_hash2",
        },
    ]

    # Insert test data
    with Session(engine) as session:
        for company_data in companies_data:
            company = CompanySQL(**company_data)
            session.add(company)
        for job_data in jobs_data:
            job = JobSQL(**job_data)
            session.add(job)
        session.commit()

    # Create analytics and cost monitor services
    analytics_service = AnalyticsService(db_path=str(jobs_db))
    cost_monitor = CostMonitor(db_path=str(costs_db))

    # Add some cost data
    cost_monitor.track_ai_cost("test_model", 1000, 5.00, "ui_test_operation")
    cost_monitor.track_proxy_cost(50, 2.50, "ui_test_proxy")
    cost_monitor.track_scraping_cost("UI Test Corp", 10, 1.25)

    return {
        "analytics_service": analytics_service,
        "cost_monitor": cost_monitor,
        "jobs_db": str(jobs_db),
        "costs_db": str(costs_db),
    }


class TestAnalyticsDashboardPageBasics:
    """Test basic analytics dashboard page functionality."""

    @patch("src.ui.pages.analytics.AnalyticsService")
    @patch("src.ui.pages.analytics.CostMonitor")
    def test_render_analytics_page_initializes_services(
        self, mock_cost_monitor_cls, mock_analytics_cls, mock_streamlit
    ):
        """Test that the analytics page initializes services correctly."""
        from src.ui.pages.analytics import render_analytics_page

        # Mock service instances
        mock_analytics = Mock()
        mock_cost_monitor = Mock()
        mock_analytics_cls.return_value = mock_analytics
        mock_cost_monitor_cls.return_value = mock_cost_monitor

        # Mock service responses
        mock_cost_monitor.get_monthly_summary.return_value = {
            "total_cost": 10.0,
            "monthly_budget": 50.0,
            "budget_status": "within_budget",
            "costs_by_service": {"ai": 5.0, "proxy": 3.0, "scraping": 2.0},
            "utilization_percent": 20.0,
            "remaining": 40.0,
            "month_year": "January 2024",
        }
        mock_cost_monitor.get_cost_alerts.return_value = []

        mock_analytics.get_job_trends.return_value = {
            "status": "success",
            "trends": [{"date": "2024-01-15", "job_count": 5}],
            "total_jobs": 5,
        }
        mock_analytics.get_company_analytics.return_value = {
            "status": "success",
            "companies": [{"company": "Test Corp", "total_jobs": 5}],
        }

        # Call the function
        render_analytics_page()

        # Verify services were initialized
        mock_analytics_cls.assert_called_once()
        mock_cost_monitor_cls.assert_called_once()

        # Verify page title was set
        mock_streamlit.title.assert_called_with("📊 Analytics Dashboard")

        # Verify services were added to session state
        assert "analytics_service" in mock_streamlit.session_state
        assert "cost_monitor" in mock_streamlit.session_state

    def test_analytics_dashboard_session_state_reuse(
        self, mock_streamlit, test_analytics_data
    ):
        """Test that services are reused from session state on subsequent calls."""
        from src.ui.pages.analytics import render_analytics_page

        analytics_service = test_analytics_data["analytics_service"]
        cost_monitor = test_analytics_data["cost_monitor"]

        # Pre-populate session state
        mock_streamlit.session_state["analytics_service"] = analytics_service
        mock_streamlit.session_state["cost_monitor"] = cost_monitor

        # Mock service methods to return data
        with (
            patch.object(cost_monitor, "get_monthly_summary") as mock_summary,
            patch.object(cost_monitor, "get_cost_alerts") as mock_alerts,
            patch.object(analytics_service, "get_job_trends") as mock_trends,
            patch.object(analytics_service, "get_company_analytics") as mock_companies,
        ):
            mock_summary.return_value = {"total_cost": 8.75, "monthly_budget": 50.0}
            mock_alerts.return_value = []
            mock_trends.return_value = {"status": "success", "trends": []}
            mock_companies.return_value = {"status": "success", "companies": []}

            # Call render function
            render_analytics_page()

            # Verify session state services were used
            mock_summary.assert_called()
            mock_alerts.assert_called()

    @patch("src.ui.pages.analytics.is_streamlit_context", return_value=True)
    def test_streamlit_context_detection(self, mock_context_check, mock_streamlit):
        """Test that Streamlit context is properly detected."""
        from src.ui.pages.analytics import render_analytics_page

        # Mock service initialization and responses
        with (
            patch("src.ui.pages.analytics.AnalyticsService") as mock_analytics_cls,
            patch("src.ui.pages.analytics.CostMonitor") as mock_cost_monitor_cls,
        ):
            mock_analytics = Mock()
            mock_cost_monitor = Mock()
            mock_analytics_cls.return_value = mock_analytics
            mock_cost_monitor_cls.return_value = mock_cost_monitor

            # Mock successful responses
            mock_cost_monitor.get_monthly_summary.return_value = {"total_cost": 0.0}
            mock_cost_monitor.get_cost_alerts.return_value = []
            mock_analytics.get_job_trends.return_value = {
                "status": "success",
                "trends": [],
            }
            mock_analytics.get_company_analytics.return_value = {
                "status": "success",
                "companies": [],
            }

            render_analytics_page()

            # Verify context was checked
            mock_context_check.assert_called_once()


class TestCostMonitoringDashboardSection:
    """Test cost monitoring dashboard section rendering."""

    def test_cost_monitoring_section_renders_budget_info(
        self, mock_streamlit, test_analytics_data
    ):
        """Test cost monitoring section renders budget information correctly."""
        from src.ui.pages.analytics import _render_cost_monitoring_section

        cost_monitor = test_analytics_data["cost_monitor"]

        # Mock cost summary data
        cost_summary = {
            "total_cost": 25.50,
            "monthly_budget": 50.0,
            "remaining": 24.50,
            "utilization_percent": 51.0,
            "budget_status": "moderate_usage",
            "costs_by_service": {"ai": 15.00, "proxy": 7.50, "scraping": 3.00},
            "operation_counts": {"ai": 10, "proxy": 5, "scraping": 3},
            "month_year": "January 2024",
        }

        with patch.object(
            cost_monitor, "get_monthly_summary", return_value=cost_summary
        ):
            with patch.object(cost_monitor, "get_cost_alerts", return_value=[]):
                _render_cost_monitoring_section(cost_monitor)

        # Verify section header was rendered
        mock_streamlit.header.assert_called_with("💰 Cost Monitoring")

        # Verify budget metrics were displayed
        assert mock_streamlit.metric.call_count >= 3  # Should show multiple metrics

        # Verify columns were created for layout
        mock_streamlit.columns.assert_called()

    def test_cost_monitoring_section_handles_budget_alerts(
        self, mock_streamlit, test_analytics_data
    ):
        """Test cost monitoring section handles budget alerts properly."""
        from src.ui.pages.analytics import _render_cost_monitoring_section

        cost_monitor = test_analytics_data["cost_monitor"]

        # Mock approaching limit scenario
        cost_summary = {
            "total_cost": 42.50,
            "monthly_budget": 50.0,
            "budget_status": "approaching_limit",
            "utilization_percent": 85.0,
            "costs_by_service": {"ai": 30.0, "proxy": 8.50, "scraping": 4.0},
            "operation_counts": {"ai": 15, "proxy": 8, "scraping": 5},
        }

        cost_alerts = [
            {"type": "warning", "message": "Approaching budget limit: 85% used"}
        ]

        with patch.object(
            cost_monitor, "get_monthly_summary", return_value=cost_summary
        ):
            with patch.object(
                cost_monitor, "get_cost_alerts", return_value=cost_alerts
            ):
                _render_cost_monitoring_section(cost_monitor)

        # Verify warning was displayed
        mock_streamlit.warning.assert_called()

    def test_cost_monitoring_section_over_budget_handling(self, mock_streamlit):
        """Test cost monitoring section handles over-budget scenarios."""
        from src.ui.pages.analytics import _render_cost_monitoring_section

        # Mock cost monitor with over-budget data
        mock_cost_monitor = Mock()

        cost_summary = {
            "total_cost": 65.00,
            "monthly_budget": 50.0,
            "budget_status": "over_budget",
            "utilization_percent": 130.0,
            "costs_by_service": {"ai": 45.0, "proxy": 15.0, "scraping": 5.0},
            "operation_counts": {"ai": 25, "proxy": 12, "scraping": 8},
        }

        cost_alerts = [
            {"type": "error", "message": "Monthly budget exceeded: $65.00 / $50.00"}
        ]

        mock_cost_monitor.get_monthly_summary.return_value = cost_summary
        mock_cost_monitor.get_cost_alerts.return_value = cost_alerts

        _render_cost_monitoring_section(mock_cost_monitor)

        # Verify error alert was displayed
        mock_streamlit.error.assert_called()

    def test_cost_monitoring_section_error_handling(self, mock_streamlit):
        """Test cost monitoring section handles service errors gracefully."""
        from src.ui.pages.analytics import _render_cost_monitoring_section

        # Mock cost monitor that raises an error
        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.side_effect = Exception(
            "Database connection failed"
        )

        _render_cost_monitoring_section(mock_cost_monitor)

        # Verify error message was displayed
        mock_streamlit.error.assert_called()
        error_message = mock_streamlit.error.call_args[0][0]
        assert "Failed to load cost monitoring" in error_message


class TestJobTrendsDashboardSection:
    """Test job trends dashboard section rendering."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_job_trends_section_renders_charts(
        self, mock_streamlit, test_analytics_data
    ):
        """Test job trends section renders charts correctly."""
        from src.ui.pages.analytics import _render_job_trends_section

        analytics_service = test_analytics_data["analytics_service"]

        # Mock trends data
        trends_data = {
            "status": "success",
            "trends": [
                {"date": "2024-01-15", "job_count": 5},
                {"date": "2024-01-16", "job_count": 8},
                {"date": "2024-01-17", "job_count": 3},
            ],
            "total_jobs": 16,
            "method": "duckdb_sqlite_scanner",
        }

        with patch.object(
            analytics_service, "get_job_trends", return_value=trends_data
        ):
            _render_job_trends_section(analytics_service)

        # Verify section header was rendered
        mock_streamlit.header.assert_called_with("📈 Job Market Trends")

        # Verify chart was displayed
        mock_streamlit.plotly_chart.assert_called()

    def test_job_trends_section_handles_empty_data(self, mock_streamlit):
        """Test job trends section handles empty data gracefully."""
        from src.ui.pages.analytics import _render_job_trends_section

        mock_analytics = Mock()
        trends_data = {"status": "success", "trends": [], "total_jobs": 0}

        mock_analytics.get_job_trends.return_value = trends_data

        _render_job_trends_section(mock_analytics)

        # Should show info message about no data
        mock_streamlit.info.assert_called()

    def test_job_trends_section_error_handling(self, mock_streamlit):
        """Test job trends section handles service errors."""
        from src.ui.pages.analytics import _render_job_trends_section

        mock_analytics = Mock()
        trends_data = {"status": "error", "error": "DuckDB unavailable", "trends": []}

        mock_analytics.get_job_trends.return_value = trends_data

        _render_job_trends_section(mock_analytics)

        # Should display error message
        mock_streamlit.error.assert_called()
        error_message = mock_streamlit.error.call_args[0][0]
        assert "DuckDB unavailable" in error_message

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_job_trends_section_plotly_integration(
        self, mock_streamlit, test_analytics_data
    ):
        """Test job trends section creates proper Plotly charts."""
        from src.ui.pages.analytics import _render_job_trends_section

        analytics_service = test_analytics_data["analytics_service"]

        # Mock trends data with realistic values
        trends_data = {
            "status": "success",
            "trends": [
                {"date": "2024-01-15", "job_count": 12},
                {"date": "2024-01-16", "job_count": 15},
                {"date": "2024-01-17", "job_count": 8},
            ],
            "total_jobs": 35,
        }

        with patch.object(
            analytics_service, "get_job_trends", return_value=trends_data
        ):
            with patch("src.ui.pages.analytics.px.line") as mock_px_line:
                mock_fig = Mock()
                mock_px_line.return_value = mock_fig

                _render_job_trends_section(analytics_service)

                # Verify Plotly line chart was created
                mock_px_line.assert_called_once()
                mock_streamlit.plotly_chart.assert_called_with(
                    mock_fig, use_container_width=True
                )


class TestCompanyAnalyticsDashboardSection:
    """Test company analytics dashboard section rendering."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_company_analytics_section_renders_data(
        self, mock_streamlit, test_analytics_data
    ):
        """Test company analytics section renders company data correctly."""
        from src.ui.pages.analytics import _render_company_analytics_section

        analytics_service = test_analytics_data["analytics_service"]

        # Mock company analytics data
        company_data = {
            "status": "success",
            "companies": [
                {
                    "company": "UI Test Corp",
                    "total_jobs": 15,
                    "avg_min_salary": 95000.0,
                    "avg_max_salary": 125000.0,
                    "last_job_posted": "2024-01-17",
                },
                {
                    "company": "Dashboard Inc",
                    "total_jobs": 8,
                    "avg_min_salary": 105000.0,
                    "avg_max_salary": 140000.0,
                    "last_job_posted": "2024-01-16",
                },
            ],
            "total_companies": 2,
        }

        with patch.object(
            analytics_service, "get_company_analytics", return_value=company_data
        ):
            _render_company_analytics_section(analytics_service)

        # Verify section header was rendered
        mock_streamlit.header.assert_called_with("🏢 Company Hiring Analysis")

        # Verify dataframe was displayed
        mock_streamlit.dataframe.assert_called()

    def test_company_analytics_section_handles_no_data(self, mock_streamlit):
        """Test company analytics section handles empty company data."""
        from src.ui.pages.analytics import _render_company_analytics_section

        mock_analytics = Mock()
        company_data = {"status": "success", "companies": [], "total_companies": 0}

        mock_analytics.get_company_analytics.return_value = company_data

        _render_company_analytics_section(mock_analytics)

        # Should show info message about no data
        mock_streamlit.info.assert_called()

    def test_company_analytics_section_error_handling(self, mock_streamlit):
        """Test company analytics section handles service errors."""
        from src.ui.pages.analytics import _render_company_analytics_section

        mock_analytics = Mock()
        company_data = {
            "status": "error",
            "error": "Database query failed",
            "companies": [],
        }

        mock_analytics.get_company_analytics.return_value = company_data

        _render_company_analytics_section(mock_analytics)

        # Should display error message
        mock_streamlit.error.assert_called()


class TestSalaryAnalyticsDashboardSection:
    """Test salary analytics dashboard section rendering."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_salary_analytics_section_renders_metrics(
        self, mock_streamlit, test_analytics_data
    ):
        """Test salary analytics section renders salary metrics correctly."""
        from src.ui.pages.analytics import _render_salary_analytics_section

        analytics_service = test_analytics_data["analytics_service"]

        # Mock salary analytics data
        salary_data = {
            "status": "success",
            "salary_data": {
                "total_jobs_with_salary": 25,
                "avg_min_salary": 98000.50,
                "avg_max_salary": 128500.75,
                "min_salary": 75000.0,
                "max_salary": 180000.0,
                "salary_std_dev": 15500.25,
                "analysis_period_days": 90,
            },
        }

        with patch.object(
            analytics_service, "get_salary_analytics", return_value=salary_data
        ):
            _render_salary_analytics_section(analytics_service)

        # Verify section header was rendered
        mock_streamlit.header.assert_called_with("💰 Salary Analytics")

        # Verify salary metrics were displayed
        assert (
            mock_streamlit.metric.call_count >= 4
        )  # Should show multiple salary metrics

    def test_salary_analytics_section_handles_no_salary_data(self, mock_streamlit):
        """Test salary analytics section handles jobs without salary data."""
        from src.ui.pages.analytics import _render_salary_analytics_section

        mock_analytics = Mock()
        salary_data = {
            "status": "success",
            "salary_data": {
                "total_jobs_with_salary": 0,
                "avg_min_salary": 0,
                "avg_max_salary": 0,
                "min_salary": 0,
                "max_salary": 0,
                "salary_std_dev": 0,
                "analysis_period_days": 90,
            },
        }

        mock_analytics.get_salary_analytics.return_value = salary_data

        _render_salary_analytics_section(mock_analytics)

        # Should show info message about no salary data
        mock_streamlit.info.assert_called()

    def test_salary_analytics_section_error_handling(self, mock_streamlit):
        """Test salary analytics section handles service errors."""
        from src.ui.pages.analytics import _render_salary_analytics_section

        mock_analytics = Mock()
        salary_data = {
            "status": "error",
            "error": "Statistical calculation failed",
            "salary_data": {},
        }

        mock_analytics.get_salary_analytics.return_value = salary_data

        _render_salary_analytics_section(mock_analytics)

        # Should display error message
        mock_streamlit.error.assert_called()


class TestAnalyticsDashboardIntegration:
    """Test full analytics dashboard integration scenarios."""

    def test_complete_dashboard_rendering_workflow(
        self, mock_streamlit, test_analytics_data
    ):
        """Test complete dashboard rendering workflow with real services."""
        from src.ui.pages.analytics import render_analytics_page

        analytics_service = test_analytics_data["analytics_service"]
        cost_monitor = test_analytics_data["cost_monitor"]

        # Pre-populate session state to avoid service initialization
        mock_streamlit.session_state["analytics_service"] = analytics_service
        mock_streamlit.session_state["cost_monitor"] = cost_monitor

        # Mock all service calls to return realistic data
        with (
            patch.object(cost_monitor, "get_monthly_summary") as mock_cost_summary,
            patch.object(cost_monitor, "get_cost_alerts") as mock_cost_alerts,
            patch.object(analytics_service, "get_job_trends") as mock_job_trends,
            patch.object(
                analytics_service, "get_company_analytics"
            ) as mock_company_analytics,
        ):
            # Mock responses
            mock_cost_summary.return_value = {
                "total_cost": 8.75,
                "monthly_budget": 50.0,
                "budget_status": "within_budget",
                "costs_by_service": {"ai": 5.0, "proxy": 2.5, "scraping": 1.25},
                "utilization_percent": 17.5,
            }
            mock_cost_alerts.return_value = []

            mock_job_trends.return_value = {
                "status": "success",
                "trends": [{"date": "2024-01-17", "job_count": 2}],
                "total_jobs": 2,
            }

            mock_company_analytics.return_value = {
                "status": "success",
                "companies": [
                    {"company": "UI Test Corp", "total_jobs": 1},
                    {"company": "Dashboard Inc", "total_jobs": 1},
                ],
            }

            # Render the full dashboard
            render_analytics_page()

            # Verify all sections were rendered
            assert mock_streamlit.title.call_count >= 1
            assert (
                mock_streamlit.header.call_count >= 3
            )  # Cost, trends, companies sections

            # Verify all services were called
            mock_cost_summary.assert_called_once()
            mock_cost_alerts.assert_called_once()
            if DUCKDB_AVAILABLE:
                mock_job_trends.assert_called_once()
                mock_company_analytics.assert_called_once()

    def test_dashboard_error_recovery(self, mock_streamlit):
        """Test dashboard handles section errors without failing completely."""
        from src.ui.pages.analytics import render_analytics_page

        # Mock services with mixed success/failure
        mock_analytics = Mock()
        mock_cost_monitor = Mock()

        # Cost monitor succeeds
        mock_cost_monitor.get_monthly_summary.return_value = {"total_cost": 0.0}
        mock_cost_monitor.get_cost_alerts.return_value = []

        # Analytics service fails
        mock_analytics.get_job_trends.return_value = {
            "status": "error",
            "error": "Service unavailable",
        }
        mock_analytics.get_company_analytics.return_value = {
            "status": "error",
            "error": "Service unavailable",
        }

        # Pre-populate session state
        mock_streamlit.session_state["analytics_service"] = mock_analytics
        mock_streamlit.session_state["cost_monitor"] = mock_cost_monitor

        # Render dashboard
        render_analytics_page()

        # Dashboard should still render despite analytics failures
        mock_streamlit.title.assert_called_with("📊 Analytics Dashboard")

        # Error messages should be shown for failed sections
        mock_streamlit.error.assert_called()

    def test_dashboard_performance_monitoring(
        self, mock_streamlit, test_analytics_data
    ):
        """Test dashboard rendering performance characteristics."""
        import time

        from src.ui.pages.analytics import render_analytics_page

        analytics_service = test_analytics_data["analytics_service"]
        cost_monitor = test_analytics_data["cost_monitor"]

        # Pre-populate session state
        mock_streamlit.session_state["analytics_service"] = analytics_service
        mock_streamlit.session_state["cost_monitor"] = cost_monitor

        # Measure rendering time
        start_time = time.time()
        render_analytics_page()
        render_time = time.time() - start_time

        # Dashboard should render quickly (allowing for mock overhead)
        assert render_time < 2.0  # Should complete in under 2 seconds

    @patch("src.ui.pages.analytics.pd.DataFrame")
    def test_dashboard_data_formatting(
        self, mock_dataframe, mock_streamlit, test_analytics_data
    ):
        """Test that dashboard properly formats data for display."""
        from src.ui.pages.analytics import _render_company_analytics_section

        analytics_service = test_analytics_data["analytics_service"]

        company_data = {
            "status": "success",
            "companies": [
                {
                    "company": "Test Corp",
                    "total_jobs": 5,
                    "avg_min_salary": 100000.50,
                    "avg_max_salary": 130000.75,
                    "last_job_posted": "2024-01-17",
                }
            ],
        }

        with patch.object(
            analytics_service, "get_company_analytics", return_value=company_data
        ):
            _render_company_analytics_section(analytics_service)

        # Verify DataFrame was created with proper data
        mock_dataframe.assert_called_once()
        df_call_args = mock_dataframe.call_args[0][0]
        assert len(df_call_args) == 1  # One company
        assert df_call_args[0]["company"] == "Test Corp"

    def test_dashboard_responsive_layout(self, mock_streamlit):
        """Test dashboard creates responsive layout with proper columns."""
        from src.ui.pages.analytics import _render_cost_monitoring_section

        mock_cost_monitor = Mock()
        mock_cost_monitor.get_monthly_summary.return_value = {
            "total_cost": 25.0,
            "monthly_budget": 50.0,
            "budget_status": "within_budget",
            "costs_by_service": {"ai": 20.0, "proxy": 5.0},
        }
        mock_cost_monitor.get_cost_alerts.return_value = []

        # Mock columns to return column objects
        col1, col2, col3 = Mock(), Mock(), Mock()
        mock_streamlit.columns.return_value = [col1, col2, col3]

        _render_cost_monitoring_section(mock_cost_monitor)

        # Verify responsive column layout was created
        mock_streamlit.columns.assert_called()

        # Verify metrics were added to columns
        for col in [col1, col2, col3]:
            col.metric.assert_called() if hasattr(col, "metric") else None
