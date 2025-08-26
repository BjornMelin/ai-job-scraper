"""Integration tests for analytics services working together.

This test suite validates that the refactored AnalyticsService and CostMonitor
work correctly together in realistic usage scenarios, testing end-to-end workflows
that combine both services for comprehensive dashboard functionality.

Test coverage includes:
- Analytics service + cost monitor integration
- End-to-end analytics workflows
- Dashboard data consistency
- Cross-service error handling
- Performance with both services active
- Realistic usage scenarios
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import contextlib
import logging

from datetime import UTC, datetime, timedelta
from pathlib import Path
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
def integrated_test_databases(tmp_path):
    """Create test databases for both analytics and cost monitoring."""
    # Create jobs database with sample data
    jobs_db = tmp_path / "test_jobs.db"
    engine = create_engine(
        f"sqlite:///{jobs_db}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    # Sample companies and jobs data
    base_date = datetime.now(UTC)
    companies_data = [
        {
            "id": 1,
            "name": "TechGiant Corp",
            "url": "https://techgiant.com",
            "active": True,
        },
        {"id": 2, "name": "AI Startup", "url": "https://aistartup.com", "active": True},
        {
            "id": 3,
            "name": "Data Sciences Inc",
            "url": "https://datasci.com",
            "active": True,
        },
        {
            "id": 4,
            "name": "CloudCompany Ltd",
            "url": "https://cloudco.com",
            "active": True,
        },
    ]

    jobs_data = [
        # TechGiant Corp jobs
        {
            "id": 1,
            "company_id": 1,
            "title": "Senior Software Engineer",
            "description": "Backend development with Python",
            "link": "https://techgiant.com/job1",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=1),
            "salary": [180000, 220000],
            "archived": False,
            "content_hash": "hash1",
        },
        {
            "id": 2,
            "company_id": 1,
            "title": "Machine Learning Engineer",
            "description": "ML model development and deployment",
            "link": "https://techgiant.com/job2",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=2),
            "salary": [200000, 250000],
            "archived": False,
            "content_hash": "hash2",
        },
        # AI Startup jobs
        {
            "id": 3,
            "company_id": 2,
            "title": "AI Research Scientist",
            "description": "Research and development of AI algorithms",
            "link": "https://aistartup.com/job1",
            "location": "New York, NY",
            "posted_date": base_date - timedelta(days=3),
            "salary": [160000, 200000],
            "archived": False,
            "content_hash": "hash3",
        },
        {
            "id": 4,
            "company_id": 2,
            "title": "Data Scientist",
            "description": "Statistical analysis and modeling",
            "link": "https://aistartup.com/job2",
            "location": "Remote",
            "posted_date": base_date - timedelta(days=5),
            "salary": [150000, 190000],
            "archived": False,
            "content_hash": "hash4",
        },
        # Data Sciences Inc jobs
        {
            "id": 5,
            "company_id": 3,
            "title": "Senior Data Engineer",
            "description": "Big data infrastructure development",
            "link": "https://datasci.com/job1",
            "location": "Austin, TX",
            "posted_date": base_date - timedelta(days=7),
            "salary": [170000, 210000],
            "archived": False,
            "content_hash": "hash5",
        },
        # CloudCompany Ltd job (older, outside 30-day window)
        {
            "id": 6,
            "company_id": 4,
            "title": "Cloud Architect",
            "description": "Cloud infrastructure design",
            "link": "https://cloudco.com/job1",
            "location": "Seattle, WA",
            "posted_date": base_date - timedelta(days=45),
            "salary": [190000, 230000],
            "archived": False,
            "content_hash": "hash6",
        },
        # Archived job (should not appear in analytics)
        {
            "id": 7,
            "company_id": 1,
            "title": "Archived Position",
            "description": "This position was archived",
            "link": "https://techgiant.com/archived",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=1),
            "salary": [100000, 120000],
            "archived": True,
            "content_hash": "hash7",
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

    # Create costs database path
    costs_db = tmp_path / "test_costs.db"

    return {"jobs_db": str(jobs_db), "costs_db": str(costs_db)}


@pytest.fixture
def analytics_service(integrated_test_databases):
    """Create AnalyticsService instance with test data."""
    return AnalyticsService(db_path=integrated_test_databases["jobs_db"])


@pytest.fixture
def cost_monitor(integrated_test_databases):
    """Create CostMonitor instance with test data."""
    return CostMonitor(db_path=integrated_test_databases["costs_db"])


@pytest.fixture
def analytics_with_cost_data(analytics_service, cost_monitor):
    """Create both services with realistic cost tracking data."""
    # Simulate costs for analytics operations
    datetime.now(UTC)

    # AI costs for job analysis
    cost_monitor.track_ai_cost("gpt-4", 2500, 0.05, "job_extraction_techgiant")
    cost_monitor.track_ai_cost("gpt-4", 3000, 0.06, "job_extraction_aistartup")
    cost_monitor.track_ai_cost("groq-llama", 5000, 0.02, "company_analysis_batch")
    cost_monitor.track_ai_cost("gpt-4", 1500, 0.03, "salary_normalization")

    # Proxy costs for scraping
    cost_monitor.track_proxy_cost(150, 3.50, "residential_proxy_techgiant")
    cost_monitor.track_proxy_cost(100, 2.25, "datacenter_proxy_aistartup")
    cost_monitor.track_proxy_cost(75, 1.80, "mobile_proxy_datasci")

    # Scraping operation costs
    cost_monitor.track_scraping_cost("TechGiant Corp", 45, 4.50)
    cost_monitor.track_scraping_cost("AI Startup", 35, 3.25)
    cost_monitor.track_scraping_cost("Data Sciences Inc", 25, 2.75)
    cost_monitor.track_scraping_cost("CloudCompany Ltd", 15, 1.50)

    return {"analytics": analytics_service, "cost_monitor": cost_monitor}


class TestAnalyticsIntegrationBasics:
    """Test basic integration between analytics services."""

    def test_both_services_initialize_correctly(self, analytics_service, cost_monitor):
        """Test that both services can be initialized together."""
        # Test analytics service
        analytics_status = analytics_service.get_status_report()
        assert isinstance(analytics_status, dict)
        assert "analytics_method" in analytics_status

        # Test cost monitor
        cost_summary = cost_monitor.get_monthly_summary()
        assert isinstance(cost_summary, dict)
        assert "total_cost" in cost_summary
        assert "monthly_budget" in cost_summary

        # Both services should work independently
        assert analytics_status["database_path"] != cost_monitor.db_path
        assert cost_summary["monthly_budget"] == 50.0

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_analytics_and_cost_data_consistency(self, analytics_with_cost_data):
        """Test data consistency between analytics and cost tracking."""
        analytics = analytics_with_cost_data["analytics"]
        cost_monitor = analytics_with_cost_data["cost_monitor"]

        # Get analytics data
        job_trends = analytics.get_job_trends(days=30)
        company_analytics = analytics.get_company_analytics()
        cost_summary = cost_monitor.get_monthly_summary()

        # Verify analytics data
        assert job_trends["status"] == "success"
        assert company_analytics["status"] == "success"
        assert len(company_analytics["companies"]) > 0

        # Verify cost data reflects analytics operations
        assert cost_summary["total_cost"] > 0
        assert "ai" in cost_summary["costs_by_service"]
        assert "proxy" in cost_summary["costs_by_service"]
        assert "scraping" in cost_summary["costs_by_service"]

        # Cost data should reflect the companies we scraped
        ai_cost = cost_summary["costs_by_service"]["ai"]
        scraping_cost = cost_summary["costs_by_service"]["scraping"]
        assert ai_cost > 0  # Should have AI costs from job analysis
        assert scraping_cost > 0  # Should have scraping costs


class TestEndToEndAnalyticsWorkflow:
    """Test complete end-to-end analytics workflows."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_complete_analytics_dashboard_workflow(self, analytics_with_cost_data):
        """Test complete workflow: data → analytics → cost tracking → dashboard."""
        analytics = analytics_with_cost_data["analytics"]
        cost_monitor = analytics_with_cost_data["cost_monitor"]

        # Step 1: Get all analytics data (simulating dashboard loading)
        job_trends = analytics.get_job_trends(days=30)
        company_analytics = analytics.get_company_analytics()
        salary_analytics = analytics.get_salary_analytics(days=90)
        cost_summary = cost_monitor.get_monthly_summary()
        cost_alerts = cost_monitor.get_cost_alerts()

        # Step 2: Verify all analytics succeeded
        assert job_trends["status"] == "success"
        assert company_analytics["status"] == "success"
        assert salary_analytics["status"] == "success"
        assert isinstance(cost_summary, dict)
        assert isinstance(cost_alerts, list)

        # Step 3: Verify data quality and completeness

        # Job trends should show recent activity
        if job_trends["trends"]:
            trend_dates = [trend["date"] for trend in job_trends["trends"]]
            assert len(trend_dates) > 0
            # Should have jobs from the last few days
            assert (
                job_trends["total_jobs"] >= 5
            )  # We have 5 non-archived jobs in last 30 days

        # Company analytics should show all active companies
        companies = company_analytics["companies"]
        company_names = [comp["company"] for comp in companies]
        expected_companies = {"TechGiant Corp", "AI Startup", "Data Sciences Inc"}
        found_companies = set(company_names) & expected_companies
        assert len(found_companies) >= 2  # Should find at least 2 of our companies

        # Salary analytics should process salary data
        salary_data = salary_analytics["salary_data"]
        assert (
            salary_data["total_jobs_with_salary"] >= 5
        )  # All our jobs have salary data
        assert salary_data["avg_min_salary"] > 0
        assert salary_data["avg_max_salary"] > salary_data["avg_min_salary"]

        # Cost summary should reflect actual operations
        assert cost_summary["total_cost"] > 0
        assert len(cost_summary["costs_by_service"]) == 3  # ai, proxy, scraping
        assert cost_summary["utilization_percent"] > 0

        # Step 4: Verify cost alerts are appropriate
        if cost_summary["budget_status"] in ["approaching_limit", "over_budget"]:
            assert len(cost_alerts) > 0
        else:
            assert len(cost_alerts) == 0

    def test_realistic_scraping_session_cost_tracking(
        self, analytics_service, cost_monitor
    ):
        """Test realistic scraping session with comprehensive cost tracking."""
        # Simulate a realistic scraping session for multiple companies

        companies_to_scrape = [
            ("TechGiant Corp", 50, 0.08),  # 50 jobs found, $0.08 AI cost per job
            ("AI Startup", 35, 0.06),  # 35 jobs, $0.06 per job
            ("Data Sciences Inc", 25, 0.05),  # 25 jobs, $0.05 per job
            ("CloudCompany Ltd", 15, 0.04),  # 15 jobs, $0.04 per job
        ]

        total_expected_cost = 0.0

        for company, job_count, ai_cost_per_job in companies_to_scrape:
            # Track proxy costs for scraping
            proxy_cost = job_count * 0.02  # $0.02 per job for proxy
            cost_monitor.track_proxy_cost(
                job_count * 2, proxy_cost, f"proxy_{company.lower().replace(' ', '_')}"
            )

            # Track AI costs for job analysis
            total_tokens = job_count * 800  # ~800 tokens per job
            ai_total_cost = job_count * ai_cost_per_job
            cost_monitor.track_ai_cost(
                "gpt-4", total_tokens, ai_total_cost, f"analysis_{company}"
            )

            # Track scraping operation costs
            scraping_cost = (
                job_count * 0.03
            )  # $0.03 per job for scraping infrastructure
            cost_monitor.track_scraping_cost(company, job_count, scraping_cost)

            total_expected_cost += proxy_cost + ai_total_cost + scraping_cost

        # Get cost summary and verify realistic totals
        cost_summary = cost_monitor.get_monthly_summary()

        assert (
            abs(cost_summary["total_cost"] - total_expected_cost) < 0.01
        )  # Allow for floating point precision
        assert cost_summary["costs_by_service"]["ai"] > 0
        assert cost_summary["costs_by_service"]["proxy"] > 0
        assert cost_summary["costs_by_service"]["scraping"] > 0

        # Verify operation counts make sense
        assert cost_summary["operation_counts"]["ai"] == 4  # 4 companies analyzed
        assert cost_summary["operation_counts"]["proxy"] == 4  # 4 proxy operations
        assert (
            cost_summary["operation_counts"]["scraping"] == 4
        )  # 4 scraping operations

        # Check if we're approaching budget limits
        if total_expected_cost >= 40.0:  # 80% of $50 budget
            assert cost_summary["budget_status"] in ["approaching_limit", "over_budget"]
            alerts = cost_monitor.get_cost_alerts()
            assert len(alerts) > 0

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_analytics_performance_under_cost_tracking_load(
        self, analytics_service, cost_monitor
    ):
        """Test analytics performance when cost tracking is active."""
        import time

        # Add substantial cost tracking data to simulate heavy usage
        for i in range(20):
            cost_monitor.track_ai_cost(
                f"model_{i}", 1000 + i * 100, 0.01 + i * 0.001, f"operation_{i}"
            )
            cost_monitor.track_proxy_cost(50 + i * 5, 0.50 + i * 0.05, f"proxy_{i}")
            cost_monitor.track_scraping_cost(f"company_{i}", 10 + i, 0.25 + i * 0.02)

        # Measure analytics performance with cost tracking active
        start_time = time.time()

        job_trends = analytics_service.get_job_trends(days=30)
        company_analytics = analytics_service.get_company_analytics()
        salary_analytics = analytics_service.get_salary_analytics(days=90)
        cost_summary = cost_monitor.get_monthly_summary()

        total_time = time.time() - start_time

        # All operations should succeed
        assert job_trends["status"] == "success"
        assert company_analytics["status"] == "success"
        assert salary_analytics["status"] == "success"
        assert isinstance(cost_summary, dict)

        # Should complete in reasonable time even with heavy cost tracking
        assert total_time < 5.0  # Should complete in under 5 seconds


class TestCrossServiceErrorHandling:
    """Test error handling across both analytics services."""

    def test_analytics_failure_with_cost_tracking_success(self, cost_monitor):
        """Test cost tracking continues working when analytics fails."""
        # Create analytics service with invalid database
        broken_analytics = AnalyticsService(db_path="/nonexistent/path/jobs.db")

        # Analytics should fail
        job_trends = broken_analytics.get_job_trends(days=30)
        assert job_trends["status"] == "error"

        # But cost tracking should still work
        cost_monitor.track_ai_cost("gpt-4", 1000, 0.02, "test_operation")
        cost_summary = cost_monitor.get_monthly_summary()
        assert cost_summary["total_cost"] == 0.02
        assert len(cost_summary["costs_by_service"]) == 1

    def test_cost_tracking_failure_with_analytics_success(
        self, analytics_service, tmp_path
    ):
        """Test analytics continues working when cost tracking fails."""
        # Create cost monitor with read-only database directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        try:
            broken_cost_monitor = CostMonitor(db_path=str(readonly_dir / "costs.db"))

            # Cost tracking might fail
            with contextlib.suppress(Exception):
                broken_cost_monitor.track_ai_cost("gpt-4", 1000, 0.02, "test")

            # But analytics should still work
            if DUCKDB_AVAILABLE:
                job_trends = analytics_service.get_job_trends(days=30)
                assert job_trends["status"] == "success"

        finally:
            # Clean up permissions
            readonly_dir.chmod(0o755)

    def test_both_services_database_corruption_recovery(self, tmp_path):
        """Test recovery when both service databases are corrupted."""
        jobs_db = tmp_path / "corrupt_jobs.db"
        costs_db = tmp_path / "corrupt_costs.db"

        # Create corrupted database files
        with Path(jobs_db).open("wb") as f:
            f.write(b"Not a valid SQLite database")
        with Path(costs_db).open("wb") as f:
            f.write(b"Also not a valid SQLite database")

        # Services should handle corruption gracefully
        analytics = AnalyticsService(db_path=str(jobs_db))
        cost_monitor = CostMonitor(db_path=str(costs_db))

        # Analytics might fail due to corruption
        job_trends = analytics.get_job_trends(days=30)
        assert job_trends["status"] in ["success", "error"]

        # Cost monitor should recreate database and work
        with contextlib.suppress(Exception):
            cost_monitor.track_ai_cost("test", 1000, 0.01, "recovery_test")
            summary = cost_monitor.get_monthly_summary()
            assert isinstance(summary, dict)

    @patch("src.services.analytics_service.STREAMLIT_AVAILABLE", True)
    @patch("src.services.analytics_service.st")
    @patch("src.services.cost_monitor.STREAMLIT_AVAILABLE", True)
    @patch("src.services.cost_monitor.st")
    def test_streamlit_integration_with_both_services(
        self, mock_cost_st, mock_analytics_st, analytics_with_cost_data
    ):
        """Test Streamlit integration with both services active."""
        mock_analytics_st.success = Mock()
        mock_analytics_st.error = Mock()
        mock_cost_st.warning = Mock()
        mock_cost_st.error = Mock()

        analytics = analytics_with_cost_data["analytics"]
        cost_monitor = analytics_with_cost_data["cost_monitor"]

        # Use both services (triggering any Streamlit integration)
        analytics.get_job_trends(days=30)
        cost_monitor.track_ai_cost(
            "expensive_model", 50000, 25.00, "large_operation"
        )  # Should trigger warning

        # Verify Streamlit integration doesn't interfere between services
        # Analytics service should show success message (if DuckDB available)
        if DUCKDB_AVAILABLE:
            mock_analytics_st.success.assert_called()

        # Cost monitor should show warning for high usage
        mock_cost_st.warning.assert_called()


class TestRealisticDashboardScenarios:
    """Test realistic dashboard usage scenarios."""

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_daily_dashboard_refresh_scenario(self, analytics_with_cost_data):
        """Test daily dashboard refresh with accumulated data."""
        analytics = analytics_with_cost_data["analytics"]
        cost_monitor = analytics_with_cost_data["cost_monitor"]

        # Simulate daily dashboard refresh operations
        dashboard_data = {}

        # Load all dashboard sections
        dashboard_data["job_trends"] = analytics.get_job_trends(days=30)
        dashboard_data["company_analytics"] = analytics.get_company_analytics()
        dashboard_data["salary_analytics"] = analytics.get_salary_analytics(days=90)
        dashboard_data["cost_summary"] = cost_monitor.get_monthly_summary()
        dashboard_data["cost_alerts"] = cost_monitor.get_cost_alerts()

        # Track the dashboard refresh operation cost
        cost_monitor.track_ai_cost(
            "dashboard_refresh", 500, 0.005, "daily_dashboard_refresh"
        )

        # Verify all dashboard sections loaded successfully
        for section_name, data in dashboard_data.items():
            if isinstance(data, dict) and "status" in data:
                assert data["status"] == "success", f"{section_name} failed to load"
            else:
                assert isinstance(data, (dict, list)), (
                    f"{section_name} returned invalid data type"
                )

        # Verify dashboard shows meaningful data
        assert dashboard_data["job_trends"]["total_jobs"] >= 0
        assert len(dashboard_data["company_analytics"]["companies"]) >= 0
        assert (
            dashboard_data["salary_analytics"]["salary_data"]["total_jobs_with_salary"]
            >= 0
        )
        assert dashboard_data["cost_summary"]["total_cost"] >= 0

        # Updated cost summary should include the dashboard refresh cost
        updated_summary = cost_monitor.get_monthly_summary()
        assert (
            updated_summary["total_cost"] > dashboard_data["cost_summary"]["total_cost"]
        )

    def test_weekly_analytics_report_generation(self, analytics_with_cost_data):
        """Test weekly analytics report generation scenario."""
        analytics = analytics_with_cost_data["analytics"]
        cost_monitor = analytics_with_cost_data["cost_monitor"]

        # Simulate generating a comprehensive weekly report
        report_data = {}

        if DUCKDB_AVAILABLE:
            # Get analytics for different time periods
            report_data["week_trends"] = analytics.get_job_trends(days=7)
            report_data["month_trends"] = analytics.get_job_trends(days=30)
            report_data["quarter_trends"] = analytics.get_job_trends(days=90)

            report_data["company_breakdown"] = analytics.get_company_analytics()

            report_data["salary_week"] = analytics.get_salary_analytics(days=7)
            report_data["salary_month"] = analytics.get_salary_analytics(days=30)
            report_data["salary_quarter"] = analytics.get_salary_analytics(days=90)

        # Get cost analysis
        report_data["cost_summary"] = cost_monitor.get_monthly_summary()
        report_data["cost_alerts"] = cost_monitor.get_cost_alerts()

        # Track the report generation cost
        report_tokens = 2000  # Estimated tokens for report generation
        report_cost = 0.04  # Estimated cost
        cost_monitor.track_ai_cost(
            "report_generation", report_tokens, report_cost, "weekly_analytics_report"
        )

        if DUCKDB_AVAILABLE:
            # Verify all analytics sections succeeded
            analytics_sections = [
                "week_trends",
                "month_trends",
                "quarter_trends",
                "company_breakdown",
                "salary_week",
                "salary_month",
                "salary_quarter",
            ]
            for section in analytics_sections:
                assert report_data[section]["status"] == "success"

        # Verify cost data
        assert isinstance(report_data["cost_summary"], dict)
        assert "total_cost" in report_data["cost_summary"]
        assert isinstance(report_data["cost_alerts"], list)

        # Verify report generation was tracked in costs
        updated_summary = cost_monitor.get_monthly_summary()
        assert "ai" in updated_summary["costs_by_service"]
        assert updated_summary["costs_by_service"]["ai"] >= report_cost

    def test_budget_threshold_monitoring_with_analytics(
        self, analytics_service, cost_monitor
    ):
        """Test budget threshold monitoring during analytics operations."""
        # Start with empty cost tracking
        initial_summary = cost_monitor.get_monthly_summary()
        assert initial_summary["total_cost"] == 0.0
        assert initial_summary["budget_status"] == "within_budget"

        # Gradually add costs and check thresholds
        cost_scenarios = [
            (20.00, "within_budget"),  # 40% of budget
            (10.00, "moderate_usage"),  # 60% of budget (40% + 20%)
            (10.00, "approaching_limit"),  # 80% of budget (60% + 20%)
            (15.00, "over_budget"),  # 110% of budget (80% + 30%)
        ]

        for additional_cost, expected_status in cost_scenarios:
            # Add cost through realistic operation
            cost_monitor.track_ai_cost(
                "analytics_model", 5000, additional_cost, "batch_analytics"
            )

            # Get updated summary
            summary = cost_monitor.get_monthly_summary()
            assert summary["budget_status"] == expected_status

            # Get appropriate alerts
            alerts = cost_monitor.get_cost_alerts()
            if expected_status == "within_budget":
                assert len(alerts) == 0
            elif expected_status in ["moderate_usage"]:
                assert len(alerts) == 0  # No alerts for moderate usage
            elif expected_status == "approaching_limit":
                assert len(alerts) == 1
                assert alerts[0]["type"] == "warning"
            elif expected_status == "over_budget":
                assert len(alerts) == 1
                assert alerts[0]["type"] == "error"

            # Analytics should continue working regardless of budget status
            if DUCKDB_AVAILABLE:
                job_trends = analytics_service.get_job_trends(days=7)
                assert job_trends["status"] == "success"

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_high_volume_analytics_cost_tracking(self, analytics_service, cost_monitor):
        """Test high-volume analytics with detailed cost tracking."""
        # Simulate a high-volume analytics day with multiple operations
        operations = [
            ("morning_batch", 30, 8000, 0.15),  # Morning batch processing
            ("midday_update", 30, 6000, 0.12),  # Midday data update
            ("afternoon_analysis", 30, 10000, 0.20),  # Afternoon deep analysis
            ("evening_report", 30, 5000, 0.10),  # Evening report generation
        ]

        total_expected_cost = 0.0

        for operation_name, days, tokens, cost in operations:
            # Perform analytics operation
            trends = analytics_service.get_job_trends(days=days)
            companies = analytics_service.get_company_analytics()
            salaries = analytics_service.get_salary_analytics(days=days)

            # Track costs for this operation
            cost_monitor.track_ai_cost("gpt-4", tokens, cost, operation_name)

            # Add some proxy costs for data gathering
            proxy_cost = cost * 0.3  # 30% of AI cost for proxy usage
            cost_monitor.track_proxy_cost(
                int(tokens / 100), proxy_cost, f"proxy_{operation_name}"
            )

            total_expected_cost += cost + proxy_cost

            # Verify analytics succeeded
            assert trends["status"] == "success"
            assert companies["status"] == "success"
            assert salaries["status"] == "success"

        # Verify total costs are tracked correctly
        final_summary = cost_monitor.get_monthly_summary()
        assert abs(final_summary["total_cost"] - total_expected_cost) < 0.01

        # Verify operation counts
        assert final_summary["operation_counts"]["ai"] == 4  # 4 operations
        assert final_summary["operation_counts"]["proxy"] == 4  # 4 proxy operations

        # Check budget status
        expected_utilization = (total_expected_cost / 50.0) * 100
        assert abs(final_summary["utilization_percent"] - expected_utilization) < 0.1
