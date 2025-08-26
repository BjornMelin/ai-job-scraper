"""Shared fixtures for analytics testing.

This module provides reusable pytest fixtures for testing the analytics
architecture including AnalyticsService, CostMonitor, and UI components.

Fixtures include:
- Database setup with sample data
- Service initialization with test configurations
- Mock Streamlit components
- Sample analytics and cost data
- Performance testing utilities
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

from datetime import UTC, datetime, timedelta

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.models import CompanySQL, JobSQL
from src.services.analytics_service import AnalyticsService
from src.services.cost_monitor import CostMonitor


@pytest.fixture(scope="session")
def analytics_test_db_schema():
    """Create database schema for analytics testing (session scope)."""
    # Use in-memory database for schema creation
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def analytics_sample_companies():
    """Sample company data for analytics testing."""
    return [
        {
            "id": 1,
            "name": "Analytics Corp",
            "url": "https://analytics.com",
            "active": True,
        },
        {
            "id": 2,
            "name": "Data Insights Ltd",
            "url": "https://datainsights.com",
            "active": True,
        },
        {
            "id": 3,
            "name": "AI Solutions Inc",
            "url": "https://aisolutions.com",
            "active": True,
        },
        {
            "id": 4,
            "name": "Tech Innovations",
            "url": "https://techinnovations.com",
            "active": True,
        },
        {
            "id": 5,
            "name": "Startup Unicorn",
            "url": "https://unicorn.com",
            "active": True,
        },
    ]


@pytest.fixture
def analytics_sample_jobs():
    """Sample job data for analytics testing."""
    base_date = datetime.now(UTC)

    return [
        # Recent jobs (within 30 days) - these should appear in trends
        {
            "id": 1,
            "company_id": 1,
            "title": "Senior Data Scientist",
            "description": "Advanced analytics and ML",
            "link": "https://analytics.com/job1",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=1),
            "salary": [150000, 190000],
            "archived": False,
            "content_hash": "analytics_hash1",
        },
        {
            "id": 2,
            "company_id": 1,
            "title": "Analytics Engineer",
            "description": "Data pipeline development",
            "link": "https://analytics.com/job2",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=2),
            "salary": [130000, 160000],
            "archived": False,
            "content_hash": "analytics_hash2",
        },
        {
            "id": 3,
            "company_id": 2,
            "title": "Business Intelligence Analyst",
            "description": "Dashboard and reporting",
            "link": "https://datainsights.com/job1",
            "location": "New York, NY",
            "posted_date": base_date - timedelta(days=3),
            "salary": [110000, 140000],
            "archived": False,
            "content_hash": "analytics_hash3",
        },
        {
            "id": 4,
            "company_id": 3,
            "title": "Machine Learning Engineer",
            "description": "ML model deployment",
            "link": "https://aisolutions.com/job1",
            "location": "Seattle, WA",
            "posted_date": base_date - timedelta(days=5),
            "salary": [140000, 180000],
            "archived": False,
            "content_hash": "analytics_hash4",
        },
        {
            "id": 5,
            "company_id": 4,
            "title": "Data Engineer",
            "description": "Big data infrastructure",
            "link": "https://techinnovations.com/job1",
            "location": "Austin, TX",
            "posted_date": base_date - timedelta(days=7),
            "salary": [125000, 155000],
            "archived": False,
            "content_hash": "analytics_hash5",
        },
        {
            "id": 6,
            "company_id": 5,
            "title": "Research Scientist",
            "description": "AI research and development",
            "link": "https://unicorn.com/job1",
            "location": "Remote",
            "posted_date": base_date - timedelta(days=10),
            "salary": [160000, 200000],
            "archived": False,
            "content_hash": "analytics_hash6",
        },
        {
            "id": 7,
            "company_id": 2,
            "title": "Senior Analytics Manager",
            "description": "Analytics team leadership",
            "link": "https://datainsights.com/job2",
            "location": "New York, NY",
            "posted_date": base_date - timedelta(days=12),
            "salary": [180000, 220000],
            "archived": False,
            "content_hash": "analytics_hash7",
        },
        {
            "id": 8,
            "company_id": 3,
            "title": "AI Product Manager",
            "description": "AI product strategy",
            "link": "https://aisolutions.com/job2",
            "location": "Seattle, WA",
            "posted_date": base_date - timedelta(days=15),
            "salary": [170000, 210000],
            "archived": False,
            "content_hash": "analytics_hash8",
        },
        # Older jobs (outside 30-day window) - for testing date filtering
        {
            "id": 9,
            "company_id": 1,
            "title": "Junior Data Scientist",
            "description": "Entry level analytics",
            "link": "https://analytics.com/job3",
            "location": "San Francisco, CA",
            "posted_date": base_date - timedelta(days=45),
            "salary": [90000, 120000],
            "archived": False,
            "content_hash": "analytics_hash9",
        },
        {
            "id": 10,
            "company_id": 4,
            "title": "Data Analyst Intern",
            "description": "Summer internship program",
            "link": "https://techinnovations.com/job2",
            "location": "Austin, TX",
            "posted_date": base_date - timedelta(days=60),
            "salary": [60000, 75000],
            "archived": False,
            "content_hash": "analytics_hash10",
        },
        # Archived jobs (should not appear in analytics)
        {
            "id": 11,
            "company_id": 2,
            "title": "Archived Analytics Role",
            "description": "This position was filled",
            "link": "https://datainsights.com/archived",
            "location": "New York, NY",
            "posted_date": base_date - timedelta(days=1),
            "salary": [100000, 130000],
            "archived": True,
            "content_hash": "analytics_hash11",
        },
        # Jobs without salary data (for testing salary analytics edge cases)
        {
            "id": 12,
            "company_id": 5,
            "title": "Volunteer Data Scientist",
            "description": "Non-profit analytics work",
            "link": "https://unicorn.com/volunteer",
            "location": "Remote",
            "posted_date": base_date - timedelta(days=8),
            "salary": None,
            "archived": False,
            "content_hash": "analytics_hash12",
        },
        {
            "id": 13,
            "company_id": 3,
            "title": "Salary TBD Position",
            "description": "Competitive salary",
            "link": "https://aisolutions.com/tbd",
            "location": "Seattle, WA",
            "posted_date": base_date - timedelta(days=14),
            "salary": None,
            "archived": False,
            "content_hash": "analytics_hash13",
        },
    ]


@pytest.fixture
def analytics_test_database(
    tmp_path, analytics_sample_companies, analytics_sample_jobs
):
    """Create a test database with sample analytics data."""
    db_path = tmp_path / "analytics_test.db"

    # Create database with schema
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    # Insert sample data
    with Session(engine) as session:
        # Add companies
        for company_data in analytics_sample_companies:
            company = CompanySQL(**company_data)
            session.add(company)

        # Add jobs
        for job_data in analytics_sample_jobs:
            job = JobSQL(**job_data)
            session.add(job)

        session.commit()

    return str(db_path)


@pytest.fixture
def analytics_service(analytics_test_database):
    """Create AnalyticsService instance with test database."""
    return AnalyticsService(db_path=analytics_test_database)


@pytest.fixture
def cost_test_database(tmp_path):
    """Create a test database for cost monitoring."""
    db_path = tmp_path / "cost_test.db"
    return str(db_path)


@pytest.fixture
def cost_monitor(cost_test_database):
    """Create CostMonitor instance with test database."""
    return CostMonitor(db_path=cost_test_database)


@pytest.fixture
def cost_monitor_with_sample_data(cost_test_database):
    """Create CostMonitor with realistic sample cost data."""
    monitor = CostMonitor(db_path=cost_test_database)

    # Add varied cost data representing different scenarios

    # AI costs - various models and operations
    monitor.track_ai_cost("gpt-4", 2500, 0.05, "job_extraction")
    monitor.track_ai_cost("gpt-4", 1500, 0.03, "company_analysis")
    monitor.track_ai_cost("groq-llama", 8000, 0.02, "batch_processing")
    monitor.track_ai_cost("gpt-4", 3000, 0.06, "salary_normalization")
    monitor.track_ai_cost("claude-3", 2000, 0.04, "content_summarization")

    # Proxy costs - different endpoints and usage patterns
    monitor.track_proxy_cost(150, 3.75, "residential_proxy_premium")
    monitor.track_proxy_cost(200, 2.50, "datacenter_proxy_bulk")
    monitor.track_proxy_cost(75, 1.80, "mobile_proxy_rotation")
    monitor.track_proxy_cost(300, 4.20, "residential_proxy_geo")

    # Scraping costs - various companies and success rates
    monitor.track_scraping_cost("Analytics Corp", 45, 4.50)
    monitor.track_scraping_cost("Data Insights Ltd", 35, 3.25)
    monitor.track_scraping_cost("AI Solutions Inc", 28, 2.75)
    monitor.track_scraping_cost("Tech Innovations", 22, 2.40)
    monitor.track_scraping_cost("Startup Unicorn", 18, 1.95)

    return monitor


@pytest.fixture
def analytics_integration_setup(analytics_service, cost_monitor_with_sample_data):
    """Create setup for integration testing with both services configured."""
    return {
        "analytics": analytics_service,
        "cost_monitor": cost_monitor_with_sample_data,
    }


@pytest.fixture
def performance_test_database(tmp_path):
    """Create a larger test database for performance testing."""
    db_path = tmp_path / "performance_test.db"

    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    base_date = datetime.now(UTC)

    # Create larger dataset for performance testing
    companies_data = []
    for i in range(50):  # 50 companies
        companies_data.append(
            {
                "id": i + 1,
                "name": f"Performance Test Corp {i:03d}",
                "url": f"https://perftest{i:03d}.com",
                "active": True,
            }
        )

    jobs_data = []
    for i in range(500):  # 500 jobs
        company_id = (i % 50) + 1  # Distribute jobs across companies
        days_back = i % 90  # Jobs spread over 90 days

        jobs_data.append(
            {
                "id": i + 1,
                "company_id": company_id,
                "title": f"Performance Test Job {i:03d}",
                "description": f"Performance testing job {i}",
                "link": f"https://perftest{company_id:03d}.com/job/{i}",
                "location": [
                    "Remote",
                    "San Francisco",
                    "New York",
                    "Seattle",
                    "Austin",
                ][i % 5],
                "posted_date": base_date - timedelta(days=days_back),
                "salary": [80000 + (i * 200), 120000 + (i * 300)]
                if i % 10 != 0
                else None,
                "archived": i % 20 == 0,  # 5% archived
                "content_hash": f"perf_hash_{i:03d}",
            }
        )

    # Insert performance test data
    with Session(engine) as session:
        for company_data in companies_data:
            company = CompanySQL(**company_data)
            session.add(company)

        for job_data in jobs_data:
            job = JobSQL(**job_data)
            session.add(job)

        session.commit()

    return str(db_path)


@pytest.fixture
def performance_analytics_service(performance_test_database):
    """Create AnalyticsService with performance test database."""
    return AnalyticsService(db_path=performance_test_database)


@pytest.fixture
def mock_streamlit_components():
    """Mock Streamlit components for UI testing without importing streamlit."""
    from unittest.mock import Mock

    # Create comprehensive mock of Streamlit API
    mock_st = Mock()

    # Basic display functions
    mock_st.title = Mock()
    mock_st.header = Mock()
    mock_st.subheader = Mock()
    mock_st.write = Mock()
    mock_st.text = Mock()
    mock_st.markdown = Mock()

    # Status messages
    mock_st.info = Mock()
    mock_st.success = Mock()
    mock_st.warning = Mock()
    mock_st.error = Mock()
    mock_st.exception = Mock()

    # Metrics and data display
    mock_st.metric = Mock()
    mock_st.dataframe = Mock()
    mock_st.table = Mock()
    mock_st.json = Mock()

    # Charts and visualizations
    mock_st.line_chart = Mock()
    mock_st.bar_chart = Mock()
    mock_st.area_chart = Mock()
    mock_st.plotly_chart = Mock()

    # Layout components
    mock_st.columns = Mock(return_value=[Mock() for _ in range(3)])
    mock_st.container = Mock(return_value=Mock())
    mock_st.expander = Mock(return_value=Mock())
    mock_st.tabs = Mock(return_value=[Mock() for _ in range(3)])
    mock_st.sidebar = Mock()

    # Session state
    mock_st.session_state = {}

    # Caching decorators
    mock_st.cache_data = Mock(return_value=lambda f: f)  # Pass-through decorator
    mock_st.cache_resource = Mock(return_value=lambda f: f)

    return mock_st


@pytest.fixture
def sample_analytics_responses():
    """Sample analytics service responses for testing."""
    base_date = datetime.now(UTC)

    return {
        "job_trends_success": {
            "status": "success",
            "method": "duckdb_sqlite_scanner",
            "trends": [
                {
                    "date": (base_date - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "job_count": 12,
                },
                {
                    "date": (base_date - timedelta(days=6)).strftime("%Y-%m-%d"),
                    "job_count": 8,
                },
                {
                    "date": (base_date - timedelta(days=5)).strftime("%Y-%m-%d"),
                    "job_count": 15,
                },
                {
                    "date": (base_date - timedelta(days=4)).strftime("%Y-%m-%d"),
                    "job_count": 6,
                },
                {
                    "date": (base_date - timedelta(days=3)).strftime("%Y-%m-%d"),
                    "job_count": 11,
                },
                {
                    "date": (base_date - timedelta(days=2)).strftime("%Y-%m-%d"),
                    "job_count": 9,
                },
                {
                    "date": (base_date - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "job_count": 14,
                },
            ],
            "total_jobs": 75,
        },
        "job_trends_empty": {
            "status": "success",
            "method": "duckdb_sqlite_scanner",
            "trends": [],
            "total_jobs": 0,
        },
        "job_trends_error": {
            "status": "error",
            "method": "duckdb_sqlite_scanner",
            "error": "DuckDB connection failed",
            "trends": [],
        },
        "company_analytics_success": {
            "status": "success",
            "method": "duckdb_sqlite_scanner",
            "companies": [
                {
                    "company": "Analytics Corp",
                    "total_jobs": 25,
                    "avg_min_salary": 140000.0,
                    "avg_max_salary": 175000.0,
                    "last_job_posted": base_date.strftime("%Y-%m-%d"),
                },
                {
                    "company": "Data Insights Ltd",
                    "total_jobs": 18,
                    "avg_min_salary": 125000.0,
                    "avg_max_salary": 160000.0,
                    "last_job_posted": (base_date - timedelta(days=2)).strftime(
                        "%Y-%m-%d"
                    ),
                },
                {
                    "company": "AI Solutions Inc",
                    "total_jobs": 15,
                    "avg_min_salary": 150000.0,
                    "avg_max_salary": 190000.0,
                    "last_job_posted": (base_date - timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    ),
                },
            ],
            "total_companies": 3,
        },
        "salary_analytics_success": {
            "status": "success",
            "method": "duckdb_sqlite_scanner",
            "salary_data": {
                "total_jobs_with_salary": 58,
                "avg_min_salary": 132500.75,
                "avg_max_salary": 168750.25,
                "min_salary": 90000.0,
                "max_salary": 220000.0,
                "salary_std_dev": 18500.50,
                "analysis_period_days": 90,
            },
        },
    }


@pytest.fixture
def sample_cost_responses():
    """Sample cost monitor responses for testing."""
    return {
        "monthly_summary_within_budget": {
            "total_cost": 28.50,
            "monthly_budget": 50.0,
            "remaining": 21.50,
            "utilization_percent": 57.0,
            "budget_status": "moderate_usage",
            "costs_by_service": {
                "ai": 16.00,
                "proxy": 8.50,
                "scraping": 4.00,
            },
            "operation_counts": {
                "ai": 12,
                "proxy": 8,
                "scraping": 5,
            },
            "month_year": "January 2024",
        },
        "monthly_summary_approaching_limit": {
            "total_cost": 42.75,
            "monthly_budget": 50.0,
            "remaining": 7.25,
            "utilization_percent": 85.5,
            "budget_status": "approaching_limit",
            "costs_by_service": {
                "ai": 28.00,
                "proxy": 10.75,
                "scraping": 4.00,
            },
            "operation_counts": {
                "ai": 20,
                "proxy": 12,
                "scraping": 6,
            },
            "month_year": "January 2024",
        },
        "monthly_summary_over_budget": {
            "total_cost": 67.25,
            "monthly_budget": 50.0,
            "remaining": -17.25,
            "utilization_percent": 134.5,
            "budget_status": "over_budget",
            "costs_by_service": {
                "ai": 45.00,
                "proxy": 15.25,
                "scraping": 7.00,
            },
            "operation_counts": {
                "ai": 35,
                "proxy": 18,
                "scraping": 10,
            },
            "month_year": "January 2024",
        },
        "cost_alerts_none": [],
        "cost_alerts_warning": [
            {
                "type": "warning",
                "message": "Approaching budget limit: 85% used",
            }
        ],
        "cost_alerts_error": [
            {
                "type": "error",
                "message": "Monthly budget exceeded: $67.25 / $50.00",
            }
        ],
    }


@pytest.fixture
def expected_test_coverage():
    """Expected test coverage metrics for analytics components."""
    return {
        "analytics_service": {
            "minimum_coverage": 90,
            "critical_methods": [
                "__init__",
                "_init_duckdb",
                "get_job_trends",
                "get_company_analytics",
                "get_salary_analytics",
                "get_status_report",
            ],
        },
        "cost_monitor": {
            "minimum_coverage": 90,
            "critical_methods": [
                "__init__",
                "track_ai_cost",
                "track_proxy_cost",
                "track_scraping_cost",
                "get_monthly_summary",
                "get_cost_alerts",
            ],
        },
        "ui_analytics": {
            "minimum_coverage": 85,
            "critical_functions": [
                "render_analytics_page",
                "_render_cost_monitoring_section",
                "_render_job_trends_section",
                "_render_company_analytics_section",
                "_render_salary_analytics_section",
            ],
        },
    }


@pytest.fixture(scope="session")
def test_performance_thresholds():
    """Performance thresholds for analytics testing."""
    return {
        "analytics_service": {
            "job_trends_max_time": 2.0,  # seconds
            "company_analytics_max_time": 2.0,
            "salary_analytics_max_time": 2.0,
        },
        "cost_monitor": {
            "monthly_summary_max_time": 1.0,
            "track_cost_max_time": 0.5,
        },
        "ui_rendering": {
            "dashboard_render_max_time": 3.0,
            "section_render_max_time": 1.0,
        },
        "integration": {
            "full_workflow_max_time": 5.0,
        },
    }


@pytest.fixture
def analytics_test_config():
    """Configuration settings for analytics testing."""
    return {
        "database": {
            "test_db_memory": True,
            "connection_pool_size": 5,
            "test_data_size": "small",  # small, medium, large
        },
        "services": {
            "enable_duckdb_tests": True,  # Set to False if DuckDB unavailable
            "enable_streamlit_tests": True,
            "mock_external_dependencies": True,
        },
        "performance": {
            "enable_performance_tests": True,
            "performance_iterations": 3,
            "benchmark_mode": False,
        },
        "coverage": {
            "minimum_line_coverage": 90,
            "minimum_branch_coverage": 85,
            "fail_under_threshold": True,
        },
    }


# Parametrized fixtures for testing different scenarios
@pytest.fixture(params=[7, 30, 90])
def analytics_time_periods(request):
    """Parametrized fixture for testing different time periods."""
    return request.param


@pytest.fixture(params=["within_budget", "approaching_limit", "over_budget"])
def cost_budget_scenarios(request, sample_cost_responses):
    """Parametrized fixture for testing different budget scenarios."""
    scenario_mapping = {
        "within_budget": sample_cost_responses["monthly_summary_within_budget"],
        "approaching_limit": sample_cost_responses["monthly_summary_approaching_limit"],
        "over_budget": sample_cost_responses["monthly_summary_over_budget"],
    }
    return request.param, scenario_mapping[request.param]


@pytest.fixture(params=["success", "empty", "error"])
def analytics_response_scenarios(request, sample_analytics_responses):
    """Parametrized fixture for testing different analytics response scenarios."""
    scenario_mapping = {
        "success": sample_analytics_responses["job_trends_success"],
        "empty": sample_analytics_responses["job_trends_empty"],
        "error": sample_analytics_responses["job_trends_error"],
    }
    return request.param, scenario_mapping[request.param]
