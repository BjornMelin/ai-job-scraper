"""Comprehensive tests for the CostMonitor service with SQLModel cost tracking.

This test suite validates the simple cost monitoring service that tracks operational
costs for a $50 monthly budget using SQLModel and Streamlit caching.

Key improvements:
- Fixed Streamlit caching compatibility by disabling cache in tests
- Modern SQLModel patterns using session.exec() instead of session.query()
- Library-first approach with real integration tests
- Comprehensive budget alert testing
- Performance validation
"""

import json
import logging
import time

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from sqlmodel import Session, select

from src.services.cost_monitor import CostEntry, CostMonitor

# Disable logging during tests
logging.disable(logging.CRITICAL)


def no_cache_decorator(*args, **kwargs):
    """Mock decorator that bypasses Streamlit caching for tests."""

    def decorator(func):
        return func

    return decorator


@pytest.fixture(autouse=True)
def disable_streamlit_cache():
    """Automatically disable Streamlit caching for all tests."""
    with patch("streamlit.cache_data", side_effect=no_cache_decorator):
        yield


@pytest.fixture
def test_cost_db(tmp_path):
    """Create a test SQLite database for cost monitoring."""
    db_path = tmp_path / "test_costs.db"
    return str(db_path)


@pytest.fixture
def cost_monitor(test_cost_db):
    """Create a CostMonitor instance with test database."""
    monitor = CostMonitor(db_path=test_cost_db)
    # Clear any cached data to ensure test isolation
    if hasattr(monitor.get_monthly_summary, "clear"):
        monitor.get_monthly_summary.clear()
    return monitor


class TestCostEntryModel:
    """Test CostEntry SQLModel functionality."""

    def test_cost_entry_creation(self, cost_monitor):
        """Test basic CostEntry model creation and persistence."""
        # Test entry creation
        entry = CostEntry(
            service="ai",
            operation="gpt4_completion",
            cost_usd=0.05,
            extra_data='{"model": "gpt-4", "tokens": 1000}',
        )

        with Session(cost_monitor.engine) as session:
            session.add(entry)
            session.commit()

            # Verify entry was saved
            saved_entry = session.exec(
                select(CostEntry).where(CostEntry.service == "ai")
            ).first()

            assert saved_entry is not None
            assert saved_entry.service == "ai"
            assert saved_entry.operation == "gpt4_completion"
            assert saved_entry.cost_usd == 0.05
            assert saved_entry.extra_data == '{"model": "gpt-4", "tokens": 1000}'

    def test_cost_entry_default_values(self, cost_monitor):
        """Test CostEntry default values and timestamps."""
        entry = CostEntry(service="proxy", operation="request", cost_usd=0.01)

        with Session(cost_monitor.engine) as session:
            session.add(entry)
            session.commit()

            saved_entry = session.exec(
                select(CostEntry).where(CostEntry.service == "proxy")
            ).first()

            assert saved_entry.id is not None
            assert saved_entry.timestamp is not None
            assert saved_entry.extra_data == ""
            # Timestamp should be recent (within last minute)
            # Handle timezone-aware/naive datetime comparison
            saved_timestamp = saved_entry.timestamp
            if saved_timestamp.tzinfo is None:
                # If saved timestamp is naive (from SQLite), assume it's UTC
                saved_timestamp = saved_timestamp.replace(tzinfo=UTC)

            assert abs((saved_timestamp - datetime.now(UTC)).total_seconds()) < 60

    def test_cost_entry_json_handling(self, cost_monitor):
        """Test handling of JSON data in extra_data field."""
        # Test valid JSON
        valid_json = '{"tokens": 500, "model": "gpt-3.5-turbo"}'
        entry1 = CostEntry(
            service="ai", operation="completion", cost_usd=0.02, extra_data=valid_json
        )

        # Test empty JSON
        entry2 = CostEntry(service="ai", operation="completion", cost_usd=0.02)

        with Session(cost_monitor.engine) as session:
            session.add_all([entry1, entry2])
            session.commit()

            entries = session.exec(
                select(CostEntry).where(CostEntry.service == "ai")
            ).all()
            assert len(entries) == 2

            # Verify JSON data
            json_entry = next(e for e in entries if e.extra_data != "")
            parsed_data = json.loads(json_entry.extra_data)
            assert parsed_data["tokens"] == 500
            assert parsed_data["model"] == "gpt-3.5-turbo"


class TestCostMonitorInitialization:
    """Test CostMonitor initialization and setup."""

    def test_init_with_default_values(self):
        """Test CostMonitor initialization with default values."""
        monitor = CostMonitor()

        assert monitor.db_path == "costs.db"
        assert monitor.monthly_budget == 50.0
        assert monitor.engine is not None

    def test_init_with_custom_values(self, test_cost_db):
        """Test CostMonitor initialization with custom values."""
        monitor = CostMonitor(db_path=test_cost_db)

        assert monitor.db_path == test_cost_db
        assert monitor.monthly_budget == 50.0

    def test_database_tables_created(self, cost_monitor):
        """Test that database tables are created on initialization."""
        # Tables should exist and be accessible
        with Session(cost_monitor.engine) as session:
            result = session.exec(select(CostEntry)).all()
            assert isinstance(result, list)  # Should return empty list, not error


class TestCostTrackingMethods:
    """Test cost tracking methods."""

    def test_track_ai_cost(self, cost_monitor):
        """Test AI cost tracking functionality."""
        cost_monitor.track_ai_cost("gpt-4", 1000, 0.02, "job_extraction")

        with Session(cost_monitor.engine) as session:
            ai_entries = session.exec(
                select(CostEntry).where(CostEntry.service == "ai")
            ).all()

            assert len(ai_entries) == 1
            entry = ai_entries[0]
            assert entry.service == "ai"
            assert entry.operation == "job_extraction"
            assert entry.cost_usd == 0.02

            # Verify extra_data JSON
            extra_data = json.loads(entry.extra_data)
            assert extra_data["model"] == "gpt-4"
            assert extra_data["tokens"] == 1000

    def test_track_proxy_cost(self, cost_monitor):
        """Test proxy cost tracking functionality."""
        cost_monitor.track_proxy_cost(100, 0.05, "iproyal-endpoint")

        with Session(cost_monitor.engine) as session:
            proxy_entries = session.exec(
                select(CostEntry).where(CostEntry.service == "proxy")
            ).all()

            assert len(proxy_entries) == 1
            entry = proxy_entries[0]
            assert entry.service == "proxy"
            assert entry.operation == "requests"
            assert entry.cost_usd == 0.05

            # Verify extra_data JSON
            extra_data = json.loads(entry.extra_data)
            assert extra_data["requests"] == 100
            assert extra_data["endpoint"] == "iproyal-endpoint"

    def test_track_scraping_cost(self, cost_monitor):
        """Test scraping cost tracking functionality."""
        cost_monitor.track_scraping_cost("TechCorp", 25, 1.50)

        with Session(cost_monitor.engine) as session:
            scraping_entries = session.exec(
                select(CostEntry).where(CostEntry.service == "scraping")
            ).all()

            assert len(scraping_entries) == 1
            entry = scraping_entries[0]
            assert entry.service == "scraping"
            assert entry.operation == "company_scrape"
            assert entry.cost_usd == 1.50

            # Verify extra_data JSON
            extra_data = json.loads(entry.extra_data)
            assert extra_data["company"] == "TechCorp"
            assert extra_data["jobs_found"] == 25

    def test_track_multiple_costs(self, cost_monitor):
        """Test tracking multiple costs of different services."""
        # Track various costs
        cost_monitor.track_ai_cost("gpt-3.5-turbo", 500, 0.01, "title_cleanup")
        cost_monitor.track_proxy_cost(50, 0.02, "residential-proxy")
        cost_monitor.track_scraping_cost("AI Solutions", 15, 0.75)
        cost_monitor.track_ai_cost("gpt-4", 200, 0.03, "salary_extraction")

        with Session(cost_monitor.engine) as session:
            # Check total entries
            all_entries = session.exec(select(CostEntry)).all()
            assert len(all_entries) == 4

            # Check by service
            ai_entries = session.exec(
                select(CostEntry).where(CostEntry.service == "ai")
            ).all()
            proxy_entries = session.exec(
                select(CostEntry).where(CostEntry.service == "proxy")
            ).all()
            scraping_entries = session.exec(
                select(CostEntry).where(CostEntry.service == "scraping")
            ).all()

            assert len(ai_entries) == 2
            assert len(proxy_entries) == 1
            assert len(scraping_entries) == 1

            # Check total cost
            total_cost = sum(entry.cost_usd for entry in all_entries)
            assert abs(total_cost - 0.81) < 0.001  # 0.01 + 0.02 + 0.75 + 0.03


class TestMonthlySummaryAndBudgetAnalysis:
    """Test monthly summary and budget analysis functionality."""

    def test_get_monthly_summary_empty(self, cost_monitor):
        """Test monthly summary with no cost entries."""
        summary = cost_monitor.get_monthly_summary()

        assert summary["costs_by_service"] == {}
        assert summary["operation_counts"] == {}
        assert summary["total_cost"] == 0.0
        assert summary["monthly_budget"] == 50.0
        assert summary["remaining"] == 50.0
        assert summary["utilization_percent"] == 0.0
        assert summary["budget_status"] == "within_budget"

    def test_get_monthly_summary_with_data(self, cost_monitor):
        """Test monthly summary with cost data."""
        # Add some costs
        cost_monitor.track_ai_cost("gpt-4", 1000, 5.00, "analysis")
        cost_monitor.track_proxy_cost(200, 2.00, "scraping")
        cost_monitor.track_scraping_cost("Company", 50, 8.00)
        cost_monitor.track_ai_cost("gpt-3.5", 500, 1.50, "cleanup")

        summary = cost_monitor.get_monthly_summary()

        assert summary["costs_by_service"]["ai"] == 6.50  # 5.00 + 1.50
        assert summary["costs_by_service"]["proxy"] == 2.00
        assert summary["costs_by_service"]["scraping"] == 8.00
        assert summary["total_cost"] == 16.50
        assert summary["remaining"] == 33.50  # 50 - 16.50
        assert summary["utilization_percent"] == 33.0  # (16.50/50) * 100
        assert summary["budget_status"] == "within_budget"

        # Check operation counts
        assert summary["operation_counts"]["ai"] == 2
        assert summary["operation_counts"]["proxy"] == 1
        assert summary["operation_counts"]["scraping"] == 1

    def test_monthly_summary_only_current_month(self, cost_monitor):
        """Test that monthly summary only includes current month data."""
        # Add cost for current month
        cost_monitor.track_ai_cost("gpt-4", 1000, 10.00, "current_month")

        # Manually add cost for previous month
        last_month = datetime.now(UTC).replace(day=1) - timedelta(days=1)
        old_entry = CostEntry(
            service="ai",
            operation="old_operation",
            cost_usd=20.00,
            timestamp=last_month,
        )

        with Session(cost_monitor.engine) as session:
            session.add(old_entry)
            session.commit()

        summary = cost_monitor.get_monthly_summary()

        # Should only include current month's cost
        assert summary["total_cost"] == 10.00
        assert summary["costs_by_service"]["ai"] == 10.00
        assert summary["operation_counts"]["ai"] == 1

    def test_budget_status_calculations(self, cost_monitor):
        """Test budget status calculation thresholds."""
        # Test within_budget (< 60%)
        cost_monitor.track_ai_cost(
            "gpt-4", 1000, 25.00, "within_budget"
        )  # 50% utilization
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "within_budget"

        # Test moderate_usage (60-80%)
        cost_monitor.track_proxy_cost(100, 10.00, "moderate")  # Total: 35.00 = 70%
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "moderate_usage"

        # Test approaching_limit (80-100%)
        cost_monitor.track_scraping_cost("Company", 10, 7.50)  # Total: 42.50 = 85%
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "approaching_limit"

        # Test over_budget (>= 100%)
        cost_monitor.track_ai_cost("gpt-4", 500, 10.00, "over")  # Total: 52.50 = 105%
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "over_budget"


class TestBudgetAlertsAndMonitoring:
    """Test budget alerts and monitoring functionality."""

    def test_get_cost_alerts_within_budget(self, cost_monitor):
        """Test cost alerts when within budget."""
        cost_monitor.track_ai_cost(
            "gpt-4", 1000, 20.00, "within_budget"
        )  # 40% utilization

        alerts = cost_monitor.get_cost_alerts()
        assert alerts == []  # No alerts for within budget

    def test_get_cost_alerts_approaching_limit(self, cost_monitor):
        """Test cost alerts when approaching budget limit."""
        cost_monitor.track_ai_cost(
            "gpt-4", 1000, 42.00, "approaching"
        )  # 84% utilization

        alerts = cost_monitor.get_cost_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "warning"
        assert "Approaching budget limit: 84% used" in alerts[0]["message"]

    def test_get_cost_alerts_over_budget(self, cost_monitor):
        """Test cost alerts when over budget."""
        cost_monitor.track_ai_cost(
            "gpt-4", 1000, 55.00, "over_budget"
        )  # 110% utilization

        alerts = cost_monitor.get_cost_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "error"
        assert "Monthly budget exceeded: $55.00 / $50.00" in alerts[0]["message"]

    def test_get_cost_alerts_error_handling(self, cost_monitor):
        """Test cost alerts error handling."""
        # Patch get_monthly_summary to raise an exception
        with patch.object(
            cost_monitor, "get_monthly_summary", side_effect=Exception("Database error")
        ):
            alerts = cost_monitor.get_cost_alerts()

            assert len(alerts) == 1
            assert alerts[0]["type"] == "error"
            assert "Cost monitoring error" in alerts[0]["message"]


class TestCostMonitorEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_cost_tracking(self, cost_monitor):
        """Test tracking zero-cost operations."""
        cost_monitor.track_ai_cost("free-model", 100, 0.00, "free_operation")

        with Session(cost_monitor.engine) as session:
            entries = session.exec(select(CostEntry)).all()
            assert len(entries) == 1
            assert entries[0].cost_usd == 0.00

        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == 0.00
        assert summary["budget_status"] == "within_budget"

    def test_large_costs_handling(self, cost_monitor):
        """Test handling of large cost values."""
        # Test large cost that exceeds budget significantly
        cost_monitor.track_scraping_cost("Expensive Company", 1000, 999.99)

        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == 999.99
        assert summary["budget_status"] == "over_budget"
        assert summary["utilization_percent"] > 1000  # Way over budget

    def test_concurrent_cost_tracking(self, cost_monitor):
        """Test concurrent cost tracking operations."""
        import threading
        import time

        def track_costs(thread_id):
            for i in range(5):
                cost_monitor.track_ai_cost(f"model-{thread_id}", 100, 0.01, f"op-{i}")
                time.sleep(0.001)  # Small delay

        # Run concurrent tracking
        threads = []
        for i in range(3):
            thread = threading.Thread(target=track_costs, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all entries were tracked
        with Session(cost_monitor.engine) as session:
            entries = session.exec(select(CostEntry)).all()
            assert len(entries) == 15  # 3 threads * 5 operations each

        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == 0.15  # 15 * 0.01

    def test_invalid_json_handling(self, cost_monitor):
        """Test handling of invalid JSON in extra_data."""
        # Directly create entry with invalid JSON (simulating corruption)
        invalid_entry = CostEntry(
            service="test",
            operation="invalid_json",
            cost_usd=1.00,
            extra_data="invalid{json}data",
        )

        with Session(cost_monitor.engine) as session:
            session.add(invalid_entry)
            session.commit()

        # Service should still work
        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == 1.00


class TestCostMonitorIntegration:
    """Test integration scenarios."""

    def test_full_workflow_integration(self, cost_monitor):
        """Test complete cost monitoring workflow."""
        # Simulate a full scraping workflow
        # 1. AI cost for job analysis
        cost_monitor.track_ai_cost("gpt-4", 2000, 4.00, "job_analysis")

        # 2. Proxy costs for scraping
        cost_monitor.track_proxy_cost(500, 5.00, "residential_proxy")

        # 3. Multiple scraping operations
        companies = ["TechCorp", "AI Solutions", "DataFlow"]
        for i, company in enumerate(companies):
            cost_monitor.track_scraping_cost(company, 20 + i * 5, 2.00 + i * 0.5)

        # 4. AI cost for post-processing
        cost_monitor.track_ai_cost("gpt-3.5-turbo", 1000, 1.50, "cleanup")

        # Verify complete workflow
        summary = cost_monitor.get_monthly_summary()

        expected_total = 18.00  # 4.00 + 5.00 + 2.00 + 2.50 + 3.00 + 1.50
        assert abs(summary["total_cost"] - expected_total) < 0.01

        # Check service breakdowns
        assert summary["costs_by_service"]["ai"] == 5.50  # 4.00 + 1.50
        assert summary["costs_by_service"]["proxy"] == 5.00
        assert summary["costs_by_service"]["scraping"] == 7.50  # 2.00 + 2.50 + 3.00

        # Check budget status
        assert summary["budget_status"] == "within_budget"  # 36% utilization
        assert summary["utilization_percent"] == 36.0

    def test_monthly_budget_cycle_simulation(self, cost_monitor):
        """Test simulated monthly budget cycle."""
        # Start of month - light usage
        cost_monitor.track_ai_cost("gpt-4", 1000, 10.00, "light_usage")
        summary1 = cost_monitor.get_monthly_summary()
        assert summary1["budget_status"] == "within_budget"

        # Mid-month - moderate usage
        cost_monitor.track_scraping_cost("Company A", 100, 20.00)
        summary2 = cost_monitor.get_monthly_summary()
        assert summary2["budget_status"] == "moderate_usage"  # 60% utilization

        # Late month - approaching limit
        cost_monitor.track_proxy_cost(1000, 15.00, "heavy_scraping")
        summary3 = cost_monitor.get_monthly_summary()
        assert summary3["budget_status"] == "approaching_limit"  # 90% utilization

        # Month end - over budget
        cost_monitor.track_ai_cost("gpt-4", 500, 10.00, "month_end")
        summary4 = cost_monitor.get_monthly_summary()
        assert summary4["budget_status"] == "over_budget"  # 110% utilization

        # Verify alerts are generated
        alerts = cost_monitor.get_cost_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "error"

    def test_performance_with_large_dataset(self, cost_monitor):
        """Test performance with larger dataset."""
        start_time = time.perf_counter()

        # Add 100 cost entries
        for i in range(100):
            service = ["ai", "proxy", "scraping"][i % 3]
            cost = 0.01 + (i % 10) * 0.01  # Vary costs

            if service == "ai":
                cost_monitor.track_ai_cost("gpt-4", 100, cost, f"operation_{i}")
            elif service == "proxy":
                cost_monitor.track_proxy_cost(10, cost, f"endpoint_{i}")
            else:
                cost_monitor.track_scraping_cost(f"Company_{i}", 5, cost)

        # Get summary (should be fast)
        summary_start = time.perf_counter()
        summary = cost_monitor.get_monthly_summary()
        summary_time = time.perf_counter() - summary_start

        total_time = time.perf_counter() - start_time

        # Performance assertions
        assert summary_time < 1.0  # Summary should complete under 1 second
        assert total_time < 5.0  # Full workflow should complete under 5 seconds

        # Verify correctness
        assert summary["operation_counts"]["ai"] > 0
        assert summary["operation_counts"]["proxy"] > 0
        assert summary["operation_counts"]["scraping"] > 0
        assert summary["total_cost"] > 0
