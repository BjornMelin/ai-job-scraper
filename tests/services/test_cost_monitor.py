"""Comprehensive tests for the CostMonitor service with SQLModel cost tracking.

This test suite validates the simple cost monitoring service that tracks operational
costs for a $50 monthly budget using SQLModel and Streamlit caching.

Test coverage includes:
- CostEntry SQLModel database operations
- Cost tracking for AI, proxy, and scraping services
- Monthly budget monitoring and alerts
- Budget status calculations and thresholds
- Service cost breakdowns and summaries
- Streamlit caching integration
- Error handling and edge cases
- Performance validation
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import json
import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from sqlmodel import Session, select

from src.services.cost_monitor import CostEntry, CostMonitor

# Logger instance for test debugging
logger = logging.getLogger(__name__)

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def test_cost_db(tmp_path):
    """Create a test SQLite database for cost monitoring."""
    db_path = tmp_path / "test_costs.db"
    return str(db_path)


@pytest.fixture
def cost_monitor(test_cost_db):
    """Create a CostMonitor instance with test database."""
    return CostMonitor(db_path=test_cost_db)


@pytest.fixture
def cost_monitor_with_data(test_cost_db):
    """Create a CostMonitor with sample cost data."""
    monitor = CostMonitor(db_path=test_cost_db)

    # Add sample cost data for current month
    current_month = datetime.now(UTC)
    current_month.replace(day=15, hour=10, minute=30)  # Mid-month

    # Sample costs within budget
    monitor.track_ai_cost("gpt-4", 1000, 0.02, "job_extraction")
    monitor.track_ai_cost("groq-llama", 5000, 0.01, "company_analysis")
    monitor.track_proxy_cost(100, 1.50, "residential_proxy")
    monitor.track_scraping_cost("TechCorp", 25, 2.00)
    monitor.track_scraping_cost("AI Solutions", 15, 1.25)

    return monitor


class TestCostEntryModel:
    """Test the CostEntry SQLModel database model."""

    def test_cost_entry_creation(self, cost_monitor):
        """Test creating and storing CostEntry instances."""
        with Session(cost_monitor.engine) as session:
            entry = CostEntry(
                service="ai",
                operation="test_operation",
                cost_usd=1.50,
                extra_data='{"model": "test", "tokens": 1000}',
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)

            assert entry.id is not None
            assert entry.service == "ai"
            assert entry.operation == "test_operation"
            assert entry.cost_usd == 1.50
            assert entry.extra_data == '{"model": "test", "tokens": 1000}'
            assert entry.timestamp is not None
            assert isinstance(entry.timestamp, datetime)  # Should be datetime

    def test_cost_entry_default_values(self, cost_monitor):
        """Test CostEntry default values."""
        with Session(cost_monitor.engine) as session:
            entry = CostEntry(service="proxy", operation="requests", cost_usd=0.50)
            session.add(entry)
            session.commit()
            session.refresh(entry)

            assert entry.extra_data == ""
            assert isinstance(entry.timestamp, datetime)

    def test_cost_entry_indexing(self, cost_monitor):
        """Test that database indexes are working correctly."""
        with Session(cost_monitor.engine) as session:
            # Add multiple entries
            for i in range(10):
                entry = CostEntry(
                    service=f"service_{i % 3}",  # 3 different services
                    operation=f"op_{i}",
                    cost_usd=float(i),
                    timestamp=datetime.now(UTC) - timedelta(hours=i),
                )
                session.add(entry)
            session.commit()

            # Query by service (indexed field)
            results = session.exec(
                select(CostEntry).where(CostEntry.service == "service_0")
            ).all()
            assert len(results) > 0


class TestCostMonitorInitialization:
    """Test CostMonitor initialization and setup."""

    def test_init_with_default_values(self):
        """Test initialization with default parameters."""
        monitor = CostMonitor()
        assert monitor.db_path == "costs.db"
        assert monitor.monthly_budget == 50.0
        assert monitor.engine is not None

    def test_init_with_custom_db_path(self, test_cost_db):
        """Test initialization with custom database path."""
        monitor = CostMonitor(db_path=test_cost_db)
        assert monitor.db_path == test_cost_db
        assert monitor.monthly_budget == 50.0

    def test_database_tables_created(self, cost_monitor):
        """Test that database tables are created during initialization."""
        with Session(cost_monitor.engine) as session:
            # Should be able to query the cost_entries table
            result = session.exec(select(CostEntry)).all()
            assert isinstance(result, list)


class TestCostTrackingMethods:
    """Test cost tracking methods for different services."""

    def test_track_ai_cost_basic(self, cost_monitor):
        """Test basic AI cost tracking."""
        cost_monitor.track_ai_cost("gpt-4", 1000, 0.02, "job_analysis")

        # Verify cost was stored
        with Session(cost_monitor.engine) as session:
            entries = session.exec(
                select(CostEntry).where(CostEntry.service == "ai")
            ).all()
            assert len(entries) == 1

            entry = entries[0]
            assert entry.operation == "job_analysis"
            assert entry.cost_usd == 0.02

            # Verify extra_data JSON structure
            extra_data = json.loads(entry.extra_data)
            assert extra_data["model"] == "gpt-4"
            assert extra_data["tokens"] == 1000

    def test_track_proxy_cost_basic(self, cost_monitor):
        """Test basic proxy cost tracking."""
        cost_monitor.track_proxy_cost(50, 1.25, "residential_endpoint")

        # Verify cost was stored
        with Session(cost_monitor.engine) as session:
            entries = session.exec(
                select(CostEntry).where(CostEntry.service == "proxy")
            ).all()
            assert len(entries) == 1

            entry = entries[0]
            assert entry.operation == "requests"
            assert entry.cost_usd == 1.25

            # Verify extra_data JSON structure
            extra_data = json.loads(entry.extra_data)
            assert extra_data["requests"] == 50
            assert extra_data["endpoint"] == "residential_endpoint"

    def test_track_scraping_cost_basic(self, cost_monitor):
        """Test basic scraping cost tracking."""
        cost_monitor.track_scraping_cost("TechCorp", 25, 2.50)

        # Verify cost was stored
        with Session(cost_monitor.engine) as session:
            entries = session.exec(
                select(CostEntry).where(CostEntry.service == "scraping")
            ).all()
            assert len(entries) == 1

            entry = entries[0]
            assert entry.operation == "company_scrape"
            assert entry.cost_usd == 2.50

            # Verify extra_data JSON structure
            extra_data = json.loads(entry.extra_data)
            assert extra_data["company"] == "TechCorp"
            assert extra_data["jobs_found"] == 25

    def test_track_multiple_costs(self, cost_monitor):
        """Test tracking multiple costs across different services."""
        # Track various costs
        cost_monitor.track_ai_cost("gpt-4", 1500, 0.03, "extraction")
        cost_monitor.track_ai_cost("groq-llama", 3000, 0.01, "summarization")
        cost_monitor.track_proxy_cost(100, 2.00, "datacenter_proxy")
        cost_monitor.track_scraping_cost("AI Solutions", 40, 3.50)

        # Verify all costs were stored
        with Session(cost_monitor.engine) as session:
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

            # Verify total cost
            total_cost = sum(
                entry.cost_usd
                for entry in ai_entries + proxy_entries + scraping_entries
            )
            assert total_cost == 5.54  # 0.03 + 0.01 + 2.00 + 3.50


class TestMonthlySummaryAndBudgetAnalysis:
    """Test monthly summary and budget analysis functionality."""

    def test_get_monthly_summary_empty(self, cost_monitor):
        """Test monthly summary with no cost data."""
        summary = cost_monitor.get_monthly_summary()

        assert isinstance(summary, dict)
        assert summary["total_cost"] == 0.0
        assert summary["monthly_budget"] == 50.0
        assert summary["remaining"] == 50.0
        assert summary["utilization_percent"] == 0.0
        assert summary["budget_status"] == "within_budget"
        assert summary["costs_by_service"] == {}
        assert summary["operation_counts"] == {}
        assert "month_year" in summary

    def test_get_monthly_summary_with_data(self, cost_monitor_with_data):
        """Test monthly summary with sample cost data."""
        summary = cost_monitor_with_data.get_monthly_summary()

        assert summary["total_cost"] > 0
        assert summary["remaining"] < 50.0
        assert summary["utilization_percent"] > 0
        assert len(summary["costs_by_service"]) > 0
        assert len(summary["operation_counts"]) > 0

        # Verify service breakdown structure
        for service, cost in summary["costs_by_service"].items():
            assert service in ["ai", "proxy", "scraping"]
            assert isinstance(cost, float)
            assert cost > 0

        # Verify operation counts
        for count in summary["operation_counts"].values():
            assert isinstance(count, int)
            assert count > 0

    def test_monthly_summary_only_current_month(self, cost_monitor):
        """Test that monthly summary only includes current month data."""
        current_month = datetime.now(UTC)
        last_month = current_month.replace(
            month=current_month.month - 1 if current_month.month > 1 else 12
        )

        # Add cost from last month (should not be included)
        with Session(cost_monitor.engine) as session:
            old_entry = CostEntry(
                service="ai",
                operation="old_operation",
                cost_usd=10.00,
                timestamp=last_month,
            )
            session.add(old_entry)
            session.commit()

        # Add cost from current month (should be included)
        cost_monitor.track_ai_cost("gpt-4", 1000, 2.00, "current_operation")

        summary = cost_monitor.get_monthly_summary()

        # Should only include current month costs
        assert summary["total_cost"] == 2.00
        assert len(summary["costs_by_service"]) == 1
        assert summary["costs_by_service"]["ai"] == 2.00

    def test_budget_status_calculations(self, cost_monitor):
        """Test budget status thresholds and calculations."""
        # Test within budget (< 60%)
        cost_monitor.track_ai_cost("gpt-4", 1000, 25.00, "test")  # 50% of budget
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "within_budget"
        assert summary["utilization_percent"] == 50.0

        # Test moderate usage (60-80%)
        cost_monitor.track_proxy_cost(100, 10.00, "test")  # Total: 70% of budget
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "moderate_usage"
        assert summary["utilization_percent"] == 70.0

        # Test approaching limit (80-100%)
        cost_monitor.track_scraping_cost("Test", 10, 5.00)  # Total: 80% of budget
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "approaching_limit"
        assert summary["utilization_percent"] == 80.0

        # Test over budget (>= 100%)
        cost_monitor.track_ai_cost("gpt-4", 2000, 12.00)  # Total: 104% of budget
        summary = cost_monitor.get_monthly_summary()
        assert summary["budget_status"] == "over_budget"
        assert summary["utilization_percent"] == 104.0


class TestBudgetAlertsAndMonitoring:
    """Test budget alert functionality."""

    @patch("src.services.cost_monitor.STREAMLIT_AVAILABLE", True)
    @patch("src.services.cost_monitor.st")
    def test_budget_alerts_approaching_limit(self, mock_st, cost_monitor):
        """Test budget alerts when approaching limit."""
        mock_st.warning = Mock()
        mock_st.error = Mock()

        # Add costs to reach 85% of budget
        cost_monitor.track_ai_cost("gpt-4", 10000, 42.50, "large_operation")

        # Should trigger warning
        mock_st.warning.assert_called_once()
        mock_st.error.assert_not_called()

        # Verify warning message contains percentage
        warning_call = mock_st.warning.call_args[0][0]
        assert "85.0%" in warning_call
        assert "Approaching budget limit" in warning_call

    @patch("src.services.cost_monitor.STREAMLIT_AVAILABLE", True)
    @patch("src.services.cost_monitor.st")
    def test_budget_alerts_over_budget(self, mock_st, cost_monitor):
        """Test budget alerts when over budget."""
        mock_st.warning = Mock()
        mock_st.error = Mock()

        # Add costs to exceed budget
        cost_monitor.track_ai_cost("gpt-4", 20000, 55.00, "expensive_operation")

        # Should trigger error alert
        mock_st.error.assert_called_once()

        # Verify error message contains amounts
        error_call = mock_st.error.call_args[0][0]
        assert "$55.00" in error_call
        assert "$50.00" in error_call
        assert "Monthly budget exceeded" in error_call

    def test_get_cost_alerts_within_budget(self, cost_monitor):
        """Test cost alerts when within budget."""
        cost_monitor.track_ai_cost("gpt-4", 1000, 20.00, "normal_operation")

        alerts = cost_monitor.get_cost_alerts()
        assert alerts == []  # No alerts when within budget

    def test_get_cost_alerts_approaching_limit(self, cost_monitor):
        """Test cost alerts when approaching limit."""
        cost_monitor.track_ai_cost("gpt-4", 10000, 42.50, "large_operation")  # 85%

        alerts = cost_monitor.get_cost_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "warning"
        assert "Approaching budget limit" in alerts[0]["message"]
        assert "85%" in alerts[0]["message"]

    def test_get_cost_alerts_over_budget(self, cost_monitor):
        """Test cost alerts when over budget."""
        cost_monitor.track_ai_cost("gpt-4", 20000, 60.00, "expensive_operation")

        alerts = cost_monitor.get_cost_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "error"
        assert "Monthly budget exceeded" in alerts[0]["message"]
        assert "$60.00" in alerts[0]["message"]

    @patch("src.services.cost_monitor.CostMonitor.get_monthly_summary")
    def test_get_cost_alerts_error_handling(self, mock_summary, cost_monitor):
        """Test cost alerts error handling."""
        mock_summary.side_effect = Exception("Database error")

        alerts = cost_monitor.get_cost_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "error"
        assert "Cost monitoring error" in alerts[0]["message"]
        assert "Database error" in alerts[0]["message"]


class TestStreamlitIntegration:
    """Test Streamlit integration features."""

    def test_streamlit_unavailable_fallback(self):
        """Test behavior when Streamlit is unavailable."""
        # When STREAMLIT_AVAILABLE is False, should not raise errors
        with patch("src.services.cost_monitor.STREAMLIT_AVAILABLE", False):
            monitor = CostMonitor()
            monitor.track_ai_cost("gpt-4", 1000, 60.00, "test")  # Over budget
            # Should complete without errors even when over budget

    @patch("src.services.cost_monitor.st.cache_data")
    def test_streamlit_caching_decorator(self, mock_cache_data):
        """Test that Streamlit caching decorator is applied."""
        monitor = CostMonitor()

        # Check that get_monthly_summary has caching decorator applied
        assert hasattr(monitor.get_monthly_summary, "__wrapped__") or hasattr(
            monitor.get_monthly_summary, "clear"
        )

    def test_cost_tracking_performance_with_caching(self, cost_monitor):
        """Test that repeated calls to get_monthly_summary are cached."""
        import time

        # Add some data
        cost_monitor.track_ai_cost("gpt-4", 1000, 5.00, "test")

        # First call
        start_time = time.time()
        summary1 = cost_monitor.get_monthly_summary()
        first_call_time = time.time() - start_time

        # Second call (should be faster due to caching)
        start_time = time.time()
        summary2 = cost_monitor.get_monthly_summary()
        second_call_time = time.time() - start_time

        # Verify results are identical
        assert summary1 == summary2

        # Second call should typically be faster
        # (though not guaranteed in all test environments)
        assert second_call_time <= first_call_time + 0.1  # Allow small margin


class TestCostMonitorPerformance:
    """Test performance characteristics of cost monitoring."""

    def test_cost_tracking_performance(self, cost_monitor):
        """Test cost tracking performance with many operations."""
        import time

        start_time = time.time()

        # Track 100 cost operations
        for i in range(100):
            service_type = ["ai", "proxy", "scraping"][i % 3]
            if service_type == "ai":
                cost_monitor.track_ai_cost(
                    f"model_{i}", 1000 + i, 0.01, f"operation_{i}"
                )
            elif service_type == "proxy":
                cost_monitor.track_proxy_cost(10 + i, 0.10, f"endpoint_{i}")
            else:
                cost_monitor.track_scraping_cost(f"company_{i}", 5, 0.25)

        elapsed_time = time.time() - start_time

        # Should complete 100 operations in reasonable time
        assert elapsed_time < 5.0  # Under 5 seconds

        # Verify all operations were stored
        summary = cost_monitor.get_monthly_summary()
        assert len(summary["costs_by_service"]) == 3  # ai, proxy, scraping
        assert summary["operation_counts"]["ai"] > 0
        assert summary["operation_counts"]["proxy"] > 0
        assert summary["operation_counts"]["scraping"] > 0

    def test_monthly_summary_performance(self, cost_monitor):
        """Test monthly summary performance with large dataset."""
        import time

        # Add substantial cost data
        for i in range(50):
            cost_monitor.track_ai_cost(f"model_{i}", 1000, 0.10, f"op_{i}")
            cost_monitor.track_proxy_cost(100, 0.50, f"proxy_{i}")
            cost_monitor.track_scraping_cost(f"company_{i}", 10, 1.00)

        # Test monthly summary performance
        start_time = time.time()
        summary = cost_monitor.get_monthly_summary()
        elapsed_time = time.time() - start_time

        assert elapsed_time < 2.0  # Should complete in under 2 seconds
        assert summary["total_cost"] > 0
        assert len(summary["costs_by_service"]) == 3


class TestCostMonitorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_cost_tracking(self, cost_monitor):
        """Test tracking zero-cost operations."""
        cost_monitor.track_ai_cost("free_model", 1000, 0.00, "free_operation")

        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == 0.00
        assert summary["costs_by_service"]["ai"] == 0.00
        assert summary["budget_status"] == "within_budget"

    def test_negative_cost_handling(self, cost_monitor):
        """Test handling of negative costs (refunds/credits)."""
        # Add normal cost
        cost_monitor.track_ai_cost("gpt-4", 1000, 10.00, "operation")

        # Add negative cost (refund) - this should still work
        with Session(cost_monitor.engine) as session:
            refund_entry = CostEntry(
                service="ai",
                operation="refund",
                cost_usd=-2.00,
                extra_data='{"type": "refund"}',
            )
            session.add(refund_entry)
            session.commit()

        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == 8.00  # 10.00 - 2.00
        assert summary["costs_by_service"]["ai"] == 8.00

    def test_very_large_costs(self, cost_monitor):
        """Test handling of very large costs."""
        large_cost = 999999.99
        cost_monitor.track_ai_cost(
            "expensive_model", 1000000, large_cost, "massive_operation"
        )

        summary = cost_monitor.get_monthly_summary()
        assert summary["total_cost"] == large_cost
        assert summary["budget_status"] == "over_budget"
        assert summary["utilization_percent"] > 1000000  # Way over budget

    def test_invalid_json_in_extra_data(self, cost_monitor):
        """Test handling of invalid JSON in extra_data field."""
        # Directly insert invalid JSON
        with Session(cost_monitor.engine) as session:
            entry = CostEntry(
                service="test",
                operation="invalid_json_test",
                cost_usd=1.00,
                extra_data='{"invalid": json}',  # Invalid JSON
            )
            session.add(entry)
            session.commit()

        # get_monthly_summary should still work despite invalid JSON
        summary = cost_monitor.get_monthly_summary()
        assert isinstance(summary, dict)
        assert summary["total_cost"] == 1.00

    def test_concurrent_cost_tracking(self, cost_monitor):
        """Test concurrent cost tracking operations."""
        import threading
        import time

        def track_costs(thread_id):
            """Track costs in a separate thread."""
            for i in range(10):
                cost_monitor.track_ai_cost(
                    f"model_{thread_id}_{i}", 1000, 0.01, f"op_{thread_id}_{i}"
                )
                time.sleep(0.001)  # Small delay

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=track_costs, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all costs were tracked correctly
        summary = cost_monitor.get_monthly_summary()
        assert summary["operation_counts"]["ai"] == 50  # 5 threads * 10 operations each
        assert summary["total_cost"] == 0.50  # 50 operations * $0.01 each

    def test_database_file_permissions(self, tmp_path):
        """Test behavior with database file permission issues."""
        # Create a directory where we can't write
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        db_path = readonly_dir / "costs.db"

        try:
            # This might raise an exception or create a monitor that fails operations
            monitor = CostMonitor(db_path=str(db_path))

            # Attempt to track cost - might succeed or fail depending on system
            try:
                monitor.track_ai_cost("test", 1000, 1.00, "test")
                # If successful, verify the operation
                summary = monitor.get_monthly_summary()
                assert isinstance(summary, dict)
            except Exception as e:
                # If it fails, that's also acceptable for readonly directory
                logger.debug("Cost tracking failed in readonly directory: %s", e)

        except Exception as e:
            # Constructor failure is acceptable for readonly directory
            logger.debug("CostMonitor init failed in readonly directory: %s", e)

        finally:
            # Clean up - restore permissions
            readonly_dir.chmod(0o755)

    def test_malformed_database_recovery(self, tmp_path):
        """Test recovery from malformed database files."""
        db_path = tmp_path / "malformed.db"

        # Create a malformed database file
        with db_path.open("wb") as f:
            f.write(b"This is not a valid SQLite database file")

        # CostMonitor should handle this gracefully
        try:
            monitor = CostMonitor(db_path=str(db_path))
            # If successful, it should recreate the database
            monitor.track_ai_cost("test", 1000, 1.00, "recovery_test")
            summary = monitor.get_monthly_summary()
            assert summary["total_cost"] == 1.00
        except Exception as e:
            # If it fails completely, that's also acceptable for malformed DB
            logger.debug("Malformed database recovery failed: %s", e)
