"""Simple cost monitoring service for $50 monthly budget tracking.

This module provides lightweight cost tracking using SQLModel for database
operations and Streamlit caching for dashboard integration. Focuses on
simplicity and maintainability with library-first approach.

Features:
- Simple SQLModel cost tracking entries
- Monthly budget monitoring ($50 limit)
- Cost alerts at 80% and 100% thresholds
- Service-based cost breakdown (AI, proxy, scraping)
- Streamlit caching for dashboard performance
"""

from __future__ import annotations

import logging

from datetime import UTC, datetime
from typing import Any

from sqlmodel import Field, Session, SQLModel, create_engine, func, select

# Import streamlit with fallback for non-Streamlit environments
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    class _DummyStreamlit:
        @staticmethod
        def cache_data(**_kwargs):
            def decorator(wrapped_func):
                return wrapped_func

            return decorator

    st = _DummyStreamlit()

logger = logging.getLogger(__name__)


class CostEntry(SQLModel, table=True):
    """Simple cost tracking model for operational expenses.

    Tracks costs by service type for budget monitoring and analysis.
    """

    __tablename__ = "cost_entries"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    service: str = Field(index=True)  # "ai", "proxy", "scraping"
    operation: str  # Description of operation
    cost_usd: float  # Cost in USD
    extra_data: str = ""  # Optional JSON string for additional details

    def __init__(self, **data):
        """Initialize with timezone-aware timestamp handling."""
        # Ensure timestamp is timezone-aware when retrieved from database
        if "timestamp" in data and data["timestamp"] is not None:
            ts = data["timestamp"]
            if isinstance(ts, datetime) and ts.tzinfo is None:
                # If timestamp is naive (from SQLite), assume it's UTC
                data["timestamp"] = ts.replace(tzinfo=UTC)
        super().__init__(**data)


class CostMonitor:
    """Simple cost monitoring service for $50 monthly budget.

    Provides basic cost tracking with budget alerts and service breakdown
    using SQLModel and Streamlit caching for optimal performance.

    Features:
    - Track costs by service (AI, proxy, scraping)
    - Monthly budget monitoring with alerts
    - Simple cost aggregation and reporting
    - Built-in $50 monthly budget limit

    Example:
        ```python
        monitor = CostMonitor()

        # Track AI operation cost
        monitor.track_ai_cost("gpt-4", 1000, 0.02, "job_extraction")

        # Get monthly summary
        summary = monitor.get_monthly_summary()
        ```
    """

    def __init__(self, db_path: str = "costs.db"):
        """Initialize cost monitor with SQLite database.

        Args:
            db_path: Path to SQLite database for cost storage.
        """
        self.db_path = db_path
        self.monthly_budget = 50.0  # $50 monthly budget
        self.engine = create_engine(f"sqlite:///{db_path}")

        # Create tables
        SQLModel.metadata.create_all(self.engine)

        logger.info(
            "Cost monitor initialized with $%.2f monthly budget", self.monthly_budget
        )

    def track_ai_cost(
        self, model: str, tokens: int, cost: float, operation: str
    ) -> None:
        """Track AI/LLM operation costs.

        Args:
            model: AI model used (e.g., "gpt-4", "groq-llama")
            tokens: Number of tokens processed
            cost: Cost in USD
            operation: Description of operation
        """
        extra_data = f'{{"model": "{model}", "tokens": {tokens}}}'

        with Session(self.engine) as session:
            entry = CostEntry(
                service="ai", operation=operation, cost_usd=cost, extra_data=extra_data
            )
            session.add(entry)
            session.commit()

        logger.info("AI cost tracked: %s, $%.4f (%s)", model, cost, operation)
        self._check_budget_alerts(cost)

    def track_proxy_cost(self, requests: int, cost: float, endpoint: str) -> None:
        """Track proxy service costs.

        Args:
            requests: Number of requests made
            cost: Cost in USD
            endpoint: Proxy endpoint used
        """
        extra_data = f'{{"requests": {requests}, "endpoint": "{endpoint}"}}'

        with Session(self.engine) as session:
            entry = CostEntry(
                service="proxy",
                operation="requests",
                cost_usd=cost,
                extra_data=extra_data,
            )
            session.add(entry)
            session.commit()

        logger.info(
            "Proxy cost tracked: %d requests, $%.4f (%s)", requests, cost, endpoint
        )
        self._check_budget_alerts(cost)

    def track_scraping_cost(self, company: str, jobs_found: int, cost: float) -> None:
        """Track scraping operation costs.

        Args:
            company: Company being scraped
            jobs_found: Number of jobs found
            cost: Cost in USD
        """
        extra_data = f'{{"company": "{company}", "jobs_found": {jobs_found}}}'

        with Session(self.engine) as session:
            entry = CostEntry(
                service="scraping",
                operation="company_scrape",
                cost_usd=cost,
                extra_data=extra_data,
            )
            session.add(entry)
            session.commit()

        logger.info(
            "Scraping cost tracked: %s, %d jobs, $%.4f", company, jobs_found, cost
        )
        self._check_budget_alerts(cost)

    @st.cache_data(ttl=60)  # 1-minute cache for real-time budget tracking
    def get_monthly_summary(_self) -> dict[str, Any]:  # noqa: N805
        """Get current month cost breakdown and budget analysis.

        Returns:
            Dict containing monthly costs, budget status, and breakdowns.
        """
        with Session(_self.engine) as session:
            # Get start of current month
            now = datetime.now(UTC)
            start_of_month = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

            # Get costs by service for current month
            results = session.exec(
                select(
                    CostEntry.service,
                    func.sum(CostEntry.cost_usd).label("total_cost"),
                    func.count(CostEntry.id).label("operation_count"),
                )
                .where(CostEntry.timestamp >= start_of_month)
                .group_by(CostEntry.service)
            ).all()

            # Build cost breakdown
            costs_by_service = {}
            operation_counts = {}
            total_cost = 0.0

            for result in results:
                service_cost = float(result.total_cost)
                costs_by_service[result.service] = service_cost
                operation_counts[result.service] = result.operation_count
                total_cost += service_cost

            # Calculate budget metrics
            remaining = _self.monthly_budget - total_cost
            utilization_percent = (total_cost / _self.monthly_budget) * 100
            budget_status = _self._get_budget_status(total_cost)

            return {
                "costs_by_service": costs_by_service,
                "operation_counts": operation_counts,
                "total_cost": total_cost,
                "monthly_budget": _self.monthly_budget,
                "remaining": remaining,
                "utilization_percent": utilization_percent,
                "budget_status": budget_status,
                "month_year": now.strftime("%B %Y"),
            }

    def _get_budget_status(self, total_cost: float) -> str:
        """Get budget status based on utilization.

        Args:
            total_cost: Current month total cost

        Returns:
            Status string indicating budget health.
        """
        utilization = total_cost / self.monthly_budget

        if utilization >= 1.0:
            return "over_budget"
        if utilization >= 0.8:
            return "approaching_limit"
        if utilization >= 0.6:
            return "moderate_usage"
        return "within_budget"

    def _check_budget_alerts(self, _new_cost: float) -> None:
        """Check if budget alerts should be triggered after cost addition.

        Args:
            _new_cost: Cost that was just added (unused but kept for API consistency)
        """
        try:
            summary = self.get_monthly_summary()
            status = summary["budget_status"]
            utilization = summary["utilization_percent"]

            # Show alerts based on budget status
            if status == "over_budget" and STREAMLIT_AVAILABLE:
                budget_msg = (
                    f"🚨 Monthly budget exceeded! "
                    f"${summary['total_cost']:.2f} / ${self.monthly_budget:.2f}"
                )
                st.error(budget_msg)
            elif status == "approaching_limit" and STREAMLIT_AVAILABLE:
                st.warning(f"⚠️ Approaching budget limit: {utilization:.1f}% used")

            # Log warnings
            if status in ["over_budget", "approaching_limit"]:
                logger.warning(
                    "Budget alert: %s - $%.2f / $%.2f (%.1f%% used)",
                    status,
                    summary["total_cost"],
                    self.monthly_budget,
                    utilization,
                )

        except Exception:
            logger.exception("Failed to check budget alerts")

    def get_cost_alerts(self) -> list[dict[str, str]]:
        """Get cost-related alerts for dashboard display.

        Returns:
            List of alert dictionaries with type and message.
        """
        try:
            summary = self.get_monthly_summary()
            alerts = []

            if summary["budget_status"] == "over_budget":
                alerts.append(
                    {
                        "type": "error",
                        "message": (
                            f"Monthly budget exceeded: ${summary['total_cost']:.2f} / "
                            f"${summary['monthly_budget']:.2f}"
                        ),
                    }
                )
            elif summary["budget_status"] == "approaching_limit":
                alerts.append(
                    {
                        "type": "warning",
                        "message": (
                            f"Approaching budget limit: "
                            f"{summary['utilization_percent']:.0f}% used"
                        ),
                    }
                )

        except Exception as e:
            logger.exception("Failed to get cost alerts")
            return [{"type": "error", "message": f"Cost monitoring error: {e}"}]

        return alerts
