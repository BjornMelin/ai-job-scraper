"""Operational cost tracking over the canonical application database."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.engine import Connection, Engine
from sqlmodel import select

from src.database import db_session
from src.database_models import CostEntry

logger = logging.getLogger(__name__)


class CostMonitor:
    """Record cost events and report current-month budget health."""

    def __init__(
        self,
        bind: Engine | Connection | None = None,
        *,
        monthly_budget: float = 50.0,
    ) -> None:
        if not math.isfinite(monthly_budget) or monthly_budget <= 0:
            raise ValueError("monthly_budget must be finite and positive")
        self._bind = bind
        self.monthly_budget = monthly_budget

    def _track(
        self,
        *,
        service: str,
        operation: str,
        cost: float,
        extra_data: dict[str, object],
    ) -> None:
        with db_session(self._bind) as session:
            session.add(
                CostEntry.create_validated(
                    service=service,
                    operation=operation,
                    cost_usd=cost,
                    extra_data=extra_data,
                )
            )
        self._check_budget_alerts()

    def track_ai_cost(
        self,
        model: str,
        tokens: int,
        cost: float,
        operation: str,
    ) -> None:
        """Record one AI operation."""
        if tokens < 0:
            raise ValueError("tokens must be nonnegative")
        self._track(
            service="ai",
            operation=operation,
            cost=cost,
            extra_data={"model": model, "tokens": tokens},
        )

    def track_proxy_cost(self, requests: int, cost: float, endpoint: str) -> None:
        """Record one proxy charge."""
        if requests < 0:
            raise ValueError("requests must be nonnegative")
        self._track(
            service="proxy",
            operation="requests",
            cost=cost,
            extra_data={"requests": requests, "endpoint": endpoint},
        )

    def track_scraping_cost(self, company: str, jobs_found: int, cost: float) -> None:
        """Record one scraping charge."""
        if jobs_found < 0:
            raise ValueError("jobs_found must be nonnegative")
        self._track(
            service="scraping",
            operation="search_run",
            cost=cost,
            extra_data={"company": company, "jobs_found": jobs_found},
        )

    def get_monthly_summary(self) -> dict[str, Any]:
        """Return current-month costs, counts, and budget status."""
        now = datetime.now(UTC)
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        with db_session(self._bind) as session:
            rows = session.exec(
                select(
                    CostEntry.service,
                    func.sum(CostEntry.cost_usd),
                    func.count(CostEntry.id),
                )
                .where(CostEntry.timestamp >= start_of_month)
                .group_by(CostEntry.service)
            ).all()

        costs_by_service: dict[str, float] = {}
        operation_counts: dict[str, int] = {}
        for service, total, count in rows:
            costs_by_service[service] = float(total)
            operation_counts[service] = count
        total_cost = sum(costs_by_service.values())
        return {
            "costs_by_service": costs_by_service,
            "operation_counts": operation_counts,
            "total_cost": total_cost,
            "monthly_budget": self.monthly_budget,
            "remaining": self.monthly_budget - total_cost,
            "utilization_percent": (total_cost / self.monthly_budget) * 100,
            "budget_status": self._get_budget_status(total_cost),
            "month_year": now.strftime("%B %Y"),
        }

    def _get_budget_status(self, total_cost: float) -> str:
        utilization = total_cost / self.monthly_budget
        if utilization >= 1:
            return "over_budget"
        if utilization >= 0.8:
            return "approaching_limit"
        if utilization >= 0.6:
            return "moderate_usage"
        return "within_budget"

    def _check_budget_alerts(self) -> None:
        """Log threshold crossings without coupling persistence to the UI."""
        try:
            summary = self.get_monthly_summary()
        except Exception:
            logger.exception("Cost persisted, but budget alert evaluation failed")
            return
        if summary["budget_status"] in {"over_budget", "approaching_limit"}:
            logger.warning(
                "Budget alert: %s, $%.2f of $%.2f",
                summary["budget_status"],
                summary["total_cost"],
                summary["monthly_budget"],
            )

    def get_cost_alerts(self) -> list[dict[str, str]]:
        """Return presentation-neutral alerts for the dashboard."""
        try:
            summary = self.get_monthly_summary()
        except Exception as error:
            logger.exception("Cost summary failed")
            return [{"type": "error", "message": f"Cost monitoring error: {error}"}]

        if summary["budget_status"] == "over_budget":
            return [
                {
                    "type": "error",
                    "message": (
                        f"Monthly budget exceeded: ${summary['total_cost']:.2f} / "
                        f"${summary['monthly_budget']:.2f}"
                    ),
                }
            ]
        if summary["budget_status"] == "approaching_limit":
            return [
                {
                    "type": "warning",
                    "message": (
                        "Approaching budget limit: "
                        f"{summary['utilization_percent']:.0f}% used"
                    ),
                }
            ]
        return []
