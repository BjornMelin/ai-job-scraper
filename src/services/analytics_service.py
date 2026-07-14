"""Small-dataset analytics over the canonical application database."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from statistics import fmean, pstdev
from typing import Any

from sqlalchemy import func
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from src.database import db_session, get_engine
from src.database_models import CompanySQL, JobSQL

logger = logging.getLogger(__name__)
METHOD = "sqlalchemy"


def _salary_pair(value: object) -> tuple[int | None, int | None]:
    if isinstance(value, list | tuple) and len(value) == 2:
        minimum, maximum = value
        return (
            int(minimum) if minimum is not None else None,
            int(maximum) if maximum is not None else None,
        )
    return (None, None)


class AnalyticsService:
    """Compute dashboard analytics without a duplicate database engine."""

    def __init__(self, bind: Engine | Connection | None = None) -> None:
        self._bind = bind

    def get_job_trends(self, days: int = 30) -> dict[str, Any]:
        if days < 1:
            raise ValueError("days must be positive")
        try:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            with db_session(self._bind) as session:
                rows = session.exec(
                    select(func.date(JobSQL.posted_date), func.count(JobSQL.id))
                    .where(
                        JobSQL.posted_date >= cutoff,
                        JobSQL.archived.is_(False),
                    )
                    .group_by(func.date(JobSQL.posted_date))
                    .order_by(func.date(JobSQL.posted_date)),
                ).all()
            trends = [
                {"date": posted_date, "job_count": count} for posted_date, count in rows
            ]
            return {
                "trends": trends,
                "method": METHOD,
                "status": "success",
                "total_jobs": sum(item["job_count"] for item in trends),
            }
        except SQLAlchemyError as error:
            logger.exception("Job trends query failed")
            return self._error("trends", [], error)

    def get_company_analytics(self) -> dict[str, Any]:
        try:
            with db_session(self._bind) as session:
                rows = session.exec(
                    select(
                        CompanySQL.name,
                        JobSQL.salary,
                        JobSQL.posted_date,
                    )
                    .join(JobSQL, JobSQL.company_id == CompanySQL.id)
                    .where(JobSQL.archived.is_(False)),
                ).all()

            grouped: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"count": 0, "minimums": [], "maximums": [], "dates": []}
            )
            for company, salary, posted_date in rows:
                grouped[company]["count"] += 1
                minimum, maximum = _salary_pair(salary)
                if minimum is not None:
                    grouped[company]["minimums"].append(minimum)
                if maximum is not None:
                    grouped[company]["maximums"].append(maximum)
                if posted_date is not None:
                    grouped[company]["dates"].append(posted_date)

            companies = [
                {
                    "company": company,
                    "total_jobs": data["count"],
                    "avg_min_salary": round(fmean(data["minimums"]), 2)
                    if data["minimums"]
                    else 0,
                    "avg_max_salary": round(fmean(data["maximums"]), 2)
                    if data["maximums"]
                    else 0,
                    "last_job_posted": max(data["dates"]) if data["dates"] else None,
                }
                for company, data in grouped.items()
            ]
            companies.sort(key=lambda item: (-item["total_jobs"], item["company"]))
            companies = companies[:20]
            return {
                "companies": companies,
                "method": METHOD,
                "status": "success",
                "total_companies": len(companies),
            }
        except SQLAlchemyError as error:
            logger.exception("Company analytics query failed")
            return self._error("companies", [], error)

    def get_salary_analytics(self, days: int = 90) -> dict[str, Any]:
        if days < 1:
            raise ValueError("days must be positive")
        try:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            with db_session(self._bind) as session:
                salaries = session.exec(
                    select(JobSQL.salary).where(
                        JobSQL.posted_date >= cutoff,
                        JobSQL.archived.is_(False),
                    ),
                ).all()
            pairs = [_salary_pair(salary) for salary in salaries]
            minimums = [minimum for minimum, _ in pairs if minimum is not None]
            maximums = [maximum for _, maximum in pairs if maximum is not None]
            salary_data = {
                "total_jobs_with_salary": sum(
                    minimum is not None or maximum is not None
                    for minimum, maximum in pairs
                ),
                "avg_min_salary": round(fmean(minimums), 2) if minimums else 0,
                "avg_max_salary": round(fmean(maximums), 2) if maximums else 0,
                "min_salary": min(minimums) if minimums else 0,
                "max_salary": max(maximums) if maximums else 0,
                "salary_std_dev": round(pstdev(minimums), 2)
                if len(minimums) > 1
                else 0,
                "analysis_period_days": days,
            }
            return {"salary_data": salary_data, "method": METHOD, "status": "success"}
        except SQLAlchemyError as error:
            logger.exception("Salary analytics query failed")
            return self._error("salary_data", {}, error)

    def get_status_report(self) -> dict[str, Any]:
        engine = self._bind.engine if isinstance(self._bind, Connection) else self._bind
        engine = engine or get_engine()
        return {
            "analytics_method": METHOD,
            "database_url": engine.url.render_as_string(hide_password=True),
            "status": "active",
            "cache_enabled": False,
        }

    @staticmethod
    def _error(key: str, empty: object, error: Exception) -> dict[str, Any]:
        return {
            key: empty,
            "status": "error",
            "error": str(error),
            "method": METHOD,
        }
