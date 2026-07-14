"""Canonical job search over the application database."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from src.database import db_session
from src.database_models import CompanySQL, JobSQL
from src.schemas import Job
from src.services.job_service import FilterDict, JobService

logger = logging.getLogger(__name__)


class JobSearchService:
    """Search persisted jobs without a second connection or cache owner."""

    def __init__(self, bind: Engine | Connection | None = None) -> None:
        self._bind = bind

    @staticmethod
    def _apply_search(statement, terms: list[str], filters: FilterDict):
        for term in terms:
            statement = statement.where(
                or_(
                    JobSQL.title.contains(term, autoescape=True),
                    JobSQL.description.contains(term, autoescape=True),
                    CompanySQL.name.contains(term, autoescape=True),
                    JobSQL.location.contains(term, autoescape=True),
                )
            )
        return JobService._apply_filters_to_query(statement, filters)

    def search_jobs(
        self,
        query: str,
        filters: FilterDict | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        """Match every query term across the canonical searchable job fields."""
        terms = query.split()
        if not terms:
            return []
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("offset must be nonnegative")

        statement = select(JobSQL, CompanySQL.name.label("company_name")).join(
            CompanySQL,
            JobSQL.company_id == CompanySQL.id,
        )
        statement = (
            self._apply_search(statement, terms, filters or {})
            .offset(offset)
            .limit(limit)
        )

        try:
            with db_session(self._bind) as session:
                rows = session.exec(statement).all()
            return [
                JobService._to_dto_with_company(job, company_name)
                for job, company_name in rows
            ]
        except SQLAlchemyError:
            logger.exception("Job search failed")
            raise

    def count_jobs(
        self,
        query: str,
        filters: FilterDict | None = None,
    ) -> int:
        """Count every job matching a text query and the shared filters."""
        terms = query.split()
        if not terms:
            return 0
        statement = select(func.count(JobSQL.id)).join(
            CompanySQL,
            JobSQL.company_id == CompanySQL.id,
        )
        statement = self._apply_search(
            statement,
            terms,
            filters or {},
        ).order_by(None)
        try:
            with db_session(self._bind) as session:
                return session.exec(statement).one()
        except SQLAlchemyError:
            logger.exception("Job search count failed")
            raise

    def get_search_stats(self) -> dict[str, Any]:
        """Return honest status for the stateless SQL search path."""
        try:
            with db_session(self._bind) as session:
                total_jobs = session.exec(select(func.count(JobSQL.id))).one()
            return {
                "search_method": "sqlalchemy_like",
                "fts_enabled": False,
                "total_jobs": total_jobs,
                "status": "active",
            }
        except SQLAlchemyError as error:
            logger.exception("Search status query failed")
            return {
                "search_method": "sqlalchemy_like",
                "fts_enabled": False,
                "total_jobs": 0,
                "status": "error",
                "error": str(error),
            }


search_service = JobSearchService()
