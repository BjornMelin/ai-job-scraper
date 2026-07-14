"""Read-only company facets derived from persisted jobs."""

from sqlalchemy import case, func
from sqlmodel import select

from src.database import db_session
from src.database_models import CompanySQL, JobSQL
from src.schemas import Company


class CompanyService:
    """Expose companies as job-derived facets, never scrape configuration."""

    def get_all_companies(self) -> list[Company]:
        with db_session() as session:
            rows = session.exec(
                select(
                    CompanySQL.id,
                    CompanySQL.name,
                    CompanySQL.url,
                    func.count(JobSQL.id),
                    func.sum(case((JobSQL.archived.is_(False), 1), else_=0)),
                    func.max(JobSQL.posted_date),
                )
                .join(JobSQL, JobSQL.company_id == CompanySQL.id)
                .group_by(CompanySQL.id)
                .order_by(func.count(JobSQL.id).desc(), CompanySQL.name),
            ).all()
            return [
                Company(
                    id=company_id,
                    name=name,
                    url=url,
                    total_jobs=total_jobs,
                    active_jobs=active_jobs or 0,
                    last_job_posted=last_job_posted,
                )
                for (
                    company_id,
                    name,
                    url,
                    total_jobs,
                    active_jobs,
                    last_job_posted,
                ) in rows
            ]


company_service = CompanyService()
