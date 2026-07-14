"""Company facets are derived from jobs, never scrape configuration."""

from datetime import UTC, datetime, timedelta

from sqlmodel import Session
from src.database_models import CompanySQL, JobSQL
from src.services.company_service import CompanyService


def test_company_facets_include_only_companies_with_jobs(session: Session) -> None:
    now = datetime.now(UTC)
    used = CompanySQL(name="Used", url="https://used.example")
    orphan = CompanySQL(name="Orphan", url=None)
    session.add_all([used, orphan])
    session.flush()
    session.add_all(
        [
            JobSQL.create_validated(
                company_id=used.id,
                title="Current",
                description="Current job",
                link="https://used.example/jobs/current",
                location="Remote",
                posted_date=now,
            ),
            JobSQL.create_validated(
                company_id=used.id,
                title="Archived",
                description="Archived job",
                link="https://used.example/jobs/archived",
                location="Remote",
                posted_date=now - timedelta(days=2),
                archived=True,
            ),
        ]
    )
    session.commit()

    assert [
        company.model_dump() for company in CompanyService().get_all_companies()
    ] == [
        {
            "id": used.id,
            "name": "Used",
            "url": "https://used.example",
            "total_jobs": 2,
            "active_jobs": 1,
            "last_job_posted": now,
        }
    ]


def test_company_facets_are_empty_without_jobs(session: Session) -> None:
    session.add(CompanySQL(name="Not a scrape source", url=None))
    session.commit()

    assert CompanyService().get_all_companies() == []
