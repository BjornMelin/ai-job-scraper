"""Analytics over the canonical SQLAlchemy database."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import DatabaseError
from sqlmodel import Session
from src.database_models import CompanySQL, JobSQL
from src.services.analytics_service import AnalyticsService


@pytest.fixture
def analytics_data(session: Session) -> None:
    now = datetime.now(UTC)
    first = CompanySQL(name="First", url=None)
    second = CompanySQL(name="Second", url=None)
    session.add_all([first, second])
    session.flush()
    session.add_all(
        [
            JobSQL.create_validated(
                company_id=first.id,
                title="ML Engineer",
                description="One",
                link="https://first.example/1",
                location="Remote",
                posted_date=now,
                salary=(100_000, 150_000),
            ),
            JobSQL.create_validated(
                company_id=first.id,
                title="AI Engineer",
                description="Two",
                link="https://first.example/2",
                location="Remote",
                posted_date=now - timedelta(days=1),
                salary=(120_000, 180_000),
            ),
            JobSQL.create_validated(
                company_id=second.id,
                title="Archived",
                description="Ignored",
                link="https://second.example/1",
                location="Remote",
                posted_date=now,
                salary=(90_000, 130_000),
                archived=True,
            ),
        ]
    )
    session.commit()


def test_job_trends_use_current_database(analytics_data) -> None:
    result = AnalyticsService().get_job_trends(days=7)

    assert result["status"] == "success"
    assert result["method"] == "sqlalchemy"
    assert result["total_jobs"] == 2
    assert sum(point["job_count"] for point in result["trends"]) == 2


def test_company_analytics_are_job_derived(analytics_data) -> None:
    result = AnalyticsService().get_company_analytics()

    assert result["status"] == "success"
    assert result["total_companies"] == 1
    company = result["companies"][0]
    assert company["last_job_posted"] is not None
    assert {
        key: value for key, value in company.items() if key != "last_job_posted"
    } == {
        "company": "First",
        "total_jobs": 2,
        "avg_min_salary": 110_000,
        "avg_max_salary": 165_000,
    }


def test_salary_analytics_exclude_archived_jobs(analytics_data) -> None:
    result = AnalyticsService().get_salary_analytics(days=7)

    assert result["status"] == "success"
    assert result["salary_data"] == {
        "total_jobs_with_salary": 2,
        "avg_min_salary": 110_000,
        "avg_max_salary": 165_000,
        "min_salary": 100_000,
        "max_salary": 180_000,
        "salary_std_dev": 10_000,
        "analysis_period_days": 7,
    }


def test_invalid_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        AnalyticsService().get_job_trends(days=0)


def test_query_failure_returns_stable_error(monkeypatch) -> None:
    class BrokenContext:
        def __enter__(self):
            raise DatabaseError("query", {}, RuntimeError("offline"))

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "src.services.analytics_service.db_session",
        lambda _bind=None: BrokenContext(),
    )

    result = AnalyticsService().get_job_trends()
    assert result["status"] == "error"
    assert result["trends"] == []
    assert result["method"] == "sqlalchemy"


def test_status_reports_uncached_canonical_engine(session: Session) -> None:
    status = AnalyticsService(session.bind).get_status_report()

    assert status["analytics_method"] == "sqlalchemy"
    assert status["cache_enabled"] is False
    assert status["status"] == "active"
