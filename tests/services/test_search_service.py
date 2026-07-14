"""Behavioral tests for canonical database-backed job search."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError
from src.models.job_models import ApplicationStage
from src.services.search_service import JobSearchService, search_service

from tests.factories import CompanyFactory, JobFactory


@pytest.fixture
def searchable_jobs(session):
    tech = CompanyFactory(name="Tech Corp")
    ai = CompanyFactory(name="AI Startup")
    jobs = [
        JobFactory(
            company_id=tech.id,
            title="Senior Python Developer",
            description="Build APIs with Django and FastAPI",
            location="San Francisco, CA",
            posted_date=datetime.now(UTC) - timedelta(days=1),
            salary=(120_000, 180_000),
        ),
        JobFactory(
            company_id=tech.id,
            title="Machine Learning Engineer",
            description="Develop Python models with PyTorch",
            location="Remote",
            posted_date=datetime.now(UTC) - timedelta(days=2),
            salary=(150_000, 220_000),
            application_status=ApplicationStage.APPLIED,
        ),
        JobFactory(
            company_id=ai.id,
            title="Data Scientist",
            description="Analyze product data",
            location="New York, NY",
            posted_date=datetime.now(UTC) - timedelta(days=3),
            salary=(130_000, 190_000),
            favorite=True,
        ),
        JobFactory(
            company_id=ai.id,
            title="Frontend Developer",
            description="Build React interfaces",
            location="Austin, TX",
            posted_date=datetime.now(UTC) - timedelta(days=4),
            archived=True,
        ),
    ]
    session.commit()
    return jobs


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Python", ["Senior Python Developer", "Machine Learning Engineer"]),
        ("FastAPI", ["Senior Python Developer"]),
        ("AI Startup", ["Data Scientist"]),
        ("Remote", ["Machine Learning Engineer"]),
        ("python pytorch", ["Machine Learning Engineer"]),
        ("missing", []),
        ("", []),
        ("   ", []),
    ],
)
def test_searches_all_fields_with_and_semantics(searchable_jobs, query, expected):
    assert [job.title for job in JobSearchService().search_jobs(query)] == expected


def test_search_reuses_job_filters(searchable_jobs):
    service = JobSearchService()
    assert [
        job.title
        for job in service.search_jobs(
            "Engineer", {"application_status": [ApplicationStage.APPLIED]}
        )
    ] == ["Machine Learning Engineer"]
    assert [
        job.title for job in service.search_jobs("Data", {"favorites_only": True})
    ] == ["Data Scientist"]
    assert [
        job.title for job in service.search_jobs("Python", {"salary_min": 200_000})
    ] == ["Machine Learning Engineer"]
    assert [
        job.title
        for job in service.search_jobs("Developer", {"include_archived": True})
    ] == ["Senior Python Developer", "Frontend Developer"]


def test_company_filter_count_limit_and_offset_are_applied(searchable_jobs):
    service = JobSearchService()
    filters = {"company": ["Tech Corp"]}
    results = service.search_jobs("Python", filters, limit=1)
    assert len(results) == 1
    assert results[0].company == "Tech Corp"
    assert service.count_jobs("Python", filters) == 2
    second_page = service.search_jobs("Python", filters, limit=1, offset=1)
    assert second_page[0].id != results[0].id


def test_search_treats_wildcards_as_literal_text(searchable_jobs):
    assert JobSearchService().search_jobs("%") == []
    assert JobSearchService().search_jobs("_") == []


@pytest.mark.parametrize("limit", [0, 1001])
def test_search_rejects_invalid_limit(searchable_jobs, limit):
    with pytest.raises(ValueError, match="limit"):
        JobSearchService().search_jobs("Python", limit=limit)


def test_search_rejects_invalid_offset(searchable_jobs):
    with pytest.raises(ValueError, match="offset"):
        JobSearchService().search_jobs("Python", offset=-1)


def test_search_stats_report_honest_stateless_method(searchable_jobs):
    stats = JobSearchService().get_search_stats()
    assert stats == {
        "search_method": "sqlalchemy_like",
        "fts_enabled": False,
        "total_jobs": 4,
        "status": "active",
    }


def test_search_propagates_database_errors():
    with (
        patch(
            "src.services.search_service.db_session",
            side_effect=OperationalError("statement", {}, RuntimeError("offline")),
        ),
        pytest.raises(OperationalError),
    ):
        JobSearchService().search_jobs("Python")


def test_search_stats_return_stable_error_envelope():
    with patch(
        "src.services.search_service.db_session",
        side_effect=OperationalError("statement", {}, RuntimeError("offline")),
    ):
        stats = JobSearchService().get_search_stats()
    assert stats["status"] == "error"
    assert stats["fts_enabled"] is False
    assert stats["total_jobs"] == 0


def test_global_search_service_uses_canonical_default():
    assert isinstance(search_service, JobSearchService)
