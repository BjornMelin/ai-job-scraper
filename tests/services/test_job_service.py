"""Behavioral tests for canonical job querying and persistence."""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlmodel import select
from src.database_models import JobSQL
from src.models.job_models import (
    ApplicationStage,
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
)
from src.schemas import Job
from src.services.job_service import JobService, job_service

from tests.factories import CompanyFactory, JobFactory


def _posting(
    *,
    identifier: str = "job-1",
    title: str = "AI Engineer",
    company: str = "Acme",
    url: str = "https://example.com/jobs/1",
    description: str = "Build useful systems",
) -> JobPosting:
    return JobPosting(
        id=identifier,
        site=JobSite.LINKEDIN,
        title=title,
        company=company,
        job_url=url,
        description=description,
        location="Remote",
        date_posted=date.today(),
        min_amount=120_000,
        max_amount=180_000,
    )


@pytest.fixture
def seeded_jobs(session):
    tech = CompanyFactory(name="TechCorp")
    startup = CompanyFactory(name="StartupCo")
    jobs = [
        JobFactory(
            company_id=tech.id,
            title="Senior Python Developer",
            posted_date=datetime.now(UTC) - timedelta(days=1),
            salary=(120_000, 180_000),
            application_status=ApplicationStage.INBOX,
        ),
        JobFactory(
            company_id=startup.id,
            title="ML Engineer",
            posted_date=datetime.now(UTC) - timedelta(days=5),
            salary=(100_000, 150_000),
            application_status=ApplicationStage.APPLIED,
            favorite=True,
        ),
        JobFactory(
            company_id=tech.id,
            title="Archived Data Scientist",
            posted_date=datetime.now(UTC) - timedelta(days=10),
            salary=(90_000, 130_000),
            application_status=ApplicationStage.CLOSED,
            archived=True,
        ),
    ]
    session.commit()
    return tech, startup, jobs


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-15", "2024-01-15"),
        ("2024-12-31T10:30:00Z", "2024-12-31"),
        ("12/31/2024", "2024-12-31"),
        ("31/12/2024", "2024-12-31"),
        ("December 31, 2024", "2024-12-31"),
        ("31 December 2024", "2024-12-31"),
    ],
)
def test_parse_date_supported_formats(value, expected):
    parsed = JobService._parse_date(value)
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed.strftime("%Y-%m-%d") == expected


@pytest.mark.parametrize("value", [None, "", "not-a-date", 123])
def test_parse_date_rejects_unsupported_values(value):
    assert JobService._parse_date(value) is None


def test_parse_date_normalizes_date_and_naive_datetime():
    assert JobService._parse_date(date(2024, 1, 15)) == datetime(
        2024, 1, 15, tzinfo=UTC
    )
    naive = datetime(2024, 1, 15, 12)
    assert JobService._parse_date(naive) == naive.replace(tzinfo=UTC)


def test_dto_conversion_uses_explicit_company_name():
    record = JobSQL.create_validated(
        id=1,
        company_id=2,
        title="Platform Engineer",
        description="Description",
        link="https://example.com/job",
        location="Remote",
        salary=(100_000, 140_000),
    )
    dto = JobService._to_dto_with_company(record, "Acme")
    assert isinstance(dto, Job)
    assert dto.company == "Acme"
    assert dto.salary == (100_000, 140_000)


def test_dto_conversion_defensively_normalizes_a_legacy_null_salary():
    record = JobSQL.create_validated(
        id=1,
        company_id=2,
        title="Platform Engineer",
        description="Description",
        link="https://example.com/job",
        location="Remote",
    )
    record.__dict__["salary"] = None

    assert JobService._to_dto_with_company(record, "Acme").salary == (None, None)


def test_filtered_jobs_apply_canonical_facets(seeded_jobs):
    unfiltered = JobService.get_filtered_jobs()
    assert [job.title for job in unfiltered] == [
        "Senior Python Developer",
        "ML Engineer",
    ]

    assert [
        job.company for job in JobService.get_filtered_jobs({"company": ["TechCorp"]})
    ] == ["TechCorp"]
    assert [
        job.application_status
        for job in JobService.get_filtered_jobs(
            {"application_status": [ApplicationStage.APPLIED]}
        )
    ] == [ApplicationStage.APPLIED]
    assert [
        job.title for job in JobService.get_filtered_jobs({"favorites_only": True})
    ] == ["ML Engineer"]
    assert [
        job.title for job in JobService.get_filtered_jobs({"salary_min": 160_000})
    ] == ["Senior Python Developer"]
    assert len(JobService.get_filtered_jobs({"include_archived": True})) == 3


def test_job_crud_and_counts_persist(seeded_jobs, session):
    _, _, jobs = seeded_jobs
    job_id = jobs[0].id
    assert job_id is not None

    assert JobService.update_job_status(job_id, ApplicationStage.APPLIED) is True
    assert JobService.toggle_favorite(job_id) is True
    assert JobService.update_notes(job_id, "Follow up Friday") is True

    job = JobService.get_job_by_id(job_id)
    assert job is not None
    assert job.application_status is ApplicationStage.APPLIED
    assert job.application_date is not None
    assert job.favorite is True
    assert job.notes == "Follow up Friday"
    assert JobService.get_job_counts_by_status() == {ApplicationStage.APPLIED: 2}

    assert JobService.archive_job(job_id) is True
    assert [item.id for item in JobService.get_filtered_jobs()] == [jobs[1].id]
    session.expire_all()


def test_job_crud_returns_false_for_missing_records(session):
    assert JobService.update_job_status(404, ApplicationStage.APPLIED) is False
    assert JobService.toggle_favorite(404) is False
    assert JobService.update_notes(404, "note") is False
    assert JobService.archive_job(404) is False
    assert JobService.get_job_by_id(404) is None


def test_bulk_update_sets_application_date(seeded_jobs):
    _, _, jobs = seeded_jobs
    assert JobService.bulk_update_jobs(
        [
            {
                "id": jobs[0].id,
                "favorite": True,
                "application_status": ApplicationStage.APPLIED,
                "notes": "Submitted",
            }
        ]
    )
    updated = JobService.get_job_by_id(jobs[0].id)
    assert updated is not None
    assert updated.favorite is True
    assert updated.notes == "Submitted"
    assert updated.application_date is not None
    assert JobService.bulk_update_jobs([]) is True


def test_recent_jobs_are_ordered_and_limited(seeded_jobs):
    recent = JobService.get_recent_jobs(days=7, limit=1)
    assert [job.title for job in recent] == ["Senior Python Developer"]


def test_persistence_reports_insert_update_and_skip(session):
    service = JobService()
    original = _posting()
    second = _posting(
        identifier="job-2",
        company="Beta",
        url="https://example.com/jobs/2",
    )

    assert service._save_jobs_to_database([original, second]) == {
        "inserted": 2,
        "updated": 0,
        "skipped": 0,
    }
    assert service._save_jobs_to_database([original, second]) == {
        "inserted": 0,
        "updated": 0,
        "skipped": 2,
    }
    changed = original.model_copy(update={"title": "Principal AI Engineer"})
    assert service._save_jobs_to_database([changed]) == {
        "inserted": 0,
        "updated": 1,
        "skipped": 0,
    }
    assert JobService.get_filtered_jobs({"company": ["Acme"]})[0].title == (
        "Principal AI Engineer"
    )


def test_sparse_rescrape_preserves_richer_stored_fields(session):
    service = JobService()
    original = _posting()
    assert service._save_jobs_to_database([original])["inserted"] == 1

    sparse = original.model_copy(
        update={
            "description": None,
            "location": None,
            "date_posted": None,
            "min_amount": None,
            "max_amount": None,
        }
    )
    assert service._save_jobs_to_database([sparse]) == {
        "inserted": 0,
        "updated": 0,
        "skipped": 1,
    }

    persisted = JobService.get_filtered_jobs()[0]
    assert persisted.description == original.description
    assert persisted.location == original.location
    assert persisted.posted_date is not None
    assert persisted.salary == (120_000, 180_000)


def test_sparse_rescrape_recovers_a_legacy_sql_null_salary(session):
    service = JobService()
    original = _posting()
    assert service._save_jobs_to_database([original])["inserted"] == 1
    existing = session.exec(select(JobSQL).where(JobSQL.link == original.job_url)).one()
    existing.__dict__["salary"] = None

    sparse = original.model_copy(update={"min_amount": None, "max_amount": None})
    assert service._persist_jobs(session, [sparse]) == {
        "inserted": 0,
        "updated": 0,
        "skipped": 1,
    }

    assert JobService._to_dto_with_company(existing, "Acme").salary == (None, None)


@pytest.mark.asyncio
async def test_search_and_save_uses_typed_request_and_exact_persistence(session):
    service = JobService()
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN],
        search_term="AI engineer",
    )
    service.scraper = AsyncMock()
    service.scraper.scrape_jobs_async.return_value = JobScrapeResult(
        jobs=[_posting()],
        total_found=1,
        request_params=request,
        metadata={"success": True},
    )

    result = await service.search_and_save_jobs(
        "AI engineer",
        location="Denver",
        sites=["linkedin", JobSite.INDEED],
        is_remote=True,
        job_type=JobType.FULLTIME,
        results_wanted=25,
    )

    sent = service.scraper.scrape_jobs_async.await_args.args[0]
    assert sent.site_name == [JobSite.LINKEDIN, JobSite.INDEED]
    assert sent.location == "Denver"
    assert sent.is_remote is True
    assert sent.job_type is JobType.FULLTIME
    assert sent.enforce_annual_salary is True
    assert result.metadata["persistence"] == {
        "inserted": 1,
        "updated": 0,
        "skipped": 0,
    }


def test_persistence_propagates_database_failure():
    with (
        patch(
            "src.services.job_service.db_session",
            side_effect=OperationalError("statement", {}, RuntimeError("offline")),
        ),
        pytest.raises(OperationalError),
    ):
        JobService()._save_jobs_to_database([_posting()])


def test_global_service_instance_is_ready():
    assert isinstance(job_service, JobService)
    assert job_service.scraper is not None
