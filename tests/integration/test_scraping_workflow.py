"""Integration tests for JobSpy conversion and atomic persistence."""

from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sqlmodel import select
from src.database_models import JobSQL
from src.models.job_models import JobPosting, JobScrapeRequest, JobSite
from src.scraping.job_scraper import JobSpyScraper
from src.services.job_service import JobService

from tests.factories import CompanyFactory, JobFactory


def _posting(
    *,
    identifier: str,
    url: str,
    title: str = "AI Engineer",
    description: str = "Original description",
) -> JobPosting:
    return JobPosting(
        id=identifier,
        site=JobSite.LINKEDIN,
        title=title,
        company="Acme",
        job_url=url,
        location="Remote",
        date_posted=date.today(),
        description=description,
    )


def test_dataframe_conversion_normalizes_real_jobspy_shapes():
    frame = pd.DataFrame(
        [
            {
                "id": None,
                "site": None,
                "title": " Platform Engineer ",
                "company": " Acme ",
                "job_url": "https://example.com/jobs/1",
                "date_posted": pd.Timestamp("2026-07-14"),
                "emails": "jobs@example.com",
                "skills": ["Python", "SQL"],
                "company_addresses": ("Denver",),
                "min_amount": "120000",
            }
        ]
    )

    jobs, invalid_rows = JobSpyScraper()._dataframe_to_models(frame, JobSite.LINKEDIN)

    assert len(jobs) == 1
    assert invalid_rows == 0
    job = jobs[0]
    assert job.id == "https://example.com/jobs/1"
    assert job.site is JobSite.LINKEDIN
    assert job.title == "Platform Engineer"
    assert job.company == "Acme"
    assert job.emails == ["jobs@example.com"]
    assert job.skills == ["Python", "SQL"]
    assert job.company_addresses == ["Denver"]
    assert job.min_amount == 120_000


def test_dataframe_conversion_skips_rows_without_persistable_identity():
    frame = pd.DataFrame(
        [
            {
                "id": "valid",
                "site": "linkedin",
                "title": "AI Engineer",
                "company": "Acme",
                "job_url": "https://example.com/valid",
            },
            {
                "id": "missing-company",
                "site": "linkedin",
                "title": "AI Engineer",
                "job_url": "https://example.com/no-company",
            },
            {
                "id": "missing-url",
                "site": "linkedin",
                "title": "AI Engineer",
                "company": "Acme",
            },
            {
                "id": "missing-title",
                "site": "linkedin",
                "title": " ",
                "company": "Acme",
                "job_url": "https://example.com/no-title",
            },
        ]
    )

    jobs, invalid_rows = JobSpyScraper()._dataframe_to_models(frame, JobSite.LINKEDIN)

    assert [job.id for job in jobs] == ["valid"]
    assert invalid_rows == 3


def test_sync_scrape_returns_explicit_failed_result_on_provider_error():
    request = JobScrapeRequest(
        site_name=JobSite.LINKEDIN,
        search_term="AI Engineer",
    )
    with patch(
        "src.scraping.job_scraper.scrape_jobs",
        side_effect=ConnectionError("offline"),
    ):
        result = JobSpyScraper().scrape_jobs_sync(request)

    assert result.jobs == []
    assert result.total_found == 0
    assert result.metadata == {
        "scraping_method": "jobspy",
        "success": False,
        "raw_found": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "error": "Scraping operation failed",
    }


@pytest.mark.asyncio
async def test_async_scrape_delegates_to_sync_implementation():
    scraper = JobSpyScraper()
    request = JobScrapeRequest(site_name=JobSite.LINKEDIN, search_term="AI")
    expected = scraper._empty_result(request)
    scraper.scrape_jobs_sync = Mock(return_value=expected)

    assert await scraper.scrape_jobs_async(request) == expected
    scraper.scrape_jobs_sync.assert_called_once_with(request)


def test_scrape_persistence_is_atomic(session):
    company = CompanyFactory(name="Acme")
    session.commit()
    service = JobService()
    jobs = [
        _posting(identifier="one", url="https://example.com/jobs/one"),
        _posting(identifier="two", url="https://example.com/jobs/two"),
    ]
    with (
        patch.object(
            service,
            "_get_or_create_company",
            side_effect=[company.id, RuntimeError("second row failed")],
        ),
        pytest.raises(RuntimeError, match="second row failed"),
    ):
        service._save_jobs_to_database(jobs)

    session.expire_all()
    assert session.exec(select(JobSQL)).all() == []


def test_rescrape_preserves_user_owned_state(session):
    company = CompanyFactory(name="Acme")
    job = JobFactory(
        company_id=company.id,
        link="https://example.com/jobs/1",
        title="AI Engineer",
        favorite=True,
        notes="Warm intro",
        application_status="Applied",
    )
    session.commit()

    result = JobService()._save_jobs_to_database(
        [
            _posting(
                identifier="one",
                url="https://example.com/jobs/1",
                title="Principal AI Engineer",
                description="Expanded scope",
            )
        ]
    )

    assert result == {"inserted": 0, "updated": 1, "skipped": 0}
    session.expire_all()
    refreshed = session.get(JobSQL, job.id)
    assert refreshed is not None
    assert refreshed.title == "Principal AI Engineer"
    assert refreshed.description == "Expanded scope"
    assert refreshed.favorite is True
    assert refreshed.notes == "Warm intro"
    assert refreshed.application_status == "Applied"


def test_direct_application_url_is_canonical(session):
    posting = _posting(identifier="one", url="https://listing.example/1")
    posting.job_url_direct = "https://apply.example/1"

    assert JobService()._save_jobs_to_database([posting])["inserted"] == 1
    saved = session.exec(select(JobSQL)).one()
    assert saved.link == "https://apply.example/1"
