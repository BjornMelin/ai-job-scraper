"""End-to-end service tests over the canonical SQLAlchemy database."""

from datetime import date
from unittest.mock import AsyncMock

import pytest
from src.models.job_models import (
    JobPosting,
    JobScrapeResult,
    JobSite,
    SavedSearchRunStatus,
)
from src.schemas import SavedSearchCreate
from src.scraping.scrape_all import run_saved_search
from src.services.analytics_service import AnalyticsService
from src.services.company_service import CompanyService
from src.services.job_service import JobService, job_service
from src.services.saved_search_service import SavedSearchService


def _scrape_result(request, *, title: str = "AI Platform Engineer"):
    posting = JobPosting(
        id="linkedin-123",
        site=JobSite.LINKEDIN,
        title=title,
        company="Signal Labs",
        company_url="https://signal.example",
        job_url="https://signal.example/jobs/123",
        location="Remote",
        date_posted=date.today(),
        min_amount=150_000,
        max_amount=210_000,
        description="Build the platform",
    )
    return JobScrapeResult(
        jobs=[posting],
        total_found=1,
        request_params=request,
        metadata={"success": True, "raw_found": 1},
    )


@pytest.mark.asyncio
async def test_saved_search_run_drives_persistence_facets_and_analytics(
    session, monkeypatch
):
    searches = SavedSearchService()
    search = searches.create(
        SavedSearchCreate(
            name="Remote AI roles",
            query="AI platform engineer",
            location="United States",
            sites=[JobSite.LINKEDIN],
            remote_only=True,
            results_limit=25,
        )
    )
    scraper = AsyncMock()
    scraper.scrape_jobs_async.side_effect = _scrape_result
    monkeypatch.setattr(job_service, "scraper", scraper)

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.SUCCEEDED
    assert completed.jobs_seen == 1
    assert completed.jobs_new == 1
    assert completed.last_error is None
    assert completed.duration_ms is not None

    sent = scraper.scrape_jobs_async.await_args.args[0]
    assert sent.search_term == "AI platform engineer"
    assert sent.is_remote is True
    assert sent.results_wanted == 25

    jobs = JobService.get_filtered_jobs()
    assert len(jobs) == 1
    assert jobs[0].company == "Signal Labs"
    assert jobs[0].posted_date is not None

    companies = CompanyService().get_all_companies()
    assert len(companies) == 1
    assert companies[0].name == "Signal Labs"
    assert companies[0].total_jobs == 1
    assert companies[0].active_jobs == 1

    analytics = AnalyticsService()
    trends = analytics.get_job_trends()
    salary = analytics.get_salary_analytics()
    assert trends["status"] == "success"
    assert trends["method"] == "sqlalchemy"
    assert trends["total_jobs"] == 1
    assert salary["salary_data"]["avg_min_salary"] == 150_000
    assert salary["salary_data"]["avg_max_salary"] == 210_000


@pytest.mark.asyncio
async def test_repeat_run_reports_no_new_jobs(session, monkeypatch):
    search = SavedSearchService().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    scraper = AsyncMock()
    scraper.scrape_jobs_async.side_effect = _scrape_result
    monkeypatch.setattr(job_service, "scraper", scraper)

    first = await run_saved_search(search.id)
    second = await run_saved_search(search.id)

    assert first is not None and first.jobs_new == 1
    assert second is not None and second.jobs_new == 0
    assert len(JobService.get_filtered_jobs()) == 1


@pytest.mark.asyncio
async def test_failed_run_records_stable_health(session, monkeypatch):
    search = SavedSearchService().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    scraper = AsyncMock()
    scraper.scrape_jobs_async.side_effect = ConnectionError("provider unavailable")
    monkeypatch.setattr(job_service, "scraper", scraper)

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.FAILED
    assert completed.jobs_seen == 0
    assert completed.jobs_new == 0
    assert completed.last_error == "provider unavailable"


@pytest.mark.asyncio
async def test_scraper_failure_result_is_not_reported_as_success(session, monkeypatch):
    search = SavedSearchService().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    scraper = AsyncMock()

    def failed_result(request):
        return JobScrapeResult(
            jobs=[],
            total_found=0,
            request_params=request,
            metadata={"success": False, "error": "rate limited"},
        )

    scraper.scrape_jobs_async.side_effect = failed_result
    monkeypatch.setattr(job_service, "scraper", scraper)

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.FAILED
    assert completed.last_error == "rate limited"


@pytest.mark.asyncio
async def test_deleting_search_never_deletes_persisted_jobs(session, monkeypatch):
    searches = SavedSearchService()
    search = searches.create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    scraper = AsyncMock()
    scraper.scrape_jobs_async.side_effect = _scrape_result
    monkeypatch.setattr(job_service, "scraper", scraper)
    await run_saved_search(search.id)

    assert searches.delete(search.id) is True
    assert searches.get(search.id) is None
    assert len(JobService.get_filtered_jobs()) == 1


def test_analytics_status_reports_canonical_engine(session):
    status = AnalyticsService().get_status_report()
    assert status["analytics_method"] == "sqlalchemy"
    assert status["cache_enabled"] is False
    assert status["database_url"].startswith("sqlite:///")
