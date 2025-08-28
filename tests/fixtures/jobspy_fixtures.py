"""Comprehensive test fixtures for JobSpy integration testing.

This module provides realistic mock data and fixtures for testing JobSpy integration
without making actual network calls. All data is completely mocked for fast,
deterministic testing.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from src.models.job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
    LocationType,
)


@pytest.fixture
def sample_jobspy_raw_data() -> list[dict[str, Any]]:
    """Raw data that JobSpy would return as DataFrame rows."""
    return [
        {
            "id": "job_001_linkedin",
            "site": "linkedin",
            "job_url": "https://linkedin.com/jobs/12345",
            "job_url_direct": "https://linkedin.com/jobs/12345?apply=true",
            "title": "Senior Python Developer",
            "company": "TechCorp Inc",
            "location": "San Francisco, CA",
            "date_posted": "2024-01-15",
            "job_type": "fulltime",
            "salary_source": "employer",
            "interval": "yearly",
            "min_amount": 120000.0,
            "max_amount": 180000.0,
            "currency": "USD",
            "is_remote": False,
            "location_type": "onsite",
            "job_level": "Senior",
            "job_function": "Engineering",
            "listing_type": "external",
            "description": "We are seeking a Senior Python Developer to join our team...",
            "emails": ["jobs@techcorp.com"],
            "skills": ["Python", "Django", "PostgreSQL", "AWS"],
            "experience_range": "5-8 years",
            "vacancy_count": 3,
            "company_industry": "Technology",
            "company_url": "https://techcorp.com",
            "company_logo": "https://techcorp.com/logo.png",
            "company_url_direct": "https://techcorp.com/careers",
            "company_addresses": ["123 Tech St, San Francisco, CA"],
            "company_num_employees": "1001-5000",
            "company_revenue": "$100M-500M",
            "company_description": "Leading technology company focused on innovation",
            "company_rating": 4.5,
            "company_reviews_count": 1250,
        },
        {
            "id": "job_002_indeed",
            "site": "indeed",
            "job_url": "https://indeed.com/jobs/67890",
            "job_url_direct": None,
            "title": "Data Scientist",
            "company": "DataCo Analytics",
            "location": "Remote",
            "date_posted": "2024-01-14",
            "job_type": "contract",
            "salary_source": None,
            "interval": None,
            "min_amount": None,
            "max_amount": None,
            "currency": None,
            "is_remote": True,
            "location_type": "remote",
            "job_level": "Mid",
            "job_function": "Data Science",
            "listing_type": "premium",
            "description": "Join our data science team to build ML models...",
            "emails": None,
            "skills": ["Python", "TensorFlow", "SQL", "Statistics"],
            "experience_range": "3-5 years",
            "vacancy_count": 1,
            "company_industry": "Analytics",
            "company_url": "https://dataco.com",
            "company_logo": None,
            "company_url_direct": "https://dataco.com/jobs",
            "company_addresses": None,
            "company_num_employees": "51-200",
            "company_revenue": None,
            "company_description": "Analytics consulting firm",
            "company_rating": None,
            "company_reviews_count": None,
        },
        {
            "id": "job_003_glassdoor",
            "site": "glassdoor",
            "job_url": "https://glassdoor.com/jobs/11111",
            "job_url_direct": "https://glassdoor.com/jobs/11111/apply",
            "title": "Frontend Developer",
            "company": "StartupCo",
            "location": "Austin, TX",
            "date_posted": "2024-01-13",
            "job_type": "fulltime",
            "salary_source": "glassdoor_estimate",
            "interval": "yearly",
            "min_amount": 80000.0,
            "max_amount": 120000.0,
            "currency": "USD",
            "is_remote": False,
            "location_type": "hybrid",
            "job_level": "Junior",
            "job_function": "Engineering",
            "listing_type": "standard",
            "description": "Looking for a creative frontend developer...",
            "emails": ["hiring@startupco.com", "tech@startupco.com"],
            "skills": ["React", "JavaScript", "CSS", "HTML"],
            "experience_range": "1-3 years",
            "vacancy_count": 2,
            "company_industry": "Software",
            "company_url": "https://startupco.com",
            "company_logo": "https://startupco.com/assets/logo.jpg",
            "company_url_direct": "https://startupco.com/careers",
            "company_addresses": ["456 Startup Blvd, Austin, TX"],
            "company_num_employees": "11-50",
            "company_revenue": "$1M-10M",
            "company_description": "Innovative startup in the fintech space",
            "company_rating": 3.8,
            "company_reviews_count": 45,
        },
    ]


@pytest.fixture
def sample_jobspy_dataframe(sample_jobspy_raw_data) -> pd.DataFrame:
    """Mock JobSpy DataFrame output with realistic data."""
    return pd.DataFrame(sample_jobspy_raw_data)


@pytest.fixture
def empty_jobspy_dataframe() -> pd.DataFrame:
    """Empty DataFrame simulating no results from JobSpy."""
    return pd.DataFrame()


@pytest.fixture
def malformed_jobspy_dataframe() -> pd.DataFrame:
    """DataFrame with malformed/missing data for edge case testing."""
    return pd.DataFrame(
        [
            {
                "id": "malformed_001",
                "site": "unknown_site",  # Invalid site
                "title": "",  # Empty title
                "company": None,  # None company
                "location": "Invalid Location Format",
                "date_posted": "invalid-date",  # Invalid date
                "job_type": "unknown_type",  # Invalid job type
                "min_amount": "not_a_number",  # Invalid salary
                "max_amount": "",  # Empty salary
                "is_remote": "maybe",  # Invalid boolean
                "company_rating": "five_stars",  # Invalid rating
            }
        ]
    )


@pytest.fixture
def sample_job_scrape_request() -> JobScrapeRequest:
    """Standard job scrape request for testing."""
    return JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.INDEED],
        search_term="Python developer",
        location="San Francisco, CA",
        distance=25,
        is_remote=False,
        job_type=JobType.FULLTIME,
        results_wanted=50,
        hours_old=24,
    )


@pytest.fixture
def remote_job_scrape_request() -> JobScrapeRequest:
    """Remote job scrape request for testing location types."""
    return JobScrapeRequest(
        site_name=JobSite.LINKEDIN,
        search_term="Data Scientist",
        location="Remote",
        is_remote=True,
        job_type=JobType.CONTRACT,
        results_wanted=25,
    )


@pytest.fixture
def sample_job_postings(sample_jobspy_raw_data) -> list[JobPosting]:
    """List of JobPosting objects for testing."""
    return [JobPosting.model_validate(data) for data in sample_jobspy_raw_data]


@pytest.fixture
def sample_job_scrape_result(
    sample_job_postings, sample_job_scrape_request
) -> JobScrapeResult:
    """Complete job scrape result for testing."""
    return JobScrapeResult(
        jobs=sample_job_postings,
        total_found=len(sample_job_postings),
        request_params=sample_job_scrape_request,
        metadata={
            "scrape_timestamp": datetime.now().isoformat(),
            "success_rate": 1.0,
            "total_sites_scraped": 2,
        },
    )


@pytest.fixture
def mock_jobspy_scrape_success(monkeypatch, sample_jobspy_dataframe):
    """Mock jobspy.scrape_jobs function to return successful results."""

    def mock_scrape_jobs(**kwargs):
        # Simulate realistic behavior based on parameters
        df = sample_jobspy_dataframe.copy()

        # Filter by site if specified
        if kwargs.get("site_name"):
            sites = (
                kwargs["site_name"]
                if isinstance(kwargs["site_name"], list)
                else [kwargs["site_name"]]
            )
            site_names = [
                site.value if hasattr(site, "value") else str(site).lower()
                for site in sites
            ]
            df = df[df["site"].isin(site_names)]

        # Limit results if specified
        if kwargs.get("results_wanted"):
            df = df.head(kwargs["results_wanted"])

        return df

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    return mock_scrape_jobs


@pytest.fixture
def mock_jobspy_scrape_empty(monkeypatch, empty_jobspy_dataframe):
    """Mock jobspy.scrape_jobs to return no results."""

    def mock_scrape_jobs(**kwargs):
        return empty_jobspy_dataframe

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    return mock_scrape_jobs


@pytest.fixture
def mock_jobspy_scrape_error(monkeypatch):
    """Mock jobspy.scrape_jobs to raise an exception."""

    def mock_scrape_jobs(**kwargs):
        raise ConnectionError("Failed to connect to job site")

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    return mock_scrape_jobs


@pytest.fixture
def mock_jobspy_scrape_malformed(monkeypatch, malformed_jobspy_dataframe):
    """Mock jobspy.scrape_jobs to return malformed data."""

    def mock_scrape_jobs(**kwargs):
        return malformed_jobspy_dataframe

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    return mock_scrape_jobs


@pytest.fixture
def jobsite_enum_values():
    """All valid JobSite enum values for parametrized testing."""
    return list(JobSite)


@pytest.fixture
def jobtype_enum_values():
    """All valid JobType enum values for parametrized testing."""
    return list(JobType)


@pytest.fixture
def locationtype_enum_values():
    """All valid LocationType enum values for parametrized testing."""
    return list(LocationType)


@pytest.fixture
def edge_case_test_data():
    """Edge cases for comprehensive validation testing."""
    return {
        "empty_strings": {"title": "", "company": "", "location": ""},
        "none_values": {"title": None, "company": None, "location": None},
        "whitespace_only": {"title": "   ", "company": "\t", "location": "\n"},
        "very_long_strings": {
            "title": "a" * 1000,
            "company": "b" * 500,
            "location": "c" * 200,
        },
        "special_characters": {
            "title": "Software Engineer @#$%^&*()",
            "company": "Company & Associates, LLC.",
            "location": "São Paulo, Brazil",
        },
        "numeric_strings": {
            "title": "123456",
            "company": "999",
            "location": "12345",
        },
    }


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    base_data = {
        "id": "perf_job_{i}",
        "site": "linkedin",
        "title": "Software Engineer {i}",
        "company": "Company {i}",
        "location": "City {i}, State",
        "date_posted": "2024-01-15",
        "job_type": "fulltime",
        "is_remote": False,
    }

    # Generate 1000 jobs for performance testing
    return [
        {
            k: v.format(i=i) if isinstance(v, str) and "{i}" in v else v
            for k, v in base_data.items()
        }
        for i in range(1000)
    ]


# Async test helpers


@pytest.fixture
async def async_mock_jobspy_success():
    """Async wrapper for successful JobSpy mocking."""

    async def mock_async_scrape(**kwargs):
        # Simulate async delay
        import asyncio

        await asyncio.sleep(0.01)  # 10ms simulated delay
        return pd.DataFrame(
            [
                {
                    "id": "async_job_001",
                    "site": "linkedin",
                    "title": "Async Developer",
                    "company": "AsyncCorp",
                    "location": "Remote",
                    "is_remote": True,
                }
            ]
        )

    return mock_async_scrape


# Parametrized test data combinations


@pytest.fixture(
    params=[
        (JobSite.LINKEDIN, "fulltime", False),
        (JobSite.INDEED, "contract", True),
        (JobSite.GLASSDOOR, "parttime", False),
        ([JobSite.LINKEDIN, JobSite.INDEED], "fulltime", True),
    ]
)
def site_jobtype_remote_combinations(request):
    """Parametrized combinations of site, job type, and remote settings."""
    site, job_type, is_remote = request.param
    return {
        "site_name": site,
        "job_type": job_type,
        "is_remote": is_remote,
    }
