"""Comprehensive JobSpy test fixtures using polyfactory for type-safe generation.

This module provides pytest fixtures for mocking JobSpy responses with comprehensive
edge case coverage including malformed data, empty responses, and realistic job
datasets.

Key Features:
- Type-safe polyfactory integration with JobPosting models
- Parametrized fixtures for different response scenarios
- Extensive edge case coverage (Unicode, extreme values, duplicates)
- Performance-optimized fixture scoping
- Realistic and malformed DataFrame generation

Usage:
    @pytest.mark.parametrize(
        "jobspy_response", ["valid", "empty", "malformed"], indirect=True
    )
    def test_with_different_responses(jobspy_response):
        # jobspy_response is a pandas DataFrame
        pass
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest
from faker import Faker
from polyfactory import Use
from polyfactory.factories.pydantic_factory import ModelFactory
from src.models.job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
    LocationType,
)

fake = Faker()

# =============================================================================
# POLYFACTORY FACTORIES FOR TYPE-SAFE GENERATION
# =============================================================================


class JobPostingFactory(ModelFactory[JobPosting]):
    """Factory for generating type-safe JobPosting instances."""

    __model__ = JobPosting
    __check_model__ = False

    # Core fields with realistic data
    id = Use(lambda: f"job_{fake.uuid4()}")
    site = Use(lambda: fake.random_element(list(JobSite)))
    job_url = Use(lambda: fake.url())
    job_url_direct = Use(lambda: fake.url())
    title = Use(lambda: fake.job())
    company = Use(lambda: fake.company())
    location = Use(lambda: fake.city())
    date_posted = Use(lambda: fake.date_between(start_date="-30d", end_date="today"))

    # Job type and location
    job_type = Use(lambda: fake.random_element(list(JobType)))
    is_remote = Use(lambda: fake.boolean())
    location_type = Use(lambda: fake.random_element(list(LocationType)))

    # Salary information
    salary_source = Use(
        lambda: fake.random_element(["Employer", "Glassdoor", "Indeed", "PayScale"])
    )
    interval = Use(lambda: fake.random_element(["yearly", "monthly", "hourly"]))
    min_amount = Use(lambda: fake.random_int(min=50000, max=120000))
    max_amount = Use(lambda: fake.random_int(min=120000, max=200000))
    currency = Use(lambda: fake.random_element(["USD", "EUR", "GBP", "CAD"]))

    # Job details
    job_level = Use(
        lambda: fake.random_element(["Entry", "Mid", "Senior", "Executive"])
    )
    job_function = Use(
        lambda: fake.random_element(
            ["Engineering", "Marketing", "Sales", "HR", "Finance"]
        )
    )
    listing_type = Use(
        lambda: fake.random_element(["External", "Internal", "Easy Apply"])
    )
    description = Use(lambda: fake.text(max_nb_chars=500))
    emails = Use(lambda: [fake.email() for _ in range(fake.random_int(min=0, max=3))])
    skills = Use(lambda: [fake.word() for _ in range(fake.random_int(min=3, max=10))])
    experience_range = Use(
        lambda: (
            f"{fake.random_int(min=1, max=3)}-{fake.random_int(min=4, max=10)} years"
        )
    )
    vacancy_count = Use(lambda: fake.random_int(min=1, max=5))

    # Company information
    company_industry = Use(
        lambda: fake.random_element(
            ["Technology", "Healthcare", "Finance", "Education", "Retail"]
        )
    )
    company_url = Use(lambda: fake.url())
    company_logo = Use(lambda: fake.image_url())
    company_url_direct = Use(lambda: fake.url())
    company_addresses = Use(
        lambda: [fake.address() for _ in range(fake.random_int(min=1, max=3))]
    )
    company_num_employees = Use(
        lambda: fake.random_element(["1-10", "11-50", "51-200", "201-1000", "1000+"])
    )
    company_revenue = Use(
        lambda: fake.random_element(
            ["<$1M", "$1M-$5M", "$5M-$50M", "$50M-$500M", "$500M+"]
        )
    )
    company_description = Use(lambda: fake.text(max_nb_chars=300))
    company_rating = Use(lambda: round(fake.random.uniform(1.0, 5.0), 1))
    company_reviews_count = Use(lambda: fake.random_int(min=10, max=5000))

    @classmethod
    def to_dataframe(cls, count: int = 10) -> pd.DataFrame:
        """Generate a pandas DataFrame with specified number of jobs.

        Args:
            count: Number of job records to generate

        Returns:
            pandas DataFrame with JobPosting-compatible structure
        """
        jobs = [cls.build() for _ in range(count)]
        return pd.DataFrame([job.model_dump() for job in jobs])

    @classmethod
    def build_malformed(cls) -> dict[str, Any]:
        """Build a malformed job record for testing error handling."""
        return {
            "id": None,  # Missing required field
            "title": "",  # Empty string
            "company": 12345,  # Wrong type
            "location": {"invalid": "object"},  # Wrong type
            "min_amount": "not_a_number",  # Invalid float
            "max_amount": Decimal("150000.50"),  # Non-standard number type
            "date_posted": "invalid_date",  # Invalid date
            "emails": "not_a_list",  # Wrong type for list field
            "skills": None,  # Null list field
            "company_rating": "five_stars",  # Invalid rating
            "is_remote": "maybe",  # Invalid boolean
            "site": "unknown_site",  # Invalid enum
            "job_type": "",  # Empty enum
        }

    @classmethod
    def build_edge_case(cls) -> dict[str, Any]:
        """Build an edge case job record with extreme/unusual values."""
        return {
            "id": "job_" + "🎯" * 10,  # Unicode in ID
            "title": "Senior 🚀 Full-Stack Engineer 💻 (Remote) - €€€",  # Unicode/emoji
            "company": "ÄÄÄÄ Corporation 中文公司",  # International characters
            "location": "New York, NY; San Francisco, CA; Remote",  # Multiple locations
            "min_amount": 1000000.0,  # Extremely high salary
            "max_amount": 999999.0,  # Invalid range (max < min)
            "date_posted": date(2030, 1, 1),  # Future date
            "emails": ["test@" + "x" * 100 + ".com"],  # Very long email
            "skills": ["Python"] * 50,  # Duplicate skills
            "company_rating": 5.5,  # Invalid rating (>5.0)
            "vacancy_count": -1,  # Negative count
            "experience_range": "0-0 years",  # Zero experience
            "company_num_employees": "Unknown",  # Non-standard format
            "site": JobSite.LINKEDIN,
            "job_type": JobType.FULLTIME,
        }


class JobScrapeRequestFactory(ModelFactory[JobScrapeRequest]):
    """Factory for generating type-safe JobScrapeRequest instances."""

    __model__ = JobScrapeRequest
    __check_model__ = False

    site_name = Use(lambda: fake.random_element(list(JobSite)))
    search_term = Use(
        lambda: fake.random_element(
            [
                "Python Developer",
                "Data Scientist",
                "Machine Learning Engineer",
                "Software Engineer",
                "DevOps Engineer",
                "Product Manager",
            ]
        )
    )
    google_search_term = Use(lambda: fake.random_element([None, fake.job()]))
    location = Use(lambda: fake.city())
    distance = Use(lambda: fake.random_int(min=10, max=100))
    is_remote = Use(lambda: fake.boolean())
    job_type = Use(lambda: fake.random_element([None, *list(JobType)]))
    easy_apply = Use(lambda: fake.random_element([None, True, False]))
    results_wanted = Use(lambda: fake.random_int(min=10, max=100))
    country_indeed = Use(
        lambda: fake.random_element(["usa", "canada", "uk", "germany"])
    )
    offset = Use(lambda: fake.random_int(min=0, max=50))
    hours_old = Use(
        lambda: fake.random_element([None, 24, 72, 168])
    )  # 1 day, 3 days, 1 week
    enforce_annual_salary = Use(lambda: fake.boolean())
    linkedin_fetch_description = Use(lambda: fake.boolean())
    description_format = Use(lambda: fake.random_element(["markdown", "html", "text"]))


# =============================================================================
# BASIC FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def empty_jobspy_response() -> pd.DataFrame:
    """Empty JobSpy DataFrame response."""
    return pd.DataFrame()


@pytest.fixture(scope="session")
def minimal_jobspy_response() -> pd.DataFrame:
    """Minimal valid JobSpy response with required fields only."""
    return pd.DataFrame(
        [
            {
                "id": "minimal_job_1",
                "site": "linkedin",
                "title": "Test Job",
                "company": "Test Company",
            }
        ]
    )


@pytest.fixture(scope="session")
def valid_jobspy_response() -> pd.DataFrame:
    """Standard valid JobSpy response with realistic data."""
    return JobPostingFactory.to_dataframe(count=5)


@pytest.fixture(scope="session")
def large_jobspy_response() -> pd.DataFrame:
    """Large JobSpy response for performance testing."""
    return JobPostingFactory.to_dataframe(count=1000)


@pytest.fixture(scope="session")
def malformed_jobspy_response() -> pd.DataFrame:
    """JobSpy response with various malformed data scenarios."""
    malformed_jobs = [
        JobPostingFactory.build_malformed(),
        JobPostingFactory.build_malformed(),
        {
            "title": None,  # Missing title
            "company": "",  # Empty company
            "location": "   ",  # Whitespace only
        },
        {
            "id": "",  # Empty ID
            "salary": "competitive",  # Non-numeric salary
            "missing_required_fields": True,
        },
    ]
    return pd.DataFrame(malformed_jobs)


@pytest.fixture(scope="session")
def edge_case_jobspy_response() -> pd.DataFrame:
    """JobSpy response with edge cases and extreme values."""
    edge_cases = [
        JobPostingFactory.build_edge_case(),
        {
            "id": "duplicate_id",  # Will be duplicated
            "title": "Job 1",
            "company": "Company A",
        },
        {
            "id": "duplicate_id",  # Duplicate ID
            "title": "Job 2",
            "company": "Company B",
        },
        {
            "id": "unicode_test_🔥",
            "title": "Software Engineer 👨‍💻",
            "company": "Tech Corp 🏢",
            "location": "San Francisco 🌉",
            "description": "Join our team! 🎉 We're looking for a passionate "
            "developer 💪",
        },
    ]
    return pd.DataFrame(edge_cases)


@pytest.fixture(scope="session")
def mixed_types_jobspy_response() -> pd.DataFrame:
    """JobSpy response with mixed and inconsistent data types."""
    mixed_data = [
        {
            "id": 12345,  # Integer instead of string
            "title": ["Job Title as List"],  # List instead of string
            "company": True,  # Boolean instead of string
            "min_amount": "50k",  # String instead of float
            "max_amount": None,  # None value
            "date_posted": 1640995200,  # Unix timestamp
            "skills": "Python,Java,SQL",  # Comma-separated string instead of list
            "is_remote": 1,  # Integer instead of boolean
            "company_rating": "4.5 stars",  # String with units
        },
        {
            "id": None,  # None ID
            "title": "",  # Empty string
            "company": 0,  # Zero as company
            "location": {},  # Empty dict
            "emails": "single@email.com",  # String instead of list
        },
    ]
    return pd.DataFrame(mixed_data)


# =============================================================================
# PARAMETRIZED FIXTURES
# =============================================================================


@pytest.fixture(
    params=["empty", "minimal", "valid", "malformed", "edge_case", "mixed_types"]
)
def jobspy_response(request) -> pd.DataFrame:
    """Parametrized fixture returning different JobSpy response types.

    Available params:
        - empty: Empty DataFrame
        - minimal: Minimal valid data
        - valid: Standard realistic data
        - malformed: Various malformed data scenarios
        - edge_case: Extreme values and edge cases
        - mixed_types: Mixed and inconsistent data types
    """
    if request.param == "empty":
        return pd.DataFrame()
    if request.param == "minimal":
        return pd.DataFrame(
            [
                {
                    "id": "minimal_job_1",
                    "site": "linkedin",
                    "title": "Test Job",
                    "company": "Test Company",
                }
            ]
        )
    if request.param == "valid":
        return JobPostingFactory.to_dataframe(count=5)
    if request.param == "malformed":
        malformed_jobs = [
            JobPostingFactory.build_malformed(),
            JobPostingFactory.build_malformed(),
            {
                "title": None,  # Missing title
                "company": "",  # Empty company
                "location": "   ",  # Whitespace only
            },
        ]
        return pd.DataFrame(malformed_jobs)
    if request.param == "edge_case":
        edge_cases = [
            JobPostingFactory.build_edge_case(),
            {
                "id": "duplicate_id",  # Will be duplicated
                "title": "Job 1",
                "company": "Company A",
            },
            {
                "id": "duplicate_id",  # Duplicate ID
                "title": "Job 2",
                "company": "Company B",
            },
        ]
        return pd.DataFrame(edge_cases)
    if request.param == "mixed_types":
        mixed_data = [
            {
                "id": 12345,  # Integer instead of string
                "title": ["Job Title as List"],  # List instead of string
                "company": True,  # Boolean instead of string
                "min_amount": "50k",  # String instead of float
            },
            {
                "id": None,  # None ID
                "title": "",  # Empty string
                "company": 0,  # Zero as company
            },
        ]
        return pd.DataFrame(mixed_data)
    return pd.DataFrame()


@pytest.fixture(params=[5, 50, 100, 1000])
def jobspy_response_sized(request) -> pd.DataFrame:
    """Parametrized fixture for testing different dataset sizes."""
    return JobPostingFactory.to_dataframe(count=request.param)


@pytest.fixture(params=list(JobSite))
def jobspy_response_by_site(request) -> pd.DataFrame:
    """Parametrized fixture for testing different job sites."""
    factory = JobPostingFactory.build()
    factory.site = request.param
    return pd.DataFrame([factory.model_dump()])


# =============================================================================
# JOB SCRAPE REQUEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_scrape_request() -> JobScrapeRequest:
    """Standard JobScrapeRequest for testing."""
    return JobScrapeRequestFactory.build()


@pytest.fixture
def linkedin_scrape_request() -> JobScrapeRequest:
    """LinkedIn-specific scrape request."""
    return JobScrapeRequestFactory.build(
        site_name=JobSite.LINKEDIN,
        linkedin_fetch_description=True,
        search_term="Python Developer",
        location="San Francisco, CA",
    )


@pytest.fixture
def remote_scrape_request() -> JobScrapeRequest:
    """Remote job scrape request."""
    return JobScrapeRequestFactory.build(
        is_remote=True,
        location=None,
        search_term="Remote Software Engineer",
    )


@pytest.fixture
def multi_site_scrape_request() -> JobScrapeRequest:
    """Multi-site scrape request."""
    return JobScrapeRequestFactory.build(
        site_name=[JobSite.LINKEDIN, JobSite.INDEED, JobSite.GLASSDOOR],
        search_term="Data Scientist",
        results_wanted=50,
    )


# =============================================================================
# JOB SCRAPE RESULT FIXTURES
# =============================================================================


@pytest.fixture
def successful_scrape_result(
    valid_jobspy_response, sample_scrape_request
) -> JobScrapeResult:
    """Successful JobScrapeResult with valid data."""
    return JobScrapeResult.from_pandas(
        df=valid_jobspy_response,
        request=sample_scrape_request,
        metadata={"scraping_method": "jobspy", "success": True, "duration": 2.5},
    )


@pytest.fixture
def empty_scrape_result(sample_scrape_request) -> JobScrapeResult:
    """Empty JobScrapeResult (no jobs found)."""
    return JobScrapeResult(
        jobs=[],
        total_found=0,
        request_params=sample_scrape_request,
        metadata={
            "scraping_method": "jobspy",
            "success": True,
            "message": "No jobs found",
        },
    )


@pytest.fixture
def failed_scrape_result(sample_scrape_request) -> JobScrapeResult:
    """Failed JobScrapeResult with error metadata."""
    return JobScrapeResult(
        jobs=[],
        total_found=0,
        request_params=sample_scrape_request,
        metadata={
            "scraping_method": "jobspy",
            "success": False,
            "error": "Network timeout",
            "error_type": "ConnectionError",
        },
    )


# =============================================================================
# SPECIALIZED FIXTURES FOR EDGE CASES
# =============================================================================


@pytest.fixture
def duplicate_jobs_response() -> pd.DataFrame:
    """JobSpy response with duplicate job IDs for testing deduplication."""
    job_data = JobPostingFactory.build().model_dump()
    # Create duplicates with same ID but different data
    duplicates = [
        {**job_data, "id": "duplicate_test", "title": "Job Title 1"},
        {**job_data, "id": "duplicate_test", "title": "Job Title 2"},
        {**job_data, "id": "duplicate_test", "title": "Job Title 3"},
    ]
    return pd.DataFrame(duplicates)


@pytest.fixture
def salary_edge_cases_response() -> pd.DataFrame:
    """JobSpy response with various salary edge cases."""
    base_job = JobPostingFactory.build().model_dump()
    salary_cases = [
        {**base_job, "id": "salary_1", "min_amount": 0, "max_amount": 0},  # Zero salary
        {
            **base_job,
            "id": "salary_2",
            "min_amount": 200000,
            "max_amount": 150000,
        },  # Invalid range
        {
            **base_job,
            "id": "salary_3",
            "min_amount": 1000000,
            "max_amount": 2000000,
        },  # Extreme values
        {
            **base_job,
            "id": "salary_4",
            "min_amount": None,
            "max_amount": None,
        },  # No salary info
        {
            **base_job,
            "id": "salary_5",
            "min_amount": 50.5,
            "max_amount": 75.75,
        },  # Hourly wages
    ]
    return pd.DataFrame(salary_cases)


@pytest.fixture
def date_edge_cases_response() -> pd.DataFrame:
    """JobSpy response with various date edge cases."""
    base_job = JobPostingFactory.build().model_dump()
    date_cases = [
        {**base_job, "id": "date_1", "date_posted": date(2030, 12, 31)},  # Future date
        {**base_job, "id": "date_2", "date_posted": date(1900, 1, 1)},  # Very old date
        {**base_job, "id": "date_3", "date_posted": None},  # No date
        {
            **base_job,
            "id": "date_4",
            "date_posted": datetime.now(UTC),
        },  # Datetime instead of date
    ]
    return pd.DataFrame(date_cases)


@pytest.fixture
def unicode_jobs_response() -> pd.DataFrame:
    """JobSpy response with Unicode and emoji characters."""
    unicode_jobs = [
        {
            "id": "unicode_1",
            "title": "Développeur Python 🐍",
            "company": "Société Française 🇫🇷",
            "location": "Paris, France 🗼",
            "description": "Nous recherchons un développeur passionné! 🚀✨",
        },
        {
            "id": "unicode_2",
            "title": "软件工程师 💻",
            "company": "中国科技公司 🏢",
            "location": "北京, 中国 🇨🇳",
            "description": "加入我们的团队! 🎯💪",
        },
        {
            "id": "unicode_3",
            "title": "Инженер-программист 🔧",
            "company": "Российская компания 🏭",
            "location": "Москва, Россия 🇷🇺",
            "description": "Присоединяйтесь к нашей команде! 🌟⭐",
        },
    ]
    return pd.DataFrame(unicode_jobs)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def null_jobspy_response() -> None:
    """None response from JobSpy (simulates complete failure)."""
    return


@pytest.fixture
def jobspy_timeout_simulation() -> Exception:
    """Exception fixture for simulating JobSpy timeout."""
    return TimeoutError("JobSpy request timed out after 30 seconds")


@pytest.fixture
def jobspy_connection_error() -> Exception:
    """Exception fixture for simulating JobSpy connection error."""
    return ConnectionError("Failed to connect to job site")


@pytest.fixture
def random_job_posting() -> JobPosting:
    """Generate a single random JobPosting instance."""
    return JobPostingFactory.build()


@pytest.fixture
def job_posting_batch() -> list[JobPosting]:
    """Generate a batch of JobPosting instances."""
    return JobPostingFactory.batch(size=10)


# =============================================================================
# FIXTURE EXAMPLE USAGE
# =============================================================================


def pytest_configure() -> None:
    """Configure pytest with fixture documentation."""
    pytest.main.__doc__ = """
    JobSpy Fixtures Usage Examples:

    # Basic usage with different response types
    @pytest.mark.parametrize(
        "jobspy_response", ["valid", "empty", "malformed"], indirect=True
    )
    def test_scraper_handles_responses(jobspy_response):
        result = scraper.process(jobspy_response)
        assert isinstance(result, JobScrapeResult)

    # Size-based testing
    @pytest.mark.parametrize("jobspy_response_sized", [10, 100], indirect=True)
    def test_performance(jobspy_response_sized):
        start_time = time.time()
        process_jobs(jobspy_response_sized)
        assert time.time() - start_time < 1.0

    # Site-specific testing
    def test_linkedin_specific(linkedin_scrape_request):
        assert linkedin_scrape_request.site_name == JobSite.LINKEDIN
        assert linkedin_scrape_request.linkedin_fetch_description is True

    # Edge case testing
    def test_unicode_handling(unicode_jobs_response):
        for _, job in unicode_jobs_response.iterrows():
            assert isinstance(job['title'], str)
            # Verify Unicode characters are preserved
    """
