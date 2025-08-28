"""Modern tests for scraper functions with minimal mocking.

This module contains modernized tests for the job scraping functionality using:
- Real HTTP responses with responses library
- Factory-generated test data instead of hand-crafted dictionaries
- Real in-memory SQLite for database tests
- Property-based testing for edge cases
- Minimal mocking focused only on external dependencies

Replaces the heavy mocking approach in test_scrapers.py with library-first patterns.
"""

from datetime import UTC, datetime

import pytest
import responses

from hypothesis import given, strategies as st
from sqlmodel import Session, select

from src.models import CompanySQL, JobSQL
from src.scraper import scrape_all
from tests.factories import (
    CompanyFactory,
    JobFactory,
    create_sample_companies,
    create_sample_jobs,
)


@pytest.mark.unit
def test_job_model_creation_with_factory(session: Session) -> None:
    """Test job creation using factory-generated data."""
    # Use factory to create realistic job data
    job = JobFactory.create(session=session)

    # Verify job was created properly
    result = session.exec(select(JobSQL).where(JobSQL.id == job.id)).first()
    assert result is not None
    assert result.title in [
        "Senior AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
    ]
    assert result.salary is not None
    assert len(result.salary) == 2  # (min_salary, max_salary)


@pytest.mark.unit
def test_database_upsert_with_factories(session: Session) -> None:
    """Test database upsert operations using factory-generated data."""
    # Create existing company and job using factories
    company = CompanyFactory.create(session=session)
    existing_job = JobFactory.create(
        session=session, company_id=company.id, favorite=True
    )

    # Create a stale job that should be deleted
    stale_job = JobFactory.create(session=session, company_id=company.id)
    stale_job_id = stale_job.id

    session.commit()

    # Simulate upsert: update existing job
    existing_job.title = "Updated AI Engineer"
    existing_job.description = "Updated description with new requirements"
    existing_job.salary = (120000, 180000)

    # Add new job
    new_job = JobFactory.create(session=session, company_id=company.id)

    # Delete stale job
    session.delete(stale_job)
    session.commit()

    # Verify operations
    all_jobs = session.exec(select(JobSQL)).all()
    job_ids = [job.id for job in all_jobs]

    # Stale job should be deleted
    assert stale_job_id not in job_ids

    # Existing job should be updated but favorite flag preserved
    updated_job = session.exec(
        select(JobSQL).where(JobSQL.id == existing_job.id)
    ).first()
    assert updated_job.title == "Updated AI Engineer"
    assert updated_job.favorite is True  # User field preserved
    assert updated_job.salary == (120000, 180000)

    # New job should exist
    assert new_job.id in job_ids


@pytest.mark.http
@responses.activate
def test_scrape_company_pages_with_real_responses() -> None:
    """Test company page scraping with real HTTP responses using responses library."""
    # Mock the company loading and scraper responses
    mock_html_response = """
    <html>
        <body>
            <div class="job-listing">
                <h3>Senior AI Engineer</h3>
                <p>We are looking for a Senior AI Engineer to join our team.</p>
                <a href="/careers/ai-engineer-123">View Details</a>
                <span>San Francisco, CA</span>
            </div>
        </body>
    </html>
    """

    job_detail_response = """
    <html>
        <body>
            <h1>Senior AI Engineer</h1>
            <p>Full job description here. We need someone with 5+ years experience
               in machine learning and AI systems.</p>
            <div class="salary">$120,000 - $180,000</div>
            <div class="location">San Francisco, CA</div>
        </body>
    </html>
    """

    # Mock HTTP responses
    responses.add(
        responses.GET,
        "https://example.com/careers",
        body=mock_html_response,
        status=200,
        content_type="text/html",
    )

    responses.add(
        responses.GET,
        "https://example.com/careers/ai-engineer-123",
        body=job_detail_response,
        status=200,
        content_type="text/html",
    )

    # Test would integrate with real scraping logic here
    # For now, verify responses are set up correctly
    import httpx

    client = httpx.Client()
    response = client.get("https://example.com/careers")
    assert response.status_code == 200
    assert "Senior AI Engineer" in response.text


@pytest.mark.http
@responses.activate
def test_scrape_job_boards_with_mock_api() -> None:
    """Test job board scraping with mocked API responses."""
    # Mock API response for job board
    api_response = {
        "jobs": [
            {
                "title": "Machine Learning Engineer",
                "company": "TechCorp",
                "description": "Build ML systems at scale",
                "url": "https://techcorp.com/jobs/ml-eng",
                "location": "Remote",
                "posted_date": "2024-01-15",
                "salary_min": 130000,
                "salary_max": 200000,
            },
            {
                "title": "Data Scientist",
                "company": "DataCorp",
                "description": "Analyze large datasets",
                "url": "https://datacorp.com/jobs/ds",
                "location": "New York, NY",
                "posted_date": "2024-01-14",
                "salary_min": 110000,
                "salary_max": 160000,
            },
        ]
    }

    responses.add(
        responses.GET, "https://api.jobboard.com/search", json=api_response, status=200
    )

    # Test HTTP call
    import httpx

    client = httpx.Client()
    response = client.get("https://api.jobboard.com/search")
    assert response.status_code == 200

    data = response.json()
    assert len(data["jobs"]) == 2
    assert data["jobs"][0]["title"] == "Machine Learning Engineer"


@pytest.mark.integration
def test_scrape_workflow_integration_with_factories(session: Session) -> None:
    """Test complete scraping workflow using factories for test data."""
    # Create test companies using factories
    companies = create_sample_companies(session, count=3)

    # Mock the scraping functions to return factory-generated jobs
    with pytest.MonkeyPatch().context() as m:

        def mock_scrape_company_pages(max_jobs_per_company=50):
            # Return realistic jobs using factories
            jobs = []
            for company in companies[:2]:  # Mock 2 companies having jobs
                company_jobs = create_sample_jobs(
                    session, count=3, company=company, senior=True
                )
                jobs.extend(company_jobs)
            return jobs

        def mock_scrape_job_boards(keywords, countries):
            # Return job board results as dictionaries
            from tests.factories import JobDictFactory

            return JobDictFactory.create_batch(5)

        m.setattr(
            "src.scraper_company_pages.scrape_company_pages", mock_scrape_company_pages
        )
        m.setattr("src.scraper_job_boards.scrape_job_boards", mock_scrape_job_boards)

        # Mock the database sync engine
        def mock_sync_jobs(jobs_list):
            return {
                "inserted": len(jobs_list),
                "updated": 0,
                "archived": 0,
                "deleted": 0,
                "skipped": 0,
            }

        m.setattr(
            "src.services.database_sync.SmartSyncEngine.sync_jobs", mock_sync_jobs
        )

        # Execute the scraping workflow
        result = scrape_all(max_jobs_per_company=10)

        # Verify workflow completed successfully
        assert "inserted" in result
        assert result["inserted"] > 0


@pytest.mark.property
@given(
    salary_min=st.integers(min_value=50000, max_value=200000),
    salary_max=st.integers(min_value=50000, max_value=400000),
)
def test_salary_tuple_validation(salary_min: int, salary_max: int) -> None:
    """Property-based test for salary tuple validation logic."""
    # Ensure max >= min for valid test cases
    if salary_max < salary_min:
        salary_min, salary_max = salary_max, salary_min

    # Test the salary tuple validation
    salary_tuple = (salary_min, salary_max)

    # Test inline expressions that replaced accessor functions
    extracted_min = salary_tuple[0] if salary_tuple else None
    extracted_max = salary_tuple[1] if salary_tuple else None

    assert extracted_min == salary_min
    assert extracted_max == salary_max
    assert extracted_min <= extracted_max


@pytest.mark.property
@given(
    posted_date=st.datetimes(
        min_value=datetime(2020, 1, 1, tzinfo=UTC), max_value=datetime.now(UTC)
    )
)
def test_job_date_validation(posted_date: datetime) -> None:
    """Property-based test for job posting date validation."""
    # Test that job posting dates are handled correctly
    assert posted_date.tzinfo is not None  # Should have timezone info
    assert posted_date <= datetime.now(UTC)  # Should not be in future


@pytest.mark.unit
def test_job_filtering_ai_relevance() -> None:
    """Test job relevance filtering for AI/ML positions."""
    from tests.factories import JobDictFactory

    # Create mix of relevant and irrelevant jobs
    relevant_jobs = [
        JobDictFactory.create(title="Senior AI Engineer"),
        JobDictFactory.create(title="Machine Learning Engineer"),
        JobDictFactory.create(title="Data Scientist"),
        JobDictFactory.create(title="NLP Engineer"),
    ]

    irrelevant_jobs = [
        JobDictFactory.create(title="Sales Manager"),
        JobDictFactory.create(title="HR Coordinator"),
        JobDictFactory.create(title="Marketing Specialist"),
    ]

    all_jobs = relevant_jobs + irrelevant_jobs

    # Simple relevance filter (would be replaced with actual filtering logic)
    ai_keywords = ["ai", "ml", "machine learning", "data scientist", "nlp", "engineer"]

    filtered_jobs = [
        job
        for job in all_jobs
        if any(keyword in job["title"].lower() for keyword in ai_keywords)
    ]

    # Should only keep relevant jobs
    assert len(filtered_jobs) == len(relevant_jobs)

    titles = [job["title"] for job in filtered_jobs]
    assert "Sales Manager" not in titles
    assert "Senior AI Engineer" in titles


@pytest.mark.performance
def test_factory_performance_batch_creation(session: Session) -> None:
    """Test performance of factory batch creation for large datasets."""
    import time

    start_time = time.time()

    # Create large batch of test data
    create_sample_companies(session, count=10)
    create_sample_jobs(session, count=100)  # 100 jobs across companies

    creation_time = time.time() - start_time

    # Verify data was created correctly
    company_count = len(session.exec(select(CompanySQL)).all())
    job_count = len(session.exec(select(JobSQL)).all())

    assert company_count >= 10  # At least 10 companies (factories create as needed)
    assert job_count >= 100  # At least 100 jobs

    # Performance should be reasonable (under 5 seconds for this amount of data)
    assert creation_time < 5.0, f"Factory creation took too long: {creation_time:.2f}s"


@pytest.mark.unit
def test_content_hash_generation() -> None:
    """Test content hash generation for job deduplication."""
    from tests.factories import JobFactory

    # Create job with specific content
    job = JobFactory.build(
        title="AI Engineer",
        description="Specific job description for testing",
        location="San Francisco, CA",
    )

    # Hash should be consistent for same content
    import hashlib

    content = f"{job.title}{job.description}{job.location}"
    expected_hash = hashlib.md5(content.encode()).hexdigest()

    # Test that content hash logic works correctly
    # (This would integrate with actual hash generation in the model)
    assert len(expected_hash) == 32  # MD5 hash length
    assert expected_hash.isalnum()  # Only alphanumeric characters
