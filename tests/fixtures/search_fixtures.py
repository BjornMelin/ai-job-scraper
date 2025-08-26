"""Test fixtures and utilities for search functionality tests.

This module provides common test fixtures, sample data, and utility functions
used across search service and UI component tests.

Fixtures:
- Sample job data for testing
- Search result mock data
- Performance test configurations
- Filter test scenarios
"""

import json
import tempfile

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.services.search_service import JobSearchService


@pytest.fixture
def sample_search_jobs():
    """Sample job data for search testing."""
    return [
        {
            "id": 1,
            "title": "Senior Python Developer",
            "description": (
                "We are looking for a Senior Python developer to join our engineering "
                "team. Experience with Django, Flask, and REST APIs required."
            ),
            "company": "TechCorp",
            "location": "San Francisco, CA",
            "posted_date": "2024-01-15T10:00:00Z",
            "salary": [100000, 150000],
            "application_status": "New",
            "favorite": 0,
            "archived": 0,
            "company_id": 1,
        },
        {
            "id": 2,
            "title": "Machine Learning Engineer",
            "description": (
                "Build and deploy ML models for production. Experience with PyTorch, "
                "TensorFlow, and MLOps required."
            ),
            "company": "AI Startup",
            "location": "Remote",
            "posted_date": "2024-01-20T14:30:00Z",
            "salary": [120000, 180000],
            "application_status": "Applied",
            "favorite": 1,
            "archived": 0,
            "company_id": 2,
        },
        {
            "id": 3,
            "title": "Data Scientist",
            "description": (
                "Analyze large datasets and develop predictive models. Strong "
                "statistics and Python skills required."
            ),
            "company": "DataCorp",
            "location": "New York, NY",
            "posted_date": "2024-01-10T09:15:00Z",
            "salary": [90000, 140000],
            "application_status": "Interested",
            "favorite": 1,
            "archived": 0,
            "company_id": 3,
        },
        {
            "id": 4,
            "title": "Full Stack Developer",
            "description": (
                "Work with Python backend (Django/Flask) and React frontend. "
                "Full-stack experience required."
            ),
            "company": "WebCompany",
            "location": "Austin, TX",
            "posted_date": "2024-01-25T16:45:00Z",
            "salary": [85000, 130000],
            "application_status": "New",
            "favorite": 0,
            "archived": 0,
            "company_id": 4,
        },
        {
            "id": 5,
            "title": "DevOps Engineer",
            "description": (
                "Manage cloud infrastructure and CI/CD pipelines. Experience with "
                "AWS, Docker, and Kubernetes."
            ),
            "company": "CloudCorp",
            "location": "Seattle, WA",
            "posted_date": "2024-01-18T11:20:00Z",
            "salary": [110000, 160000],
            "application_status": "Rejected",
            "favorite": 0,
            "archived": 1,
            "company_id": 5,
        },
        {
            "id": 6,
            "title": "Research Scientist - ML",
            "description": (
                "Conduct research in machine learning and artificial intelligence. "
                "PhD in related field preferred."
            ),
            "company": "Research Lab",
            "location": "Boston, MA",
            "posted_date": "2024-01-12T13:00:00Z",
            "salary": [130000, 200000],
            "application_status": "New",
            "favorite": 1,
            "archived": 0,
            "company_id": 6,
        },
        {
            "id": 7,
            "title": "Backend Python Engineer",
            "description": (
                "Develop high-performance Python APIs and microservices. Experience "
                "with FastAPI and PostgreSQL."
            ),
            "company": "StartupXYZ",
            "location": "Los Angeles, CA",
            "posted_date": "2024-01-22T08:30:00Z",
            "salary": [95000, 145000],
            "application_status": "Applied",
            "favorite": 0,
            "archived": 0,
            "company_id": 7,
        },
        {
            "id": 8,
            "title": "AI Engineer",
            "description": (
                "Build AI-powered applications and integrate LLMs. Experience with "
                "OpenAI API and Langchain preferred."
            ),
            "company": "InnovateAI",
            "location": "Remote",
            "posted_date": "2024-01-28T10:15:00Z",
            "salary": [140000, 200000],
            "application_status": "New",
            "favorite": 1,
            "archived": 0,
            "company_id": 8,
        },
    ]


@pytest.fixture
def sample_companies():
    """Sample company data for search testing."""
    return [
        {"id": 1, "name": "TechCorp"},
        {"id": 2, "name": "AI Startup"},
        {"id": 3, "name": "DataCorp"},
        {"id": 4, "name": "WebCompany"},
        {"id": 5, "name": "CloudCorp"},
        {"id": 6, "name": "Research Lab"},
        {"id": 7, "name": "StartupXYZ"},
        {"id": 8, "name": "InnovateAI"},
    ]


@pytest.fixture
def search_test_database(sample_search_jobs, sample_companies):
    """Create a temporary SQLite database with test data for search testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        service = JobSearchService(tmp.name)

        # Create database schema
        service.db.executescript("""
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY,
                title TEXT,
                description TEXT,
                company TEXT,
                location TEXT,
                posted_date TEXT,
                salary TEXT,
                application_status TEXT DEFAULT 'New',
                favorite INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                company_id INTEGER
            );

            CREATE TABLE companysql (
                id INTEGER PRIMARY KEY,
                name TEXT
            );
        """)

        # Insert companies
        for company in sample_companies:
            service.db.execute(
                "INSERT INTO companysql (id, name) VALUES (?, ?)",
                [company["id"], company["name"]],
            )

        # Insert jobs
        for job in sample_search_jobs:
            service.db.execute(
                """
                INSERT INTO jobs (
                    id, title, description, company, location, posted_date,
                    salary, application_status, favorite, archived, company_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    job["id"],
                    job["title"],
                    job["description"],
                    job["company"],
                    job["location"],
                    job["posted_date"],
                    json.dumps(job["salary"]),
                    job["application_status"],
                    job["favorite"],
                    job["archived"],
                    job["company_id"],
                ],
            )

        # Setup FTS5
        service._setup_search_index()

        yield service

    # Cleanup
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def search_filters_test_cases():
    """Test cases for search filter validation."""
    return [
        # Basic filters
        {
            "name": "location_filter",
            "filters": {"location": "San Francisco"},
            "expected_matches": 1,  # Only TechCorp job
        },
        {
            "name": "remote_filter",
            "filters": {"location": "Remote"},
            "expected_matches": 2,  # AI Startup and InnovateAI
        },
        {
            "name": "salary_min_filter",
            "filters": {"salary_min": 120000},
            "expected_matches": 4,  # Jobs with max salary >= 120k
        },
        {
            "name": "salary_max_filter",
            "filters": {"salary_max": 150000},
            "expected_matches": 6,  # Jobs with min salary <= 150k
        },
        {
            "name": "favorites_only_filter",
            "filters": {"favorites_only": True},
            "expected_matches": 4,  # Jobs marked as favorite
        },
        {
            "name": "application_status_filter",
            "filters": {"application_status": ["Applied"]},
            "expected_matches": 2,  # ML Engineer and Backend Engineer
        },
        {
            "name": "exclude_archived_filter",
            "filters": {"include_archived": False},
            "expected_matches": 7,  # All except DevOps Engineer
        },
        # Combined filters
        {
            "name": "combined_filters",
            "filters": {
                "salary_min": 100000,
                "favorites_only": True,
                "application_status": ["New", "Applied", "Interested"],
            },
            "expected_matches": 3,  # ML Engineer, Data Scientist, AI Engineer
        },
        # Date filters
        {
            "name": "date_from_filter",
            "filters": {"date_from": "2024-01-20"},
            "expected_matches": 4,  # Jobs posted on or after Jan 20
        },
        {
            "name": "date_to_filter",
            "filters": {"date_to": "2024-01-15"},
            "expected_matches": 3,  # Jobs posted on or before Jan 15
        },
    ]


@pytest.fixture
def search_performance_test_data():
    """Generate large dataset for performance testing."""
    jobs = []
    companies = []

    # Create companies
    for i in range(50):
        companies.append({"id": i + 1, "name": f"Company_{i:03d}"})

    # Create jobs with varied content
    job_titles = [
        "Python Developer",
        "Java Engineer",
        "Data Scientist",
        "ML Engineer",
        "DevOps Engineer",
        "Full Stack Developer",
        "Backend Developer",
        "Frontend Developer",
        "AI Researcher",
        "Software Architect",
    ]

    locations = [
        "San Francisco, CA",
        "New York, NY",
        "Seattle, WA",
        "Austin, TX",
        "Boston, MA",
        "Los Angeles, CA",
        "Denver, CO",
        "Remote",
        "Chicago, IL",
    ]

    descriptions = [
        "Develop and maintain high-quality software applications",
        "Work with cutting-edge technologies and frameworks",
        "Collaborate with cross-functional teams to deliver solutions",
        "Build scalable and performant systems",
        "Research and implement new technologies",
    ]

    for i in range(1000):
        company_id = (i % 50) + 1
        job = {
            "id": i + 1,
            "title": f"{job_titles[i % len(job_titles)]} {i}",
            "description": f"{descriptions[i % len(descriptions)]} for job {i}",
            "company": f"Company_{(i % 50):03d}",
            "location": locations[i % len(locations)],
            "posted_date": (datetime.now(UTC) - timedelta(days=i % 30)).isoformat(),
            "salary": [60000 + (i % 50) * 2000, 100000 + (i % 50) * 3000],
            "application_status": ["New", "Applied", "Interested", "Rejected"][i % 4],
            "favorite": i % 5 == 0,  # Every 5th job is a favorite
            "archived": i % 20 == 0,  # Every 20th job is archived
            "company_id": company_id,
        }
        jobs.append(job)

    return {"jobs": jobs, "companies": companies}


@pytest.fixture
def performance_test_database(search_performance_test_data):
    """Create database with large dataset for performance testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        service = JobSearchService(tmp.name)

        # Create schema
        service.db.executescript("""
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY,
                title TEXT,
                description TEXT,
                company TEXT,
                location TEXT,
                posted_date TEXT,
                salary TEXT,
                application_status TEXT DEFAULT 'New',
                favorite INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                company_id INTEGER
            );

            CREATE TABLE companysql (
                id INTEGER PRIMARY KEY,
                name TEXT
            );
        """)

        # Insert data in batches for better performance
        companies = search_performance_test_data["companies"]
        for company in companies:
            service.db.execute(
                "INSERT INTO companysql (id, name) VALUES (?, ?)",
                [company["id"], company["name"]],
            )

        jobs = search_performance_test_data["jobs"]
        batch_size = 100
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i : i + batch_size]
            service.db.executemany(
                """
                INSERT INTO jobs (
                    id, title, description, company, location, posted_date,
                    salary, application_status, favorite, archived, company_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    [
                        job["id"],
                        job["title"],
                        job["description"],
                        job["company"],
                        job["location"],
                        job["posted_date"],
                        json.dumps(job["salary"]),
                        job["application_status"],
                        job["favorite"],
                        job["archived"],
                        job["company_id"],
                    ]
                    for job in batch
                ],
            )

        # Setup FTS5
        service._setup_search_index()

        yield service

    # Cleanup
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def mock_search_results():
    """Mock search results for UI testing."""
    return [
        {
            "id": 1,
            "title": "Senior Python Developer",
            "description": "Great Python role with competitive benefits",
            "company": "TechCorp",
            "location": "San Francisco, CA",
            "salary": "[100000, 150000]",
            "posted_date": "2024-01-15T10:00:00Z",
            "application_status": "New",
            "favorite": 0,
            "rank": 0.95,
        },
        {
            "id": 2,
            "title": "Machine Learning Engineer",
            "description": "Build ML models for production deployment",
            "company": "AI Startup",
            "location": "Remote",
            "salary": "[120000, 180000]",
            "posted_date": "2024-01-20T14:30:00Z",
            "application_status": "Applied",
            "favorite": 1,
            "rank": 0.87,
        },
    ]


@pytest.fixture
def search_error_test_cases():
    """Test cases for search error scenarios."""
    return [
        {
            "name": "database_connection_error",
            "error_type": "ConnectionError",
            "error_message": "Unable to connect to database",
            "expected_result": [],
        },
        {
            "name": "search_timeout",
            "error_type": "TimeoutError",
            "error_message": "Search operation timed out",
            "expected_result": [],
        },
        {
            "name": "invalid_query_syntax",
            "error_type": "ValueError",
            "error_message": "Invalid FTS5 query syntax",
            "expected_result": [],
        },
        {
            "name": "fts_index_corruption",
            "error_type": "sqlite3.OperationalError",
            "error_message": "FTS5 index is corrupted",
            "expected_result": [],
        },
    ]


class SearchTestUtils:
    """Utility functions for search testing."""

    @staticmethod
    def create_minimal_test_db():
        """Create minimal test database with single job."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            service.db.executescript("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    company TEXT,
                    location TEXT
                );
            """)

            service.db.execute("""
                INSERT INTO jobs (title, description, company, location)
                VALUES ('Test Job', 'Test Description', 'Test Company', 'Test Location')
            """)

            service._setup_search_index()

            return service, tmp.name

    @staticmethod
    def assert_search_results_valid(results: list[dict[str, Any]]):
        """Validate search results format and content."""
        assert isinstance(results, list)

        for result in results:
            assert isinstance(result, dict)

            # Check required fields
            required_fields = ["id", "title", "description", "company", "location"]
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"

            # Check data types
            assert isinstance(result["id"], int)
            assert isinstance(result["title"], str)
            assert isinstance(result["description"], str)

            # Check rank field if present (FTS5 search)
            if "rank" in result:
                assert isinstance(result["rank"], (int, float))

    @staticmethod
    def assert_performance_under_threshold(
        duration_ms: float, threshold_ms: float = 100.0
    ):
        """Assert that operation completed within performance threshold."""
        assert duration_ms < threshold_ms, (
            f"Operation took {duration_ms:.2f}ms, should be under {threshold_ms}ms"
        )

    @staticmethod
    def generate_search_query_variations():
        """Generate various search query patterns for testing."""
        return [
            # Simple queries
            "python",
            "developer",
            "machine learning",
            # Phrase queries
            '"python developer"',
            '"machine learning engineer"',
            '"full stack developer"',
            # Boolean queries
            "python AND developer",
            "machine OR learning",
            "python NOT junior",
            # Wildcard queries
            "develop*",
            "engineer*",
            "data*",
            # Complex queries
            '"senior python" AND (django OR flask)',
            'machine learning NOT "junior level"',
            "remote AND (python* OR java*)",
        ]

    @staticmethod
    def create_filter_combinations():
        """Generate various filter combination scenarios."""
        base_date = datetime(2024, 1, 15, tzinfo=UTC)
        return [
            # Single filters
            {"location": "Remote"},
            {"salary_min": 100000},
            {"favorites_only": True},
            {"application_status": ["Applied"]},
            # Multiple filters
            {"salary_min": 80000, "salary_max": 150000},
            {"location": "San Francisco", "favorites_only": True},
            {"application_status": ["New", "Applied"], "salary_min": 100000},
            # Date filters
            {"date_from": base_date - timedelta(days=7)},
            {"date_to": base_date},
            {"date_from": base_date - timedelta(days=14), "date_to": base_date},
            # Complex combinations
            {
                "location": "Remote",
                "salary_min": 120000,
                "favorites_only": True,
                "application_status": ["New", "Applied"],
                "date_from": base_date - timedelta(days=30),
            },
        ]


@pytest.fixture
def search_test_utils():
    """Provide search test utilities."""
    return SearchTestUtils
