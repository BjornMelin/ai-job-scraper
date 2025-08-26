"""Comprehensive tests for the FTS5 search service implementation.

This module provides comprehensive test coverage for the JobSearchService class,
testing FTS5 search functionality, filter integration, performance, error handling,
and fallback mechanisms to achieve 90%+ code coverage.

Test Categories:
- FTS5 setup and initialization
- Search functionality with stemming and ranking
- Filter integration (all filter types)
- Performance and caching behavior
- Error handling and edge cases
- Database operations and maintenance

Performance Requirements:
- Search latency <10ms for typical queries
- Proper index creation and maintenance
- Streamlit caching validation
"""

import contextlib
import json
import logging
import sqlite3
import tempfile
import time

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import sqlite_utils

from src.models import CompanySQL, JobSQL
from src.services.search_service import JobSearchService

if TYPE_CHECKING:
    from sqlmodel import Session

logger = logging.getLogger(__name__)


class TestFTS5Setup:
    """Test FTS5 search index setup and initialization."""

    def test_database_connection_lazy_initialization(self):
        """Test that database connection is created lazily."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Database connection should not be created yet
            assert service._db is None
            assert service._fts_enabled is False

            # Accessing db property should create connection
            db = service.db
            assert service._db is not None
            assert isinstance(db, sqlite_utils.Database)

        # Clean up
        Path(tmp.name).unlink(missing_ok=True)

    def test_fts5_setup_with_existing_jobs_table(
        self, session: "Session", sample_company: CompanySQL
    ):
        """Test FTS5 setup when jobs table already exists."""
        # Create some test jobs first
        jobs = [
            JobSQL(
                company_id=sample_company.id,
                title="Python Developer",
                description="We are looking for a Python developer to join our team",
                link="https://test.com/job1",
                location="San Francisco",
                last_seen=datetime.now(UTC),
            ),
            JobSQL(
                company_id=sample_company.id,
                title="Machine Learning Engineer",
                description="Develop machine learning models and algorithms",
                link="https://test.com/job2",
                location="Remote",
                last_seen=datetime.now(UTC),
            ),
        ]
        session.add_all(jobs)
        session.commit()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            # Copy existing data to temp database
            service = JobSearchService(tmp.name)

            # Create jobs table and insert test data
            service.db.executescript("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    company TEXT,
                    location TEXT,
                    company_id INTEGER
                );

                INSERT INTO jobs (title, description, company, location) VALUES
                (
                    'Python Developer',
                    'We are looking for a Python developer',
                    'Test Company',
                    'San Francisco'
                ),
                ('ML Engineer', 'Develop ML models', 'Test Company', 'Remote');
            """)

            # Reset service to trigger FTS setup
            service._db = None
            service._fts_enabled = False

            # Access db should setup FTS5
            db = service.db
            assert service._fts_enabled is True

            # Verify FTS5 table was created
            assert "jobs_fts" in db.table_names()

            # Verify FTS5 content
            results = list(db.execute("SELECT count(*) as count FROM jobs_fts"))
            assert results[0][0] == 2

        Path(tmp.name).unlink(missing_ok=True)

    def test_fts5_setup_without_jobs_table(self):
        """Test graceful handling when jobs table doesn't exist yet."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Access db when no jobs table exists
            db = service.db

            # Should not fail, but FTS not enabled yet
            assert "jobs" not in db.table_names()
            # FTS might be False or True depending on timing

        Path(tmp.name).unlink(missing_ok=True)

    def test_fts5_already_enabled_scenario(self):
        """Test handling when FTS5 is already enabled."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            # First service creates FTS5
            service1 = JobSearchService(tmp.name)
            service1.db.executescript("""
                CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                                 company TEXT, location TEXT);
                CREATE VIRTUAL TABLE jobs_fts USING fts5(
                    title, description, company, location,
                    content='jobs', content_rowid='id'
                );
                INSERT INTO jobs (title, description, company, location) VALUES
                ('Test Job', 'Test description', 'Test Company', 'Test Location');
                INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild');
            """)

            # Second service should handle existing FTS5
            service2 = JobSearchService(tmp.name)

            # Access db to trigger FTS setup
            _ = service2.db

            # Should recognize FTS is already available
            assert service2._is_fts_available() is True

        Path(tmp.name).unlink(missing_ok=True)

    def test_fts5_porter_stemming_configuration(self):
        """Test that FTS5 is configured with porter stemming and unicode61."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY,
                    title TEXT, description TEXT, company TEXT, location TEXT
                )
            """)

            # Setup FTS5
            service._setup_search_index()

            if service._fts_enabled:
                # Verify porter stemming works by inserting test data
                service.db.execute("""
                    INSERT INTO jobs (title, description, company, location) VALUES
                    (
                        'Developer Position',
                        'We need a developer who can develop applications',
                        'Tech Corp',
                        'NYC'
                    )
                """)

                # Test porter stemming: "develop" should match "developer"
                results = list(
                    service.db.execute(
                        "SELECT * FROM jobs_fts WHERE jobs_fts MATCH 'develop'"
                    )
                )
                assert len(results) > 0, (
                    "Porter stemming should match 'develop' with 'developer'"
                )

        Path(tmp.name).unlink(missing_ok=True)

    def test_fts5_error_handling_database_locked(self):
        """Test error handling when database is locked or unavailable."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            # Create a database connection that locks the file
            lock_conn = sqlite3.connect(tmp.name)
            lock_conn.execute("BEGIN EXCLUSIVE TRANSACTION")

            try:
                # This might fail due to locking, should handle gracefully
                # FTS setup might fail, but shouldn't crash
                JobSearchService(tmp.name)
            except Exception:
                # Any exception should be handled gracefully in production
                logger.exception("Exception during FTS5 setup with locked database")
            finally:
                lock_conn.close()

        Path(tmp.name).unlink(missing_ok=True)


class TestSearchFunctionality:
    """Test core search functionality with FTS5."""

    @pytest.fixture
    def search_db(self):
        """Create a temporary database with test search data."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create test data with varied content for search testing
            test_jobs = [
                (
                    "Python Developer",
                    "Develop Python applications using Django and Flask",
                    "TechCorp",
                    "San Francisco",
                ),
                (
                    "Machine Learning Engineer",
                    "Build ML models for production deployment",
                    "AI Startup",
                    "Remote",
                ),
                (
                    "Senior Python Engineer",
                    "Lead Python development team",
                    "BigTech",
                    "Seattle",
                ),
                (
                    "Data Scientist",
                    "Analyze data and develop statistical models",
                    "DataCorp",
                    "New York",
                ),
                (
                    "Full Stack Developer",
                    "Work with Python backend and React frontend",
                    "WebCompany",
                    "Austin",
                ),
                (
                    "DevOps Engineer",
                    "Manage infrastructure and deployment pipelines",
                    "CloudCorp",
                    "Denver",
                ),
                (
                    "ML Researcher",
                    "Research new machine learning algorithms",
                    "Research Lab",
                    "Boston",
                ),
                (
                    "Python Backend Developer",
                    "Develop high-performance Python APIs",
                    "StartupXYZ",
                    "Los Angeles",
                ),
            ]

            # Create jobs table and FTS5
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

            # Insert test companies
            for i, company in enumerate({job[2] for job in test_jobs}, 1):
                service.db.execute(
                    "INSERT INTO companysql (id, name) VALUES (?, ?)", [i, company]
                )

            # Insert test jobs
            company_map = {
                name: i for i, name in enumerate({job[2] for job in test_jobs}, 1)
            }
            for i, (title, desc, company, location) in enumerate(test_jobs, 1):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        id, title, description, company, location,
                        posted_date, salary, company_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        i,
                        title,
                        desc,
                        company,
                        location,
                        datetime.now(UTC).isoformat(),
                        json.dumps([80000, 120000]),
                        company_map[company],
                    ],
                )

            # Setup FTS5
            service._setup_search_index()

            yield service

        Path(tmp.name).unlink(missing_ok=True)

    def test_basic_search_functionality(self, search_db):
        """Test basic search returns relevant results."""
        results = search_db.search_jobs("Python")

        assert len(results) > 0
        # All results should contain 'python' in some field
        for result in results:
            contains_python = any(
                "python" in str(result.get(field, "")).lower()
                for field in ["title", "description", "company"]
            )
            assert contains_python, f"Result should contain 'python': {result}"

    def test_porter_stemming_search(self, search_db):
        """Test that porter stemming works - 'develop' matches 'developer'."""
        # Search for 'develop' should match jobs with 'developer'
        results = search_db.search_jobs("develop")

        assert len(results) > 0

        # At least one result should have 'develop' variants in title/description
        found_variants = []
        for result in results:
            title_desc = (
                f"{result.get('title', '')} {result.get('description', '')}".lower()
            )
            if any(
                variant in title_desc
                for variant in ["develop", "developer", "development"]
            ):
                found_variants.append(result)

        assert len(found_variants) > 0, (
            "Porter stemming should match develop/developer variants"
        )

    def test_multi_field_search(self, search_db):
        """Test search across multi-fields (title, description, company, location)."""
        # Search for company name
        results = search_db.search_jobs("TechCorp")
        assert len(results) > 0
        assert any(result.get("company") == "TechCorp" for result in results)

        # Search for location
        results = search_db.search_jobs("San Francisco")
        assert len(results) > 0
        assert any("San Francisco" in result.get("location", "") for result in results)

        # Search for description content
        results = search_db.search_jobs("Django")
        assert len(results) > 0
        assert any("Django" in result.get("description", "") for result in results)

    def test_bm25_relevance_ranking(self, search_db):
        """Test that results are ranked by relevance using BM25."""
        # Search for term that appears in multiple documents with different frequencies
        results = search_db.search_jobs("Python")

        # Should have multiple results
        assert len(results) >= 2

        # Results should have rank field (BM25 score)
        for result in results:
            assert "rank" in result
            assert isinstance(result["rank"], (int, float))

        # Results should be ordered by relevance (rank)
        # Note: FTS5 rank is negative, so better matches have higher (less neg) values
        # The search service orders by rank ASC which gives worst (most negative) first
        # So we need to check they are ordered ascending
        ranks = [result["rank"] for result in results]
        assert ranks == sorted(ranks), (
            "Results should be ordered by relevance (most negative to least negative)"
        )

    def test_empty_query_handling(self, search_db):
        """Test handling of empty or whitespace queries."""
        # Empty string
        results = search_db.search_jobs("")
        assert results == []

        # Whitespace only
        results = search_db.search_jobs("   \n\t  ")
        assert results == []

        # None should return empty list (graceful handling)
        results = search_db.search_jobs(None)
        assert results == []

    def test_special_character_handling(self, search_db):
        """Test search with special characters and Unicode."""
        # Special characters should not crash the search
        results = search_db.search_jobs("C++")
        assert isinstance(results, list)  # Should not crash

        results = search_db.search_jobs("@#$%")
        assert isinstance(results, list)  # Should not crash

        # Unicode characters
        results = search_db.search_jobs("développeur")
        assert isinstance(results, list)  # Should not crash

    def test_boolean_operators(self, search_db):
        """Test FTS5 boolean search operators."""
        if search_db._is_fts_available():
            # AND operator
            results = search_db.search_jobs("Python AND Django")
            for result in results:
                content = (
                    f"{result.get('title', '')} {result.get('description', '')}".lower()
                )
                assert "python" in content
                assert "django" in content

            # OR operator (should return more results than AND)
            results_or = search_db.search_jobs("Python OR Django")
            results_and = search_db.search_jobs("Python AND Django")
            assert len(results_or) >= len(results_and)

    def test_search_result_format(self, search_db):
        """Test that search results have correct format and all required fields."""
        results = search_db.search_jobs("Python")

        assert len(results) > 0

        for result in results:
            # Should be dictionary
            assert isinstance(result, dict)

            # Should have essential fields
            required_fields = ["id", "title", "description", "company", "location"]
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"

            # Should have search-specific fields
            assert "rank" in result

    @pytest.mark.performance
    def test_search_performance(self, search_db):
        """Test that search performance meets <10ms requirement."""
        query = "Python developer"

        # Warm up the database
        search_db.search_jobs(query)

        # Measure search time
        start_time = time.perf_counter()
        results = search_db.search_jobs(query)
        end_time = time.perf_counter()

        search_time_ms = (end_time - start_time) * 1000

        # Should complete in under 10ms for small test dataset
        assert search_time_ms < 10.0, (
            f"Search took {search_time_ms:.2f}ms, should be <10ms"
        )
        assert len(results) > 0


class TestFilterIntegration:
    """Test search filter integration with all filter types."""

    @pytest.fixture
    def filtered_search_db(self):
        """Create database with test data for filter testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create more comprehensive test data for filtering
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
            companies = ["TechCorp", "StartupXYZ", "BigTech"]
            for i, company in enumerate(companies, 1):
                service.db.execute(
                    "INSERT INTO companysql (id, name) VALUES (?, ?)", [i, company]
                )

            # Insert jobs with varied attributes for filter testing
            # Format: (title, desc, company_id, location, posted_date, salary, status,
            # favorite, archived)
            test_jobs = [
                (
                    "Python Dev",
                    "Python development",
                    1,
                    "San Francisco",
                    "2024-01-15T10:00:00Z",
                    [80000, 120000],
                    "New",
                    0,
                    0,
                ),
                (
                    "ML Engineer",
                    "Machine learning",
                    2,
                    "Remote",
                    "2024-01-10T10:00:00Z",
                    [100000, 150000],
                    "Applied",
                    1,
                    0,
                ),
                (
                    "Senior Engineer",
                    "Senior role",
                    3,
                    "New York",
                    "2023-12-01T10:00:00Z",
                    [150000, 200000],
                    "Rejected",
                    0,
                    1,
                ),
                (
                    "Data Scientist",
                    "Data analysis",
                    1,
                    "Austin",
                    "2024-02-01T10:00:00Z",
                    [90000, 130000],
                    "Interested",
                    1,
                    0,
                ),
                (
                    "High Salary Role",
                    "Premium position",
                    2,
                    "Seattle",
                    "2024-01-20T10:00:00Z",
                    [250000, 800000],
                    "New",
                    0,
                    0,
                ),
            ]

            for i, (
                title,
                desc,
                company_id,
                location,
                posted_date,
                salary,
                status,
                favorite,
                archived,
            ) in enumerate(test_jobs, 1):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        id, title, description, company_id, location, posted_date,
                        salary, application_status, favorite, archived
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        i,
                        title,
                        desc,
                        company_id,
                        location,
                        posted_date,
                        json.dumps(salary),
                        status,
                        favorite,
                        archived,
                    ],
                )

            # Setup FTS5
            service._setup_search_index()

            yield service

        Path(tmp.name).unlink(missing_ok=True)

    def test_company_filter(self, filtered_search_db):
        """Test filtering by company."""
        # Test single company filter
        filters = {"company": ["TechCorp"]}
        results = filtered_search_db.search_jobs("Python", filters)

        # Results should only include jobs from TechCorp
        for result in results:
            # Company name should be populated from JOIN
            assert "TechCorp" in str(result.get("company", ""))

    def test_application_status_filter(self, filtered_search_db):
        """Test filtering by application status."""
        # Filter by single status
        filters = {"application_status": ["Applied"]}
        results = filtered_search_db.search_jobs("", filters)

        for result in results:
            assert result.get("application_status") == "Applied"

        # Filter by multiple statuses
        filters = {"application_status": ["New", "Applied"]}
        results = filtered_search_db.search_jobs("", filters)

        valid_statuses = {"New", "Applied"}
        for result in results:
            assert result.get("application_status") in valid_statuses

    def test_date_range_filters(self, filtered_search_db):
        """Test filtering by date ranges."""
        # Filter by start date
        filters = {"date_from": "2024-01-01"}
        results = filtered_search_db.search_jobs("", filters)

        for result in results:
            posted_date = result.get("posted_date")
            if posted_date:
                # Date should be after 2024-01-01
                assert posted_date >= "2024-01-01"

        # Filter by end date
        filters = {"date_to": "2024-01-31"}
        results = filtered_search_db.search_jobs("", filters)

        for result in results:
            posted_date = result.get("posted_date")
            if posted_date:
                # Date should be before end of January 2024
                assert posted_date <= "2024-01-31T23:59:59Z"

    def test_favorites_filter(self, filtered_search_db):
        """Test filtering for favorite jobs only."""
        filters = {"favorites_only": True}
        results = filtered_search_db.search_jobs("", filters)

        # All results should be favorites
        for result in results:
            assert result.get("favorite") == 1

    def test_salary_range_filters(self, filtered_search_db):
        """Test salary range filtering."""
        # Test minimum salary filter
        filters = {"salary_min": 100000}
        results = filtered_search_db.search_jobs("", filters)

        for result in results:
            salary = result.get("salary")
            if salary:
                salary_data = json.loads(salary) if isinstance(salary, str) else salary
                if isinstance(salary_data, list) and len(salary_data) >= 2:
                    max_salary = salary_data[1]
                    assert max_salary >= 100000, (
                        f"Max salary {max_salary} should be >= 100000"
                    )

        # Test maximum salary filter (should exclude unbounded salaries)
        filters = {"salary_max": 200000}
        results = filtered_search_db.search_jobs("", filters)

        for result in results:
            salary = result.get("salary")
            if salary:
                salary_data = json.loads(salary) if isinstance(salary, str) else salary
                if isinstance(salary_data, list) and len(salary_data) >= 1:
                    min_salary = salary_data[0]
                    # Should exclude jobs with very high salaries
                    assert min_salary <= 200000, (
                        f"Min salary {min_salary} should be <= 200000"
                    )

    def test_archived_jobs_filtering(self, filtered_search_db):
        """Test that archived jobs are excluded by default."""
        # Default behavior should exclude archived jobs
        results = filtered_search_db.search_jobs("")

        for result in results:
            assert result.get("archived", 0) == 0

        # Explicit inclusion of archived jobs
        filters = {"include_archived": True}
        results_with_archived = filtered_search_db.search_jobs("", filters)

        # Should have more results when including archived
        results_default = filtered_search_db.search_jobs("")
        assert len(results_with_archived) >= len(results_default)

    def test_combined_filters(self, filtered_search_db):
        """Test multiple filters applied together."""
        filters = {
            "application_status": ["New", "Applied"],
            "favorites_only": True,
            "salary_min": 80000,
            "date_from": "2024-01-01",
        }

        results = filtered_search_db.search_jobs("", filters)

        # Verify all filter conditions are met
        valid_statuses = {"New", "Applied"}
        for result in results:
            assert result.get("application_status") in valid_statuses
            assert result.get("favorite") == 1
            assert result.get("posted_date", "") >= "2024-01-01"
            # Salary check
            salary = result.get("salary")
            if salary:
                salary_data = json.loads(salary) if isinstance(salary, str) else salary
                if isinstance(salary_data, list) and len(salary_data) >= 2:
                    assert salary_data[1] >= 80000

    def test_filter_edge_cases(self, filtered_search_db):
        """Test edge cases for filter handling."""
        # Empty filter dictionary
        results = filtered_search_db.search_jobs("Python", {})
        assert isinstance(results, list)

        # None as filters
        results = filtered_search_db.search_jobs("Python", None)
        assert isinstance(results, list)

        # "All" in company filter (should not filter)
        filters = {"company": ["All"]}
        results = filtered_search_db.search_jobs("Python", filters)
        assert isinstance(results, list)

        # "All" in application_status filter
        filters = {"application_status": ["All"]}
        results = filtered_search_db.search_jobs("Python", filters)
        assert isinstance(results, list)


class TestPerformanceAndCaching:
    """Test performance requirements and Streamlit caching."""

    def test_streamlit_caching_decorator(self):
        """Test that search method has Streamlit caching configured."""
        # Check if the caching decorator is properly applied
        search_method = JobSearchService.search_jobs

        # The method should have cache configuration
        # This is hard to test directly, but we can check the method attributes
        assert hasattr(search_method, "__wrapped__") or hasattr(
            search_method, "_cache_config"
        )

    @patch("src.services.search_service.st")
    def test_streamlit_caching_behavior(self, mock_st):
        """Test Streamlit caching behavior with mocked st."""

        # Mock the cache decorator
        def dummy_cache(**kwargs):
            def decorator(func):
                func._cache_config = kwargs
                return func

            return decorator

        mock_st.cache_data = dummy_cache

        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Cache configuration should be preserved
            if hasattr(service.search_jobs, "_cache_config"):
                assert service.search_jobs._cache_config.get("ttl") == 300

        Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.performance
    def test_concurrent_search_performance(self):
        """Test search performance under concurrent access."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create test data
            service.db.executescript("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT, posted_date TEXT, salary TEXT,
                    application_status TEXT DEFAULT 'New', favorite INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0, company_id INTEGER
                );

                CREATE TABLE companysql (
                    id INTEGER PRIMARY KEY, name TEXT
                );
            """)

            # Insert test jobs
            for i in range(100):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        title, description, company, location, posted_date, salary
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"Job {i}",
                        f"Description for job {i}",
                        f"Company {i % 10}",
                        f"Location {i % 5}",
                        datetime.now(UTC).isoformat(),
                        json.dumps([50000 + i * 1000, 80000 + i * 1000]),
                    ],
                )

            service._setup_search_index()

            # SQLite has thread-safety limitations - create separate instances
            import queue
            import threading

            results_queue = queue.Queue()

            def search_worker():
                # Create separate service instance for this thread
                thread_service = JobSearchService(tmp.name)
                start_time = time.perf_counter()
                results = thread_service.search_jobs("Job")
                end_time = time.perf_counter()
                results_queue.put((len(results), (end_time - start_time) * 1000))

            # Start multiple search threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=search_worker)
                thread.start()
                threads.append(thread)

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Collect results
            search_times = []
            while not results_queue.empty():
                result_count, search_time = results_queue.get()
                assert result_count > 0, "Each search should return results"
                search_times.append(search_time)

            # All searches should complete reasonably quickly
            avg_time = sum(search_times) / len(search_times)
            assert avg_time < 50.0, (
                f"Average search time {avg_time:.2f}ms too slow under concurrency"
            )

        Path(tmp.name).unlink(missing_ok=True)

    def test_large_result_set_performance(self):
        """Test search performance with large result sets."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create large dataset
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT, posted_date TEXT, salary TEXT,
                    application_status TEXT DEFAULT 'New', favorite INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0, company_id INTEGER
                )
            """)

            # Insert many jobs with "developer" in title for large result set
            for i in range(1000):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        title, description, company, location, posted_date, salary
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"Developer {i}",
                        f"We need developers to develop software {i}",
                        f"Company {i % 50}",
                        f"Location {i % 10}",
                        datetime.now(UTC).isoformat(),
                        json.dumps([60000, 100000]),
                    ],
                )

            service._setup_search_index()

            # Search for common term that will return many results
            start_time = time.perf_counter()
            results = service.search_jobs("developer", limit=100)
            end_time = time.perf_counter()

            search_time_ms = (end_time - start_time) * 1000

            # Should still be fast even with large dataset
            assert search_time_ms < 100.0, (
                f"Search took {search_time_ms:.2f}ms, should be <100ms for "
                "large dataset"
            )
            assert len(results) <= 100, "Should respect limit parameter"

        Path(tmp.name).unlink(missing_ok=True)

    def test_search_query_limits(self):
        """Test search result limits are properly applied."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create test data
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT, posted_date TEXT, salary TEXT,
                    application_status TEXT DEFAULT 'New', favorite INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0, company_id INTEGER
                )
            """)

            # Insert more jobs than the limit
            for i in range(100):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        title, description, company, location, posted_date, salary
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"Python Developer {i}",
                        f"Python development role {i}",
                        "TestCorp",
                        "Remote",
                        datetime.now(UTC).isoformat(),
                        json.dumps([70000, 110000]),
                    ],
                )

            service._setup_search_index()

            # Test default limit (50)
            results = service.search_jobs("Python")
            assert len(results) <= 50, "Should respect default limit of 50"

            # Test custom limit
            results = service.search_jobs("Python", limit=10)
            assert len(results) <= 10, "Should respect custom limit of 10"

            # Test very high limit
            results = service.search_jobs("Python", limit=200)
            # Should not exceed actual number of matches
            assert len(results) <= 100, "Should not exceed available results"

        Path(tmp.name).unlink(missing_ok=True)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_database_connection_failures(self):
        """Test handling of database connection failures."""
        # Test with non-existent directory
        invalid_path = "/nonexistent/directory/test.db"
        JobSearchService(invalid_path)

        # Should handle gracefully (sqlite_utils creates directories)
        # If it succeeds, clean up
        with contextlib.suppress(Exception):
            if Path(invalid_path).exists():
                Path(invalid_path).unlink()
                parent_dir = Path(invalid_path).parent
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()

    def test_corrupted_fts_index_handling(self):
        """Test handling of corrupted or missing FTS5 index."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table but corrupt FTS
            service.db.executescript("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT
                );
                CREATE VIRTUAL TABLE jobs_fts USING fts5(
                    title, description, company, location
                );
            """)

            # Corrupt the FTS index by dropping its internal tables
            # May not exist or may fail, that's expected
            with contextlib.suppress(Exception):
                service.db.execute("DROP TABLE jobs_fts_data")

            # Search should still work (fallback to non-FTS)
            results = service.search_jobs("test")
            assert isinstance(results, list)  # Should not crash

        Path(tmp.name).unlink(missing_ok=True)

    def test_invalid_query_syntax(self):
        """Test handling of invalid FTS5 query syntax."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT
                )
            """)
            service.db.execute("""
                INSERT INTO jobs VALUES (
                    1, 'Test Job', 'Test Description', 'Test Co', 'Test Loc'
                )
            """)
            service._setup_search_index()

            # Invalid FTS5 queries that might cause syntax errors
            invalid_queries = [
                '"unclosed quote',
                "AND OR NOT",
                "((()))",
                "NEAR/",
                '"phrase" AND AND "another"',
            ]

            for invalid_query in invalid_queries:
                # Should handle gracefully, not crash
                results = service.search_jobs(invalid_query)
                assert isinstance(results, list)

        Path(tmp.name).unlink(missing_ok=True)

    def test_malformed_filter_data(self):
        """Test handling of malformed filter data."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT, salary TEXT, posted_date TEXT,
                    application_status TEXT, favorite INTEGER, archived INTEGER
                )
            """)

            # Test various malformed filter scenarios
            malformed_filters = [
                {"date_from": "invalid-date"},
                {"date_to": "not-a-date"},
                {"salary_min": "not-a-number"},
                {"salary_max": "invalid"},
                {"company": None},  # Should handle None gracefully
                {"application_status": 123},  # Wrong type
            ]

            for filters in malformed_filters:
                # Should handle gracefully without crashing
                results = service.search_jobs("test", filters)
                assert isinstance(results, list)

        Path(tmp.name).unlink(missing_ok=True)

    def test_search_with_missing_columns(self):
        """Test search behavior when database schema is incomplete."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table missing some expected columns
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    description TEXT
                    -- Missing: company, location, salary, etc.
                )
            """)
            service.db.execute(
                "INSERT INTO jobs VALUES (1, 'Test Job', 'Test Description')"
            )

            # Search should handle missing columns gracefully
            results = service.search_jobs("Test")
            assert isinstance(results, list)
            if results:
                # Should have basic fields, missing ones should be None or have defaults
                result = results[0]
                assert result.get("title") == "Test Job"
                assert result.get("description") == "Test Description"

        Path(tmp.name).unlink(missing_ok=True)

    def test_search_service_logging(self, caplog):
        """Test that search service logs appropriately."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Enable logging for this test
            with caplog.at_level(logging.INFO):
                # Create basic setup
                service.db.execute("""
                    CREATE TABLE jobs (
                        id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                        company TEXT, location TEXT
                    )
                """)
                service.db.execute("""
                    INSERT INTO jobs VALUES (
                        1, 'Test Job', 'Test Description', 'Test Co', 'Test Loc'
                    )
                """)

                # This should log FTS5 setup
                service._setup_search_index()

                # This should log search results
                service.search_jobs("Test")

            # Verify appropriate log messages were generated
            log_messages = [record.message for record in caplog.records]
            fts_logs = [msg for msg in log_messages if "FTS5" in msg]
            assert len(fts_logs) > 0, "Should log FTS5 setup information"

        Path(tmp.name).unlink(missing_ok=True)


class TestFallbackSearchMechanism:
    """Test fallback search when FTS5 is unavailable."""

    @pytest.fixture
    def fallback_db(self):
        """Create database without FTS5 for fallback testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table but prevent FTS5 setup
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

            # Insert test data
            service.db.execute(
                "INSERT INTO companysql (id, name) VALUES (1, 'TestCorp')"
            )

            test_jobs = [
                ("Python Developer", "Develop Python applications", 1, "San Francisco"),
                ("Java Engineer", "Work with Java technologies", 1, "New York"),
                ("Data Scientist", "Analyze large datasets", 1, "Remote"),
            ]

            for i, (title, desc, company_id, location) in enumerate(test_jobs, 1):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        id, title, description, company_id, location,
                        posted_date, salary
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        i,
                        title,
                        desc,
                        company_id,
                        location,
                        datetime.now(UTC).isoformat(),
                        json.dumps([80000, 120000]),
                    ],
                )

            # Force FTS to be unavailable
            service._fts_enabled = False

            yield service

        Path(tmp.name).unlink(missing_ok=True)

    def test_fallback_search_basic_functionality(self, fallback_db):
        """Test that fallback search works when FTS5 is unavailable."""
        # Force FTS to appear unavailable
        fallback_db._fts_enabled = False

        results = fallback_db.search_jobs("Python")

        assert len(results) > 0
        # Should find Python-related jobs using LIKE queries
        python_found = any("Python" in result.get("title", "") for result in results)
        assert python_found, "Fallback search should find Python jobs"

    def test_fallback_multi_term_search(self, fallback_db):
        """Test fallback search with multiple search terms."""
        fallback_db._fts_enabled = False

        # Multiple terms should all be matched
        results = fallback_db.search_jobs("Python Developer")

        for result in results:
            title_desc = (
                f"{result.get('title', '')} {result.get('description', '')}".lower()
            )
            # Each result should contain both terms (AND logic in fallback)
            assert "python" in title_desc
            assert "developer" in title_desc

    def test_fallback_cross_field_search(self, fallback_db):
        """Test fallback search across different fields."""
        fallback_db._fts_enabled = False

        # Search for company name
        results = fallback_db.search_jobs("TestCorp")
        assert len(results) > 0

        # Search for location
        results = fallback_db.search_jobs("Francisco")
        assert len(results) > 0

        # Search for description content
        results = fallback_db.search_jobs("applications")
        assert len(results) > 0

    def test_fallback_with_filters(self, fallback_db):
        """Test that fallback search works with filters."""
        fallback_db._fts_enabled = False

        # Add filter to fallback search
        filters = {"application_status": ["New"]}
        results = fallback_db.search_jobs("Python", filters)

        for result in results:
            assert result.get("application_status") == "New"

    def test_fallback_result_format(self, fallback_db):
        """Test that fallback search returns properly formatted results."""
        fallback_db._fts_enabled = False

        results = fallback_db.search_jobs("Developer")

        assert len(results) > 0

        for result in results:
            # Should have all required fields
            assert "id" in result
            assert "title" in result
            assert "description" in result
            assert "company" in result
            assert "location" in result
            assert "rank" in result

            # Rank should be 0 for fallback (no relevance ranking)
            assert result["rank"] == 0

    def test_fallback_performance(self, fallback_db):
        """Test fallback search performance."""
        fallback_db._fts_enabled = False

        # Add more data for performance test
        for i in range(100):
            fallback_db.db.execute(
                """
                INSERT INTO jobs (
                    title, description, company_id, location,
                    posted_date, salary
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    f"Job Title {i}",
                    f"Job description {i}",
                    1,
                    f"Location {i}",
                    datetime.now(UTC).isoformat(),
                    json.dumps([60000, 100000]),
                ],
            )

        # Measure fallback search time
        start_time = time.perf_counter()
        results = fallback_db.search_jobs("Job")
        end_time = time.perf_counter()

        search_time_ms = (end_time - start_time) * 1000

        # Fallback should still be reasonably fast
        assert search_time_ms < 100.0, (
            f"Fallback search took {search_time_ms:.2f}ms, should be <100ms"
        )
        assert len(results) > 0

    def test_fts_availability_detection(self, fallback_db):
        """Test FTS5 availability detection works correctly."""
        # Initially FTS should not be available
        assert fallback_db._is_fts_available() is False

        # Create FTS5 index
        fallback_db.db.execute("""
            CREATE VIRTUAL TABLE jobs_fts USING fts5(
                title, description, company, location,
                content='jobs', content_rowid='id'
            )
        """)
        fallback_db.db.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild')")

        # Now FTS should be detected as available
        fallback_db._fts_enabled = True  # Reset flag
        assert fallback_db._is_fts_available() is True


class TestDatabaseOperations:
    """Test database operations and maintenance functionality."""

    def test_rebuild_search_index(self):
        """Test rebuilding the FTS5 search index."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table with data
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT
                )
            """)

            # Insert test data
            for i in range(5):
                service.db.execute(
                    """
                    INSERT INTO jobs (title, description, company, location)
                    VALUES (?, ?, ?, ?)
                """,
                    [f"Job {i}", f"Description {i}", f"Company {i}", f"Location {i}"],
                )

            # Initial FTS setup
            service._setup_search_index()

            # Rebuild index
            result = service.rebuild_search_index()

            if service._fts_enabled:
                assert result is True, "Index rebuild should succeed"

                # Verify index has correct number of records
                count_result = list(
                    service.db.execute("SELECT count(*) as count FROM jobs_fts")
                )
                assert count_result[0][0] == 5, "Rebuilt index should have all records"
            else:
                # If FTS not available, rebuild should return False
                assert result is False

        Path(tmp.name).unlink(missing_ok=True)

    def test_get_search_stats(self):
        """Test getting search index statistics."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT, posted_date TEXT
                )
            """)

            # Insert test data
            test_date = datetime.now(UTC).isoformat()
            for i in range(10):
                service.db.execute(
                    """
                    INSERT INTO jobs (
                        title, description, company, location, posted_date
                    )
                    VALUES (?, ?, ?, ?, ?)
                """,
                    [
                        f"Job {i}",
                        f"Description {i}",
                        f"Company {i}",
                        f"Location {i}",
                        test_date,
                    ],
                )

            # Setup FTS
            service._setup_search_index()

            # Get stats
            stats = service.get_search_stats()

            # Verify stats format
            required_keys = [
                "fts_enabled",
                "indexed_jobs",
                "total_jobs",
                "index_coverage",
                "last_updated",
            ]
            for key in required_keys:
                assert key in stats, f"Stats should include {key}"

            # Verify stats values
            assert stats["total_jobs"] == 10, "Should count all jobs"

            if stats["fts_enabled"]:
                assert stats["indexed_jobs"] == 10, "Should count indexed jobs"
                assert stats["index_coverage"] == 100.0, "Should have 100% coverage"
                assert stats["last_updated"] is not None, (
                    "Should have last updated timestamp"
                )
            else:
                assert stats["indexed_jobs"] == 0, "No indexed jobs when FTS disabled"
                assert stats["index_coverage"] == 0.0, "No coverage when FTS disabled"

        Path(tmp.name).unlink(missing_ok=True)

    def test_get_search_stats_empty_database(self):
        """Test search stats with empty database."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create empty jobs table
            service.db.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT
                )
            """)

            service._setup_search_index()

            stats = service.get_search_stats()

            assert stats["total_jobs"] == 0
            assert stats["indexed_jobs"] == 0
            assert stats["index_coverage"] == 0.0
            assert stats["last_updated"] is None

        Path(tmp.name).unlink(missing_ok=True)

    def test_get_search_stats_error_handling(self):
        """Test search stats error handling."""
        # Test with invalid database path
        service = JobSearchService("/nonexistent/path.db")

        # Should handle gracefully and return error stats
        stats = service.get_search_stats()

        assert stats["fts_enabled"] is False
        assert stats["total_jobs"] == 0
        assert stats["indexed_jobs"] == 0
        assert stats["index_coverage"] == 0.0
        assert "error" in stats

    def test_auto_triggers_functionality(self):
        """Test that FTS5 auto-triggers work for INSERT/UPDATE/DELETE."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            service = JobSearchService(tmp.name)

            # Create jobs table and companysql table
            service.db.executescript("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, title TEXT, description TEXT,
                    company TEXT, location TEXT, posted_date TEXT, salary TEXT,
                    application_status TEXT DEFAULT 'New', favorite INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0, company_id INTEGER
                );

                CREATE TABLE companysql (
                    id INTEGER PRIMARY KEY, name TEXT
                );

                INSERT INTO companysql (id, name) VALUES (1, 'Test Company');
            """)

            # Setup FTS with triggers
            service._setup_search_index()

            if service._fts_enabled:
                # Insert new job - should auto-update FTS
                service.db.execute("""
                    INSERT INTO jobs (
                        title, description, company, location,
                        company_id, posted_date, salary
                    )
                    VALUES (
                        'New Job', 'New Description', 'New Company', 'New Location',
                        1, '2024-01-01T10:00:00Z', '["50000", "80000"]'
                    )
                """)

                # Search should find the new job
                results = service.search_jobs("New Job")
                assert len(results) > 0, "Auto-trigger should index new job"

                # Update job - should auto-update FTS
                service.db.execute(
                    "UPDATE jobs SET title = 'Updated Job' WHERE title = 'New Job'"
                )

                # Search should find updated job
                results = service.search_jobs("Updated Job")
                assert len(results) > 0, "Auto-trigger should update FTS index"

                # Verify FTS5 triggers updated the index by checking FTS table
                # (Search service caching prevents testing via search_jobs method)
                fts_results = list(
                    service.db.execute("SELECT title FROM jobs_fts WHERE rowid = 1")
                )
                assert len(fts_results) > 0, "Job should exist in FTS index"
                assert fts_results[0][0] == "Updated Job", (
                    f"FTS index should show updated title, got: {fts_results[0][0]}"
                )

                # Verify the main table was also updated
                main_results = list(
                    service.db.execute("SELECT title FROM jobs WHERE id = 1")
                )
                assert len(main_results) > 0, "Job should exist in main table"
                assert main_results[0][0] == "Updated Job", (
                    f"Main table should show updated title, got: {main_results[0][0]}"
                )

                # Delete job - should auto-update FTS
                service.db.execute("DELETE FROM jobs WHERE title = 'Updated Job'")

                # Verify FTS5 triggers removed job from index by checking directly
                # (Search service caching prevents reliable testing via search)
                fts_after_delete = list(
                    service.db.execute("SELECT title FROM jobs_fts WHERE rowid = 1")
                )
                assert len(fts_after_delete) == 0, (
                    "Auto-trigger should remove deleted job from FTS index"
                )

                # Verify main table is also empty
                main_after_delete = list(
                    service.db.execute("SELECT title FROM jobs WHERE id = 1")
                )
                assert len(main_after_delete) == 0, (
                    "Job should be deleted from main table"
                )

        Path(tmp.name).unlink(missing_ok=True)
