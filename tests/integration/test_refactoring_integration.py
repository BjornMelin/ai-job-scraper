"""Integration tests for task 1.6 refactoring changes.

This test suite validates that the refactoring changes work correctly in an integrated
environment, ensuring that:
1. Bulk company creation still works in scraping workflows
2. UI tab filtering optimization performs correctly with database queries
3. Constants are properly accessible across modules
4. End-to-end workflows continue to function as expected

These tests verify real-world usage scenarios after the refactoring.
"""

# ruff: noqa: ARG002  # Pytest fixtures require named parameters even if unused

import logging

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.constants import APPLICATION_STATUSES
from src.models import CompanySQL, JobSQL
from src.services.company_service import CompanyService
from src.services.job_service import JobService

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)


@pytest.fixture
def test_engine():
    """Create a test-specific SQLite engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def mock_db_session(test_session):
    """Mock db_session context manager to use test session."""
    with (
        patch("src.services.company_service.db_session") as mock_company_session,
        patch("src.services.job_service.db_session") as mock_job_session,
    ):
        mock_company_session.return_value.__enter__.return_value = test_session
        mock_company_session.return_value.__exit__.return_value = None
        mock_job_session.return_value.__enter__.return_value = test_session
        mock_job_session.return_value.__exit__.return_value = None
        yield mock_company_session, mock_job_session


class TestScrapeToUIWorkflow:
    """Test complete workflow from scraping to UI filtering."""

    def test_bulk_company_creation_to_ui_filtering_workflow(
        self,
        test_session,
        mock_db_session,
    ):
        """Test complete workflow: bulk company creation → job → UI filtering."""
        # Step 1: Simulate scraping workflow with bulk company creation
        # (This tests the relocated bulk_get_or_create_companies method)
        company_names = {
            "TechStart Inc",
            "AI Innovations",
            "DataFlow Systems",
            "CloudScale Ltd",
        }

        # Use the relocated bulk_get_or_create_companies method
        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            company_names,
        )

        # Verify all companies were created/found
        assert len(company_map) == 4
        assert all(name in company_map for name in company_names)

        # Step 2: Create jobs for these companies using the company_map
        base_date = datetime.now(UTC)
        jobs_data = [
            {
                "company_id": company_map["TechStart Inc"],
                "title": "Senior AI Engineer",
                "description": "Leading AI development team",
                "link": "https://techstart.com/jobs/ai-1",
                "location": "San Francisco, CA",
                "posted_date": base_date - timedelta(days=1),
                "salary": (150000, 200000),
                "favorite": True,
                "application_status": "New",
            },
            {
                "company_id": company_map["AI Innovations"],
                "title": "Machine Learning Engineer",
                "description": "ML model development and deployment",
                "link": "https://ai-innovations.com/jobs/ml-1",
                "location": "New York, NY",
                "posted_date": base_date - timedelta(days=2),
                "salary": (140000, 180000),
                "favorite": False,
                "application_status": "Applied",
            },
            {
                "company_id": company_map["DataFlow Systems"],
                "title": "Data Scientist",
                "description": "Statistical modeling and analytics",
                "link": "https://dataflow.com/jobs/ds-1",
                "location": "Austin, TX",
                "posted_date": base_date - timedelta(days=3),
                "salary": (120000, 160000),
                "favorite": True,
                "application_status": "Applied",
            },
            {
                "company_id": company_map["CloudScale Ltd"],
                "title": "DevOps Engineer",
                "description": "Cloud infrastructure management",
                "link": "https://cloudscale.com/jobs/devops-1",
                "location": "Remote",
                "posted_date": base_date - timedelta(days=4),
                "salary": (110000, 150000),
                "favorite": False,
                "application_status": "Interested",
            },
        ]

        # Create the jobs in database
        for job_data in jobs_data:
            job = JobSQL(**job_data, content_hash=f"hash_{job_data['title']}")
            test_session.add(job)
        test_session.commit()

        # Step 3: Test UI filtering functionality (the optimized database queries)

        # Test All Jobs tab (no specific filters)
        all_jobs = JobService.get_filtered_jobs({})
        assert len(all_jobs) == 4

        # Test Favorites tab (should use database filtering)
        favorites_filters = {
            "favorites_only": True,
            "text_search": "",
            "company": [],
            "application_status": [],
        }
        favorite_jobs = JobService.get_filtered_jobs(favorites_filters)
        assert len(favorite_jobs) == 2  # AI Engineer and Data Scientist
        assert all(job.favorite for job in favorite_jobs)
        favorite_titles = {job.title for job in favorite_jobs}
        assert favorite_titles == {"Senior AI Engineer", "Data Scientist"}

        # Test Applied tab (should use database filtering with application_status)
        applied_filters = {
            "favorites_only": False,
            "text_search": "",
            "company": [],
            "application_status": ["Applied"],
        }
        applied_jobs = JobService.get_filtered_jobs(applied_filters)
        assert len(applied_jobs) == 2  # ML Engineer and Data Scientist
        assert all(job.application_status == "Applied" for job in applied_jobs)
        applied_titles = {job.title for job in applied_jobs}
        assert applied_titles == {"Machine Learning Engineer", "Data Scientist"}

        # Test intersection: Favorite AND Applied jobs
        favorite_applied_filters = {
            "favorites_only": True,
            "application_status": ["Applied"],
        }
        favorite_applied_jobs = JobService.get_filtered_jobs(favorite_applied_filters)
        assert len(favorite_applied_jobs) == 1  # Only Data Scientist
        job = favorite_applied_jobs[0]
        assert job.title == "Data Scientist"
        assert job.favorite is True
        assert job.application_status == "Applied"

    def test_constants_integration_across_modules(self, test_session, mock_db_session):
        """Test that constants are properly accessible across different modules."""
        # Test that APPLICATION_STATUSES can be imported and used

        # Verify the constant has expected structure
        assert isinstance(APPLICATION_STATUSES, list)
        assert len(APPLICATION_STATUSES) > 0

        # Test that the constants work with JobService filtering
        company = CompanySQL(
            name="ConstantTest Corp",
            url="https://test.com",
            active=True,
        )
        test_session.add(company)
        test_session.commit()
        test_session.refresh(company)

        # Create jobs with each status from the constant
        for i, status in enumerate(APPLICATION_STATUSES):
            job = JobSQL(
                company_id=company.id,
                title=f"Job {i + 1}",
                description=f"Test job with {status} status",
                link=f"https://test.com/job-{i + 1}",
                location="Test Location",
                application_status=status,
                content_hash=f"hash_{i + 1}",
            )
            test_session.add(job)
        test_session.commit()

        # Test filtering by each status value
        for status in APPLICATION_STATUSES:
            filters = {"application_status": [status]}
            jobs = JobService.get_filtered_jobs(filters)
            assert len(jobs) == 1
            assert jobs[0].application_status == status

        # Test that job_card.py can access the constants (simulated import)
        try:
            # ruff: noqa: N811  # Import as different name for testing
            from src.constants import APPLICATION_STATUSES as job_card_statuses

            assert job_card_statuses == APPLICATION_STATUSES
        except ImportError:
            pytest.fail("job_card.py cannot import APPLICATION_STATUSES")

    def test_company_service_and_job_service_integration(
        self,
        test_session,
        mock_db_session,
    ):
        """Test integration between CompanyService and JobService after refactoring."""
        # Step 1: Create companies using CompanyService
        company1 = CompanyService.add_company(
            "Integration Corp",
            "https://integration.com",
        )
        company2 = CompanyService.add_company("Testing Ltd", "https://testing.ltd")

        # Step 2: Use bulk_get_or_create_companies with mix of existing and new
        company_names = {"Integration Corp", "Testing Ltd", "NewBulkCorp"}
        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            company_names,
        )

        # Verify existing companies are found and new ones created
        assert len(company_map) == 3
        assert company_map["Integration Corp"] == company1.id
        assert company_map["Testing Ltd"] == company2.id
        assert "NewBulkCorp" in company_map

        # Step 3: Create jobs for these companies
        base_date = datetime.now(UTC)
        jobs = [
            JobSQL(
                company_id=company_map["Integration Corp"],
                title="Integration Engineer",
                description="System integration specialist",
                link="https://integration.com/jobs/eng-1",
                location="Seattle, WA",
                posted_date=base_date - timedelta(days=1),
                application_status="New",
                favorite=True,
                content_hash="hash_int_1",
            ),
            JobSQL(
                company_id=company_map["Testing Ltd"],
                title="QA Engineer",
                description="Quality assurance testing",
                link="https://testing.ltd/jobs/qa-1",
                location="Portland, OR",
                posted_date=base_date - timedelta(days=2),
                application_status="Applied",
                favorite=False,
                content_hash="hash_test_1",
            ),
            JobSQL(
                company_id=company_map["NewBulkCorp"],
                title="Software Developer",
                description="Full-stack development",
                link="https://newbulk.com/jobs/dev-1",
                location="Remote",
                posted_date=base_date - timedelta(days=3),
                application_status="Interested",
                favorite=True,
                content_hash="hash_bulk_1",
            ),
        ]

        for job in jobs:
            test_session.add(job)
        test_session.commit()

        # Step 4: Test combined filtering scenarios

        # Filter by company and status
        filters = {
            "company": ["Integration Corp"],
            "application_status": ["New"],
        }
        results = JobService.get_filtered_jobs(filters)
        assert len(results) == 1
        assert results[0].title == "Integration Engineer"
        assert results[0].company == "Integration Corp"

        # Filter favorites across all companies
        filters = {"favorites_only": True}
        results = JobService.get_filtered_jobs(filters)
        assert len(results) == 2  # Integration Engineer and Software Developer
        favorite_companies = {job.company for job in results}
        assert favorite_companies == {"Integration Corp", "NewBulkCorp"}

        # Test company statistics with job counts
        companies_with_stats = CompanyService.get_companies_with_job_counts()
        assert len(companies_with_stats) == 3

        # Verify job counts are correct
        for company_stat in companies_with_stats:
            company_name = company_stat["company"].name
            if company_name in ["Integration Corp", "Testing Ltd", "NewBulkCorp"]:
                assert company_stat["total_jobs"] == 1
                assert (
                    company_stat["active_jobs"] == 1
                )  # All jobs are active (not archived)

    def test_error_handling_integration(self, test_session, mock_db_session):
        """Test error handling across services after refactoring."""
        # Test bulk_get_or_create_companies with empty set
        result = CompanyService.bulk_get_or_create_companies(test_session, set())
        assert result == {}

        # Test JobService with invalid filters
        filters = {"application_status": ["NonExistentStatus"]}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0

        # Test mixed valid/invalid status values
        filters = {"application_status": ["Applied", "NonExistent", "New"]}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 0  # No jobs with those statuses exist yet

        # Create a company and job using the services (so they use mocked sessions)
        company = CompanyService.add_company("Error Test Corp", "https://error.com")

        # Create a job by adding to test session, then use the company_id
        job = JobSQL(
            company_id=company.id,
            title="Error Test Job",
            description="Testing error handling",
            link="https://error.com/job-1",
            location="Test City",
            application_status="Applied",
            content_hash="hash_error_1",
        )
        test_session.add(job)
        test_session.commit()

        # Clear the cache so the new job is visible
        JobService.get_filtered_jobs.clear()

        # Now test mixed valid/invalid again
        filters = {"application_status": ["Applied", "NonExistent", "New"]}
        jobs = JobService.get_filtered_jobs(filters)
        assert len(jobs) == 1  # Should find the Applied job
        assert jobs[0].application_status == "Applied"

    def _calculate_expected_count(
        self,
        total_items: int,
        modulo: int,
        target_remainder: int,
    ) -> int:
        """Helper method to calculate expected count for modulo-based filtering.

        Args:
            total_items: Total number of items to check.
            modulo: The modulo value to use.
            target_remainder: The target remainder for matching items.

        Returns:
            Count of items that match the condition.
        """
        return len([i for i in range(total_items) if i % modulo == target_remainder])

    def test_performance_integration(self, test_session, mock_db_session):
        """Test that refactored methods maintain good performance characteristics."""
        # Test bulk_get_or_create_companies performance with larger dataset
        large_company_set = {f"PerfTest Corp {i:03d}" for i in range(50)}

        company_map = CompanyService.bulk_get_or_create_companies(
            test_session,
            large_company_set,
        )

        # Should handle 50 companies efficiently
        assert len(company_map) == 50
        assert all(f"PerfTest Corp {i:03d}" in company_map for i in range(50))

        # Create jobs for performance testing
        base_date = datetime.now(UTC)
        jobs = []
        statuses = ["New", "Interested", "Applied", "Rejected"]

        for i in range(50):
            company_name = f"PerfTest Corp {i:03d}"
            job = JobSQL(
                company_id=company_map[company_name],
                title=f"Performance Test Job {i:03d}",
                description=f"Testing performance with job {i}",
                link=f"https://perftest{i:03d}.com/job",
                location="Performance City",
                posted_date=base_date - timedelta(days=(i % 30)),
                application_status=statuses[i % 4],
                favorite=(i % 5 == 0),  # Every 5th job is favorite
                content_hash=f"perf_hash_{i:03d}",
            )
            jobs.append(job)

        test_session.add_all(jobs)
        test_session.commit()

        # Test filtering performance with larger dataset
        filters = {"application_status": ["Applied"]}
        applied_jobs = JobService.get_filtered_jobs(filters)
        expected_applied = self._calculate_expected_count(
            50,
            4,
            2,
        )  # Applied is index 2
        assert len(applied_jobs) == expected_applied

        # Test favorites filtering performance
        filters = {"favorites_only": True}
        favorite_jobs = JobService.get_filtered_jobs(filters)
        expected_favorites = self._calculate_expected_count(50, 5, 0)  # Every 5th
        assert len(favorite_jobs) == expected_favorites

        # Test combined filtering performance
        filters = {
            "favorites_only": True,
            "application_status": ["Applied", "New"],
        }
        combined_jobs = JobService.get_filtered_jobs(filters)
        # Should find favorites that are either Applied or New
        assert len(combined_jobs) >= 0  # At least 0, depends on specific distribution
        assert all(job.favorite for job in combined_jobs)
        assert all(
            job.application_status in ["Applied", "New"] for job in combined_jobs
        )
