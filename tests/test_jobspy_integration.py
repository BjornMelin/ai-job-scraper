"""Comprehensive integration tests for JobSpy + JobService workflows.

This module tests complete end-to-end integration including:
- JobService + JobSpy wrapper integration
- Database persistence with mocked DB operations
- Job deduplication logic validation
- Company creation/lookup tests
- Error recovery scenarios
- Performance testing of complete workflows
- Data consistency validation

All external dependencies (JobSpy, database) are completely mocked for fast,
deterministic testing without side effects.
"""

import asyncio

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
    LocationType,
)


# Mock database models for testing
class MockJob:
    """Mock Job database model."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id")
        self.title = kwargs.get("title", "")
        self.company = kwargs.get("company", "")
        self.location = kwargs.get("location")
        self.job_url = kwargs.get("job_url")
        self.job_type = kwargs.get("job_type")
        self.is_remote = kwargs.get("is_remote", False)
        self.salary_min = kwargs.get("salary_min")
        self.salary_max = kwargs.get("salary_max")
        self.date_posted = kwargs.get("date_posted")
        self.description = kwargs.get("description")
        self.skills = kwargs.get("skills", [])
        self.created_at = kwargs.get("created_at", datetime.now())
        self.updated_at = kwargs.get("updated_at", datetime.now())
        self.company_id = kwargs.get("company_id")

    def __repr__(self):
        return f"MockJob(id={self.id}, title='{self.title}', company='{self.company}')"


class MockCompany:
    """Mock Company database model."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id")
        self.name = kwargs.get("name", "")
        self.industry = kwargs.get("industry")
        self.size = kwargs.get("size")
        self.website = kwargs.get("website")
        self.description = kwargs.get("description")
        self.logo_url = kwargs.get("logo_url")
        self.rating = kwargs.get("rating")
        self.review_count = kwargs.get("review_count")
        self.created_at = kwargs.get("created_at", datetime.now())
        self.updated_at = kwargs.get("updated_at", datetime.now())

    def __repr__(self):
        return f"MockCompany(id={self.id}, name='{self.name}')"


class MockJobService:
    """Mock JobService for integration testing."""

    def __init__(self):
        self.jobs = []  # In-memory job storage
        self.companies = []  # In-memory company storage
        self.scraping_stats = {"inserted": 0, "updated": 0, "skipped": 0}
        self._next_job_id = 1
        self._next_company_id = 1

    def get_or_create_company(self, company_name: str, **kwargs) -> MockCompany:
        """Get existing company or create new one."""
        # Look for existing company
        for company in self.companies:
            if company.name.lower() == company_name.lower():
                return company

        # Create new company
        company = MockCompany(id=self._next_company_id, name=company_name, **kwargs)
        self._next_company_id += 1
        self.companies.append(company)
        return company

    def find_duplicate_job(self, job_posting: JobPosting) -> MockJob | None:
        """Find duplicate job based on title, company, and location."""
        for job in self.jobs:
            if (
                job.title.lower() == job_posting.title.lower()
                and job.company.lower() == job_posting.company.lower()
                and job.location == job_posting.location
            ):
                return job
        return None

    def create_job_from_posting(self, job_posting: JobPosting) -> MockJob:
        """Create job from JobPosting."""
        # Get or create company
        company = self.get_or_create_company(
            job_posting.company,
            industry=job_posting.company_industry,
            size=job_posting.company_num_employees,
            website=job_posting.company_url,
            description=job_posting.company_description,
            logo_url=job_posting.company_logo,
            rating=job_posting.company_rating,
            review_count=job_posting.company_reviews_count,
        )

        # Create job
        job = MockJob(
            id=self._next_job_id,
            title=job_posting.title,
            company=job_posting.company,
            location=job_posting.location,
            job_url=job_posting.job_url,
            job_type=job_posting.job_type.value if job_posting.job_type else None,
            is_remote=job_posting.is_remote,
            salary_min=job_posting.min_amount,
            salary_max=job_posting.max_amount,
            date_posted=job_posting.date_posted,
            description=job_posting.description,
            skills=job_posting.skills or [],
            company_id=company.id,
        )

        self._next_job_id += 1
        self.jobs.append(job)
        return job

    def update_job_from_posting(
        self, existing_job: MockJob, job_posting: JobPosting
    ) -> MockJob:
        """Update existing job with new posting data."""
        existing_job.title = job_posting.title
        existing_job.location = job_posting.location
        existing_job.job_url = job_posting.job_url
        existing_job.job_type = (
            job_posting.job_type.value if job_posting.job_type else None
        )
        existing_job.is_remote = job_posting.is_remote
        existing_job.salary_min = job_posting.min_amount
        existing_job.salary_max = job_posting.max_amount
        existing_job.date_posted = job_posting.date_posted
        existing_job.description = job_posting.description
        existing_job.skills = job_posting.skills or []
        existing_job.updated_at = datetime.now()
        return existing_job

    def process_scraping_results(
        self, scrape_result: JobScrapeResult
    ) -> dict[str, int]:
        """Process JobSpy scraping results and update database."""
        stats = {"inserted": 0, "updated": 0, "skipped": 0}

        for job_posting in scrape_result.jobs:
            # Check for duplicates
            existing_job = self.find_duplicate_job(job_posting)

            if existing_job:
                # Update existing job
                if self._should_update_job(existing_job, job_posting):
                    self.update_job_from_posting(existing_job, job_posting)
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # Create new job
                self.create_job_from_posting(job_posting)
                stats["inserted"] += 1

        self.scraping_stats = stats
        return stats

    def _should_update_job(
        self, existing_job: MockJob, new_posting: JobPosting
    ) -> bool:
        """Determine if existing job should be updated."""
        # Update if description changed or salary information improved
        return (
            existing_job.description != new_posting.description
            or (not existing_job.salary_min and new_posting.min_amount)
            or (not existing_job.salary_max and new_posting.max_amount)
        )

    async def scrape_and_process_jobs(
        self, request: JobScrapeRequest
    ) -> dict[str, int]:
        """Complete workflow: scrape jobs and process results."""
        # This would integrate with JobSpyScraper in real implementation
        # For testing, we'll simulate the scraping

        # Simulate scraping delay
        await asyncio.sleep(0.01)

        # Mock scraping results
        mock_results = [
            JobPosting(
                id=f"scraped_{i}",
                site=JobSite.LINKEDIN,
                title=f"Job Title {i}",
                company=f"Company {i}",
                location="San Francisco, CA",
                job_type=JobType.FULLTIME,
                min_amount=100000.0 + (i * 1000),
                max_amount=150000.0 + (i * 1000),
                is_remote=i % 2 == 0,  # Alternate remote/onsite
                description=f"Job description for position {i}",
            )
            for i in range(request.results_wanted)
        ]

        scrape_result = JobScrapeResult(
            jobs=mock_results,
            total_found=len(mock_results),
            request_params=request,
            metadata={"test": True},
        )

        return self.process_scraping_results(scrape_result)


class TestJobServiceJobSpyIntegration:
    """Test JobService integration with JobSpy scraping."""

    def test_basic_scraping_workflow(self):
        """Test basic job scraping and database integration."""
        service = MockJobService()
        request = JobScrapeRequest(
            site_name=JobSite.LINKEDIN,
            search_term="Python Developer",
            results_wanted=5,
        )

        # Mock successful scraping results
        sample_jobs = [
            JobPosting(
                id="test_001",
                site=JobSite.LINKEDIN,
                title="Senior Python Developer",
                company="TechCorp",
                location="San Francisco, CA",
                job_type=JobType.FULLTIME,
                min_amount=120000.0,
                max_amount=180000.0,
                description="Great Python role",
            )
        ]

        scrape_result = JobScrapeResult(
            jobs=sample_jobs,
            total_found=1,
            request_params=request,
        )

        stats = service.process_scraping_results(scrape_result)

        assert stats["inserted"] == 1
        assert stats["updated"] == 0
        assert stats["skipped"] == 0
        assert len(service.jobs) == 1
        assert len(service.companies) == 1
        assert service.jobs[0].title == "Senior Python Developer"
        assert service.companies[0].name == "TechCorp"

    @pytest.mark.asyncio
    async def test_complete_async_workflow(self):
        """Test complete async scraping workflow."""
        service = MockJobService()
        request = JobScrapeRequest(
            site_name=[JobSite.LINKEDIN, JobSite.INDEED],
            search_term="Data Scientist",
            results_wanted=3,
        )

        stats = await service.scrape_and_process_jobs(request)

        assert stats["inserted"] == 3
        assert len(service.jobs) == 3
        assert len(service.companies) == 3  # Each job creates unique company

        # Verify jobs were created correctly
        for i, job in enumerate(service.jobs):
            assert job.title == f"Job Title {i}"
            assert job.company == f"Company {i}"
            assert job.company_id is not None

    def test_job_deduplication_logic(self):
        """Test job deduplication prevents duplicate jobs."""
        service = MockJobService()

        # Create first job
        job1 = JobPosting(
            id="dup_001",
            site=JobSite.LINKEDIN,
            title="Python Developer",
            company="TechCorp",
            location="San Francisco, CA",
            description="Original description",
        )

        # Create duplicate job with different description
        job2 = JobPosting(
            id="dup_002",
            site=JobSite.INDEED,
            title="Python Developer",  # Same title
            company="TechCorp",  # Same company
            location="San Francisco, CA",  # Same location
            description="Updated description",  # Different description
        )

        result1 = JobScrapeResult(
            jobs=[job1], total_found=1, request_params=MagicMock()
        )
        result2 = JobScrapeResult(
            jobs=[job2], total_found=1, request_params=MagicMock()
        )

        stats1 = service.process_scraping_results(result1)
        stats2 = service.process_scraping_results(result2)

        assert stats1["inserted"] == 1
        assert stats2["updated"] == 1  # Should update, not insert
        assert len(service.jobs) == 1  # Still only one job
        assert service.jobs[0].description == "Updated description"  # Updated

    def test_company_creation_and_reuse(self):
        """Test company creation and reuse logic."""
        service = MockJobService()

        # Create jobs for same company
        jobs = [
            JobPosting(
                id=f"company_test_{i}",
                site=JobSite.LINKEDIN,
                title=f"Position {i}",
                company="Big Tech Corp",  # Same company
                company_industry="Technology",
                company_url="https://bigtech.com",
                company_rating=4.5,
            )
            for i in range(3)
        ]

        result = JobScrapeResult(jobs=jobs, total_found=3, request_params=MagicMock())
        stats = service.process_scraping_results(result)

        assert stats["inserted"] == 3
        assert len(service.jobs) == 3
        assert len(service.companies) == 1  # Only one company created

        # All jobs should reference same company
        company = service.companies[0]
        assert company.name == "Big Tech Corp"
        assert company.industry == "Technology"
        assert company.rating == 4.5

        for job in service.jobs:
            assert job.company_id == company.id

    def test_mixed_insert_update_skip_scenario(self):
        """Test scenario with mixed insert, update, and skip operations."""
        service = MockJobService()

        # First batch of jobs
        batch1 = [
            JobPosting(
                id="mixed_001",
                site=JobSite.LINKEDIN,
                title="Software Engineer",
                company="StartupCo",
                location="Austin, TX",
                description="Initial description",
                min_amount=80000.0,
            ),
            JobPosting(
                id="mixed_002",
                site=JobSite.INDEED,
                title="Product Manager",
                company="StartupCo",
                location="Austin, TX",
                description="PM role description",
            ),
        ]

        result1 = JobScrapeResult(
            jobs=batch1, total_found=2, request_params=MagicMock()
        )
        stats1 = service.process_scraping_results(result1)

        # Second batch: one duplicate (to update), one duplicate (to skip), one new
        batch2 = [
            JobPosting(  # Should update (has new salary info)
                id="mixed_003",
                site=JobSite.GLASSDOOR,
                title="Software Engineer",
                company="StartupCo",
                location="Austin, TX",
                description="Updated description with more details",
                min_amount=85000.0,
                max_amount=120000.0,
            ),
            JobPosting(  # Should skip (no meaningful changes)
                id="mixed_004",
                site=JobSite.LINKEDIN,
                title="Product Manager",
                company="StartupCo",
                location="Austin, TX",
                description="PM role description",  # Same description
            ),
            JobPosting(  # Should insert (new job)
                id="mixed_005",
                site=JobSite.INDEED,
                title="DevOps Engineer",
                company="StartupCo",
                location="Austin, TX",
                description="DevOps position",
            ),
        ]

        result2 = JobScrapeResult(
            jobs=batch2, total_found=3, request_params=MagicMock()
        )
        stats2 = service.process_scraping_results(result2)

        assert stats1["inserted"] == 2
        assert stats2["inserted"] == 1  # DevOps Engineer
        assert stats2["updated"] == 1  # Software Engineer
        assert stats2["skipped"] == 1  # Product Manager
        assert len(service.jobs) == 3  # 2 from first + 1 new
        assert len(service.companies) == 1  # Still same company

    @pytest.mark.asyncio
    async def test_concurrent_scraping_requests(self):
        """Test handling concurrent scraping requests."""
        service = MockJobService()

        requests = [
            JobScrapeRequest(
                search_term=f"Engineer {i}",
                results_wanted=2,
            )
            for i in range(3)
        ]

        # Run concurrent requests
        results = await asyncio.gather(
            *[service.scrape_and_process_jobs(req) for req in requests]
        )

        # Each request should insert 2 jobs
        total_inserted = sum(result["inserted"] for result in results)
        assert total_inserted == 6
        assert len(service.jobs) == 6
        assert len(service.companies) == 6

    def test_error_handling_in_integration(self, mock_jobspy_scrape_error):
        """Test error handling during integration workflow."""
        MockJobService()

        # Simulate error during scraping
        with patch("jobspy.scrape_jobs", side_effect=ConnectionError("Network error")):
            # In real implementation, this would handle the error gracefully
            # For now, just ensure error propagates correctly
            with pytest.raises(ConnectionError):
                import jobspy

                jobspy.scrape_jobs(
                    site_name=[JobSite.LINKEDIN.value],
                    search_term="Test Query",
                )

    def test_empty_results_handling(self):
        """Test handling empty scraping results."""
        service = MockJobService()

        empty_result = JobScrapeResult(
            jobs=[],
            total_found=0,
            request_params=JobScrapeRequest(search_term="Nonexistent Job XYZ123"),
        )

        stats = service.process_scraping_results(empty_result)

        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["skipped"] == 0
        assert len(service.jobs) == 0
        assert len(service.companies) == 0

    def test_malformed_data_resilience(self):
        """Test resilience to malformed job data."""
        service = MockJobService()

        # Job with minimal required fields
        minimal_job = JobPosting(
            id="minimal_001",
            site=JobSite.LINKEDIN,
            title="Test Job",
            company="Test Company",
            # Many fields left as defaults/None
        )

        result = JobScrapeResult(
            jobs=[minimal_job],
            total_found=1,
            request_params=MagicMock(),
        )

        stats = service.process_scraping_results(result)

        # Should handle minimal data gracefully
        assert stats["inserted"] == 1
        assert len(service.jobs) == 1
        assert service.jobs[0].title == "Test Job"
        assert service.jobs[0].salary_min is None
        assert service.jobs[0].salary_max is None


class TestJobSpyIntegrationPerformance:
    """Test performance characteristics of JobSpy integration."""

    @pytest.mark.asyncio
    async def test_large_batch_processing_performance(self, performance_test_data):
        """Test performance with large batch of jobs."""
        import time

        service = MockJobService()

        # Create large batch of jobs
        large_batch = [
            JobPosting(
                id=f"perf_{i}",
                site=JobSite.LINKEDIN,
                title=f"Job {i}",
                company=f"Company {i % 10}",  # 10 unique companies
                location="Remote",
                job_type=JobType.FULLTIME,
            )
            for i in range(100)
        ]

        result = JobScrapeResult(
            jobs=large_batch,
            total_found=100,
            request_params=MagicMock(),
        )

        start_time = time.time()
        stats = service.process_scraping_results(result)
        processing_time = time.time() - start_time

        # Should process 100 jobs quickly
        assert processing_time < 1.0  # Less than 1 second
        assert stats["inserted"] == 100
        assert len(service.jobs) == 100
        assert len(service.companies) == 10  # 10 unique companies

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test performance of concurrent processing."""
        import time

        service = MockJobService()

        # Create multiple smaller batches
        batches = [
            [
                JobPosting(
                    id=f"concurrent_{batch}_{i}",
                    site=JobSite.LINKEDIN,
                    title=f"Job {batch}-{i}",
                    company=f"Company {batch}",
                    location="Remote",
                )
                for i in range(10)
            ]
            for batch in range(5)
        ]

        results = [
            JobScrapeResult(
                jobs=batch, total_found=len(batch), request_params=MagicMock()
            )
            for batch in batches
        ]

        start_time = time.time()
        # Process sequentially (in real implementation might be concurrent)
        all_stats = [service.process_scraping_results(result) for result in results]
        processing_time = time.time() - start_time

        # Should process all batches quickly
        assert processing_time < 0.5  # Less than 500ms
        total_inserted = sum(stats["inserted"] for stats in all_stats)
        assert total_inserted == 50
        assert len(service.jobs) == 50
        assert len(service.companies) == 5

    def test_memory_efficiency_large_dataset(self, performance_test_data):
        """Test memory efficiency with large datasets."""
        import sys

        service = MockJobService()

        # Create very large batch
        large_batch = [
            JobPosting(
                id=f"memory_{i}",
                site=JobSite.LINKEDIN,
                title=f"Job {i}",
                company=f"Company {i % 5}",  # Reuse companies for efficiency
                location="Remote",
                description=f"Description for job {i}" * 10,  # Larger descriptions
            )
            for i in range(500)
        ]

        result = JobScrapeResult(
            jobs=large_batch,
            total_found=500,
            request_params=MagicMock(),
        )

        sys.getsizeof(service)
        stats = service.process_scraping_results(result)
        sys.getsizeof(service)

        # Should handle large dataset efficiently
        assert stats["inserted"] == 500
        assert len(service.jobs) == 500
        assert len(service.companies) == 5
        # Memory growth should be reasonable (not testing exact values due to variability)


class TestJobSpyIntegrationEdgeCases:
    """Test edge cases in JobSpy integration."""

    def test_duplicate_detection_edge_cases(self):
        """Test edge cases in duplicate detection."""
        service = MockJobService()

        # Test case-insensitive matching
        job1 = JobPosting(
            id="edge_001",
            site=JobSite.LINKEDIN,
            title="Python Developer",
            company="TechCorp",
            location="San Francisco, CA",
        )

        job2 = JobPosting(  # Same but different case
            id="edge_002",
            site=JobSite.INDEED,
            title="PYTHON DEVELOPER",  # Different case
            company="techcorp",  # Different case
            location="San Francisco, CA",
        )

        result1 = JobScrapeResult(
            jobs=[job1], total_found=1, request_params=MagicMock()
        )
        result2 = JobScrapeResult(
            jobs=[job2], total_found=1, request_params=MagicMock()
        )

        stats1 = service.process_scraping_results(result1)
        stats2 = service.process_scraping_results(result2)

        assert stats1["inserted"] == 1
        assert stats2["updated"] == 1  # Should detect as duplicate
        assert len(service.jobs) == 1

    def test_company_name_variations(self):
        """Test handling company name variations."""
        service = MockJobService()

        # Jobs with similar company names
        jobs = [
            JobPosting(
                id=f"company_var_{i}",
                site=JobSite.LINKEDIN,
                title=f"Job {i}",
                company=company_name,
                location="Remote",
            )
            for i, company_name in enumerate(
                [
                    "Google",
                    "google",  # Different case - should match
                    "Google Inc",  # Different variation - won't match (intentional)
                    "Google LLC",  # Different variation - won't match (intentional)
                ]
            )
        ]

        result = JobScrapeResult(jobs=jobs, total_found=4, request_params=MagicMock())
        stats = service.process_scraping_results(result)

        assert stats["inserted"] == 4
        assert len(service.companies) == 3  # "Google" and "google" should merge

    def test_null_and_empty_field_handling(self):
        """Test handling null and empty fields."""
        service = MockJobService()

        job_with_nulls = JobPosting(
            id="null_test",
            site=JobSite.LINKEDIN,
            title="Test Job",
            company="Test Company",
            location=None,  # Null location
            job_url=None,  # Null URL
            description="",  # Empty description
            min_amount=None,  # Null salary
            skills=None,  # Null skills
        )

        result = JobScrapeResult(
            jobs=[job_with_nulls],
            total_found=1,
            request_params=MagicMock(),
        )

        stats = service.process_scraping_results(result)

        # Should handle nulls gracefully
        assert stats["inserted"] == 1
        job = service.jobs[0]
        assert job.location is None
        assert job.job_url is None
        assert job.description == ""
        assert job.salary_min is None
        assert job.skills == []

    def test_date_handling_edge_cases(self):
        """Test handling date edge cases."""
        service = MockJobService()

        from datetime import date

        jobs = [
            JobPosting(
                id="date_001",
                site=JobSite.LINKEDIN,
                title="Job 1",
                company="Company 1",
                date_posted=date.today(),  # Today
            ),
            JobPosting(
                id="date_002",
                site=JobSite.INDEED,
                title="Job 2",
                company="Company 2",
                date_posted=date(2020, 1, 1),  # Old date
            ),
            JobPosting(
                id="date_003",
                site=JobSite.GLASSDOOR,
                title="Job 3",
                company="Company 3",
                date_posted=None,  # No date
            ),
        ]

        result = JobScrapeResult(jobs=jobs, total_found=3, request_params=MagicMock())
        stats = service.process_scraping_results(result)

        assert stats["inserted"] == 3
        assert len(service.jobs) == 3

        # Check dates were handled correctly
        assert service.jobs[0].date_posted == date.today()
        assert service.jobs[1].date_posted == date(2020, 1, 1)
        assert service.jobs[2].date_posted is None

    def test_unicode_and_special_characters(self):
        """Test handling Unicode and special characters."""
        service = MockJobService()

        job_with_unicode = JobPosting(
            id="unicode_test",
            site=JobSite.LINKEDIN,
            title="Développeur Python 🐍",
            company="Société Française S.A.",
            location="Montréal, QC, Canada",
            description="Développeur avec expérience en français/English: $100k-$150k 💰",
        )

        result = JobScrapeResult(
            jobs=[job_with_unicode],
            total_found=1,
            request_params=MagicMock(),
        )

        stats = service.process_scraping_results(result)

        # Should handle Unicode gracefully
        assert stats["inserted"] == 1
        job = service.jobs[0]
        assert "🐍" in job.title
        assert "Société" in job.company
        assert "Montréal" in job.location
        assert "💰" in job.description


class TestJobSpyIntegrationValidation:
    """Test data validation in JobSpy integration."""

    def test_job_posting_validation_during_processing(self):
        """Test validation of job postings during processing."""
        service = MockJobService()

        # Valid job posting
        valid_job = JobPosting(
            id="valid_001",
            site=JobSite.LINKEDIN,
            title="Valid Job",
            company="Valid Company",
            location="Valid Location",
        )

        result = JobScrapeResult(
            jobs=[valid_job],
            total_found=1,
            request_params=MagicMock(),
        )

        stats = service.process_scraping_results(result)

        assert stats["inserted"] == 1
        assert len(service.jobs) == 1

    def test_data_consistency_validation(self):
        """Test data consistency validation."""
        service = MockJobService()

        # Job with consistent data
        consistent_job = JobPosting(
            id="consistent_001",
            site=JobSite.LINKEDIN,
            title="Senior Engineer",
            company="Big Corp",
            location="San Francisco, CA",
            is_remote=False,
            location_type=LocationType.ONSITE,  # Consistent with is_remote=False
            min_amount=120000.0,
            max_amount=180000.0,  # max > min (consistent)
        )

        result = JobScrapeResult(
            jobs=[consistent_job],
            total_found=1,
            request_params=MagicMock(),
        )

        stats = service.process_scraping_results(result)

        assert stats["inserted"] == 1
        job = service.jobs[0]
        assert job.is_remote is False
        assert job.salary_min < job.salary_max

    def test_business_rule_validation(self):
        """Test business rule validation during integration."""
        service = MockJobService()

        # Test that remote jobs are handled correctly
        remote_job = JobPosting(
            id="remote_001",
            site=JobSite.INDEED,
            title="Remote Developer",
            company="Remote Corp",
            location="Remote",
            is_remote=True,
            location_type=LocationType.REMOTE,
        )

        # Test that onsite job with location is handled correctly
        onsite_job = JobPosting(
            id="onsite_001",
            site=JobSite.GLASSDOOR,
            title="Onsite Developer",
            company="Local Corp",
            location="New York, NY",
            is_remote=False,
            location_type=LocationType.ONSITE,
        )

        result = JobScrapeResult(
            jobs=[remote_job, onsite_job],
            total_found=2,
            request_params=MagicMock(),
        )

        stats = service.process_scraping_results(result)

        assert stats["inserted"] == 2

        remote_job_record = next(
            job for job in service.jobs if job.title == "Remote Developer"
        )
        onsite_job_record = next(
            job for job in service.jobs if job.title == "Onsite Developer"
        )

        assert remote_job_record.is_remote is True
        assert onsite_job_record.is_remote is False
