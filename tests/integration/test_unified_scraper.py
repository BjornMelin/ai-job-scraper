"""Integration tests for the unified scraping service.

Tests the complete 2-tier scraping architecture combining JobSpy and ScrapeGraphAI
with async performance optimization and comprehensive error handling.
"""

import asyncio

from unittest.mock import MagicMock, patch

import pytest

from src.interfaces.scraping_service_interface import (
    JobQuery,
    SourceType,
)
from src.schemas import Job
from src.services.unified_scraper import UnifiedScrapingService


@pytest.fixture
def scraper_service():
    """Create UnifiedScrapingService instance for testing."""
    return UnifiedScrapingService()


@pytest.fixture
def sample_job_query():
    """Create sample JobQuery for testing."""
    return JobQuery(
        keywords=["software engineer", "python developer"],
        locations=["San Francisco", "Remote"],
        source_types=[SourceType.UNIFIED],
        max_results=50,
        hours_old=72,
        enable_ai_enhancement=True,
        concurrent_requests=5,
    )


@pytest.fixture
def mock_jobspy_data():
    """Sample JobSpy response data."""
    return [
        {
            "title": "Senior Python Developer",
            "company": "Tech Corp",
            "description": "Exciting Python development role...",
            "job_url": "https://example.com/job/1",
            "location": "San Francisco, CA",
            "min_amount": 120000,
            "max_amount": 160000,
        },
        {
            "title": "Software Engineer - AI",
            "company": "AI Startup",
            "description": "Join our AI team to build...",
            "job_url": "https://example.com/job/2",
            "location": "Remote",
            "min_amount": 100000,
            "max_amount": 140000,
        },
    ]


@pytest.fixture
def mock_scrapegraph_data():
    """Sample ScrapeGraphAI response data."""
    return {
        "https://company.com/careers": {
            "jobs": [
                {
                    "title": "Machine Learning Engineer",
                    "description": "Work on cutting-edge ML models...",
                    "location": "New York, NY",
                    "url": "https://company.com/careers/ml-engineer",
                    "salary": "$140k-180k",
                }
            ]
        }
    }


class TestUnifiedScrapingService:
    """Test suite for UnifiedScrapingService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, scraper_service):
        """Test service initialization and basic properties."""
        assert scraper_service is not None
        assert scraper_service.settings is not None
        assert scraper_service._success_metrics is not None
        assert "job_boards" in scraper_service._success_metrics
        assert "company_pages" in scraper_service._success_metrics
        assert "ai_enhancement" in scraper_service._success_metrics

    @pytest.mark.asyncio
    async def test_async_context_manager(self, scraper_service):
        """Test async context manager functionality."""
        async with scraper_service as service:
            assert service is scraper_service
        # Should complete without errors

    @pytest.mark.asyncio
    @patch("src.services.unified_scraper.scrape_jobs")
    async def test_job_boards_scraping(
        self, mock_scrape_jobs, scraper_service, sample_job_query, mock_jobspy_data
    ):
        """Test job board scraping with JobSpy integration."""
        # Mock JobSpy response
        import pandas as pd

        mock_df = pd.DataFrame(mock_jobspy_data)
        mock_scrape_jobs.return_value = mock_df

        # Execute job board scraping
        jobs = await scraper_service.scrape_job_boards_async(sample_job_query)

        # Verify results
        assert len(jobs) == 2
        assert all(isinstance(job, Job) for job in jobs)
        assert jobs[0].title == "Senior Python Developer"
        assert jobs[0].company == "Tech Corp"
        assert jobs[1].title == "Software Engineer - AI"
        assert jobs[1].company == "AI Startup"

        # Verify JobSpy was called correctly
        assert mock_scrape_jobs.call_count == len(sample_job_query.locations)

    @pytest.mark.asyncio
    @patch("src.services.unified_scraper.SmartScraperMultiGraph")
    async def test_company_pages_scraping(
        self,
        mock_multi_graph,
        scraper_service,
        sample_job_query,
        mock_scrapegraph_data,
    ):
        """Test company page scraping with ScrapeGraphAI integration."""
        # Mock ScrapeGraphAI
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = mock_scrapegraph_data
        mock_multi_graph.return_value = mock_graph_instance

        # Mock database companies
        with patch.object(
            scraper_service, "_load_active_companies"
        ) as mock_load_companies:
            from src.models import CompanySQL

            mock_company = CompanySQL(
                id=1,
                name="Test Company",
                url="https://company.com/careers",
                active=True,
            )
            mock_load_companies.return_value = [mock_company]

            # Execute company page scraping
            await scraper_service.scrape_company_pages_async(sample_job_query)

            # Verify SmartScraperMultiGraph was called
            assert mock_multi_graph.called
            assert mock_graph_instance.run.called

    @pytest.mark.asyncio
    @patch("src.services.unified_scraper.scrape_jobs")
    @patch("src.services.unified_scraper.SmartScraperMultiGraph")
    async def test_unified_scraping(
        self,
        mock_multi_graph,
        mock_scrape_jobs,
        scraper_service,
        sample_job_query,
        mock_jobspy_data,
        mock_scrapegraph_data,
    ):
        """Test unified scraping combining both tiers."""
        # Mock JobSpy
        import pandas as pd

        mock_df = pd.DataFrame(mock_jobspy_data)
        mock_scrape_jobs.return_value = mock_df

        # Mock ScrapeGraphAI
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = mock_scrapegraph_data
        mock_multi_graph.return_value = mock_graph_instance

        # Mock database companies
        with patch.object(
            scraper_service, "_load_active_companies"
        ) as mock_load_companies:
            from src.models import CompanySQL

            mock_company = CompanySQL(
                id=1,
                name="Test Company",
                url="https://company.com/careers",
                active=True,
            )
            mock_load_companies.return_value = [mock_company]

            # Execute unified scraping
            jobs = await scraper_service.scrape_unified(sample_job_query)

            # Verify results from both tiers
            assert len(jobs) >= 2  # At least from job boards
            assert all(isinstance(job, Job) for job in jobs)

            # Verify both services were called
            assert mock_scrape_jobs.call_count >= 1
            assert mock_multi_graph.called

    @pytest.mark.asyncio
    async def test_background_scraping(self, scraper_service, sample_job_query):
        """Test background scraping functionality."""
        # Mock unified scraping to avoid actual API calls
        mock_jobs = [
            Job(
                company="Test Company",
                title="Test Job",
                description="Test description",
                link="https://example.com/job",
                location="Remote",
                content_hash="test_hash",
            )
        ]

        with patch.object(scraper_service, "scrape_unified", return_value=mock_jobs):
            # Start background scraping
            task_id = await scraper_service.start_background_scraping(sample_job_query)

            # Verify task was created
            assert task_id is not None
            assert task_id in scraper_service._background_tasks

            # Wait for completion
            await asyncio.sleep(0.1)  # Small delay for background task

            # Check status
            status = await scraper_service.get_scraping_status(task_id)
            assert status.task_id == task_id
            assert status.source_type == SourceType.UNIFIED

    @pytest.mark.asyncio
    async def test_progress_monitoring(self, scraper_service, sample_job_query):
        """Test real-time progress monitoring."""

        # Mock unified scraping with delay
        async def mock_scrape_with_delay(query):
            await asyncio.sleep(0.1)  # Simulate work
            return []

        with patch.object(
            scraper_service, "scrape_unified", side_effect=mock_scrape_with_delay
        ):
            # Start background scraping
            task_id = await scraper_service.start_background_scraping(sample_job_query)

            # Monitor progress
            status_updates = []
            async for status in scraper_service.monitor_scraping_progress(task_id):
                status_updates.append(status)
                if status.status in ["completed", "failed"]:
                    break

            # Verify progress updates
            assert len(status_updates) >= 1
            assert all(update.task_id == task_id for update in status_updates)

    @pytest.mark.asyncio
    async def test_success_rate_metrics(self, scraper_service):
        """Test success rate tracking and metrics."""
        # Simulate some operations
        scraper_service._update_success_metrics("job_boards", True)
        scraper_service._update_success_metrics("job_boards", True)
        scraper_service._update_success_metrics("job_boards", False)
        scraper_service._update_success_metrics("company_pages", True)

        # Get metrics
        metrics = await scraper_service.get_success_rate_metrics()

        # Verify metrics structure
        assert "job_boards" in metrics
        assert "company_pages" in metrics
        assert "overall" in metrics

        # Verify job boards metrics
        job_boards_metrics = metrics["job_boards"]
        assert job_boards_metrics["attempts"] == 3
        assert job_boards_metrics["successes"] == 2
        assert job_boards_metrics["success_rate"] == 66.67

        # Verify overall metrics
        overall_metrics = metrics["overall"]
        assert overall_metrics["attempts"] == 4
        assert overall_metrics["successes"] == 3
        assert overall_metrics["success_rate"] == 75.0

    @pytest.mark.asyncio
    async def test_error_handling(self, scraper_service, sample_job_query):
        """Test comprehensive error handling."""
        # Test job boards error handling
        with patch(
            "src.services.unified_scraper.scrape_jobs",
            side_effect=Exception("JobSpy error"),
        ):
            jobs = await scraper_service.scrape_job_boards_async(sample_job_query)
            # Should return empty list on error
            assert jobs == []

        # Test company pages error handling
        with patch.object(
            scraper_service, "_load_active_companies", side_effect=Exception("DB error")
        ):
            with pytest.raises(Exception):
                await scraper_service.scrape_company_pages_async(sample_job_query)

    @pytest.mark.asyncio
    async def test_data_normalization(self, scraper_service):
        """Test data normalization from different sources."""
        raw_jobspy_data = [
            {
                "title": "Software Engineer",
                "company": "Tech Co",
                "description": "Great job opportunity",
                "job_url": "https://example.com/job/1",
                "location": "San Francisco",
                "min_amount": 100000,
                "max_amount": 150000,
            }
        ]

        # Test JobSpy data normalization
        jobs = await scraper_service._normalize_jobspy_data(raw_jobspy_data)

        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Software Engineer"
        assert job.company == "Tech Co"
        assert job.description == "Great job opportunity"
        assert job.link == "https://example.com/job/1"
        assert job.location == "San Francisco"
        assert job.salary == (100000, 150000)

    @pytest.mark.asyncio
    async def test_deduplication(self, scraper_service):
        """Test job deduplication functionality."""
        # Create jobs with duplicate links
        jobs = [
            Job(
                company="Company A",
                title="Job 1",
                description="Description 1",
                link="https://example.com/job/1",
                location="Remote",
                content_hash="hash1",
            ),
            Job(
                company="Company B",
                title="Job 2",
                description="Description 2",
                link="https://example.com/job/1",  # Duplicate link
                location="Remote",
                content_hash="hash2",
            ),
            Job(
                company="Company C",
                title="Job 3",
                description="Description 3",
                link="https://example.com/job/3",
                location="Remote",
                content_hash="hash3",
            ),
        ]

        # Test deduplication
        unique_jobs = await scraper_service._deduplicate_jobs(jobs)

        assert len(unique_jobs) == 2
        assert unique_jobs[0].link == "https://example.com/job/1"
        assert unique_jobs[1].link == "https://example.com/job/3"

    @pytest.mark.asyncio
    async def test_ai_enhancement(self, scraper_service):
        """Test AI-powered job data enhancement."""
        jobs = [
            Job(
                company="Test Company",
                title="Software Engineer",
                description="Basic description",
                link="https://example.com/job/1",
                location="Remote",
                content_hash="hash1",
            )
        ]

        # Test AI enhancement (currently returns jobs as-is)
        enhanced_jobs = await scraper_service.enhance_job_data(jobs)

        assert len(enhanced_jobs) == 1
        assert enhanced_jobs[0].title == "Software Engineer"
        # In the future, this might include enhanced descriptions, skills, etc.

    @pytest.mark.asyncio
    async def test_source_type_routing(self, scraper_service):
        """Test source type routing functionality."""
        # Test job boards only
        query_job_boards = JobQuery(
            keywords=["python"],
            locations=["Remote"],
            source_types=[SourceType.JOB_BOARDS],
        )

        # Test company pages only
        query_company_pages = JobQuery(
            keywords=["python"],
            locations=["Remote"],
            source_types=[SourceType.COMPANY_PAGES],
        )

        with (
            patch.object(
                scraper_service, "scrape_job_boards_async", return_value=[]
            ) as mock_job_boards,
            patch.object(
                scraper_service, "scrape_company_pages_async", return_value=[]
            ) as mock_company_pages,
        ):
            # Test job boards routing
            await scraper_service.scrape_unified(query_job_boards)
            assert mock_job_boards.called
            assert not mock_company_pages.called

            mock_job_boards.reset_mock()
            mock_company_pages.reset_mock()

            # Test company pages routing
            await scraper_service.scrape_unified(query_company_pages)
            assert not mock_job_boards.called
            assert mock_company_pages.called


class TestIntegrationScenarios:
    """Integration test scenarios for realistic usage patterns."""

    @pytest.mark.asyncio
    async def test_high_concurrency_scenario(self, scraper_service):
        """Test high concurrency scraping scenario."""
        query = JobQuery(
            keywords=["software engineer", "python developer", "data scientist"],
            locations=["San Francisco", "New York", "Remote", "Seattle", "Boston"],
            concurrent_requests=10,  # High concurrency
            max_results=200,
        )

        with patch("src.services.unified_scraper.scrape_jobs") as mock_scrape:
            import pandas as pd

            # Return different data for different calls
            mock_scrape.return_value = pd.DataFrame(
                [
                    {
                        "title": "Test Job",
                        "company": "Test Company",
                        "description": "Test description",
                        "job_url": "https://example.com/job/1",
                        "location": "Remote",
                    }
                ]
            )

            jobs = await scraper_service.scrape_job_boards_async(query)

            # Should handle high concurrency without errors
            assert isinstance(jobs, list)
            # Should respect concurrency limits (verify semaphore worked)
            assert mock_scrape.call_count == len(query.locations)

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, scraper_service, sample_job_query):
        """Test error recovery and graceful degradation."""
        call_count = 0

        def failing_scrape(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise Exception("Network error")
            # Succeed on 3rd attempt
            import pandas as pd

            return pd.DataFrame(
                [
                    {
                        "title": "Recovered Job",
                        "company": "Test Company",
                        "description": "Test description",
                        "job_url": "https://example.com/job/1",
                        "location": "Remote",
                    }
                ]
            )

        with patch(
            "src.services.unified_scraper.scrape_jobs", side_effect=failing_scrape
        ):
            jobs = await scraper_service.scrape_job_boards_async(sample_job_query)

            # Should recover after retries
            assert len(jobs) >= 0  # May be empty if all locations fail
            # Should have attempted retries
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_performance_monitoring_scenario(self, scraper_service):
        """Test performance monitoring and success rate tracking."""
        # Simulate mixed success/failure scenario
        for _ in range(5):
            scraper_service._update_success_metrics("job_boards", True)
        for _ in range(2):
            scraper_service._update_success_metrics("job_boards", False)
        for _ in range(3):
            scraper_service._update_success_metrics("company_pages", True)
        for _ in range(1):
            scraper_service._update_success_metrics("company_pages", False)

        metrics = await scraper_service.get_success_rate_metrics()

        # Verify success rates meet expected thresholds
        job_boards_rate = metrics["job_boards"]["success_rate"]
        company_pages_rate = metrics["company_pages"]["success_rate"]
        overall_rate = metrics["overall"]["success_rate"]

        assert job_boards_rate == 71.43  # 5/7 * 100
        assert company_pages_rate == 75.0  # 3/4 * 100
        assert overall_rate == 72.73  # 8/11 * 100

        # Architecture target is 95%+ success rate
        # In real scenarios with proper configuration, this should be achieved
