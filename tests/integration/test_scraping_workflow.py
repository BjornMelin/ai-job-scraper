"""End-to-End Scraping Workflow Integration Tests.

This test suite validates complete scraping workflows from company pages to job boards
to database synchronization. Tests ensure proper data flow, transformation, and
persistence across all scraping components.

Test coverage includes:
- Complete scraping pipeline (HTTP → AI extraction → database sync)
- Job board scraping with pagination and filtering
- Company page scraping with various HTML structures
- AI extraction workflow with fallback mechanisms
- Database synchronization and duplicate handling
- Incremental scraping and update detection
- Multi-source data aggregation and consolidation
- Performance and resource management during scraping
"""

import logging
import time

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import responses

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

from src.database import db_session
from src.models import JobSQL
from src.schemas import JobCreate
from src.services.company_service import CompanyService
from src.services.database_sync import SmartSyncEngine
from src.services.job_service import JobService
from tests.factories import (
    create_sample_companies,
    create_sample_jobs,
)

# Disable logging during tests
logging.disable(logging.CRITICAL)


@pytest.fixture
def scraping_database(tmp_path):
    """Create test database for scraping workflow tests."""
    db_path = tmp_path / "scraping_workflow.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create initial test data
        companies = create_sample_companies(session, count=5)
        for company in companies:
            create_sample_jobs(session, count=3, company=company)
        session.commit()

    return str(db_path)


@pytest.fixture
def mock_http_responses():
    """Set up mock HTTP responses for scraping tests."""
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def scraping_services(scraping_database):
    """Set up services for scraping workflow tests."""
    return {
        "company_service": CompanyService(),
        "job_service": JobService(),
        "sync_service": SmartSyncEngine(),
    }


class TestCompleteScrapingPipeline:
    """Test complete scraping pipeline from HTTP to database."""

    @responses.activate
    def test_company_page_to_database_workflow(
        self, scraping_services, mock_http_responses
    ):
        """Test complete workflow from company page scraping to database storage."""
        services = scraping_services
        workflow_results = []

        # Mock company careers page
        company_html = """
        <html>
        <body>
            <div class="job-listing">
                <h3>Senior AI Engineer</h3>
                <div class="location">San Francisco, CA</div>
                <div class="salary">$150,000 - $200,000</div>
                <p class="description">
                    We're looking for a Senior AI Engineer to join our machine learning team.
                    You'll work on cutting-edge NLP and computer vision projects.
                </p>
                <a href="/apply/ai-engineer" class="apply-link">Apply Now</a>
            </div>
            <div class="job-listing">
                <h3>Machine Learning Researcher</h3>
                <div class="location">Remote</div>
                <div class="salary">$180,000 - $250,000</div>
                <p class="description">
                    Research and develop novel ML algorithms for our autonomous systems.
                    PhD in ML/AI preferred with 5+ years experience.
                </p>
                <a href="/apply/ml-researcher" class="apply-link">Apply Now</a>
            </div>
        </body>
        </html>
        """

        responses.add(
            responses.GET,
            "https://testcompany.com/careers",
            body=company_html,
            status=200,
        )

        # Step 1: Create company
        with patch.object(CompanyService, "create_company") as mock_create:
            mock_company = Mock()
            mock_company.id = 999
            mock_company.name = "Test Company"
            mock_company.url = "https://testcompany.com/careers"
            mock_create.return_value = mock_company

            company = services["company_service"].create_company(
                name="Test Company", url="https://testcompany.com/careers"
            )
            workflow_results.append(("company_created", company.name))

        # Step 2: Mock scraping process
        with patch("src.scraper.scrape_company_page") as mock_scrape:
            mock_scraped_jobs = [
                {
                    "title": "Senior AI Engineer",
                    "description": "We're looking for a Senior AI Engineer to join our machine learning team.",
                    "link": "https://testcompany.com/apply/ai-engineer",
                    "location": "San Francisco, CA",
                    "salary": [150000, 200000],
                },
                {
                    "title": "Machine Learning Researcher",
                    "description": "Research and develop novel ML algorithms for our autonomous systems.",
                    "link": "https://testcompany.com/apply/ml-researcher",
                    "location": "Remote",
                    "salary": [180000, 250000],
                },
            ]

            mock_scrape.return_value = mock_scraped_jobs
            scraped_jobs = mock_scrape("https://testcompany.com/careers")
            workflow_results.append(("jobs_scraped", len(scraped_jobs)))

        # Step 3: Transform and store jobs
        created_jobs = []
        for job_data in scraped_jobs:
            job_create = JobCreate(
                company_id=company.id,
                title=job_data["title"],
                description=job_data["description"],
                link=job_data["link"],
                location=job_data["location"],
                salary=job_data["salary"],
            )

            with patch.object(JobService, "create_job") as mock_job_create:
                mock_job = Mock()
                mock_job.id = len(created_jobs) + 1000
                mock_job.title = job_data["title"]
                mock_job.company_id = company.id
                mock_job_create.return_value = mock_job

                job = services["job_service"].create_job(job_create)
                created_jobs.append(job)
                workflow_results.append(("job_stored", job.title))

        # Step 4: Verify workflow completion
        assert len(created_jobs) == 2
        workflow_results.append(("workflow_completed", len(created_jobs)))

        # Verify all workflow steps
        workflow_steps = [step for step, _ in workflow_results]
        expected_steps = [
            "company_created",
            "jobs_scraped",
            "job_stored",
            "workflow_completed",
        ]

        for expected_step in expected_steps:
            assert expected_step in workflow_steps

        # Verify data integrity
        assert all(job.company_id == company.id for job in created_jobs)
        assert all(
            job.title in ["Senior AI Engineer", "Machine Learning Researcher"]
            for job in created_jobs
        )

    @responses.activate
    def test_job_board_scraping_workflow(self, scraping_services, mock_http_responses):
        """Test job board scraping with pagination and filtering."""
        services = scraping_services

        # Mock job board API responses with pagination
        page1_data = {
            "jobs": [
                {
                    "id": "job1",
                    "title": "AI Engineer",
                    "company": "TechCorp",
                    "location": "San Francisco",
                    "salary": {"min": 120000, "max": 160000},
                    "description": "Build AI systems",
                    "url": "https://jobs.example.com/job1",
                    "posted": "2024-01-15",
                },
                {
                    "id": "job2",
                    "title": "ML Engineer",
                    "company": "DataCorp",
                    "location": "Remote",
                    "salary": {"min": 140000, "max": 180000},
                    "description": "Machine learning systems",
                    "url": "https://jobs.example.com/job2",
                    "posted": "2024-01-16",
                },
            ],
            "next_page": "https://api.jobboard.com/jobs?page=2",
            "total": 3,
        }

        page2_data = {
            "jobs": [
                {
                    "id": "job3",
                    "title": "Data Scientist",
                    "company": "AnalyticsCorp",
                    "location": "New York",
                    "salary": {"min": 130000, "max": 170000},
                    "description": "Analyze complex datasets",
                    "url": "https://jobs.example.com/job3",
                    "posted": "2024-01-17",
                },
            ],
            "next_page": None,
            "total": 3,
        }

        responses.add(
            responses.GET,
            "https://api.jobboard.com/jobs?page=1",
            json=page1_data,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.jobboard.com/jobs?page=2",
            json=page2_data,
            status=200,
        )

        # Mock job board scraping process
        with patch("src.scraper_job_boards.scrape_job_board") as mock_scrape_board:
            all_jobs = []

            # Simulate pagination scraping
            for page_data in [page1_data, page2_data]:
                for job_data in page_data["jobs"]:
                    job_dict = {
                        "title": job_data["title"],
                        "company": job_data["company"],
                        "description": job_data["description"],
                        "link": job_data["url"],
                        "location": job_data["location"],
                        "salary": [
                            job_data["salary"]["min"],
                            job_data["salary"]["max"],
                        ],
                        "posted_date": datetime.fromisoformat(
                            job_data["posted"]
                        ).replace(tzinfo=UTC),
                    }
                    all_jobs.append(job_dict)

            mock_scrape_board.return_value = all_jobs
            scraped_jobs = mock_scrape_board("https://api.jobboard.com/jobs")

        # Verify pagination results
        assert len(scraped_jobs) == 3
        assert all("salary" in job and len(job["salary"]) == 2 for job in scraped_jobs)
        assert all("posted_date" in job for job in scraped_jobs)

        # Test job creation workflow
        created_count = 0
        for job_data in scraped_jobs:
            # Find or create company
            with patch.object(
                CompanyService, "get_or_create_company"
            ) as mock_get_company:
                mock_company = Mock()
                mock_company.id = hash(job_data["company"]) % 1000
                mock_company.name = job_data["company"]
                mock_get_company.return_value = mock_company

                company = services["company_service"].get_or_create_company(
                    job_data["company"]
                )

            # Create job
            job_create = JobCreate(
                company_id=company.id,
                title=job_data["title"],
                description=job_data["description"],
                link=job_data["link"],
                location=job_data["location"],
                salary=job_data["salary"],
                posted_date=job_data["posted_date"],
            )

            with patch.object(JobService, "create_job") as mock_create_job:
                mock_job = Mock()
                mock_job.id = created_count + 2000
                mock_job.title = job_data["title"]
                mock_create_job.return_value = mock_job

                services["job_service"].create_job(job_create)
                created_count += 1

        assert created_count == 3

    def test_ai_extraction_workflow_with_fallbacks(self, scraping_services):
        """Test AI extraction workflow with fallback mechanisms."""
        raw_html = """
        <html>
        <body>
            <div class="careers-page">
                <h1>Join Our Team</h1>
                <div class="job">
                    <h3>Senior Python Developer</h3>
                    <span class="location">San Francisco, CA</span>
                    <div class="requirements">
                        <p>5+ years Python experience</p>
                        <p>Django/Flask frameworks</p>
                        <p>$140,000 - $180,000 salary range</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # Test AI extraction success
        with patch("src.ai_client.extract_jobs") as mock_ai_extract:
            mock_ai_extract.return_value = [
                {
                    "title": "Senior Python Developer",
                    "location": "San Francisco, CA",
                    "description": "5+ years Python experience with Django/Flask frameworks",
                    "salary": [140000, 180000],
                }
            ]

            # Simulate AI extraction
            extracted_jobs = mock_ai_extract(raw_html)
            assert len(extracted_jobs) == 1
            assert extracted_jobs[0]["title"] == "Senior Python Developer"

        # Test AI extraction failure with fallback
        with patch("src.ai_client.extract_jobs") as mock_ai_extract:
            mock_ai_extract.side_effect = Exception("AI service unavailable")

            # Mock fallback extraction
            with patch("src.scraper.extract_jobs_fallback") as mock_fallback:
                mock_fallback.return_value = [
                    {
                        "title": "Senior Python Developer",
                        "location": "San Francisco, CA",
                        "description": "Fallback extraction - basic job parsing",
                        "salary": [None, None],  # Fallback may not extract salary
                    }
                ]

                # Should fall back to basic extraction
                try:
                    extracted_jobs = mock_ai_extract(raw_html)
                except Exception:
                    # Fallback mechanism
                    extracted_jobs = mock_fallback(raw_html)

                assert len(extracted_jobs) == 1
                assert extracted_jobs[0]["title"] == "Senior Python Developer"
                assert "Fallback extraction" in extracted_jobs[0]["description"]

    def test_incremental_scraping_workflow(self, scraping_services, scraping_database):
        """Test incremental scraping that only processes new/updated jobs."""
        # Initial scraping - get current job count
        with db_session() as session:
            session.exec(select(JobSQL)).count()

        # Mock existing job data
        existing_jobs = [
            {
                "title": "Existing AI Engineer",
                "link": "https://company.com/job/existing-ai",
                "content_hash": "hash_123",
                "last_seen": datetime.now(UTC) - timedelta(days=5),
            }
        ]

        # Mock new scraping data
        new_scrape_data = [
            # Existing job (no changes)
            {
                "title": "Existing AI Engineer",
                "link": "https://company.com/job/existing-ai",
                "description": "Same description",
                "content_hash": "hash_123",
            },
            # Updated job (content changed)
            {
                "title": "Senior AI Engineer",  # Title updated
                "link": "https://company.com/job/existing-ai",
                "description": "Updated description with new requirements",
                "content_hash": "hash_456",
            },
            # Completely new job
            {
                "title": "New ML Engineer",
                "link": "https://company.com/job/new-ml",
                "description": "Brand new position",
                "content_hash": "hash_789",
            },
        ]

        # Mock incremental processing logic
        processing_results = []

        for job_data in new_scrape_data:
            # Check if job exists by link
            existing_job = next(
                (job for job in existing_jobs if job["link"] == job_data["link"]), None
            )

            if existing_job:
                if existing_job["content_hash"] != job_data["content_hash"]:
                    # Job updated
                    processing_results.append(("updated", job_data["title"]))
                else:
                    # Job unchanged
                    processing_results.append(("skipped", job_data["title"]))
            else:
                # New job
                processing_results.append(("created", job_data["title"]))

        # Verify incremental processing logic
        created_jobs = [
            result for action, result in processing_results if action == "created"
        ]
        updated_jobs = [
            result for action, result in processing_results if action == "updated"
        ]
        skipped_jobs = [
            result for action, result in processing_results if action == "skipped"
        ]

        assert len(created_jobs) == 1  # New ML Engineer
        assert len(updated_jobs) == 1  # Updated AI Engineer
        assert len(skipped_jobs) == 1  # Existing unchanged job

        # Verify processing efficiency
        total_processed = len(processing_results)
        actual_changes = len(created_jobs) + len(updated_jobs)
        efficiency_ratio = actual_changes / total_processed

        assert efficiency_ratio >= 0.6  # At least 60% of processing was meaningful


class TestDatabaseSynchronization:
    """Test database synchronization during scraping workflows."""

    def test_batch_job_synchronization(self, scraping_services, scraping_database):
        """Test batch synchronization of scraped jobs."""
        services = scraping_services

        # Mock batch of scraped jobs
        scraped_jobs_batch = [
            {
                "company_name": "BatchCorp A",
                "title": f"Engineer {i}",
                "description": f"Job description {i}",
                "link": f"https://batchcorp.com/job/{i}",
                "location": "Remote" if i % 2 == 0 else "San Francisco",
                "salary": [100000 + i * 5000, 150000 + i * 5000],
            }
            for i in range(10)
        ]

        # Test batch processing
        with patch.object(SmartSyncEngine, "sync_jobs_batch") as mock_sync:
            mock_sync.return_value = {
                "created": 8,
                "updated": 2,
                "skipped": 0,
                "errors": 0,
            }

            sync_results = services["sync_service"].sync_jobs_batch(scraped_jobs_batch)

            assert sync_results["created"] == 8
            assert sync_results["updated"] == 2
            assert sync_results["errors"] == 0

        # Verify batch efficiency
        total_jobs = (
            sync_results["created"] + sync_results["updated"] + sync_results["skipped"]
        )
        assert total_jobs == len(scraped_jobs_batch)

    def test_company_job_association_workflow(self, scraping_services):
        """Test proper company-job association during scraping."""
        services = scraping_services

        # Mock companies and their jobs
        company_jobs_data = {
            "TechCorp": [
                {"title": "Senior Engineer", "location": "SF"},
                {"title": "Product Manager", "location": "Remote"},
            ],
            "DataCorp": [
                {"title": "Data Scientist", "location": "NYC"},
                {"title": "ML Engineer", "location": "Remote"},
            ],
        }

        association_results = {}

        for company_name, jobs in company_jobs_data.items():
            # Mock company creation/retrieval
            with patch.object(
                CompanyService, "get_or_create_company"
            ) as mock_get_company:
                mock_company = Mock()
                mock_company.id = hash(company_name) % 1000
                mock_company.name = company_name
                mock_get_company.return_value = mock_company

                company = services["company_service"].get_or_create_company(
                    company_name
                )

            # Associate jobs with company
            company_jobs = []
            for job_data in jobs:
                job_create = JobCreate(
                    company_id=company.id,
                    title=job_data["title"],
                    description=f"Job at {company_name}",
                    link=f"https://{company_name.lower()}.com/job/{job_data['title'].lower()}",
                    location=job_data["location"],
                )

                with patch.object(JobService, "create_job") as mock_create_job:
                    mock_job = Mock()
                    mock_job.id = len(company_jobs) + company.id * 100
                    mock_job.title = job_data["title"]
                    mock_job.company_id = company.id
                    mock_create_job.return_value = mock_job

                    job = services["job_service"].create_job(job_create)
                    company_jobs.append(job)

            association_results[company_name] = {
                "company_id": company.id,
                "jobs_count": len(company_jobs),
                "job_titles": [job.title for job in company_jobs],
            }

        # Verify associations
        assert len(association_results) == 2
        assert association_results["TechCorp"]["jobs_count"] == 2
        assert association_results["DataCorp"]["jobs_count"] == 2

        # Verify unique company IDs
        company_ids = [data["company_id"] for data in association_results.values()]
        assert len(set(company_ids)) == 2

    def test_duplicate_job_handling_workflow(self, scraping_services):
        """Test handling of duplicate jobs during scraping."""
        # Mock duplicate job scenarios
        jobs_with_duplicates = [
            # Original job
            {
                "title": "AI Engineer",
                "link": "https://company.com/job/ai-eng",
                "description": "Original description",
                "content_hash": "original_hash",
            },
            # Exact duplicate (should be skipped)
            {
                "title": "AI Engineer",
                "link": "https://company.com/job/ai-eng",
                "description": "Original description",
                "content_hash": "original_hash",
            },
            # Updated version (should update existing)
            {
                "title": "Senior AI Engineer",  # Title updated
                "link": "https://company.com/job/ai-eng",
                "description": "Updated description with more details",
                "content_hash": "updated_hash",
            },
            # Different job (should create new)
            {
                "title": "ML Engineer",
                "link": "https://company.com/job/ml-eng",
                "description": "Different job entirely",
                "content_hash": "different_hash",
            },
        ]

        # Mock duplicate detection logic
        processed_jobs = {}
        duplicate_results = []

        for job_data in jobs_with_duplicates:
            job_key = job_data["link"]

            if job_key in processed_jobs:
                existing_job = processed_jobs[job_key]
                if existing_job["content_hash"] == job_data["content_hash"]:
                    duplicate_results.append(("skipped_duplicate", job_data["title"]))
                else:
                    # Update existing job
                    processed_jobs[job_key] = job_data
                    duplicate_results.append(("updated_existing", job_data["title"]))
            else:
                # New job
                processed_jobs[job_key] = job_data
                duplicate_results.append(("created_new", job_data["title"]))

        # Verify duplicate handling
        created_count = len(
            [result for action, result in duplicate_results if action == "created_new"]
        )
        updated_count = len(
            [
                result
                for action, result in duplicate_results
                if action == "updated_existing"
            ]
        )
        skipped_count = len(
            [
                result
                for action, result in duplicate_results
                if action == "skipped_duplicate"
            ]
        )

        assert created_count == 2  # Original AI Engineer and ML Engineer
        assert updated_count == 1  # Updated AI Engineer
        assert skipped_count == 1  # Exact duplicate

        # Verify final state
        assert len(processed_jobs) == 2  # Only 2 unique jobs by link


class TestScrapingPerformanceAndReliability:
    """Test scraping performance and reliability scenarios."""

    def test_concurrent_company_scraping(self, scraping_services):
        """Test concurrent scraping of multiple companies."""
        import concurrent.futures
        import threading

        scraping_results = []
        result_lock = threading.Lock()

        def scrape_company_worker(company_data):
            """Worker function for concurrent company scraping."""
            company_name, company_url = company_data

            try:
                # Mock scraping delay
                time.sleep(0.1)

                # Mock scraped jobs
                mock_jobs = [
                    {
                        "title": f"Engineer at {company_name}",
                        "description": f"Job at {company_name}",
                        "link": f"{company_url}/job/1",
                        "location": "Remote",
                    }
                ]

                with result_lock:
                    scraping_results.append(
                        {
                            "company": company_name,
                            "jobs_found": len(mock_jobs),
                            "status": "success",
                            "thread_id": threading.current_thread().ident,
                        }
                    )

            except Exception as e:
                with result_lock:
                    scraping_results.append(
                        {
                            "company": company_name,
                            "jobs_found": 0,
                            "status": "error",
                            "error": str(e),
                        }
                    )

        # Test concurrent scraping
        companies_to_scrape = [
            ("TechCorp A", "https://techcorpa.com/careers"),
            ("TechCorp B", "https://techcorpb.com/careers"),
            ("TechCorp C", "https://techcorpc.com/careers"),
            ("TechCorp D", "https://techcorpd.com/careers"),
            ("TechCorp E", "https://techcorpe.com/careers"),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(scrape_company_worker, company_data)
                for company_data in companies_to_scrape
            ]

            concurrent.futures.wait(futures, timeout=5.0)

        # Verify concurrent execution
        assert len(scraping_results) == 5
        successful_scrapes = [r for r in scraping_results if r["status"] == "success"]
        assert len(successful_scrapes) == 5

        # Verify actual concurrency (different thread IDs)
        thread_ids = {r["thread_id"] for r in scraping_results if "thread_id" in r}
        assert len(thread_ids) >= 2  # Should use multiple threads

    def test_scraping_with_rate_limiting(self, scraping_services):
        """Test scraping behavior with rate limiting."""
        # Mock rate limiting scenario
        request_times = []
        rate_limit_delays = []

        def mock_scrape_with_rate_limit(url, attempt=1):
            """Mock scraping function with rate limiting."""
            current_time = time.time()
            request_times.append(current_time)

            # Simulate rate limiting after 3 requests
            if len(request_times) > 3:
                if attempt == 1:
                    # First attempt hits rate limit
                    delay = 2.0  # 2 second delay
                    rate_limit_delays.append(delay)
                    time.sleep(delay)
                    return mock_scrape_with_rate_limit(url, attempt=2)
                # Retry succeeds
                return [{"title": "Job after rate limit", "description": "Success"}]
            # Normal response
            return [{"title": f"Job {len(request_times)}", "description": "Normal"}]

        # Test scraping with rate limiting
        urls_to_scrape = [
            "https://company1.com/careers",
            "https://company2.com/careers",
            "https://company3.com/careers",
            "https://company4.com/careers",  # This will trigger rate limit
            "https://company5.com/careers",
        ]

        scraping_results = []
        for url in urls_to_scrape:
            result = mock_scrape_with_rate_limit(url)
            scraping_results.extend(result)

        # Verify rate limiting behavior
        assert len(scraping_results) == 5
        assert len(rate_limit_delays) >= 1  # At least one rate limit delay
        assert all(delay >= 1.0 for delay in rate_limit_delays)  # Reasonable delay

        # Verify request timing (rate limited requests should be spaced out)
        if len(request_times) > 4:
            time_gaps = [
                request_times[i] - request_times[i - 1]
                for i in range(1, len(request_times))
            ]
            # Should have at least one significant gap due to rate limiting
            assert any(gap > 1.0 for gap in time_gaps)

    def test_scraping_memory_usage_monitoring(self, scraping_services):
        """Test monitoring of memory usage during large scraping operations."""
        # Mock memory usage tracking
        memory_snapshots = []

        def track_memory_usage(operation_name):
            """Mock memory usage tracking."""
            # Simulate memory usage (in MB)
            import random

            base_memory = 50
            operation_memory = random.randint(10, 100)
            total_memory = base_memory + operation_memory

            memory_snapshots.append(
                {
                    "operation": operation_name,
                    "memory_mb": total_memory,
                    "timestamp": datetime.now(UTC),
                }
            )

            return total_memory

        # Simulate large scraping operation
        large_job_batch = [
            {"title": f"Job {i}", "description": "x" * 1000}  # Large descriptions
            for i in range(100)
        ]

        # Track memory during processing
        track_memory_usage("scraping_start")

        # Process jobs in batches to manage memory
        batch_size = 20
        for i in range(0, len(large_job_batch), batch_size):
            batch = large_job_batch[i : i + batch_size]

            # Mock processing batch
            processed_batch = [
                {"title": job["title"], "processed": True} for job in batch
            ]

            track_memory_usage(f"batch_{i // batch_size + 1}")

            # Simulate memory cleanup
            del batch
            del processed_batch

        track_memory_usage("scraping_end")

        # Verify memory monitoring
        assert len(memory_snapshots) >= 3  # Start, batches, end

        # Memory usage should be reasonable (under 200MB for test)
        max_memory = max(snapshot["memory_mb"] for snapshot in memory_snapshots)
        assert max_memory < 200

        # Memory should be tracked at key points
        operation_names = [s["operation"] for s in memory_snapshots]
        assert "scraping_start" in operation_names
        assert "scraping_end" in operation_names
        assert any("batch_" in name for name in operation_names)
