"""Integration tests for session isolation and DetachedInstanceError prevention.

This module contains integration tests that verify the DetachedInstanceError is
completely resolved by testing:
1. Session lifecycle management with DTOs
2. Cross-layer data flow scenarios
3. Concurrent session access patterns
4. Real-world serialization scenarios that previously caused DetachedInstanceError

Note: This module intentionally uses loops and conditionals in tests because these are
INTEGRATION TESTS that need to verify real-world scenarios like:
- Pagination with multiple pages
- Bulk operations with multiple entities
- Concurrent access patterns with multiple threads
- Sequential session creation and management

Unlike unit tests where loops should be avoided, integration tests require loops
to test realistic usage patterns and data volumes. The loops here are essential
for testing the system's behavior under real conditions.
"""

import contextlib
import json
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy.orm.exc import DetachedInstanceError
from sqlmodel import Session

from src.models import CompanySQL, JobSQL
from src.schemas import Company, Job


class TestSessionLifecycle:
    """Test session lifecycle scenarios that previously caused DetachedInstanceError."""

    def test_dto_access_after_session_close(self, engine):
        """Test that DTOs remain accessible after the database session closes."""
        # Create data in one session
        with Session(engine) as session:
            company = CompanySQL(
                name="Session Test Co",
                url="https://sessiontest.com/careers",
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            job = JobSQL(
                company_id=company.id,
                title="Session Test Engineer",
                description="Testing session lifecycle",
                link="https://sessiontest.com/jobs/123",
                location="Test City",
                content_hash="session123",
                salary=(80000, 120000),
            )
            session.add(job)
            session.commit()
            session.refresh(job)

            # Convert to DTOs while session is active
            company_dto = Company.model_validate(company)
            job_dto = Job.model_validate(
                {
                    **job.model_dump(),
                    # job.company is a computed field returning string, not relationship
                    "company": job.company,  # Returns company name as string
                },
            )

        # Session is now closed - verify DTOs still work
        assert company_dto.name == "Session Test Co"
        assert company_dto.url == "https://sessiontest.com/careers"

        assert job_dto.title == "Session Test Engineer"
        assert job_dto.company == "Session Test Co"
        assert job_dto.salary == (80000, 120000)

        # Test JSON serialization after session close
        company_json = company_dto.model_dump_json()
        job_json = job_dto.model_dump_json()

        # Verify serialization contains expected data
        company_data = json.loads(company_json)
        job_data = json.loads(job_json)

        assert company_data["name"] == "Session Test Co"
        assert job_data["company"] == "Session Test Co"
        assert job_data["salary"] == [80000, 120000]

    def test_multiple_session_contexts(self, engine):
        """Test DTOs work correctly across multiple session contexts."""
        # Create company in first session
        with Session(engine) as session1:
            company = CompanySQL(
                name="Multi Session Co",
                url="https://multisession.com/careers",
            )
            session1.add(company)
            session1.commit()
            session1.refresh(company)
            company_id = company.id
            company_dto = Company.model_validate(company)

        # Access company DTO after first session closes
        assert company_dto.name == "Multi Session Co"

        # Create job in second session, using company from first session
        with Session(engine) as session2:
            job = JobSQL(
                company_id=company_id,
                title="Multi Session Engineer",
                description="Testing multi-session access",
                link="https://multisession.com/jobs/123",
                location="Multi City",
                content_hash="multi123",
            )
            session2.add(job)
            session2.commit()
            session2.refresh(job)

            # Get company relationship for DTO conversion
            job_dto = Job.model_validate({**job.model_dump(), "company": job.company})

        # Both DTOs should work after both sessions are closed
        assert company_dto.name == "Multi Session Co"
        assert job_dto.title == "Multi Session Engineer"
        assert job_dto.company == "Multi Session Co"

    def test_dto_survives_sqlmodel_object_access_error(self, engine):
        """Test DTOs remain functional when SQLModel objects cause errors."""
        job_dto = None
        sql_job = None

        # Create objects and DTOs in session scope
        with Session(engine) as session:
            company = CompanySQL(
                name="Detached Test Co",
                url="https://detached.com/careers",
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            sql_job = JobSQL(
                company_id=company.id,
                title="Detached Test Engineer",
                description="Testing detached instance handling",
                link="https://detached.com/jobs/123",
                location="Detached City",
                content_hash="detached123",
            )
            session.add(sql_job)
            session.commit()
            session.refresh(sql_job)

            # Create DTO while relationship is accessible
            job_dto = Job.model_validate(
                {**sql_job.model_dump(), "company": sql_job.company},
            )

        # Session is now closed
        # SQLModel object should cause DetachedInstanceError when accessing relationship
        with contextlib.suppress(DetachedInstanceError):
            _ = sql_job.company  # This might fail with DetachedInstanceError
            # If it doesn't fail, that's fine - SQLModel might handle this gracefully

        # But the DTO should work fine
        assert job_dto.title == "Detached Test Engineer"
        assert job_dto.company == "Detached Test Co"  # Company name preserved as string
        assert job_dto.location == "Detached City"

        # DTO serialization should also work
        serialized = job_dto.model_dump_json()
        assert "Detached Test Co" in serialized

    def test_lazy_loading_prevention(self, engine):
        """Test DTOs prevent lazy loading issues by not maintaining relationships."""
        # Create test data
        with Session(engine) as session:
            company = CompanySQL(
                name="Lazy Load Co",
                url="https://lazyload.com/careers",
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            jobs = []
            for i in range(3):
                job = JobSQL(
                    company_id=company.id,
                    title=f"Engineer {i}",
                    description=f"Job {i} description",
                    link=f"https://lazyload.com/jobs/{i}",
                    location="Lazy City",
                    content_hash=f"lazy{i}",
                )
                jobs.append(job)
                session.add(job)

            session.commit()
            for job in jobs:
                session.refresh(job)

            # Convert all jobs to DTOs
            job_dtos = []
            for job in jobs:
                dto = Job.model_validate({**job.model_dump(), "company": job.company})
                job_dtos.append(dto)

        # Session closed - verify all DTOs work without lazy loading
        for i, dto in enumerate(job_dtos):
            assert dto.title == f"Engineer {i}"
            assert dto.company == "Lazy Load Co"  # No lazy loading needed

            # Serialization should work
            json_data = dto.model_dump(mode="json")
            assert json_data["company"] == "Lazy Load Co"


class TestCrossLayerDataFlow:
    """Test data flow scenarios across service and UI layers."""

    def test_service_to_ui_data_flow_simulation(self, engine):
        """Simulate service to UI layer data flow without DetachedInstanceError."""

        # Simulate service layer operations
        def service_get_jobs_with_companies():
            """Simulate service layer method that returns DTOs."""
            with Session(engine) as session:
                # Create test data
                company = CompanySQL(
                    name="Service Co",
                    url="https://service.com/careers",
                )
                session.add(company)
                session.commit()
                session.refresh(company)

                job = JobSQL(
                    company_id=company.id,
                    title="Service Engineer",
                    description="Service layer testing",
                    link="https://service.com/jobs/123",
                    location="Service City",
                    content_hash="service123",
                    salary=(90000, 130000),
                )
                session.add(job)
                session.commit()
                session.refresh(job)

                # Service layer returns DTOs, not SQLModel objects
                return [
                    Job.model_validate({**job.model_dump(), "company": job.company}),
                ]

        # Simulate UI layer receiving data
        def ui_render_jobs(job_dtos):
            """Simulate UI layer rendering job DTOs."""
            rendered_data = []
            for job_dto in job_dtos:
                # UI can freely access DTO properties without session concerns
                ui_job = {
                    "title": job_dto.title,
                    "company": job_dto.company,
                    "location": job_dto.location,
                    "salary_range": f"${job_dto.salary[0]:,} - ${job_dto.salary[1]:,}"
                    if job_dto.salary[0] and job_dto.salary[1]
                    else "Not specified",
                    "status": job_dto.status,
                }
                rendered_data.append(ui_job)
            return rendered_data

        # Execute the flow
        job_dtos = service_get_jobs_with_companies()
        ui_data = ui_render_jobs(job_dtos)

        # Verify UI data is correct
        assert len(ui_data) == 1
        ui_job = ui_data[0]
        assert ui_job["title"] == "Service Engineer"
        assert ui_job["company"] == "Service Co"
        assert ui_job["location"] == "Service City"
        assert ui_job["salary_range"] == "$90,000 - $130,000"
        assert ui_job["status"] == "New"

    def test_complex_filtering_with_dtos(self, engine):
        """Test complex filtering that would previously cause DetachedInstanceError."""
        # Create test data
        with Session(engine) as session:
            companies = []
            for i in range(3):
                company = CompanySQL(
                    name=f"Filter Co {i}",
                    url=f"https://filter{i}.com/careers",
                    active=i % 2 == 0,  # Alternate active status
                )
                companies.append(company)
                session.add(company)
            session.commit()

            jobs = []
            for i, company in enumerate(companies):
                session.refresh(company)
                for j in range(2):
                    job = JobSQL(
                        company_id=company.id,
                        title=f"Engineer {i}-{j}",
                        description=f"Job at company {i}",
                        link=f"https://filter{i}.com/jobs/{j}",
                        location="Filter City",
                        content_hash=f"filter{i}{j}",
                        salary=(70000 + i * 10000, 110000 + i * 10000),
                        application_status="Applied" if j == 0 else "New",
                    )
                    jobs.append(job)
                    session.add(job)

            session.commit()
            for job in jobs:
                session.refresh(job)

            # Convert to DTOs while relationships are accessible
            job_dtos = []
            for job in jobs:
                dto = Job.model_validate({**job.model_dump(), "company": job.company})
                job_dtos.append(dto)

        # Session closed - perform complex filtering on DTOs
        # Filter by application status
        applied_jobs = [dto for dto in job_dtos if dto.application_status == "Applied"]
        assert len(applied_jobs) == 3

        # Filter by salary range
        high_salary_jobs = [
            dto for dto in job_dtos if dto.salary[0] and dto.salary[0] >= 80000
        ]
        assert len(high_salary_jobs) == 4  # Jobs from companies 1 and 2

        # Filter by company name pattern
        company_0_jobs = [dto for dto in job_dtos if "Co 0" in dto.company]
        assert len(company_0_jobs) == 2

        # Complex multi-criteria filtering
        filtered_jobs = [
            dto
            for dto in job_dtos
            if dto.application_status == "New"
            and dto.salary[0]
            and dto.salary[0] >= 70000
        ]
        assert len(filtered_jobs) == 3

        # Verify DTOs can still be serialized after filtering
        for dto in filtered_jobs:
            json_data = dto.model_dump_json()
            assert "Engineer" in json_data
            assert "Filter Co" in json_data


class TestConcurrentAccess:
    """Test concurrent access patterns that could expose DetachedInstanceError."""

    def test_concurrent_dto_access(self, engine):
        """Test concurrent access to DTOs from multiple threads."""
        # Create test data
        with Session(engine) as session:
            company = CompanySQL(
                name="Concurrent Co",
                url="https://concurrent.com/careers",
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            job = JobSQL(
                company_id=company.id,
                title="Concurrent Engineer",
                description="Testing concurrent access",
                link="https://concurrent.com/jobs/123",
                location="Concurrent City",
                content_hash="concurrent123",
                salary=(100000, 150000),
            )
            session.add(job)
            session.commit()
            session.refresh(job)

            # Create DTO while session is active
            job_dto = Job.model_validate({**job.model_dump(), "company": job.company})

        # Session closed - test concurrent access to DTO
        results = []
        errors = []

        def access_dto(thread_id):
            """Access DTO from multiple threads."""
            try:
                # Access various DTO properties
                title = job_dto.title
                company = job_dto.company
                salary = job_dto.salary
                json_data = job_dto.model_dump_json()

                results.append(
                    {
                        "thread_id": thread_id,
                        "title": title,
                        "company": company,
                        "salary": salary,
                        "json_length": len(json_data),
                    },
                )

                # Small delay to increase chance of concurrent access
                time.sleep(0.01)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_dto, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors in concurrent access: {errors}"
        assert len(results) == 10

        # Verify all threads got consistent data
        for result in results:
            assert result["title"] == "Concurrent Engineer"
            assert result["company"] == "Concurrent Co"
            assert result["salary"] == (100000, 150000)
            assert result["json_length"] > 0

    def test_sequential_session_creation_with_dtos(self, engine):
        """Test sequential session creation and DTO conversion (safer approach)."""
        job_dtos = []

        # Create jobs sequentially to avoid threading issues
        for thread_id in range(3):
            with Session(engine) as session:
                company = CompanySQL(
                    name=f"Sequential Co {thread_id}",
                    url=f"https://sequential{thread_id}.com/careers",
                )
                session.add(company)
                session.commit()
                session.refresh(company)

                # Create job
                job = JobSQL(
                    company_id=company.id,
                    title=f"Sequential Engineer {thread_id}",
                    description=f"Job created sequentially {thread_id}",
                    link=f"https://sequential{thread_id}.com/jobs/123",
                    location=f"Sequential City {thread_id}",
                    content_hash=f"sequential{thread_id}",
                    salary=(80000 + thread_id * 1000, 120000 + thread_id * 1000),
                )
                session.add(job)
                session.commit()
                session.refresh(job)

                # Convert to DTO before session closes
                job_dto = Job.model_validate(
                    {**job.model_dump(), "company": job.company},
                )
                job_dtos.append(job_dto)

        # Verify all DTOs were created successfully
        assert len(job_dtos) == 3

        # Verify each DTO works independently
        for i, dto in enumerate(job_dtos):
            assert f"Sequential Engineer {i}" == dto.title
            assert f"Sequential Co {i}" == dto.company
            assert dto.salary[0] == 80000 + i * 1000

            # Test serialization
            json_data = dto.model_dump_json()
            assert f"Sequential Engineer {i}" in json_data
            assert f"Sequential Co {i}" in json_data


class TestRealWorldScenarios:
    """Test real-world scenarios that previously caused DetachedInstanceError."""

    def test_pagination_with_dtos(self, engine):
        """Test pagination scenarios that previously caused DetachedInstanceError."""
        # Create large dataset
        with Session(engine) as session:
            company = CompanySQL(
                name="Pagination Co",
                url="https://pagination.com/careers",
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            # Create many jobs
            jobs = []
            for i in range(50):
                job = JobSQL(
                    company_id=company.id,
                    title=f"Paginated Engineer {i}",
                    description=f"Job number {i}",
                    link=f"https://pagination.com/jobs/{i}",
                    location="Pagination City",
                    content_hash=f"page{i:02d}",
                    salary=(70000 + i * 1000, 100000 + i * 1000),
                )
                jobs.append(job)
                session.add(job)

            session.commit()
            for job in jobs:
                session.refresh(job)

            # Convert to DTOs
            job_dtos = []
            for job in jobs:
                dto = Job.model_validate({**job.model_dump(), "company": job.company})
                job_dtos.append(dto)

        # Session closed - simulate pagination
        page_size = 10
        total_pages = len(job_dtos) // page_size

        for page in range(total_pages):
            start_idx = page * page_size
            end_idx = start_idx + page_size
            page_jobs = job_dtos[start_idx:end_idx]

            # Verify page data
            assert len(page_jobs) == page_size

            for dto in page_jobs:
                assert dto.company == "Pagination Co"
                assert "Paginated Engineer" in dto.title

                # Test serialization for each paginated item
                json_data = dto.model_dump(mode="json")
                assert json_data["company"] == "Pagination Co"

    def test_search_filtering_serialization_combo(self, engine):
        """Test search, filtering, and serialization combo that stressed old system."""
        # Create diverse test data
        companies_data = [
            ("Tech Corp", "https://tech.com/careers", True),
            ("Finance Inc", "https://finance.com/careers", False),
            ("Startup LLC", "https://startup.com/careers", True),
        ]

        jobs_data = [
            ("Senior Python Developer", "Python", "Applied"),
            ("Data Scientist", "Python", "New"),
            ("Frontend Engineer", "JavaScript", "Interview"),
            ("Backend Engineer", "Python", "Applied"),
            ("DevOps Engineer", "Infrastructure", "New"),
        ]

        with Session(engine) as session:
            # Create companies
            created_companies = []
            for name, url, active in companies_data:
                company = CompanySQL(name=name, url=url, active=active)
                session.add(company)
                created_companies.append(company)
            session.commit()

            # Create jobs
            job_dtos = []
            for i, (title, tech, status) in enumerate(jobs_data):
                company = created_companies[i % len(created_companies)]
                session.refresh(company)

                job = JobSQL(
                    company_id=company.id,
                    title=title,
                    description=f"Looking for {tech} expertise",
                    link=f"https://example.com/jobs/{i}",
                    location="Remote" if i % 2 == 0 else "On-site",
                    content_hash=f"search{i}",
                    application_status=status,
                    salary=(80000 + i * 5000, 120000 + i * 5000),
                )
                session.add(job)
                session.commit()
                session.refresh(job)

                # Convert to DTO immediately
                dto = Job.model_validate({**job.model_dump(), "company": job.company})
                job_dtos.append(dto)

        # Session closed - perform complex search and filtering
        # Search for Python jobs
        python_jobs = [dto for dto in job_dtos if "Python" in dto.description]
        assert len(python_jobs) == 3

        # Filter by status
        applied_jobs = [dto for dto in job_dtos if dto.application_status == "Applied"]
        assert len(applied_jobs) == 2

        # Complex filtering with serialization
        filtered_serialized = []
        for dto in job_dtos:
            if (
                dto.application_status in ["Applied", "Interview"]
                and dto.location == "Remote"
                and dto.salary[0]
                and dto.salary[0] >= 80000
            ):
                # Serialize filtered results
                json_data = dto.model_dump(mode="json")
                filtered_serialized.append(json_data)

        # Verify serialized data
        assert len(filtered_serialized) > 0
        for job_data in filtered_serialized:
            assert job_data["application_status"] in ["Applied", "Interview"]
            assert job_data["location"] == "Remote"
            assert job_data["salary"][0] >= 80000
            assert isinstance(job_data["company"], str)  # Company is string, not object

    def test_dto_data_consistency_after_updates(self, engine):
        """Test that DTOs maintain data consistency even when source data changes."""
        # Create initial data
        with Session(engine) as session:
            company = CompanySQL(
                name="Consistency Co",
                url="https://consistency.com/careers",
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            job = JobSQL(
                company_id=company.id,
                title="Original Engineer",
                description="Original description",
                link="https://consistency.com/jobs/123",
                location="Original City",
                content_hash="original123",
                application_status="New",
            )
            session.add(job)
            session.commit()
            session.refresh(job)

            # Create DTO with original data
            original_dto = Job.model_validate(
                {**job.model_dump(), "company": job.company},
            )

            # Update the database record and store job_id for later use
            job.title = "Updated Engineer"
            job.application_status = "Applied"
            job_id = job.id  # Store job ID before session closes
            session.commit()

        # Session closed - original DTO should maintain original data
        assert original_dto.title == "Original Engineer"  # Original value preserved
        assert original_dto.application_status == "New"  # Original value preserved
        assert original_dto.company == "Consistency Co"

        # Create new DTO with updated data
        with Session(engine) as session:
            updated_job = session.get(JobSQL, job_id)
            updated_dto = Job.model_validate(
                {**updated_job.model_dump(), "company": updated_job.company},
            )

        # New DTO should have updated values
        assert updated_dto.title == "Updated Engineer"
        assert updated_dto.application_status == "Applied"
        assert updated_dto.company == "Consistency Co"

        # Both DTOs should be serializable independently
        original_json = original_dto.model_dump_json()
        updated_json = updated_dto.model_dump_json()

        assert "Original Engineer" in original_json
        assert "Updated Engineer" in updated_json
        assert '"application_status":"New"' in original_json
        assert '"application_status":"Applied"' in updated_json
