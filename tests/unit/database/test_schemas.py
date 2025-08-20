"""Test suite for Pydantic DTOs in schemas.py.

This module tests the Pydantic Data Transfer Objects (DTOs) to ensure:
1. Correct field validation and type handling
2. Proper from_attributes conversion from SQLModel objects
3. JSON serialization/deserialization works correctly
4. DateTime handling and encoding
5. Backward compatibility features
"""

import json

from datetime import UTC, datetime

import pytest

from pydantic import ValidationError
from sqlmodel import Session

from src.models import CompanySQL, JobSQL
from src.schemas import Company, Job


class TestCompanyDTO:
    """Test suite for Company Pydantic DTO."""

    def test_company_creation_with_required_fields(self):
        """Test creating Company DTO with required fields only."""
        company = Company(name="Test Company", url="https://example.com/careers")

        assert company.name == "Test Company"
        assert company.url == "https://example.com/careers"
        assert company.active is True  # Default value
        assert company.id is None  # Default value
        assert company.last_scraped is None  # Default value
        assert company.scrape_count == 0  # Default value
        assert company.success_rate == 1.0  # Default value

    def test_company_creation_with_all_fields(self):
        """Test creating Company DTO with all fields provided."""
        last_scraped = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        company = Company(
            id=1,
            name="Test Company",
            url="https://example.com/careers",
            active=False,
            last_scraped=last_scraped,
            scrape_count=5,
            success_rate=0.8,
        )

        assert company.id == 1
        assert company.name == "Test Company"
        assert company.url == "https://example.com/careers"
        assert company.active is False
        assert company.last_scraped == last_scraped
        assert company.scrape_count == 5
        assert company.success_rate == 0.8

    def test_company_field_validation(self):
        """Test Company DTO field validation."""
        # Test missing required field
        with pytest.raises(ValidationError) as exc_info:
            Company(url="https://example.com/careers")  # Missing name

        assert "name" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Company(name="Test Company")  # Missing url

        assert "url" in str(exc_info.value)

    def test_company_from_sqlmodel_conversion(self, session):
        """Test converting CompanySQL to Company DTO using from_attributes."""
        # Create a CompanySQL instance
        sql_company = CompanySQL(
            id=1,
            name="SQLModel Company",
            url="https://sqlmodel.com/careers",
            active=True,
            scrape_count=3,
            success_rate=0.9,
        )
        session.add(sql_company)
        session.commit()
        session.refresh(sql_company)

        # Convert to DTO
        dto_company = Company.model_validate(sql_company)

        assert dto_company.id == sql_company.id
        assert dto_company.name == sql_company.name
        assert dto_company.url == sql_company.url
        assert dto_company.active == sql_company.active
        assert dto_company.scrape_count == sql_company.scrape_count
        assert dto_company.success_rate == sql_company.success_rate

    def test_company_json_serialization(self):
        """Test JSON serialization of Company DTO."""
        last_scraped = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        company = Company(
            id=1,
            name="Test Company",
            url="https://example.com/careers",
            active=True,
            last_scraped=last_scraped,
            scrape_count=5,
            success_rate=0.8,
        )

        # Test model_dump with JSON serialization
        json_data = company.model_dump(mode="json")

        assert json_data["id"] == 1
        assert json_data["name"] == "Test Company"
        assert json_data["url"] == "https://example.com/careers"
        assert json_data["active"] is True
        assert json_data["last_scraped"] == "2024-01-15T10:30:00+00:00"
        assert json_data["scrape_count"] == 5
        assert json_data["success_rate"] == 0.8

    def test_company_json_serialization_none_datetime(self):
        """Test JSON serialization with None datetime values."""
        company = Company(name="Test Company", url="https://example.com/careers")

        json_data = company.model_dump(mode="json")
        assert json_data["last_scraped"] is None

    def test_company_json_round_trip(self):
        """Test JSON serialization and deserialization round trip."""
        original = Company(
            id=1,
            name="Round Trip Company",
            url="https://roundtrip.com/careers",
            active=False,
            scrape_count=10,
            success_rate=0.95,
        )

        # Serialize to JSON string
        json_str = original.model_dump_json()

        # Parse back to dict and create new instance
        json_dict = json.loads(json_str)
        restored = Company(**json_dict)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.url == original.url
        assert restored.active == original.active
        assert restored.scrape_count == original.scrape_count
        assert restored.success_rate == original.success_rate


class TestJobDTO:
    """Test suite for Job Pydantic DTO."""

    def test_job_creation_with_required_fields(self):
        """Test creating Job DTO with required fields only."""
        job = Job(
            company="Test Company",
            title="Software Engineer",
            description="A great software engineering position",
            link="https://example.com/jobs/123",
            location="San Francisco, CA",
            content_hash="abc123def456",
        )

        assert job.company == "Test Company"
        assert job.title == "Software Engineer"
        assert job.description == "A great software engineering position"
        assert job.link == "https://example.com/jobs/123"
        assert job.location == "San Francisco, CA"
        assert job.content_hash == "abc123def456"

        # Test default values
        assert job.id is None
        assert job.company_id is None
        assert job.posted_date is None
        assert job.salary == (None, None)
        assert job.favorite is False
        assert job.notes == ""
        assert job.application_status == "New"
        assert job.application_date is None
        assert job.archived is False
        assert job.last_seen is None

    def test_job_creation_with_all_fields(self):
        """Test creating Job DTO with all fields provided."""
        posted_date = datetime(2024, 1, 10, 9, 0, 0, tzinfo=UTC)
        app_date = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
        last_seen = datetime(2024, 1, 20, 12, 0, 0, tzinfo=UTC)

        job = Job(
            id=1,
            company_id=5,
            company="Full Company",
            title="Senior Engineer",
            description="Senior engineering role",
            link="https://example.com/jobs/456",
            location="Remote",
            posted_date=posted_date,
            salary=(100000, 150000),
            favorite=True,
            notes="This looks promising",
            content_hash="xyz789abc123",
            application_status="Applied",
            application_date=app_date,
            archived=True,
            last_seen=last_seen,
        )

        assert job.id == 1
        assert job.company_id == 5
        assert job.company == "Full Company"
        assert job.title == "Senior Engineer"
        assert job.description == "Senior engineering role"
        assert job.link == "https://example.com/jobs/456"
        assert job.location == "Remote"
        assert job.posted_date == posted_date
        assert job.salary == (100000, 150000)
        assert job.favorite is True
        assert job.notes == "This looks promising"
        assert job.content_hash == "xyz789abc123"
        assert job.application_status == "Applied"
        assert job.application_date == app_date
        assert job.archived is True
        assert job.last_seen == last_seen

    def test_job_field_validation(self):
        """Test Job DTO field validation."""
        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            Job(
                title="Software Engineer",
                description="A great position",
                link="https://example.com/jobs/123",
                location="San Francisco, CA",
                content_hash="abc123def456",
            )  # Missing company

        assert "company" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Job(
                company="Test Company",
                description="A great position",
                link="https://example.com/jobs/123",
                location="San Francisco, CA",
                content_hash="abc123def456",
            )  # Missing title

        assert "title" in str(exc_info.value)

    def test_job_salary_tuple_handling(self):
        """Test Job DTO salary tuple field handling."""
        # Test with tuple
        job = Job(
            company="Test Company",
            title="Engineer",
            description="Job description",
            link="https://example.com/jobs/123",
            location="SF",
            content_hash="hash123",
            salary=(80000, 120000),
        )
        assert job.salary == (80000, 120000)

        # Test with partial tuple (one None)
        job2 = Job(
            company="Test Company",
            title="Engineer",
            description="Job description",
            link="https://example.com/jobs/456",
            location="SF",
            content_hash="hash456",
            salary=(80000, None),
        )
        assert job2.salary == (80000, None)

        # Test default value
        job3 = Job(
            company="Test Company",
            title="Engineer",
            description="Job description",
            link="https://example.com/jobs/789",
            location="SF",
            content_hash="hash789",
        )
        assert job3.salary == (None, None)

    def test_job_status_backward_compatibility(self):
        """Test Job DTO backward compatibility alias for status property."""
        job = Job(
            company="Test Company",
            title="Engineer",
            description="Job description",
            link="https://example.com/jobs/123",
            location="SF",
            content_hash="hash123",
            application_status="Interview",
        )

        # Test that status property returns application_status value
        assert job.status == "Interview"
        assert job.application_status == "Interview"

        # Test default status
        job2 = Job(
            company="Test Company",
            title="Engineer 2",
            description="Job description",
            link="https://example.com/jobs/456",
            location="SF",
            content_hash="hash456",
        )
        assert job2.status == "New"
        assert job2.application_status == "New"

    def test_job_from_sqlmodel_conversion(self, session):
        """Test converting JobSQL to Job DTO using from_attributes."""
        # Create a CompanySQL for the relationship
        sql_company = CompanySQL(
            name="SQLModel Job Test Company",
            url="https://sqlmodel.com/careers",
        )
        session.add(sql_company)
        session.commit()
        session.refresh(sql_company)

        # Create a JobSQL instance with relationship
        posted_date = datetime(2024, 1, 10, 9, 0, 0, tzinfo=UTC)
        sql_job = JobSQL(
            id=1,
            company_id=sql_company.id,
            title="SQLModel Engineer",
            description="Working with SQLModel",
            link="https://sqlmodel.com/jobs/123",
            location="Remote",
            posted_date=posted_date,
            salary=(90000, 130000),
            favorite=True,
            notes="Interesting role",
            content_hash="sqlmodel123",
            application_status="Applied",
        )
        session.add(sql_job)
        session.commit()
        session.refresh(sql_job)

        # Convert to DTO - need to manually set company string since
        # from_attributes won't automatically resolve the relationship
        dto_job = Job.model_validate(
            {
                **sql_job.model_dump(),
                "company": sql_job.company,  # Use computed property
            },
        )

        assert dto_job.id == sql_job.id
        assert dto_job.company_id == sql_job.company_id
        assert dto_job.company == "SQLModel Job Test Company"  # Company name as string
        assert dto_job.title == sql_job.title
        assert dto_job.description == sql_job.description
        assert dto_job.link == sql_job.link
        assert dto_job.location == sql_job.location
        # Handle timezone-aware comparison
        if sql_job.posted_date and dto_job.posted_date:
            expected_posted_date = (
                sql_job.posted_date.replace(tzinfo=UTC)
                if sql_job.posted_date.tzinfo is None
                else sql_job.posted_date
            )
            assert dto_job.posted_date == expected_posted_date
        else:
            assert dto_job.posted_date == sql_job.posted_date
        assert (
            dto_job.salary == tuple(sql_job.salary)
            if isinstance(sql_job.salary, list)
            else sql_job.salary
        )
        assert dto_job.favorite == sql_job.favorite
        assert dto_job.notes == sql_job.notes
        assert dto_job.content_hash == sql_job.content_hash
        assert dto_job.application_status == sql_job.application_status

    def test_job_json_serialization(self):
        """Test JSON serialization of Job DTO."""
        posted_date = datetime(2024, 1, 10, 9, 0, 0, tzinfo=UTC)
        app_date = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)

        job = Job(
            id=1,
            company_id=5,
            company="JSON Company",
            title="JSON Engineer",
            description="Working with JSON APIs",
            link="https://json.com/jobs/123",
            location="San Francisco, CA",
            posted_date=posted_date,
            salary=(100000, 140000),
            favorite=True,
            notes="Great benefits",
            content_hash="json123",
            application_status="Applied",
            application_date=app_date,
            archived=False,
        )

        # Test model_dump with JSON serialization
        json_data = job.model_dump(mode="json")

        assert json_data["id"] == 1
        assert json_data["company_id"] == 5
        assert json_data["company"] == "JSON Company"
        assert json_data["title"] == "JSON Engineer"
        assert json_data["description"] == "Working with JSON APIs"
        assert json_data["link"] == "https://json.com/jobs/123"
        assert json_data["location"] == "San Francisco, CA"
        assert json_data["posted_date"] == "2024-01-10T09:00:00+00:00"
        assert json_data["salary"] == [100000, 140000]
        assert json_data["favorite"] is True
        assert json_data["notes"] == "Great benefits"
        assert json_data["content_hash"] == "json123"
        assert json_data["application_status"] == "Applied"
        assert json_data["application_date"] == "2024-01-15T14:30:00+00:00"
        assert json_data["archived"] is False

    def test_job_json_serialization_none_values(self):
        """Test JSON serialization with None datetime and salary values."""
        job = Job(
            company="None Company",
            title="None Engineer",
            description="Testing None values",
            link="https://none.com/jobs/123",
            location="Anywhere",
            content_hash="none123",
        )

        json_data = job.model_dump(mode="json")

        assert json_data["id"] is None
        assert json_data["company_id"] is None
        assert json_data["posted_date"] is None
        assert json_data["salary"] == [None, None]
        assert json_data["application_date"] is None
        assert json_data["last_seen"] is None

    def test_job_json_round_trip(self):
        """Test JSON serialization and deserialization round trip."""
        original = Job(
            id=1,
            company="Round Trip Company",
            title="Round Trip Engineer",
            description="Testing round trip serialization",
            link="https://roundtrip.com/jobs/123",
            location="Remote",
            salary=(95000, 125000),
            favorite=True,
            content_hash="roundtrip123",
            application_status="Interview",
        )

        # Serialize to JSON string
        json_str = original.model_dump_json()

        # Parse back to dict and create new instance
        json_dict = json.loads(json_str)
        restored = Job(**json_dict)

        assert restored.id == original.id
        assert restored.company == original.company
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.link == original.link
        assert restored.location == original.location
        assert restored.salary == original.salary
        assert restored.favorite == original.favorite
        assert restored.content_hash == original.content_hash
        assert restored.application_status == original.application_status
        assert restored.status == original.status  # Test backward compatibility


class TestDTOIntegration:
    """Test integration scenarios between DTOs and SQLModel objects."""

    def test_company_to_dto_conversion_preserves_data(self, session):
        """Test that Company SQLModel to DTO conversion preserves all data."""
        # Create company with all fields
        last_scraped = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        sql_company = CompanySQL(
            name="Integration Company",
            url="https://integration.com/careers",
            active=False,
            last_scraped=last_scraped,
            scrape_count=7,
            success_rate=0.85,
        )
        session.add(sql_company)
        session.commit()
        session.refresh(sql_company)

        # Convert to DTO
        dto_company = Company.model_validate(sql_company)

        # Verify all data is preserved
        assert dto_company.name == sql_company.name
        assert dto_company.url == sql_company.url
        assert dto_company.active == sql_company.active
        # Handle timezone-aware comparison
        if sql_company.last_scraped and dto_company.last_scraped:
            expected_last_scraped = (
                sql_company.last_scraped.replace(tzinfo=UTC)
                if sql_company.last_scraped.tzinfo is None
                else sql_company.last_scraped
            )
            assert dto_company.last_scraped == expected_last_scraped
        else:
            assert dto_company.last_scraped == sql_company.last_scraped
        assert dto_company.scrape_count == sql_company.scrape_count
        assert dto_company.success_rate == sql_company.success_rate

    def test_job_to_dto_conversion_with_company_relationship(self, session):
        """Test Job SQLModel to DTO conversion with company relationship handling."""
        # Create company
        sql_company = CompanySQL(
            name="Relationship Company",
            url="https://relationship.com/careers",
        )
        session.add(sql_company)
        session.commit()
        session.refresh(sql_company)

        # Create job with relationship
        sql_job = JobSQL(
            company_id=sql_company.id,
            title="Relationship Engineer",
            description="Testing relationships",
            link="https://relationship.com/jobs/123",
            location="Test City",
            salary=(80000, 110000),
            content_hash="relationship123",
        )
        session.add(sql_job)
        session.commit()
        session.refresh(sql_job)

        # Convert to DTO with manual company name resolution
        dto_job = Job.model_validate(
            {
                **sql_job.model_dump(),
                "company": sql_job.company,  # Use computed property
            },
        )

        # Verify relationship is resolved to string
        assert dto_job.company == "Relationship Company"
        assert dto_job.company_id == sql_company.id
        assert isinstance(dto_job.company, str)  # Company is string, not object

    def test_dto_serialization_after_session_close(self, engine):
        """Test that DTOs can be serialized after database session is closed."""
        # Create objects within a session scope
        with Session(engine) as session:
            sql_company = CompanySQL(
                name="Session Test Company",
                url="https://session.com/careers",
            )
            session.add(sql_company)
            session.commit()
            session.refresh(sql_company)

            sql_job = JobSQL(
                company_id=sql_company.id,
                title="Session Engineer",
                description="Testing session boundaries",
                link="https://session.com/jobs/123",
                location="Session City",
                content_hash="session123",
            )
            session.add(sql_job)
            session.commit()
            session.refresh(sql_job)

            # Convert to DTOs while session is still open
            company_dto = Company.model_validate(sql_company)
            job_dto = Job.model_validate(
                {**sql_job.model_dump(), "company": sql_job.company},
            )

        # Session is now closed - test that DTOs still work
        assert company_dto.name == "Session Test Company"
        assert company_dto.url == "https://session.com/careers"

        assert job_dto.title == "Session Engineer"
        assert job_dto.company == "Session Test Company"

        # Test JSON serialization after session close
        company_json = company_dto.model_dump_json()
        job_json = job_dto.model_dump_json()

        assert "Session Test Company" in company_json
        assert "Session Engineer" in job_json
        assert "Session Test Company" in job_json  # Company name in job


class TestJobComputedFields:
    """Test computed fields in Job DTO for display purposes."""

    def test_job_salary_range_display(self):
        """Test salary_range_display computed field."""
        # Test with range
        job = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            salary=(50000, 80000),
        )
        assert job.salary_range_display == "$50,000 - $80,000"

        # Test with min only
        job_min_only = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            salary=(60000, None),
        )
        assert job_min_only.salary_range_display == "$60,000+"

        # Test with max only
        job_max_only = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            salary=(None, 90000),
        )
        assert job_max_only.salary_range_display == "Up to $90,000"

        # Test with no salary
        job_no_salary = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            salary=(None, None),
        )
        assert job_no_salary.salary_range_display == "Not specified"

    def test_job_days_since_posted(self):
        """Test days_since_posted computed field."""
        from datetime import datetime, timedelta, timezone as tz

        # Test with recent date
        recent_date = datetime.now(tz.utc) - timedelta(days=3)
        job = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            posted_date=recent_date,
        )
        assert job.days_since_posted == 3

        # Test with None date
        job_no_date = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
        )
        assert job_no_date.days_since_posted is None

    def test_job_is_recently_posted(self):
        """Test is_recently_posted computed field."""
        from datetime import datetime, timedelta, timezone as tz

        # Test recently posted (within 7 days)
        recent_date = datetime.now(tz.utc) - timedelta(days=5)
        job_recent = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            posted_date=recent_date,
        )
        assert job_recent.is_recently_posted is True

        # Test not recently posted (over 7 days)
        old_date = datetime.now(tz.utc) - timedelta(days=10)
        job_old = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
            posted_date=old_date,
        )
        assert job_old.is_recently_posted is False

        # Test with None date
        job_no_date = Job(
            company="Test",
            title="Dev",
            description="Test job",
            link="https://test.com",
            location="SF",
            content_hash="hash",
        )
        assert job_no_date.is_recently_posted is False


class TestSafeIntValidator:
    """Test new SafeInt Pydantic validator type."""

    def test_safe_int_conversion(self):
        """Test SafeInt converts various inputs to non-negative integers."""
        from pydantic import BaseModel

        from src.ui.utils.validators import SafeInt

        class TestModel(BaseModel):
            count: SafeInt

        # Test valid inputs
        assert TestModel(count=5).count == 5
        assert TestModel(count="10").count == 10
        assert TestModel(count=3.7).count == 3
        assert TestModel(count=True).count == 1
        assert TestModel(count=False).count == 0

        # Test edge cases that should become 0
        assert TestModel(count=None).count == 0
        assert TestModel(count=-5).count == 0
        assert TestModel(count="invalid").count == 0
        assert TestModel(count="").count == 0

    def test_job_count_validator(self):
        """Test JobCount validator with job-specific logic."""
        from pydantic import BaseModel

        from src.ui.utils.validators import JobCount

        class TestModel(BaseModel):
            jobs: JobCount

        # Test valid job counts
        assert TestModel(jobs=0).jobs == 0
        assert TestModel(jobs=25).jobs == 25
        assert TestModel(jobs="15").jobs == 15

        # Test invalid inputs become 0
        assert TestModel(jobs=None).jobs == 0
        assert TestModel(jobs=-10).jobs == 0
        assert TestModel(jobs="invalid").jobs == 0
