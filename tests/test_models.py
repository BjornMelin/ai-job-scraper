"""Tests for database models and Pydantic validation."""

from datetime import datetime

import pytest

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from src.models import CompanySQL, JobPydantic, JobSQL


class TestJobPydantic:
    """Test cases for JobPydantic validation model."""

    def test_valid_job_creation(self, sample_job_dict):
        """Test creating a valid job with all fields."""
        job = JobPydantic(**sample_job_dict)

        assert job.company == "Test Company"
        assert job.title == "Senior AI Engineer"
        assert job.link == "https://test.com/careers/ai-engineer-123"
        assert job.location == "San Francisco, CA"
        assert job.favorite is False
        assert job.status == "New"
        assert job.notes == ""

    def test_minimal_valid_job(self):
        """Test creating a job with only required fields."""
        job = JobPydantic(
            company="Minimal Co",
            title="AI Engineer",
            description="Basic AI engineering role with machine learning focus.",
            link="https://minimal.com/job/123",
        )

        assert job.company == "Minimal Co"
        assert job.title == "AI Engineer"
        assert job.location == "Unknown"  # Default value
        assert job.favorite is False
        assert job.status == "New"

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from string fields."""
        job = JobPydantic(
            company="  Whitespace Company  ",
            title="   AI Engineer   ",
            description="   Great opportunity for AI engineers with experience.   ",
            link="https://example.com/job/123",
        )

        assert job.company == "Whitespace Company"
        assert job.title == "AI Engineer"
        assert job.description == "Great opportunity for AI engineers with experience."

    @pytest.mark.parametrize(
        "invalid_title",
        [
            "",  # Too short
            "AI",  # Too short (< 3 chars)
            "A" * 201,  # Too long (> 200 chars)
        ],
    )
    def test_invalid_title_validation(self, invalid_title):
        """Test validation fails for invalid titles."""
        with pytest.raises(ValidationError):
            JobPydantic(
                company="Test Company",
                title=invalid_title,
                description="Valid description for testing purposes.",
                link="https://example.com/job/123",
            )

    @pytest.mark.parametrize(
        "invalid_description",
        [
            "",  # Too short
            "Short",  # Too short (< 10 chars)
            "A" * 1001,  # Too long (> 1000 chars)
        ],
    )
    def test_invalid_description_validation(self, invalid_description):
        """Test validation fails for invalid descriptions."""
        with pytest.raises(ValidationError):
            JobPydantic(
                company="Test Company",
                title="AI Engineer",
                description=invalid_description,
                link="https://example.com/job/123",
            )

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "ftp://example.com/job/123",  # Invalid protocol
            "not-a-url",  # Not a URL
            "http://",  # Too short
            "https://example.com/" + "a" * 500,  # Too long
        ],
    )
    def test_invalid_url_validation(self, invalid_url):
        """Test validation fails for invalid URLs."""
        with pytest.raises(ValidationError):
            JobPydantic(
                company="Test Company",
                title="AI Engineer",
                description="Valid description for testing purposes.",
                link=invalid_url,
            )

    @pytest.mark.parametrize(
        "valid_url",
        [
            "https://example.com/job/123",
            "http://example.com/careers/ai-engineer",
            "https://subdomain.example.com/path/to/job",
        ],
    )
    def test_valid_url_validation(self, valid_url):
        """Test validation passes for valid URLs."""
        job = JobPydantic(
            company="Test Company",
            title="AI Engineer",
            description="Valid description for testing purposes.",
            link=valid_url,
        )
        assert job.link == valid_url

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            JobPydantic(
                company="Test Company",
                title="AI Engineer",
                description="Valid description for testing purposes.",
                link="https://example.com/job/123",
                extra_field="This should not be allowed",  # Extra field
            )

    def test_default_values(self):
        """Test that default values are set correctly."""
        job = JobPydantic(
            company="Test Company",
            title="AI Engineer",
            description="Valid description for testing purposes.",
            link="https://example.com/job/123",
        )

        assert job.location == "Unknown"
        assert job.posted_date is None
        assert job.hash is None
        assert job.last_seen is None
        assert job.favorite is False
        assert job.status == "New"
        assert job.notes == ""


class TestJobSQL:
    """Test cases for JobSQL database model."""

    def test_job_sql_creation(self, temp_db, sample_job):
        """Test creating and saving a JobSQL instance."""
        session = temp_db()

        session.add(sample_job)
        session.commit()

        # Query back the job
        retrieved_job = (
            session.query(JobSQL).filter_by(title="Senior AI Engineer").first()
        )

        assert retrieved_job is not None
        assert retrieved_job.company == "Test Company"
        assert retrieved_job.title == "Senior AI Engineer"
        assert retrieved_job.link == "https://test.com/careers/ai-engineer-123"
        assert retrieved_job.favorite is False
        assert retrieved_job.status == "New"

        session.close()

    def test_job_unique_link_constraint(self, temp_db, sample_job):
        """Test that job links must be unique."""
        session = temp_db()

        # Add first job
        session.add(sample_job)
        session.commit()

        # Try to add another job with same link
        duplicate_job = JobSQL(
            company="Another Company",
            title="Different Title",
            description="Different description",
            link="https://test.com/careers/ai-engineer-123",  # Same link
            location="Different Location",
            posted_date=datetime.now(),
            hash="different_hash",
            last_seen=datetime.now(),
        )

        session.add(duplicate_job)

        with pytest.raises(IntegrityError):  # Should raise integrity error
            session.commit()

        session.close()

    def test_job_indexes_exist(self, temp_db, sample_job):
        """Test that database indexes work correctly."""
        session = temp_db()

        # Add multiple jobs for testing indexes
        jobs = [
            sample_job,
            JobSQL(
                company="Another Company",
                title="ML Engineer",
                description="Machine learning engineering position.",
                link="https://another.com/careers/ml-engineer",
                location="Remote",
                posted_date=datetime.now(),
                hash="hash_2",
                last_seen=datetime.now(),
                favorite=True,
                status="Applied",
                notes="Interesting role",
            ),
        ]

        session.add_all(jobs)
        session.commit()

        # Test company index
        company_jobs = (
            session.query(JobSQL).filter(JobSQL.company == "Test Company").all()
        )
        assert len(company_jobs) == 1

        # Test title index
        title_jobs = session.query(JobSQL).filter(JobSQL.title.like("%Engineer%")).all()
        assert len(title_jobs) == 2

        # Test posted_date index
        recent_jobs = (
            session.query(JobSQL).filter(JobSQL.posted_date.is_not(None)).all()
        )
        assert len(recent_jobs) == 2

        session.close()


class TestCompanySQL:
    """Test cases for CompanySQL database model."""

    def test_company_sql_creation(self, temp_db, sample_company):
        """Test creating and saving a CompanySQL instance."""
        session = temp_db()

        session.add(sample_company)
        session.commit()

        # Query back the company
        retrieved_company = (
            session.query(CompanySQL).filter_by(name="Test Company").first()
        )

        assert retrieved_company is not None
        assert retrieved_company.name == "Test Company"
        assert retrieved_company.url == "https://test.com/careers"
        assert retrieved_company.active is True

        session.close()

    def test_company_unique_name_constraint(self, temp_db, sample_company):
        """Test that company names must be unique."""
        session = temp_db()

        # Add first company
        session.add(sample_company)
        session.commit()

        # Try to add another company with same name
        duplicate_company = CompanySQL(
            name="Test Company",  # Same name
            url="https://different.com/careers",
            active=False,
        )

        session.add(duplicate_company)

        with pytest.raises(IntegrityError):
            session.commit()

        session.close()

    def test_company_active_index(self, temp_db):
        """Test that the active index works correctly."""
        session = temp_db()

        # Add companies with different active status
        companies = [
            CompanySQL(name="Active Company", url="https://active.com", active=True),
            CompanySQL(
                name="Inactive Company", url="https://inactive.com", active=False
            ),
            CompanySQL(name="Another Active", url="https://active2.com", active=True),
        ]

        session.add_all(companies)
        session.commit()

        # Test active index
        active_companies = session.query(CompanySQL).filter_by(active=True).all()
        assert len(active_companies) == 2

        inactive_companies = session.query(CompanySQL).filter_by(active=False).all()
        assert len(inactive_companies) == 1

        session.close()
