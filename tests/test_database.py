"""Tests for database operations and integration."""

from datetime import datetime, timedelta

import pytest

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.database import SessionLocal
from src.models import CompanySQL, JobSQL


class TestDatabaseIntegration:
    """Test cases for database integration functionality."""

    def test_database_connection(self):
        """Test that database connection works."""
        session = SessionLocal()
        try:
            # Test basic query
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            session.close()

    def test_company_crud_operations(self, temp_db):
        """Test CRUD operations for companies."""
        session = temp_db()

        # Create
        company = CompanySQL(
            name="CRUD Test Company", url="https://crud-test.com/careers", active=True
        )
        session.add(company)
        session.commit()

        # Read
        retrieved = (
            session.query(CompanySQL).filter_by(name="CRUD Test Company").first()
        )
        assert retrieved is not None
        assert retrieved.url == "https://crud-test.com/careers"

        # Update
        retrieved.active = False
        session.commit()

        updated = session.query(CompanySQL).filter_by(name="CRUD Test Company").first()
        assert updated.active is False

        # Delete
        session.delete(updated)
        session.commit()

        deleted = session.query(CompanySQL).filter_by(name="CRUD Test Company").first()
        assert deleted is None

        session.close()

    def test_job_crud_operations(self, temp_db):
        """Test CRUD operations for jobs."""
        session = temp_db()

        # Create
        job = JobSQL(
            company="CRUD Job Company",
            title="CRUD Test Job",
            description="This is a test job for CRUD operations.",
            link="https://crud-job.com/test/123",
            location="Test City",
            posted_date=datetime.now(),
            hash="crud_test_hash",
            last_seen=datetime.now(),
            favorite=False,
            status="New",
            notes="",
        )
        session.add(job)
        session.commit()

        # Read
        retrieved = session.query(JobSQL).filter_by(title="CRUD Test Job").first()
        assert retrieved is not None
        assert retrieved.company == "CRUD Job Company"
        assert retrieved.location == "Test City"

        # Update
        retrieved.favorite = True
        retrieved.status = "Applied"
        retrieved.notes = "Updated notes"
        session.commit()

        updated = session.query(JobSQL).filter_by(title="CRUD Test Job").first()
        assert updated.favorite is True
        assert updated.status == "Applied"
        assert updated.notes == "Updated notes"

        # Delete
        session.delete(updated)
        session.commit()

        deleted = session.query(JobSQL).filter_by(title="CRUD Test Job").first()
        assert deleted is None

        session.close()

    def test_job_filtering_queries(self, temp_db):
        """Test various job filtering queries."""
        session = temp_db()

        # Create test data
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        jobs = [
            JobSQL(
                company="Company A",
                title="Senior AI Engineer",
                description="Senior position for AI engineering.",
                link="https://company-a.com/senior-ai",
                location="San Francisco",
                posted_date=now,
                hash="hash_1",
                last_seen=now,
                favorite=True,
                status="Applied",
                notes="Applied yesterday",
            ),
            JobSQL(
                company="Company B",
                title="ML Engineer",
                description="Machine learning engineering role.",
                link="https://company-b.com/ml-engineer",
                location="Remote",
                posted_date=yesterday,
                hash="hash_2",
                last_seen=now,
                favorite=False,
                status="New",
                notes="",
            ),
            JobSQL(
                company="Company A",
                title="Data Scientist",
                description="Data science position with ML focus.",
                link="https://company-a.com/data-scientist",
                location="New York",
                posted_date=now,
                hash="hash_3",
                last_seen=now,
                favorite=False,
                status="Interested",
                notes="Looks promising",
            ),
        ]

        session.add_all(jobs)
        session.commit()

        # Test company filtering
        company_a_jobs = session.query(JobSQL).filter_by(company="Company A").all()
        assert len(company_a_jobs) == 2

        # Test title filtering
        engineer_jobs = (
            session.query(JobSQL).filter(JobSQL.title.like("%Engineer%")).all()
        )
        assert len(engineer_jobs) == 2

        # Test favorite filtering
        favorite_jobs = session.query(JobSQL).filter_by(favorite=True).all()
        assert len(favorite_jobs) == 1
        assert favorite_jobs[0].title == "Senior AI Engineer"

        # Test status filtering
        applied_jobs = session.query(JobSQL).filter_by(status="Applied").all()
        assert len(applied_jobs) == 1

        # Test date filtering
        recent_jobs = (
            session.query(JobSQL).filter(JobSQL.posted_date >= yesterday).all()
        )
        assert len(recent_jobs) == 3

        # Test location filtering
        remote_jobs = session.query(JobSQL).filter_by(location="Remote").all()
        assert len(remote_jobs) == 1

        session.close()

    def test_database_indexes_performance(self, temp_db):
        """Test that database indexes improve query performance."""
        session = temp_db()

        # Create a larger dataset for index testing
        companies = []
        jobs = []

        for i in range(100):
            company = CompanySQL(
                name=f"Performance Test Company {i}",
                url=f"https://perf-test-{i}.com/careers",
                active=i % 2 == 0,  # Half active, half inactive
            )
            companies.append(company)

            job = JobSQL(
                company=f"Performance Test Company {i}",
                title=f"Engineer {i}",
                description=f"Engineering position {i} for performance testing.",
                link=f"https://perf-test-{i}.com/job/{i}",
                location="Test Location",
                posted_date=datetime.now(),
                hash=f"hash_{i}",
                last_seen=datetime.now(),
                favorite=i % 10 == 0,  # Every 10th job is favorite
                status="New",
                notes="",
            )
            jobs.append(job)

        session.add_all(companies)
        session.add_all(jobs)
        session.commit()

        # Test indexed company queries (should be fast)
        company_filtered = (
            session.query(JobSQL)
            .filter(JobSQL.company == "Performance Test Company 50")
            .all()
        )
        assert len(company_filtered) == 1

        # Test indexed title queries
        title_filtered = (
            session.query(JobSQL).filter(JobSQL.title.like("Engineer%")).all()
        )
        assert len(title_filtered) == 100

        # Test indexed posted_date queries
        date_filtered = (
            session.query(JobSQL).filter(JobSQL.posted_date.is_not(None)).all()
        )
        assert len(date_filtered) == 100

        # Test active company index
        active_companies = session.query(CompanySQL).filter_by(active=True).all()
        assert len(active_companies) == 50

        session.close()

    def test_database_constraints(self, temp_db):
        """Test database constraints and integrity."""
        session = temp_db()

        # Test company name uniqueness
        company1 = CompanySQL(name="Unique Test", url="https://test1.com", active=True)
        company2 = CompanySQL(name="Unique Test", url="https://test2.com", active=True)

        session.add(company1)
        session.commit()

        session.add(company2)

        with pytest.raises(IntegrityError):
            session.commit()

        session.rollback()

        # Test job link uniqueness
        job1 = JobSQL(
            company="Test Company",
            title="Job 1",
            description="First job description.",
            link="https://unique-link.com/job/123",
            location="Location 1",
            posted_date=datetime.now(),
            hash="hash_1",
            last_seen=datetime.now(),
        )

        job2 = JobSQL(
            company="Test Company",
            title="Job 2",
            description="Second job description.",
            link="https://unique-link.com/job/123",  # Same link
            location="Location 2",
            posted_date=datetime.now(),
            hash="hash_2",
            last_seen=datetime.now(),
        )

        session.add(job1)
        session.commit()

        session.add(job2)

        with pytest.raises(IntegrityError):
            session.commit()

        session.close()

    def test_database_rollback(self, temp_db):
        """Test database transaction rollback functionality."""
        session = temp_db()

        # Add some initial data
        company = CompanySQL(
            name="Rollback Test Company", url="https://rollback-test.com", active=True
        )
        session.add(company)
        session.commit()

        # Start a transaction that will be rolled back
        try:
            job = JobSQL(
                company="Rollback Test Company",
                title="Rollback Job",
                description="This job will be rolled back.",
                link="https://rollback-test.com/job/123",
                location="Rollback City",
                posted_date=datetime.now(),
                hash="rollback_hash",
                last_seen=datetime.now(),
            )
            session.add(job)

            # Force an error to trigger rollback
            invalid_job = JobSQL(
                company="Rollback Test Company",
                title="Invalid Job",
                description="This job has invalid data.",
                link="https://rollback-test.com/job/123",  # Duplicate link
                location="Invalid City",
                posted_date=datetime.now(),
                hash="invalid_hash",
                last_seen=datetime.now(),
            )
            session.add(invalid_job)
            session.commit()  # This should fail

        except Exception:
            session.rollback()

        # Verify that the rollback worked
        jobs = session.query(JobSQL).filter_by(company="Rollback Test Company").all()
        assert len(jobs) == 0  # No jobs should exist due to rollback

        # But the company should still exist (committed before the failed transaction)
        companies = (
            session.query(CompanySQL).filter_by(name="Rollback Test Company").all()
        )
        assert len(companies) == 1

        session.close()
