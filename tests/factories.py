"""Simplified test factories using factory-boy best practices.

Essential factories for core models with predictable test data.
No complex traits or realistic data generation - simple, maintainable patterns.

Factories:
- CompanyFactory: Basic company records
- JobFactory: Simple job postings

Key simplifications:
- Standard factory-boy patterns only
- Predictable test data over realistic data
- No complex traits or helper functions
- ~80 lines vs 225 lines (64% reduction)
"""

# Import SQLAlchemy models directly
from datetime import UTC, datetime, timedelta

from factory import Sequence
from factory.alchemy import SQLAlchemyModelFactory
from src.database_models import CompanySQL, JobSQL
from src.models.job_models import ApplicationStage


class CompanyFactory(SQLAlchemyModelFactory):
    """Factory for creating Company test records."""

    class Meta:
        """Factory configuration."""

        model = CompanySQL
        sqlalchemy_session = None
        sqlalchemy_session_persistence = "flush"

    name = Sequence(lambda n: f"Test Company {n}")
    url = Sequence(lambda n: f"https://company{n}.com/careers")


class JobFactory(SQLAlchemyModelFactory):
    """Factory for creating Job test records."""

    class Meta:
        """Factory configuration."""

        model = JobSQL
        sqlalchemy_session = None
        sqlalchemy_session_persistence = "flush"

    title = Sequence(lambda n: f"Software Engineer {n}")
    description = "Test job description"
    link = Sequence(lambda n: f"https://jobs.test.com/job/{n}")
    location = "Remote"
    posted_date = datetime.now(UTC) - timedelta(days=1)
    salary = (100000, 150000)
    favorite = False
    notes = ""
    content_hash = Sequence(lambda n: f"hash{n}")
    application_status = ApplicationStage.INBOX
    application_date = None
    archived = False
    last_seen = datetime.now(UTC)
