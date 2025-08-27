"""Test data factories using factory_boy for realistic test data generation.

This module provides factory classes for generating test data with Faker for:
- CompanySQL: Company records with realistic business data
- JobSQL: Job postings with varied salaries, locations, and descriptions

Factories support batch creation, traits for different scenarios,
and integration with SQLModel sessions for database tests.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import factory

from factory import Faker, LazyFunction, Sequence, SubFactory, fuzzy
from factory.alchemy import SQLAlchemyModelFactory
from faker import Faker as FakerInstance

from src.models import CompanySQL, JobSQL

# Initialize faker with seed for reproducible tests
fake = FakerInstance()
fake.seed_instance(42)

# Common application statuses for realistic variety
APPLICATION_STATUSES = [
    "New",
    "Interested",
    "Applied",
    "Interview Scheduled",
    "Interviewed",
    "Offer Extended",
    "Rejected",
    "Withdrawn",
]

# Common tech locations for realistic job data
TECH_LOCATIONS = [
    "San Francisco, CA",
    "New York, NY",
    "Seattle, WA",
    "Austin, TX",
    "Boston, MA",
    "Remote",
    "Los Angeles, CA",
    "Chicago, IL",
    "Denver, CO",
    "Atlanta, GA",
]

# Tech job titles with AI/ML focus
AI_ML_TITLES = [
    "Senior AI Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "ML Research Scientist",
    "AI Product Manager",
    "Computer Vision Engineer",
    "NLP Engineer",
    "Deep Learning Researcher",
    "AI Platform Engineer",
    "MLOps Engineer",
    "Principal AI Engineer",
    "Staff ML Engineer",
]


class CompanyFactory(SQLAlchemyModelFactory):
    """Factory for creating Company test records with realistic tech company data."""

    class Meta:
        """Factory configuration for CompanySQL model."""
        model = CompanySQL
        sqlalchemy_session_persistence = "commit"
        # Will be set by calling code
        sqlalchemy_session = None

    id = Sequence(lambda n: n)
    name = Faker("company")
    url = Faker("url", schemes=["https"])
    active = fuzzy.FuzzyChoice([True, True, True, False])  # 75% active
    last_scraped = fuzzy.FuzzyDateTime(
        start_dt=datetime.now(UTC) - timedelta(days=30), end_dt=datetime.now(UTC)
    )
    scrape_count = fuzzy.FuzzyInteger(0, 50)
    success_rate = fuzzy.FuzzyFloat(0.5, 1.0)

    class Params:
        """Factory parameters for different company types."""

        # Trait for inactive companies
        inactive = factory.Trait(
            active=False, last_scraped=None, scrape_count=0, success_rate=1.0
        )

        # Trait for well-established companies with high scrape counts
        established = factory.Trait(
            scrape_count=fuzzy.FuzzyInteger(20, 100),
            success_rate=fuzzy.FuzzyFloat(0.8, 1.0),
            last_scraped=fuzzy.FuzzyDateTime(
                start_dt=datetime.now(UTC) - timedelta(days=7), end_dt=datetime.now(UTC)
            ),
        )

    @factory.post_generation
    def fix_url(obj, create, extracted, **kwargs):  # noqa: N805
        """Ensure company URLs end with /careers for realism."""
        if create and obj.url and not obj.url.endswith("/careers"):
            obj.url = f"{obj.url.rstrip('/')}/careers"


class JobFactory(SQLAlchemyModelFactory):
    """Factory for creating Job test records with realistic tech job data."""

    class Meta:
        """Factory configuration for JobSQL model."""
        model = JobSQL
        sqlalchemy_session_persistence = "commit"
        # Will be set by calling code
        sqlalchemy_session = None

    id = Sequence(lambda n: n)
    company_id = SubFactory(CompanyFactory)
    title = fuzzy.FuzzyChoice(AI_ML_TITLES)
    description = Faker("text", max_nb_chars=800)
    link = Faker("url", schemes=["https"])
    location = fuzzy.FuzzyChoice(TECH_LOCATIONS)

    # Realistic posting dates - mostly recent with some older
    posted_date = fuzzy.FuzzyDateTime(
        start_dt=datetime.now(UTC) - timedelta(days=45),
        end_dt=datetime.now(UTC) - timedelta(days=1),
    )

    # Salary ranges appropriate for AI/ML roles
    salary = LazyFunction(lambda: _generate_realistic_salary())

    favorite = fuzzy.FuzzyChoice([True, False, False, False])  # 25% favorited
    notes = Faker("sentence", nb_words=10)
    content_hash = Faker("md5")
    application_status = fuzzy.FuzzyChoice(APPLICATION_STATUSES)
    application_date = fuzzy.FuzzyDateTime(
        start_dt=datetime.now(UTC) - timedelta(days=30),
        end_dt=datetime.now(UTC),
    )
    archived = fuzzy.FuzzyChoice([True, False, False, False, False])  # 20% archived
    last_seen = fuzzy.FuzzyDateTime(
        start_dt=datetime.now(UTC) - timedelta(days=7), end_dt=datetime.now(UTC)
    )

    class Params:
        """Factory parameters for different job scenarios."""

        # Trait for senior-level positions
        senior = factory.Trait(
            title=fuzzy.FuzzyChoice(
                [
                    "Senior AI Engineer",
                    "Principal AI Engineer",
                    "Staff ML Engineer",
                    "Lead Data Scientist",
                ]
            ),
            salary=LazyFunction(lambda: _generate_senior_salary()),
        )

        # Trait for entry-level positions
        junior = factory.Trait(
            title=fuzzy.FuzzyChoice(
                [
                    "AI Engineer I",
                    "Junior ML Engineer",
                    "Associate Data Scientist",
                    "ML Engineer - New Grad",
                ]
            ),
            salary=LazyFunction(lambda: _generate_junior_salary()),
        )

        # Trait for remote jobs
        remote = factory.Trait(location="Remote")

        # Trait for favorited jobs with notes
        favorited = factory.Trait(
            favorite=True,
            notes=Faker("sentence", nb_words=15),
            application_status="Interested",
        )

        # Trait for applied jobs
        applied = factory.Trait(
            application_status="Applied",
            application_date=fuzzy.FuzzyDateTime(
                start_dt=datetime.now(UTC) - timedelta(days=14),
                end_dt=datetime.now(UTC) - timedelta(days=1),
            ),
        )


def _generate_realistic_salary() -> tuple[int | None, int | None]:
    """Generate realistic salary ranges for AI/ML roles."""
    # Base salaries for different experience levels
    base_ranges = [
        (90_000, 130_000),  # Junior
        (120_000, 180_000),  # Mid-level
        (160_000, 250_000),  # Senior
        (200_000, 350_000),  # Staff/Principal
    ]

    # Choose a range and add some variation
    base_min, base_max = fake.random_element(base_ranges)
    variation = fake.random_int(-10_000, 20_000)

    min_salary = base_min + variation
    max_salary = base_max + variation + fake.random_int(0, 50_000)

    return (max(min_salary, 70_000), min(max_salary, 400_000))


def _generate_senior_salary() -> tuple[int | None, int | None]:
    """Generate salary ranges specifically for senior roles."""
    base_min = fake.random_int(150_000, 200_000)
    base_max = base_min + fake.random_int(80_000, 150_000)
    return (base_min, min(base_max, 400_000))


def _generate_junior_salary() -> tuple[int | None, int | None]:
    """Generate salary ranges specifically for junior roles."""
    base_min = fake.random_int(70_000, 110_000)
    base_max = base_min + fake.random_int(40_000, 80_000)
    return (base_min, min(base_max, 180_000))


# Session-less factories for cases where we don't need database persistence
class CompanyDictFactory(factory.Factory):
    """Factory for creating company dictionaries without database persistence."""

    class Meta:
        """Factory configuration for dictionary model."""
        model = dict

    name = Faker("company")
    url = Faker("url", schemes=["https"])
    active = True
    last_scraped = fuzzy.FuzzyDateTime(
        start_dt=datetime.now(UTC) - timedelta(days=30), end_dt=datetime.now(UTC)
    )
    scrape_count = fuzzy.FuzzyInteger(1, 25)
    success_rate = fuzzy.FuzzyFloat(0.7, 1.0)


class JobDictFactory(factory.Factory):
    """Factory for creating job dictionaries without database persistence."""

    class Meta:
        """Factory configuration for dictionary model."""
        model = dict

    company = Faker("company")
    title = fuzzy.FuzzyChoice(AI_ML_TITLES)
    description = Faker("text", max_nb_chars=600)
    job_url = Faker("url", schemes=["https"])
    location = fuzzy.FuzzyChoice(TECH_LOCATIONS)
    date_posted = fuzzy.FuzzyDateTime(
        start_dt=datetime.now(UTC) - timedelta(days=30), end_dt=datetime.now(UTC)
    )
    min_amount = fuzzy.FuzzyInteger(80_000, 200_000)
    max_amount = factory.LazyAttribute(
        lambda obj: obj.min_amount + fake.random_int(20_000, 100_000)
    )


def create_sample_companies(session: Any, count: int = 5, **traits) -> list[CompanySQL]:
    """Create multiple companies with a shared session.

    Args:
        session: SQLAlchemy session to use
        count: Number of companies to create
        **traits: Factory traits to apply (e.g., inactive=True)

    Returns:
        List of created CompanySQL objects
    """
    CompanyFactory._meta.sqlalchemy_session = session
    return CompanyFactory.create_batch(count, **traits)


def create_sample_jobs(
    session: Any, count: int = 10, company: CompanySQL | None = None, **traits
) -> list[JobSQL]:
    """Create multiple jobs with a shared session.

    Args:
        session: SQLAlchemy session to use
        count: Number of jobs to create
        company: Specific company to associate jobs with
        **traits: Factory traits to apply (e.g., senior=True, remote=True)

    Returns:
        List of created JobSQL objects
    """
    JobFactory._meta.sqlalchemy_session = session

    if company:
        # Create jobs for specific company
        return JobFactory.create_batch(count, company_id=company.id, **traits)
    # Let factory create companies as needed
    return JobFactory.create_batch(count, **traits)
