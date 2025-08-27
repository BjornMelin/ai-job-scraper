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
    link = Sequence(lambda n: f"https://jobs.example{n % 50}.com/job/{n}")
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


def create_realistic_dataset(
    session: Any,
    companies: int = 10,
    jobs_per_company: int = 5,
    include_inactive_companies: bool = True,
    include_archived_jobs: bool = True,
    senior_ratio: float = 0.3,
    remote_ratio: float = 0.4,
    favorited_ratio: float = 0.1,
    **kwargs,
) -> dict[str, Any]:
    """Create a comprehensive, realistic test dataset with companies and jobs.

    This function generates a complete test dataset with realistic distributions
    of companies and jobs, including various states and scenarios for testing.

    Args:
        session: SQLAlchemy session to use.
        companies: Number of companies to create.
        jobs_per_company: Average number of jobs per company.
        include_inactive_companies: Whether to include some inactive companies.
        include_archived_jobs: Whether to include some archived jobs.
        senior_ratio: Ratio of senior-level positions (0.0 to 1.0).
        remote_ratio: Ratio of remote positions (0.0 to 1.0).
        favorited_ratio: Ratio of favorited jobs (0.0 to 1.0).
        **kwargs: Additional keyword arguments passed to factories.

    Returns:
        Dictionary containing:
            - companies: List of created CompanySQL objects
            - jobs: List of created JobSQL objects
            - stats: Statistics about the generated dataset
    """
    import random

    random.seed(42)  # For reproducibility in tests

    # Configure factories with session
    CompanyFactory._meta.sqlalchemy_session = session
    JobFactory._meta.sqlalchemy_session = session

    # Create companies with various states
    companies_list = []

    # Calculate company distribution
    active_companies = int(companies * 0.8) if include_inactive_companies else companies
    inactive_companies = (
        companies - active_companies if include_inactive_companies else 0
    )

    # Create active companies with varying success rates
    for i in range(active_companies):
        if i < active_companies * 0.3:  # 30% are well-established
            company = CompanyFactory.create(established=True, **kwargs)
        else:
            company = CompanyFactory.create(active=True, **kwargs)
        companies_list.append(company)

    # Create inactive companies if requested
    for _ in range(inactive_companies):
        company = CompanyFactory.create(inactive=True, **kwargs)
        companies_list.append(company)

    # Create jobs for each company
    jobs_list = []
    total_senior = 0
    total_remote = 0
    total_favorited = 0
    total_archived = 0

    for company in companies_list:
        # Skip job creation for some inactive companies
        if not company.active and random.random() < 0.5:
            continue

        # Vary job count per company (normal distribution around mean)
        job_count = max(1, int(random.gauss(jobs_per_company, jobs_per_company * 0.3)))

        for _ in range(job_count):
            # Determine job traits based on ratios
            traits = {}

            # Senior level positions
            if random.random() < senior_ratio:
                traits["senior"] = True
                total_senior += 1
            elif random.random() < 0.2:  # 20% junior positions
                traits["junior"] = True

            # Remote positions
            if random.random() < remote_ratio:
                traits["remote"] = True
                total_remote += 1

            # Favorited jobs
            if random.random() < favorited_ratio:
                traits["favorited"] = True
                total_favorited += 1

            # Applied jobs (subset of favorited)
            if traits.get("favorited") and random.random() < 0.5:
                traits["applied"] = True

            # Create the job
            job = JobFactory.create(company_id=company.id, **traits, **kwargs)

            # Archive some jobs if requested
            if include_archived_jobs and random.random() < 0.15:  # 15% archived
                job.archived = True
                total_archived += 1

            jobs_list.append(job)

    # Commit all changes
    session.commit()

    # Calculate statistics
    avg_jobs_per_company = len(jobs_list) / len(companies_list) if companies_list else 0

    stats = {
        "total_companies": len(companies_list),
        "active_companies": active_companies,
        "inactive_companies": inactive_companies,
        "total_jobs": len(jobs_list),
        "avg_jobs_per_company": round(avg_jobs_per_company, 2),
        "senior_jobs": total_senior,
        "remote_jobs": total_remote,
        "favorited_jobs": total_favorited,
        "archived_jobs": total_archived,
        "senior_ratio_actual": round(total_senior / len(jobs_list), 2)
        if jobs_list
        else 0,
        "remote_ratio_actual": round(total_remote / len(jobs_list), 2)
        if jobs_list
        else 0,
        "favorited_ratio_actual": round(total_favorited / len(jobs_list), 2)
        if jobs_list
        else 0,
        "archived_ratio_actual": round(total_archived / len(jobs_list), 2)
        if jobs_list
        else 0,
    }

    return {
        "companies": companies_list,
        "jobs": jobs_list,
        "stats": stats,
    }
