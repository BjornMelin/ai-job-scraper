"""Database utilities for testing.

This module provides helper functions for database setup, data generation,
and test database management to streamline testing workflows.
"""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel
from tests.factories import (
    create_realistic_dataset,
    create_sample_companies,
    create_sample_jobs,
)


def create_test_engine(database_url: str = "sqlite:///:memory:") -> Any:
    """Create a test database engine with optimized settings.

    Args:
        database_url: Database URL to connect to (defaults to in-memory SQLite)

    Returns:
        Configured SQLAlchemy engine for testing
    """
    engine = create_engine(
        database_url,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            "isolation_level": None,  # Autocommit mode for faster tests
        },
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,  # Set to True for SQL debugging
    )

    # Enable foreign key constraints for SQLite
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Enable foreign key constraints and optimize SQLite for testing."""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=OFF")  # Faster writes
            cursor.execute("PRAGMA journal_mode=MEMORY")  # In-memory journaling
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.close()

    return engine


def setup_test_database(engine: Any) -> None:
    """Set up test database with all tables created.

    Args:
        engine: SQLAlchemy engine to create tables on
    """
    SQLModel.metadata.create_all(engine)


def teardown_test_database(engine: Any) -> None:
    """Clean up test database by dropping all tables.

    Args:
        engine: SQLAlchemy engine to drop tables from
    """
    SQLModel.metadata.drop_all(engine)


@contextmanager
def test_session_context(engine: Any):
    """Context manager for test database sessions with automatic rollback.

    Creates a transactional session that automatically rolls back
    changes at the end, ensuring test isolation.

    Args:
        engine: SQLAlchemy engine to create session from

    Yields:
        Database session with transaction rollback on exit
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


def create_test_data(
    session: Session, preset: str = "default", **kwargs
) -> dict[str, Any]:
    """Create test data using predefined presets.

    Args:
        session: Database session for persistence
        preset: Data preset to use ("minimal", "default", "large", "realistic")
        **kwargs: Override parameters for data generation

    Returns:
        Dictionary containing created data and metadata
    """
    presets = {
        "minimal": {"num_companies": 2, "jobs_per_company": 3},
        "default": {"num_companies": 5, "jobs_per_company": 10},
        "large": {"num_companies": 20, "jobs_per_company": 25},
        "realistic": {"num_companies": 8, "jobs_per_company": 15},
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    # Merge preset with custom overrides
    params = {**presets[preset], **kwargs}

    return create_realistic_dataset(session, **params)


def benchmark_data_creation(session: Session, scale: int = 1000) -> dict[str, Any]:
    """Create large-scale data for performance benchmarking.

    Args:
        session: Database session for persistence
        scale: Scale factor for data creation (default: 1000 jobs)

    Returns:
        Dictionary containing benchmark data and timing metrics
    """
    import time

    start_time = time.time()

    # Create companies first
    num_companies = max(10, scale // 100)
    jobs_per_company = scale // num_companies

    companies = create_sample_companies(session, num_companies)
    companies_time = time.time()

    # Create jobs distributed across companies
    all_jobs = []
    for company in companies:
        jobs = create_sample_jobs(session, jobs_per_company, company=company)
        all_jobs.extend(jobs)

    jobs_time = time.time()

    return {
        "companies": companies,
        "jobs": all_jobs,
        "scale": scale,
        "timing": {
            "companies_created": companies_time - start_time,
            "jobs_created": jobs_time - companies_time,
            "total_time": jobs_time - start_time,
        },
        "metrics": {
            "companies_per_second": len(companies) / (companies_time - start_time),
            "jobs_per_second": len(all_jobs) / (jobs_time - companies_time),
        },
    }


def reset_test_data(session: Session) -> None:
    """Reset all test data by truncating tables.

    Args:
        session: Database session to reset
    """
    # Get all table names
    tables = SQLModel.metadata.tables.keys()

    # Disable foreign key checks temporarily
    session.execute("PRAGMA foreign_keys=OFF")

    # Delete all data from all tables
    for table_name in tables:
        session.execute(f"DELETE FROM {table_name}")  # noqa: S608

    # Re-enable foreign key checks
    session.execute("PRAGMA foreign_keys=ON")
    session.commit()


def assert_database_state(session: Session, expected: dict[str, int]) -> None:
    """Assert that database tables have expected row counts.

    Args:
        session: Database session to check
        expected: Dictionary mapping table names to expected row counts

    Raises:
        AssertionError: If actual counts don't match expected counts
    """
    for table_name, expected_count in expected.items():
        result = session.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
        actual_count = result.scalar()
        assert actual_count == expected_count, (
            f"Table {table_name}: expected {expected_count} rows, "
            f"got {actual_count} rows"
        )


def with_test_data(preset: str = "default", **kwargs) -> Callable:
    """Decorator to automatically create test data for test functions.

    Args:
        preset: Data preset to use
        **kwargs: Additional parameters for data creation

    Returns:
        Decorator function that injects test data
    """

    def decorator(test_func: Callable) -> Callable:
        def wrapper(*args, **test_kwargs):
            # Assume first argument is session (pytest fixture)
            session = args[0] if args else test_kwargs.get("session")
            if not session:
                raise ValueError("No session found in test arguments")

            # Create test data
            test_data = create_test_data(session, preset, **kwargs)

            # Inject test data into test kwargs
            test_kwargs["test_data"] = test_data

            return test_func(*args, **test_kwargs)

        return wrapper

    return decorator


class DatabaseTestCase:
    """Base class for database test cases with common utilities.

    Provides common database testing utilities and setup/teardown methods
    that can be inherited by test classes requiring database access.
    """

    def setup_method(self, method):
        """Set up method called before each test method."""
        self.engine = create_test_engine()
        setup_test_database(self.engine)

    def teardown_method(self, method):
        """Tear down method called after each test method."""
        teardown_test_database(self.engine)

    def get_session(self) -> Session:
        """Get a new database session for testing.

        Returns:
            New database session
        """
        return Session(self.engine)

    def create_test_data(self, preset: str = "default", **kwargs) -> dict[str, Any]:
        """Create test data using the current engine.

        Args:
            preset: Data preset to use
            **kwargs: Additional parameters for data creation

        Returns:
            Dictionary containing created data and metadata
        """
        with self.get_session() as session:
            return create_test_data(session, preset, **kwargs)
