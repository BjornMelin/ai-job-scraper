"""Pytest configuration and fixtures for AI Job Scraper tests.

This module provides test fixtures and configuration for the AI Job Scraper
test suite, including database session management, sample data creation,
and test settings configuration.
"""

import os
import tempfile

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sqlalchemy import create_engine, event, pool
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from src.config import Settings
from tests.factories import CompanyFactory, JobFactory
from tests.fixtures.vcr import get_cassettes_dir


@pytest.fixture(scope="session")
def engine(worker_id, tmp_path_factory):
    """Create a temporary SQLite engine for the test session.

    For parallel execution with pytest-xdist, each worker gets its own
    database file to prevent conflicts. Uses StaticPool to ensure schema
    and data persist across session connections.

    Args:
        worker_id: pytest-xdist worker ID (e.g., 'gw0', 'gw1', 'master')
        tmp_path_factory: Pytest factory for creating temp directories
    """
    if worker_id == "master":
        # Single-threaded execution - use in-memory database
        db_url = "sqlite:///:memory:"
    else:
        # Parallel execution - use worker-specific database file
        tmp_path = tmp_path_factory.mktemp("db_worker")
        db_path = tmp_path / f"test_{worker_id}.db"
        db_url = f"sqlite:///{db_path}"

    engine = create_engine(
        db_url,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            # Performance optimizations for testing
            "isolation_level": None,  # Autocommit mode for faster tests
        },
        # Reduce connection overhead
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,  # Set via pytest --sql-echo for debugging
    )

    # Configure SQLite for testing performance
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record):
        """Optimize SQLite for testing performance."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")  # Enforce constraints
        cursor.execute("PRAGMA synchronous=OFF")  # Faster writes
        cursor.execute("PRAGMA journal_mode=MEMORY")  # In-memory journaling
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")  # Memory temp storage
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
        cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine) -> Generator[Session, None, None]:
    """Create a new database session for each test.

    Uses transaction rollback for isolation without recreation overhead.
    Provides complete test isolation with automatic rollback.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    # Configure factories to use this session
    CompanyFactory._meta.sqlalchemy_session = session
    JobFactory._meta.sqlalchemy_session = session

    try:
        yield session
    finally:
        # Reset factory sequences to prevent ID conflicts
        CompanyFactory.reset_sequence()
        JobFactory.reset_sequence()

        # Clean up session and transaction
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="session")
def parallel_engine_pool():
    """Create a pool of engines for parallel test execution.

    Returns a factory function that creates isolated engines
    for parallel test runners to avoid connection conflicts.
    """

    def create_parallel_engine():
        return create_engine(
            "sqlite:///:memory:",
            poolclass=pool.StaticPool,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

    return create_parallel_engine


@pytest.fixture
def transactional_session(session) -> Generator[Session, None, None]:
    """Session fixture with explicit transaction control.

    Useful for tests that need to test transaction behavior
    or rollback specific operations.
    """
    # Start a nested transaction (savepoint)
    nested = session.begin_nested()

    try:
        yield session
    finally:
        nested.rollback()


@pytest.fixture
def test_settings():
    """Create test settings with temporary values."""
    return Settings(
        openai_api_key="test-key-123",
        groq_api_key="test-groq-key",
        use_groq=False,
        proxy_pool=[],
        use_proxies=False,
        use_checkpointing=False,
        db_url="sqlite:///:memory:",
        extraction_model="gpt-4o-mini",
    )


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts.

    Automatically cleaned up after the test session.
    """
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def test_data_dir(temp_dir) -> Path:
    """Create a directory for test data files."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def cassettes_cleanup():
    """Clean up VCR cassettes after each test.

    Ensures fresh recordings for each test run when needed.
    """
    yield

    # Optional cleanup based on environment variable
    if os.getenv("CLEAN_CASSETTES", "false").lower() == "true":
        cassettes_dir = get_cassettes_dir()
        for cassette_file in cassettes_dir.rglob("*.yaml"):
            cassette_file.unlink(missing_ok=True)


@pytest.fixture
def sample_company(session: Session):
    """Create and insert a sample company for testing."""
    CompanyFactory._meta.sqlalchemy_session = session
    return CompanyFactory.create()


@pytest.fixture
def sample_job(session: Session):
    """Create and insert a sample job for testing."""
    JobFactory._meta.sqlalchemy_session = session
    return JobFactory.create()


@pytest.fixture
def multiple_companies(session: Session):
    """Create multiple companies for testing list operations."""
    CompanyFactory._meta.sqlalchemy_session = session
    return CompanyFactory.create_batch(5)


@pytest.fixture
def multiple_jobs(session: Session):
    """Create multiple jobs for testing pagination and filtering."""
    return JobFactory.create_batch(20, session=session)


@pytest.fixture
def realistic_dataset(session: Session):
    """Create a realistic dataset for integration testing."""
    from tests.utils.db_utils import create_test_data

    return create_test_data(session, preset="realistic")


@pytest.fixture
def sample_job_dict():
    """Create a sample job dictionary for testing."""
    return {
        "company": "Test Company",
        "title": "Senior AI Engineer",
        "description": "We are looking for an experienced AI engineer.",
        "link": "https://test.com/careers/ai-engineer-123",
        "location": "San Francisco, CA",
        "posted_date": datetime.now(UTC),
        "salary": "$100k-150k",
    }


# Configuration fixtures for different test environments


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure the test environment with proper settings."""
    # Set test-specific environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce test noise

    yield

    # Cleanup
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)


@pytest.fixture
def isolated_test_db():
    """Create a completely isolated test database for specific tests.

    Use this for tests that need a fresh database without any
    session-level state or when testing database migrations.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)
        yield engine
    finally:
        Path(db_path).unlink(missing_ok=True)


# Utility context managers for tests


@contextmanager
def assert_database_changes(session: Session, table_name: str, expected_change: int):
    """Context manager to assert database row count changes.

    Args:
        session: Database session
        table_name: Table to monitor
        expected_change: Expected change in row count (can be negative)
    """
    initial_count = session.execute(f"SELECT COUNT(*) FROM {table_name}").scalar()

    yield

    final_count = session.execute(f"SELECT COUNT(*) FROM {table_name}").scalar()

    actual_change = final_count - initial_count
    assert actual_change == expected_change, (
        f"Expected {expected_change} row change in {table_name}, "
        f"got {actual_change} (from {initial_count} to {final_count})"
    )


# Performance monitoring fixtures


@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage."""
    import os
    import time

    import psutil

    process = psutil.Process(os.getpid())

    start_time = time.time()
    start_memory = process.memory_info().rss

    yield

    end_time = time.time()
    end_memory = process.memory_info().rss
    end_cpu = process.cpu_percent()

    # Store metrics for analysis
    test_metrics = {
        "duration": end_time - start_time,
        "memory_delta": end_memory - start_memory,
        "cpu_usage": end_cpu,
    }

    # Can be accessed in test cleanup or reporting
    return test_metrics
