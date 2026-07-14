"""Simplified pytest configuration for factory/fixture minimalist approach.

Standard pytest fixtures with minimal factory-boy integration. No custom frameworks.
Supports spec testing requirements with st.testing.AppTest fixtures.

Key simplifications:
- Standard pytest patterns only
- Essential database fixtures
- st.testing.AppTest for Streamlit testing (SPEC-004)
- Minimal factory integration without auto-registration
- <60 lines vs 188 lines (68% reduction)
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session
from src.config import Settings
from src.database import _create_engine_impl
from src.database_models import AppSQLModel

pytest_plugins = ["tests.fixtures.jobspy_fixtures"]

# =============================================================================
# DATABASE FIXTURES
# =============================================================================


@pytest.fixture
def test_engine(tmp_path: Path) -> Generator[Engine, None, None]:
    """Fresh SQLite engine for each test."""
    engine = _create_engine_impl(f"sqlite:///{tmp_path / 'test.db'}")
    AppSQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(test_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Generator[Session]:
    """Session bound to the same engine used by application services."""
    from src import database

    from tests.factories import CompanyFactory, JobFactory

    monkeypatch.setattr(database, "get_engine", lambda database_url=None: test_engine)
    with Session(test_engine, expire_on_commit=False) as db_session:
        CompanyFactory._meta.sqlalchemy_session = db_session
        JobFactory._meta.sqlalchemy_session = db_session
        CompanyFactory.reset_sequence()
        JobFactory.reset_sequence()
        try:
            yield db_session
        finally:
            db_session.rollback()
            CompanyFactory._meta.sqlalchemy_session = None
            JobFactory._meta.sqlalchemy_session = None


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def test_settings():
    """Test configuration."""
    return Settings(
        db_url="sqlite:///:memory:",
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture(autouse=True)
def configure_test_environment():
    """Configure test environment."""
    os.environ["TESTING"] = "true"
    yield
    os.environ.pop("TESTING", None)


# =============================================================================
# STREAMLIT TESTING FIXTURES
# =============================================================================


@pytest.fixture
def app_test():
    """Streamlit's native application test harness."""
    from streamlit.testing.v1 import AppTest

    return AppTest


@pytest.fixture
def jobs_app_test(app_test):
    """AppTest instance for jobs page testing."""
    return app_test.from_file("src/ui/pages/jobs.py")


@pytest.fixture
def searches_app_test(app_test):
    """AppTest instance for saved-search testing."""
    return app_test.from_file("src/ui/pages/searches.py")


@pytest.fixture
def insights_app_test(app_test):
    """AppTest instance for read-only insights testing."""
    return app_test.from_file("src/ui/pages/insights.py")
