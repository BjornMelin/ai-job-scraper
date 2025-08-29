"""Example tests demonstrating simplified test infrastructure.

Shows proper usage of:
- Standard pytest fixtures (not custom frameworks)
- Factory-boy patterns for test data
- st.testing.AppTest for Streamlit testing (SPEC-004)
- Simple, maintainable test patterns

These examples follow library-first directive and KISS principles.
"""

import pytest


def test_database_session(session):
    """Test basic database session fixture."""
    assert session is not None
    assert session.bind is not None


def test_company_factory(session):
    """Test CompanyFactory creates valid test data."""
    from tests.factories import CompanyFactory

    # Set session for factory
    CompanyFactory._meta.sqlalchemy_session = session

    company = CompanyFactory.create()
    assert company.name.startswith("Test Company")
    assert company.url.endswith("/careers")
    assert company.active is True


def test_job_factory(session):
    """Test JobFactory with company relationship."""
    from tests.factories import CompanyFactory, JobFactory

    # Set session for factories
    CompanyFactory._meta.sqlalchemy_session = session
    JobFactory._meta.sqlalchemy_session = session

    job = JobFactory.create()
    assert job.title.startswith("Software Engineer")
    assert job.location == "Remote"
    assert job.company_id is not None


@pytest.mark.skip(reason="AppTest requires valid Streamlit files")
def test_app_test_fixture(app_test):
    """Test st.testing.AppTest fixture (SPEC-004)."""
    assert app_test is not None


def test_settings_fixture(test_settings):
    """Test configuration fixture."""
    assert test_settings.openai_api_key == "test-key"
    assert test_settings.use_groq is False
    assert test_settings.db_url == "sqlite:///:memory:"


def test_temp_dir_fixture(temp_dir):
    """Test temporary directory fixture."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()

    # Can create files in temp dir
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.read_text() == "test content"
