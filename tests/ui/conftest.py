"""UI test fixtures for Streamlit component testing.

This module provides fixtures for testing Streamlit UI components with proper
mocking of Streamlit functionality and service layer dependencies.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.schemas import Company, Job
from tests.ui.components.test_utils import MockSessionState


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions and components for UI testing."""
    mocks = {}

    # Start all the patches
    st_patches = [
        ("title", patch("streamlit.title")),
        ("markdown", patch("streamlit.markdown")),
        ("text_input", patch("streamlit.text_input")),
        ("button", patch("streamlit.button")),
        ("selectbox", patch("streamlit.selectbox")),
        ("toggle", patch("streamlit.toggle")),
        ("success", patch("streamlit.success")),
        ("error", patch("streamlit.error")),
        ("info", patch("streamlit.info")),
        ("warning", patch("streamlit.warning")),
        ("columns", patch("streamlit.columns")),
        ("container", patch("streamlit.container")),
        ("expander", patch("streamlit.expander")),
        ("form", patch("streamlit.form")),
        ("form_submit_button", patch("streamlit.form_submit_button")),
        ("tabs", patch("streamlit.tabs")),
        ("dialog", patch("streamlit.dialog")),
        ("text_area", patch("streamlit.text_area")),
        ("link_button", patch("streamlit.link_button")),
        ("metric", patch("streamlit.metric")),
        ("rerun", patch("streamlit.rerun")),
        ("spinner", patch("streamlit.spinner")),
        ("progress", patch("streamlit.progress")),
        ("data_editor", patch("streamlit.data_editor")),
        ("download_button", patch("streamlit.download_button")),
        ("caption", patch("streamlit.caption")),
        ("status", patch("streamlit.status")),
    ]

    # Start all patches and collect mocks
    started_patches = []
    for name, p in st_patches:
        mock_obj = p.start()
        mocks[name] = mock_obj
        started_patches.append(p)

    try:
        # Configure columns to return mock column objects that work as context managers
        def mock_columns_func(*args, **_kwargs):
            """Mock columns function that returns appropriate number of columns."""
            if args:
                num_cols = args[0] if isinstance(args[0], int) else len(args[0])
            else:
                num_cols = 2  # Default

            columns = []
            for _i in range(num_cols):
                col = MagicMock()
                # Configure as context manager
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                columns.append(col)
            return columns

        mocks["columns"].side_effect = mock_columns_func

        # Configure text input to return empty string by default
        # This prevents MagicMock string operation errors
        mocks["text_input"].return_value = ""
        mocks["selectbox"].return_value = None
        mocks["toggle"].return_value = False
        mocks["button"].return_value = False

        # Configure container to return mock container
        mock_container_obj = MagicMock()
        mocks["container"].return_value.__enter__ = Mock(
            return_value=mock_container_obj,
        )
        mocks["container"].return_value.__exit__ = Mock(return_value=None)

        # Configure expander to return mock expander
        mock_expander_obj = MagicMock()
        mocks["expander"].return_value.__enter__ = Mock(return_value=mock_expander_obj)
        mocks["expander"].return_value.__exit__ = Mock(return_value=None)

        # Configure form to return mock form
        mock_form_obj = MagicMock()
        mocks["form"].return_value.__enter__ = Mock(return_value=mock_form_obj)
        mocks["form"].return_value.__exit__ = Mock(return_value=None)

        # Configure tabs to return mock tab objects
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mocks["tabs"].return_value = [mock_tab1, mock_tab2, mock_tab3]

        # Configure spinner context manager
        mock_spinner_obj = MagicMock()
        mocks["spinner"].return_value.__enter__ = Mock(return_value=mock_spinner_obj)
        mocks["spinner"].return_value.__exit__ = Mock(return_value=None)

        # Configure status context manager - CRITICAL FOR BACKGROUND TASK TESTING
        mock_status_obj = MagicMock()
        mock_status_obj.write = Mock()
        mock_status_obj.update = Mock()
        mock_status_obj.progress = Mock()
        mock_status_obj.error = Mock()
        mock_status_obj.success = Mock()

        # Create a proper context manager that returns the status object
        mock_status_context = MagicMock()
        mock_status_context.__enter__ = Mock(return_value=mock_status_obj)
        mock_status_context.__exit__ = Mock(return_value=None)
        mocks["status"].return_value = mock_status_context

        # Configure dialog decorator to act as a passthrough decorator
        def mock_dialog_decorator(*_args, **_kwargs):
            """Mock dialog decorator that creates a passthrough function."""

            def decorator(func):
                # Create a wrapper that acts like both the original function
                # and has an open method
                def wrapper(_self, *args, **kwargs):
                    # Just call the original function directly
                    return func(*args, **kwargs)

                # Add an 'open' method that also calls the original function
                wrapper.open = Mock(
                    side_effect=lambda *args, **kwargs: func(*args, **kwargs),
                )

                # Copy function attributes
                wrapper.__name__ = getattr(func, "__name__", "wrapper")
                wrapper.__doc__ = func.__doc__

                return wrapper

            return decorator

        mocks["dialog"] = mock_dialog_decorator

        # Add extra references for convenience
        mocks.update(
            {
                "tab1": mock_tab1,
                "tab2": mock_tab2,
                "tab3": mock_tab3,
            },
        )

        yield mocks
    finally:
        # Stop all patches
        for p in started_patches:
            p.stop()


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state for UI testing."""
    session_state = MockSessionState()

    with patch("streamlit.session_state", session_state):
        yield session_state


@pytest.fixture
def sample_company_dto():
    """Create a sample company DTO for testing."""
    return Company(
        id=1,
        name="Tech Corp",
        url="https://techcorp.com/careers",
        active=True,
        last_scraped=datetime.now(UTC),
        scrape_count=5,
        success_rate=0.8,
    )


@pytest.fixture
def sample_companies_dto():
    """Create a list of sample company DTOs for testing."""
    return [
        Company(
            id=1,
            name="Tech Corp",
            url="https://techcorp.com/careers",
            active=True,
            last_scraped=datetime.now(UTC),
            scrape_count=5,
            success_rate=0.8,
        ),
        Company(
            id=2,
            name="DataCo",
            url="https://dataco.com/jobs",
            active=False,
            last_scraped=None,
            scrape_count=0,
            success_rate=1.0,
        ),
        Company(
            id=3,
            name="AI Solutions",
            url="https://aisolutions.com/careers",
            active=True,
            last_scraped=datetime.now(UTC),
            scrape_count=12,
            success_rate=0.92,
        ),
    ]


@pytest.fixture
def sample_job_dto():
    """Create a sample job DTO for testing."""
    return Job(
        id=1,
        company_id=1,
        company="Tech Corp",
        title="Senior AI Engineer",
        description=(
            "We are looking for an experienced AI engineer to join our team "
            "and work on cutting-edge machine learning projects."
        ),
        link="https://techcorp.com/careers/ai-engineer-123",
        location="San Francisco, CA",
        posted_date=datetime.now(UTC),
        salary=(120000, 180000),
        favorite=False,
        notes="Interesting role with good growth potential",
        content_hash="hash123",
        application_status="New",
        application_date=None,
        archived=False,
        last_seen=datetime.now(UTC),
    )


@pytest.fixture
def sample_jobs_dto():
    """Create a list of sample job DTOs for testing."""
    base_time = datetime.now(UTC)

    return [
        Job(
            id=1,
            company_id=1,
            company="Tech Corp",
            title="Senior AI Engineer",
            description=(
                "We are looking for an experienced AI engineer to join our team "
                "and work on cutting-edge machine learning projects."
            ),
            link="https://techcorp.com/careers/ai-engineer-123",
            location="San Francisco, CA",
            posted_date=base_time,
            salary=(120000, 180000),
            favorite=True,
            notes="Very interested in this role",
            content_hash="hash123",
            application_status="Interested",
            application_date=None,
            archived=False,
            last_seen=base_time,
        ),
        Job(
            id=2,
            company_id=2,
            company="DataCo",
            title="Machine Learning Specialist",
            description=(
                "Join our data science team to build predictive models "
                "and analytics solutions."
            ),
            link="https://dataco.com/jobs/ml-specialist-456",
            location="Remote",
            posted_date=base_time,
            salary=(100000, 150000),
            favorite=False,
            notes="",
            content_hash="hash456",
            application_status="Applied",
            application_date=base_time,
            archived=False,
            last_seen=base_time,
        ),
        Job(
            id=3,
            company_id=3,
            company="AI Solutions",
            title="Data Scientist",
            description=(
                "Exciting opportunity to work with large datasets "
                "and develop ML algorithms."
            ),
            link="https://aisolutions.com/careers/data-scientist-789",
            location="New York, NY",
            posted_date=base_time,
            salary=(110000, 160000),
            favorite=True,
            notes="Great company culture",
            content_hash="hash789",
            application_status="New",
            application_date=None,
            archived=False,
            last_seen=base_time,
        ),
        Job(
            id=4,
            company_id=1,
            company="Tech Corp",
            title="Python Developer",
            description="Backend development role focusing on Python and Django.",
            link="https://techcorp.com/careers/python-dev-101",
            location="Seattle, WA",
            posted_date=base_time,
            salary=(90000, 130000),
            favorite=False,
            notes="",
            content_hash="hash101",
            application_status="Rejected",
            application_date=None,
            archived=False,
            last_seen=base_time,
        ),
    ]


@pytest.fixture
def mock_company_service():
    """Mock CompanyService for testing UI components."""
    with (
        patch("src.ui.pages.companies.CompanyService") as mock_service_companies,
        patch("src.ui.pages.jobs.CompanyService") as mock_service_jobs,
        patch("src.services.company_service.CompanyService") as mock_service_core,
    ):
        # Configure all mock instances with the same behavior
        for mock_service in [
            mock_service_companies,
            mock_service_jobs,
            mock_service_core,
        ]:
            mock_service.get_all_companies.return_value = []
            mock_service.add_company.return_value = Company(
                id=1,
                name="Test Company",
                url="https://test.com",
                active=True,
            )
            mock_service.toggle_company_active.return_value = True
            mock_service.get_active_companies_count.return_value = 0
            mock_service.get_company_by_id.return_value = None

        # Return all mocks so tests can access the right one
        yield {
            "companies_page": mock_service_companies,
            "jobs_page": mock_service_jobs,
            "core": mock_service_core,
        }


@pytest.fixture
def mock_job_service():
    """Mock JobService for testing UI components."""
    # Patch in multiple locations where JobService is used
    with (
        patch("src.ui.pages.jobs.JobService") as mock_service_jobs,
        patch("src.ui.components.cards.job_card.JobService") as mock_service_cards,
        patch("src.services.job_service.JobService") as mock_service_core,
    ):
        # Configure all mock instances with the same behavior
        for mock_service in [
            mock_service_jobs,
            mock_service_cards,
            mock_service_core,
        ]:
            mock_service.get_filtered_jobs.return_value = []
            mock_service.update_job_status.return_value = True
            mock_service.toggle_favorite.return_value = True
            mock_service.update_notes.return_value = True
            mock_service.bulk_update_jobs.return_value = True
            mock_service.get_all_jobs.return_value = []
            mock_service.get_job_by_id.return_value = None
            mock_service.get_active_companies.return_value = []

        # Return all mocks so tests can access the right one
        yield {
            "jobs_page": mock_service_jobs,
            "cards": mock_service_cards,
            "core": mock_service_core,
        }


@pytest.fixture
def prevent_real_system_execution():
    """Global autouse fixture to prevent real system execution during tests.

    This fixture mocks all external dependencies and I/O operations to ensure
    complete test isolation.
    """
    with (
        # Mock all scraping and external API calls
        patch(
            "src.scraper.scrape_all",
            return_value={
                "inserted": 0,
                "updated": 0,
                "archived": 0,
                "deleted": 0,
                "skipped": 0,
            },
        ) as mock_scrape_all,
        patch(
            "src.ui.pages.jobs._execute_scraping_safely",
            return_value={
                "inserted": 0,
                "updated": 0,
                "archived": 0,
                "deleted": 0,
                "skipped": 0,
            },
        ) as mock_execute_scraping,
        patch("asyncio.run") as mock_asyncio_run,
        # Mock database connections and sessions
        patch("src.database.get_session") as mock_get_session,
        patch("sqlmodel.Session") as mock_session,
        # Mock external HTTP requests
        patch("requests.get") as mock_requests_get,
        patch("requests.post") as mock_requests_post,
        patch("httpx.AsyncClient"),
        # Mock file system operations
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("builtins.open", create=True),
        # Mock logging to prevent log spam
        patch("logging.getLogger") as mock_get_logger,
        # Mock streamlit dialog decorator at import time
        patch("streamlit.dialog") as mock_dialog_decorator,
    ):
        # Configure mock behaviors
        mock_asyncio_run.return_value = {
            "inserted": 0,
            "updated": 0,
            "archived": 0,
            "deleted": 0,
            "skipped": 0,
        }
        mock_get_logger.return_value = Mock()

        # Configure session mock
        mock_session_instance = Mock()
        mock_session.return_value.__enter__ = Mock(return_value=mock_session_instance)
        mock_session.return_value.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_session_instance

        # Configure HTTP mocks
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.text = "<html></html>"
        mock_requests_get.return_value = mock_response
        mock_requests_post.return_value = mock_response

        # Configure dialog decorator to completely bypass Streamlit dialog behavior
        def mock_dialog_func(*_args, **_kwargs):
            """Mock dialog decorator that creates a passthrough function."""

            def decorator(func):
                # Create a wrapper that acts like both the original function
                # and has an open method
                def wrapper(_self, *args, **kwargs):
                    # Just call the original function directly
                    return func(*args, **kwargs)

                # Add an 'open' method that also calls the original function
                wrapper.open = Mock(
                    side_effect=lambda *args, **kwargs: func(*args, **kwargs),
                )

                # Copy function attributes
                wrapper.__name__ = getattr(func, "__name__", "wrapper")
                wrapper.__doc__ = func.__doc__

                return wrapper

            return decorator

        mock_dialog_decorator.side_effect = mock_dialog_func

        yield {
            "scrape_all": mock_scrape_all,
            "execute_scraping": mock_execute_scraping,
            "asyncio_run": mock_asyncio_run,
            "session": mock_session_instance,
            "requests_get": mock_requests_get,
            "logger": mock_get_logger.return_value,
        }


@pytest.fixture
def ensure_proper_job_data_types():
    """Fixture to ensure Job DTOs maintain proper data types during tests.

    This prevents MagicMock string operation errors by ensuring all Job
    attributes are proper Python types, not mock objects.
    """

    def validate_job_list(jobs):
        """Validate jobs list contains proper Job objects with string attributes."""
        if not jobs:
            return jobs

        validated_jobs = []
        for job in jobs:
            # Ensure it's a proper Job DTO, not a Mock
            if (
                hasattr(job, "title")
                and hasattr(job, "company")
                and hasattr(job, "description")
            ):
                # Ensure all string attributes are actual strings
                if (
                    isinstance(job.title, str)
                    and isinstance(job.company, str)
                    and isinstance(job.description, str)
                ):
                    validated_jobs.append(job)
                else:
                    # Convert any mock attributes to strings
                    job.title = str(job.title) if job.title else ""
                    job.company = str(job.company) if job.company else ""
                    job.description = str(job.description) if job.description else ""
                    job.location = str(job.location) if job.location else ""
                    validated_jobs.append(job)
            else:
                # Skip invalid objects
                continue
        return validated_jobs

    return validate_job_list


@pytest.fixture
def mock_logging():
    """Mock logging to prevent log messages during testing."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger
