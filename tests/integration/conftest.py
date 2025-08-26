"""Integration test fixtures.

This module provides minimal fixtures needed for integration tests
without the full UI test setup from tests/ui/conftest.py.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.ui.components.test_utils import MockSessionState


@pytest.fixture
def mock_streamlit():
    """Minimal Streamlit mock for integration tests."""
    mocks = {}

    # Start basic patches needed for integration tests
    st_patches = [
        ("title", patch("streamlit.title")),
        ("markdown", patch("streamlit.markdown")),
        ("success", patch("streamlit.success")),
        ("error", patch("streamlit.error")),
        ("info", patch("streamlit.info")),
        ("warning", patch("streamlit.warning")),
        ("progress", patch("streamlit.progress")),
        ("spinner", patch("streamlit.spinner")),
        ("button", patch("streamlit.button")),
        ("columns", patch("streamlit.columns")),
        ("metric", patch("streamlit.metric")),
    ]

    # Start all patches and collect mocks
    started_patches = []
    for name, p in st_patches:
        mock_obj = p.start()
        mocks[name] = mock_obj
        started_patches.append(p)

    try:
        # Configure columns to return mock column objects
        def mock_columns_func(*args, **_kwargs):
            """Mock columns function that returns appropriate number of columns."""
            if args:
                num_cols = args[0] if isinstance(args[0], int) else len(args[0])
            else:
                num_cols = 2  # Default
            return [MagicMock() for _ in range(num_cols)]

        mocks["columns"].side_effect = mock_columns_func
        mocks["button"].return_value = False

        # Configure spinner context manager
        mock_spinner_obj = MagicMock()
        mocks["spinner"].return_value.__enter__ = Mock(return_value=mock_spinner_obj)
        mocks["spinner"].return_value.__exit__ = Mock(return_value=None)

        yield mocks
    finally:
        # Stop all patches
        for p in started_patches:
            p.stop()


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state for integration tests.

    Removed autouse to reduce overhead for tests that don't need it.
    Use this fixture explicitly when needed.
    """
    session_state = MockSessionState()
    # Mark as test environment
    session_state._test_mode = True

    with patch("streamlit.session_state", session_state):
        yield session_state
        # Clear after each test
        session_state._data.clear()


@pytest.fixture
def mock_job_service():
    """Mock JobService for integration tests."""
    with (
        patch("src.services.job_service.JobService") as mock_service_bg,
        patch("src.services.job_service.JobService") as mock_service_core,
        patch("src.ui.pages.scraping.JobService") as mock_service_scraping,
    ):
        # Configure all mock instances with the same behavior
        for mock_service in [mock_service_bg, mock_service_core, mock_service_scraping]:
            mock_service.get_active_companies.return_value = [
                "TestCompany1",
                "TestCompany2",
            ]
            mock_service.get_filtered_jobs.return_value = []
            mock_service.update_job_status.return_value = True
            mock_service.toggle_favorite.return_value = True
            mock_service.update_notes.return_value = True
            mock_service.bulk_update_jobs.return_value = True
            mock_service.get_all_jobs.return_value = []
            mock_service.get_job_by_id.return_value = None

        # Return the background tasks mock directly since most tests need that one
        yield mock_service_bg


@pytest.fixture
def prevent_real_system_execution():
    """Prevent real system execution during integration tests.

    This fixture mocks external dependencies to ensure test isolation.
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
            "src.scraper.scrape_all",
            return_value={
                "inserted": 0,
                "updated": 0,
                "archived": 0,
                "deleted": 0,
                "skipped": 0,
            },
        ) as mock_scrape_all_bg,
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

        yield {
            "scrape_all": mock_scrape_all,
            "scrape_all_bg": mock_scrape_all_bg,
            "execute_scraping": mock_execute_scraping,
            "asyncio_run": mock_asyncio_run,
            "session": mock_session_instance,
            "requests_get": mock_requests_get,
            "logger": mock_get_logger.return_value,
        }
