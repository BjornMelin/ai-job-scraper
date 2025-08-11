"""Integration test fixtures.

This module provides minimal fixtures needed for integration tests
without the full UI test setup from tests/ui/conftest.py.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


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
    ]

    # Start all patches and collect mocks
    started_patches = []
    for name, p in st_patches:
        mock_obj = p.start()
        mocks[name] = mock_obj
        started_patches.append(p)

    try:
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
    """Mock Streamlit session state for integration tests."""
    session_state = {}

    with patch("streamlit.session_state", session_state):
        yield session_state


@pytest.fixture(autouse=True)
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
            "execute_scraping": mock_execute_scraping,
            "asyncio_run": mock_asyncio_run,
            "session": mock_session_instance,
            "requests_get": mock_requests_get,
            "logger": mock_get_logger.return_value,
        }
