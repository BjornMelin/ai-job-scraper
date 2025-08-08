"""UI test fixtures for Streamlit component testing.

This module provides fixtures for testing Streamlit UI components with proper
mocking of Streamlit functionality and service layer dependencies.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.schemas import Company, Job


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
    ]

    # Start all patches and collect mocks
    started_patches = []
    for name, p in st_patches:
        mock_obj = p.start()
        mocks[name] = mock_obj
        started_patches.append(p)

    try:
        # Configure columns to return mock column objects
        def mock_columns_func(*args, **kwargs):
            """Mock columns function that returns appropriate number of columns."""
            if args:
                num_cols = args[0] if isinstance(args[0], int) else len(args[0])
            else:
                num_cols = 2  # Default
            return [MagicMock() for _ in range(num_cols)]

        mocks["columns"].side_effect = mock_columns_func

        # Configure container to return mock container
        mock_container_obj = MagicMock()
        mocks["container"].return_value.__enter__ = Mock(
            return_value=mock_container_obj
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

        # Add extra references for convenience
        mocks.update(
            {
                "tab1": mock_tab1,
                "tab2": mock_tab2,
                "tab3": mock_tab3,
            }
        )

        yield mocks
    finally:
        # Stop all patches
        for p in started_patches:
            p.stop()


class MockSessionState:
    """Mock session state that behaves like both a dict and an object."""

    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getattr__(self, name):
        return self._data.get(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def update(self, other):
        self._data.update(other)

    def __contains__(self, key):
        return key in self._data


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state for UI testing."""
    session_state = MockSessionState()

    with patch("streamlit.session_state", session_state):
        yield session_state


@pytest.fixture
def sample_company():
    """Create a sample company DTO for testing."""
    return Company(
        id=1,
        name="Tech Corp",
        url="https://techcorp.com/careers",
        active=True,
        last_scraped=datetime.now(timezone.utc),
        scrape_count=5,
        success_rate=0.8,
    )


@pytest.fixture
def sample_companies():
    """Create a list of sample company DTOs for testing."""
    return [
        Company(
            id=1,
            name="Tech Corp",
            url="https://techcorp.com/careers",
            active=True,
            last_scraped=datetime.now(timezone.utc),
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
            last_scraped=datetime.now(timezone.utc),
            scrape_count=12,
            success_rate=0.92,
        ),
    ]


@pytest.fixture
def sample_job():
    """Create a sample job DTO for testing."""
    return Job(
        id=1,
        company_id=1,
        company="Tech Corp",
        title="Senior AI Engineer",
        description="We are looking for an experienced AI engineer to join our team and work on cutting-edge machine learning projects.",
        link="https://techcorp.com/careers/ai-engineer-123",
        location="San Francisco, CA",
        posted_date=datetime.now(timezone.utc),
        salary=(120000, 180000),
        favorite=False,
        notes="Interesting role with good growth potential",
        content_hash="hash123",
        application_status="New",
        application_date=None,
        archived=False,
        last_seen=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_jobs():
    """Create a list of sample job DTOs for testing."""
    base_time = datetime.now(timezone.utc)

    return [
        Job(
            id=1,
            company_id=1,
            company="Tech Corp",
            title="Senior AI Engineer",
            description="We are looking for an experienced AI engineer to join our team and work on cutting-edge machine learning projects.",
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
            description="Join our data science team to build predictive models and analytics solutions.",
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
            description="Exciting opportunity to work with large datasets and develop ML algorithms.",
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
    with patch("src.ui.pages.companies.CompanyService") as mock_service:
        # Configure default return values for service methods
        mock_service.get_all_companies.return_value = []
        mock_service.add_company.return_value = Company(
            id=1, name="Test Company", url="https://test.com", active=True
        )
        mock_service.toggle_company_active.return_value = True
        mock_service.get_active_companies_count.return_value = 0

        yield mock_service


@pytest.fixture
def mock_job_service():
    """Mock JobService for testing UI components."""
    # Patch in multiple locations where JobService is used
    with (
        patch("src.ui.pages.jobs.JobService") as mock_service_jobs,
        patch("src.ui.components.cards.job_card.JobService") as mock_service_cards,
    ):
        # Configure both mock instances with the same behavior
        for mock_service in [mock_service_jobs, mock_service_cards]:
            mock_service.get_filtered_jobs.return_value = []
            mock_service.update_job_status.return_value = True
            mock_service.toggle_favorite.return_value = True
            mock_service.update_notes.return_value = True
            mock_service.bulk_update_jobs.return_value = True

        yield mock_service_jobs


@pytest.fixture
def mock_logging():
    """Mock logging to prevent log messages during testing."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger
