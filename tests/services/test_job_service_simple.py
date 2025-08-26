"""JobService basic validation tests.

Tests core JobService functionality that can be validated
without complex database session setup.
"""

from src.services.job_service import JobService


class TestJobServiceBasic:
    """JobService interface validation tests."""

    def test_service_exists(self):
        """Verify JobService class exists and can be imported."""
        assert JobService is not None

    def test_service_has_methods(self):
        """Verify JobService has required public methods."""
        assert hasattr(JobService, "get_filtered_jobs")
        assert hasattr(JobService, "update_job_status")
        assert hasattr(JobService, "toggle_favorite")
