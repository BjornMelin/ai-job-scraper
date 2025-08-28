"""Job scraping models package."""

from .job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
    LocationType,
)

__all__ = [
    "JobPosting",
    "JobScrapeRequest",
    "JobScrapeResult",
    "JobSite",
    "JobType",
    "LocationType",
]
