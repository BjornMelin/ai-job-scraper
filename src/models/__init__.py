"""Job scraping models package."""

from .job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
    LocationType,
)

# Import from the main models.py file for database models
try:
    from src.models import CompanySQL, JobSQL
except ImportError:
    CompanySQL = None
    JobSQL = None

__all__ = [
    "JobPosting",
    "JobScrapeRequest",
    "JobScrapeResult",
    "JobSite",
    "JobType",
    "LocationType",
]

# Add database models if available
if CompanySQL is not None:
    __all__.extend(["CompanySQL", "JobSQL"])
