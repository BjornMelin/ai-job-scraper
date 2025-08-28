"""Comprehensive Pydantic models for JobSpy integration."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

from pydantic import BaseModel, Field, field_validator


class JobSite(str, Enum):
    """Supported job sites for scraping."""

    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    ZIP_RECRUITER = "zip_recruiter"
    GOOGLE = "google"

    @classmethod
    def normalize(cls, value: str | None) -> JobSite | None:
        """Normalize job site string to enum value."""
        if not value:
            return None

        normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
        site_mapping = {
            "linkedin": cls.LINKEDIN,
            "indeed": cls.INDEED,
            "glassdoor": cls.GLASSDOOR,
            "zip_recruiter": cls.ZIP_RECRUITER,
            "ziprecruiter": cls.ZIP_RECRUITER,
            "google": cls.GOOGLE,
        }
        return site_mapping.get(normalized)


class JobType(str, Enum):
    """Job employment types."""

    FULLTIME = "fulltime"
    PARTTIME = "parttime"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"

    @classmethod
    def normalize(cls, value: str | None) -> JobType | None:
        """Normalize job type string to enum value."""
        if not value:
            return None

        normalized = (
            value.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
        )
        type_mapping = {
            "fulltime": cls.FULLTIME,
            "full": cls.FULLTIME,
            "permanent": cls.FULLTIME,
            "parttime": cls.PARTTIME,
            "part": cls.PARTTIME,
            "contract": cls.CONTRACT,
            "contractor": cls.CONTRACT,
            "internship": cls.INTERNSHIP,
            "intern": cls.INTERNSHIP,
            "temporary": cls.TEMPORARY,
            "temp": cls.TEMPORARY,
        }
        return type_mapping.get(normalized)


class LocationType(str, Enum):
    """Work location types."""

    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"

    @classmethod
    def from_remote_flag(
        cls, is_remote: bool | None, location: str | None = None
    ) -> LocationType:
        """Determine location type from remote flag and location string."""
        if is_remote:
            return cls.REMOTE
        if location and any(term in location.lower() for term in ["hybrid", "remote"]):
            return cls.HYBRID if "hybrid" in location.lower() else cls.REMOTE
        return cls.ONSITE


class JobScrapeRequest(BaseModel):
    """Request parameters for job scraping."""

    site_name: list[JobSite] | JobSite = Field(default=JobSite.LINKEDIN)
    search_term: str | None = None
    google_search_term: str | None = None
    location: str | None = None
    distance: int = Field(default=50, ge=0, le=200)
    is_remote: bool = False
    job_type: JobType | None = None
    easy_apply: bool | None = None
    results_wanted: int = Field(default=15, ge=1, le=1000)
    country_indeed: str = "usa"
    offset: int = Field(default=0, ge=0)
    hours_old: int | None = Field(default=None, ge=1)
    enforce_annual_salary: bool = False
    linkedin_fetch_description: bool = False
    description_format: str = "markdown"

    @field_validator("site_name", mode="before")
    @classmethod
    def normalize_site_name(cls, value: Any) -> Any:
        """Normalize site names to JobSite enums."""
        if isinstance(value, str):
            normalized = JobSite.normalize(value)
            return normalized if normalized else value
        if isinstance(value, list):
            return [
                JobSite.normalize(site) if isinstance(site, str) else site
                for site in value
            ]
        return value

    @field_validator("job_type", mode="before")
    @classmethod
    def normalize_job_type(cls, value: Any) -> Any:
        """Normalize job type to JobType enum."""
        if isinstance(value, str):
            return JobType.normalize(value)
        return value


class JobPosting(BaseModel):
    """Individual job posting from JobSpy."""

    id: str
    site: JobSite
    job_url: str | None = None
    job_url_direct: str | None = None
    title: str
    company: str
    location: str | None = None
    date_posted: date | None = None
    job_type: JobType | None = None

    # Salary information
    salary_source: str | None = None
    interval: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    currency: str | None = None

    # Location and work arrangement
    is_remote: bool = False
    location_type: LocationType = LocationType.ONSITE
    work_from_home_type: str | None = None

    # Job details
    job_level: str | None = None
    job_function: str | None = None
    listing_type: str | None = None
    description: str | None = None
    emails: list[str] | None = None
    skills: list[str] | None = None
    experience_range: str | None = None
    vacancy_count: int | None = None

    # Company information
    company_industry: str | None = None
    company_url: str | None = None
    company_logo: str | None = None
    company_url_direct: str | None = None
    company_addresses: list[str] | None = None
    company_num_employees: str | None = None
    company_revenue: str | None = None
    company_description: str | None = None
    company_rating: float | None = None
    company_reviews_count: int | None = None

    @field_validator("min_amount", "max_amount", "company_rating", mode="before")
    @classmethod
    def safe_float_conversion(cls, value: Any) -> float | None:
        """Safely convert values to float, handling various input types."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @field_validator("site", mode="before")
    @classmethod
    def normalize_site(cls, value: Any) -> Any:
        """Normalize site to JobSite enum."""
        if isinstance(value, str):
            return JobSite.normalize(value) or value
        return value

    @field_validator("job_type", mode="before")
    @classmethod
    def normalize_job_type_posting(cls, value: Any) -> Any:
        """Normalize job type to JobType enum."""
        if isinstance(value, str):
            return JobType.normalize(value)
        return value

    @field_validator("location_type", mode="before")
    @classmethod
    def set_location_type(cls, value: Any, info) -> LocationType:
        """Set location type based on is_remote flag and location."""
        if isinstance(value, LocationType):
            return value
        # Use other field values to determine location type
        data = info.data if info.data else {}
        is_remote = data.get("is_remote", False)
        location = data.get("location")
        return LocationType.from_remote_flag(is_remote, location)


class JobScrapeResult(BaseModel):
    """Complete job scraping results."""

    jobs: list[JobPosting]
    total_found: int
    request_params: JobScrapeRequest
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        request: JobScrapeRequest,
        metadata: dict[str, Any] | None = None,
    ) -> JobScrapeResult:
        """Create JobScrapeResult from pandas DataFrame (JobSpy output)."""
        jobs = []
        for _, row in df.iterrows():
            # Convert pandas row to dict, handling NaN/None values
            job_data = {}
            for col, value in row.items():
                if pd.isna(value):
                    job_data[col] = None
                elif isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                    job_data[col] = value.date() if hasattr(value, "date") else None
                else:
                    job_data[col] = value

            # Map DataFrame columns to JobPosting fields
            job_posting = JobPosting.model_validate(job_data)
            jobs.append(job_posting)

        return cls(
            jobs=jobs,
            total_found=len(jobs),
            request_params=request,
            metadata=metadata or {},
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert JobScrapeResult back to pandas DataFrame."""
        if not self.jobs:
            return pd.DataFrame()

        return pd.DataFrame([job.model_dump() for job in self.jobs])

    @property
    def job_count(self) -> int:
        """Get total number of jobs."""
        return len(self.jobs)

    def filter_by_location_type(self, location_type: LocationType) -> JobScrapeResult:
        """Filter jobs by location type."""
        filtered_jobs = [job for job in self.jobs if job.location_type == location_type]
        return self.model_copy(
            update={"jobs": filtered_jobs, "total_found": len(filtered_jobs)}
        )

    def filter_by_job_type(self, job_type: JobType) -> JobScrapeResult:
        """Filter jobs by job type."""
        filtered_jobs = [job for job in self.jobs if job.job_type == job_type]
        return self.model_copy(
            update={"jobs": filtered_jobs, "total_found": len(filtered_jobs)}
        )
