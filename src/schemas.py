"""Pydantic schemas (DTOs) for API responses and UI layer data transfer.

This module contains Pydantic models that mirror the SQLModel database models
but are designed for data transfer between the service layer and UI components.
These DTOs solve the DetachedInstanceError by providing clean data objects
that don't maintain database session relationships.

The schemas include read-only company facets, saved-search definitions and run
health, and job DTOs with resolved company names.

All schemas include validation, JSON encoding configuration, and proper
type hints for safe data transfer across application layers.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from src.core_utils import ensure_timezone_aware
from src.models.job_models import (
    ApplicationStage,
    JobSite,
    JobType,
    SavedSearchRunStatus,
)


class CompanyValidationError(ValueError):
    """Custom exception for company data validation errors."""


class JobValidationError(ValueError):
    """Custom exception for job data validation errors."""


class Company(BaseModel):
    """Read-only company facet computed from persisted jobs."""

    id: int | None = None
    name: str
    url: str | None = None
    total_jobs: int = 0
    active_jobs: int = 0
    last_job_posted: datetime | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that company name is not empty.

        Args:
            v: Name value to validate.

        Returns:
            Validated name string.

        Raises:
            CompanyValidationError: If name is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise CompanyValidationError("Company name cannot be empty")
        return v.strip()

    @field_validator("last_job_posted", mode="before")
    @classmethod
    def normalize_last_job_posted(cls, value) -> datetime | None:
        return ensure_timezone_aware(value)

    model_config = ConfigDict(from_attributes=True)


class SavedSearchRunHealth(BaseModel):
    """Stable health contract returned after every saved-search run."""

    last_run_at: datetime | None = None
    last_run_status: SavedSearchRunStatus = SavedSearchRunStatus.NEVER
    jobs_seen: int = Field(default=0, ge=0)
    jobs_new: int = Field(default=0, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    last_error: str | None = None

    @field_validator("last_run_at", mode="before")
    @classmethod
    def normalize_last_run_at(cls, value) -> datetime | None:
        return ensure_timezone_aware(value)


class SavedSearch(SavedSearchRunHealth):
    """Detached saved-search definition for services and UI."""

    id: int
    name: str
    query: str
    location: str
    sites: list[JobSite]
    remote_only: bool
    job_type: JobType | None
    results_limit: int
    enabled: bool

    model_config = ConfigDict(from_attributes=True)


class SavedSearchCreate(BaseModel):
    """Validated input for creating a saved search."""

    name: str
    query: str
    location: str = "United States"
    sites: list[JobSite] = Field(
        default_factory=lambda: [JobSite.LINKEDIN, JobSite.INDEED],
        min_length=1,
    )
    remote_only: bool = False
    job_type: JobType | None = None
    results_limit: int = Field(default=50, ge=1, le=1000)
    enabled: bool = True

    @field_validator("name", "query", "location")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Saved-search text fields cannot be empty")
        return value


class SavedSearchUpdate(BaseModel):
    """Validated editable fields for an existing saved search."""

    name: str | None = None
    query: str | None = None
    location: str | None = None
    sites: list[JobSite] | None = Field(default=None, min_length=1)
    remote_only: bool | None = None
    job_type: JobType | None = None
    results_limit: int | None = Field(default=None, ge=1, le=1000)
    enabled: bool | None = None

    @field_validator("name", "query", "location")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Saved-search text fields cannot be empty")
        return value


class Job(BaseModel):
    """Pydantic DTO for Job data transfer.

    Mirrors JobSQL fields but replaces company relationship with company name string,
    enabling safe data transfer across layers without session dependencies.
    This DTO eliminates DetachedInstanceError by providing clean data objects.

    Attributes:
        id: Optional unique identifier from database.
        company_id: Foreign key reference to associated company.
        company: Company name as string (resolved from relationship).
        title: Job title or position name.
        description: Full job description text.
        link: Unique application URL or job posting link.
        location: Job location (city, state, remote, etc.).
        posted_date: Date when job was originally posted.
        salary: Tuple of (min_salary, max_salary) or (None, None).
        favorite: User-defined favorite flag.
        notes: User-defined notes and comments.
        content_hash: MD5 hash for change detection and deduplication.
        application_status: Current job-search workflow stage.
        application_date: Date when application was submitted.
        archived: Soft delete flag (True = hidden from main views).
        last_seen: Timestamp of most recent scraping encounter.
    """

    id: int | None = None
    company_id: int
    company: str  # Company name as string instead of relationship
    title: str
    description: str
    link: str
    location: str
    posted_date: datetime | None = None
    salary: tuple[int | None, int | None] = (None, None)
    favorite: bool = False
    notes: str = ""
    content_hash: str
    application_status: ApplicationStage = ApplicationStage.INBOX
    application_date: datetime | None = None
    archived: bool = False
    last_seen: datetime | None = None

    @field_validator("link")
    @classmethod
    def validate_link(cls, v: str) -> str:
        """Validate that job link is not empty.

        Args:
            v: Link value to validate.

        Returns:
            Validated link string.

        Raises:
            JobValidationError: If link is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise JobValidationError("Job link cannot be empty")
        return v.strip()

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that job title is not empty.

        Args:
            v: Title value to validate.

        Returns:
            Validated title string.

        Raises:
            JobValidationError: If title is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise JobValidationError("Job title cannot be empty")
        return v.strip()

    @computed_field
    @property
    def salary_range_display(self) -> str:
        """Format salary range for display."""
        from src.ui.utils.formatters import format_salary_range

        return format_salary_range(self.salary)

    @computed_field
    @property
    def days_since_posted(self) -> int | None:
        """Calculate days since job was posted."""
        if self.posted_date is None:
            return None
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        # Ensure timezone compatibility
        posted_date = self.posted_date
        if posted_date.tzinfo is None:
            # If posted_date is naive, assume it's UTC
            posted_date = posted_date.replace(tzinfo=UTC)
        return (now - posted_date).days

    @computed_field
    @property
    def is_recently_posted(self) -> bool:
        """Check if job was posted within 7 days."""
        if self.posted_date is None:
            return False
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        # Ensure timezone compatibility
        posted_date = self.posted_date
        if posted_date.tzinfo is None:
            # If posted_date is naive, assume it's UTC
            posted_date = posted_date.replace(tzinfo=UTC)
        return (now - posted_date).days <= 7

    @field_validator("posted_date", "application_date", "last_seen", mode="before")
    @classmethod
    def ensure_datetime_timezone_aware(cls, v) -> datetime | None:
        """Ensure datetime fields are timezone-aware (UTC) - uses shared utility."""
        return ensure_timezone_aware(v)

    model_config = ConfigDict(
        from_attributes=True,  # Enable SQLModel object conversion
    )
