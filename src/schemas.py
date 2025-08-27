"""Pydantic schemas (DTOs) for API responses and UI layer data transfer.

This module contains Pydantic models that mirror the SQLModel database models
but are designed for data transfer between the service layer and UI components.
These DTOs solve the DetachedInstanceError by providing clean data objects
that don't maintain database session relationships.

The schemas include:
- Company: Company information without relationships
- CompanyCreate: Schema for creating new company records
- Job: Job posting data with resolved company name
- JobCreate: Schema for creating new job records

All schemas include validation, JSON encoding configuration, and proper
type hints for safe data transfer across application layers.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from src.core_utils import ensure_timezone_aware

# Import helper functions locally to avoid circular imports
# from src.ui.utils import (...)


class CompanyValidationError(ValueError):
    """Custom exception for company data validation errors."""


class JobValidationError(ValueError):
    """Custom exception for job data validation errors."""


class Company(BaseModel):
    """Pydantic DTO for Company data transfer.

    Mirrors CompanySQL fields but without SQLModel relationships,
    enabling safe data transfer across layers without session dependencies.
    This DTO eliminates DetachedInstanceError by providing clean data objects
    that don't maintain database session relationships.

    Attributes:
        id: Optional unique identifier from database.
        name: Company name (must be unique in database).
        url: Company careers page URL.
        active: Flag indicating if company is active for scraping.
        last_scraped: Timestamp of most recent scraping attempt.
        scrape_count: Total number of scraping attempts performed.
        success_rate: Success rate of scraping attempts (0.0 to 1.0).
    """

    id: int | None = None
    name: str
    url: str
    active: bool = True
    last_scraped: datetime | None = None
    scrape_count: int = 0
    success_rate: float = 1.0

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate and normalize URL.

        Args:
            v: URL value to validate.

        Returns:
            Validated URL string, or empty string if not provided.
        """
        if not v or not v.strip():
            return ""  # Allow empty URLs for companies without careers pages
        return v.strip()

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

    @computed_field
    @property
    def total_jobs_count(self) -> int:
        """Calculate total number of jobs."""
        # For the DTO, we don't have access to the jobs relationship
        # This would need to be set from the service layer
        return getattr(self, "_total_jobs_count", 0)

    @computed_field
    @property
    def active_jobs_count(self) -> int:
        """Calculate number of active (non-archived) jobs."""
        # For the DTO, we don't have access to the jobs relationship
        # This would need to be set from the service layer
        return getattr(self, "_active_jobs_count", 0)

    @computed_field
    @property
    def last_job_posted(self) -> datetime | None:
        """Find most recent job posting date."""
        # For the DTO, we don't have access to the jobs relationship
        # This would need to be set from the service layer
        return getattr(self, "_last_job_posted", None)

    @field_validator("last_scraped", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, v) -> datetime | None:
        """Ensure datetime is timezone-aware (UTC) - uses shared utility."""
        return ensure_timezone_aware(v)

    model_config = ConfigDict(
        from_attributes=True,  # Enable SQLModel object conversion
    )


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
        application_status: Current application status (New, Applied, etc.).
        application_date: Date when application was submitted.
        archived: Soft delete flag (True = hidden from main views).
        last_seen: Timestamp of most recent scraping encounter.
    """

    id: int | None = None
    company_id: int | None = None
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
    application_status: str = "New"
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
        from src.ui.utils import format_salary_range

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


class CompanyCreate(BaseModel):
    """Schema for creating new company records.

    This DTO is used when creating new companies via API or service layer,
    providing validation and defaults for required fields.

    Attributes:
        name: Company name (must be unique in database).
        url: Company careers page URL.
        active: Flag indicating if company is active for scraping.
    """

    name: str
    url: str
    active: bool = True

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

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate and normalize URL.

        Args:
            v: URL value to validate.

        Returns:
            Validated URL string.

        Raises:
            CompanyValidationError: If URL is invalid.
        """
        if not v or not v.strip():
            return ""  # Allow empty URLs for companies without careers pages

        url = v.strip()
        if not url.startswith(("http://", "https://")):
            raise CompanyValidationError("URL must start with http:// or https://")

        return url

    model_config = ConfigDict(
        from_attributes=True,  # Enable SQLModel object conversion
    )


class JobCreate(BaseModel):
    """Schema for creating new job records.

    This DTO is used when creating new jobs via API or service layer,
    providing validation and defaults for required fields.

    Attributes:
        company_id: Foreign key reference to associated company.
        title: Job title or position name.
        description: Full job description text.
        link: Unique application URL or job posting link.
        location: Job location (city, state, remote, etc.).
        posted_date: Date when job was originally posted.
        salary: Tuple of (min_salary, max_salary) or (None, None).
        content_hash: MD5 hash for change detection and deduplication.
        archived: Soft delete flag (False for new jobs).
    """

    company_id: int
    title: str
    description: str
    link: str
    location: str
    posted_date: datetime | None = None
    salary: tuple[int | None, int | None] = (None, None)
    content_hash: str
    archived: bool = False

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

    @field_validator("link")
    @classmethod
    def validate_link(cls, v: str) -> str:
        """Validate that job link is not empty and well-formed.

        Args:
            v: Link value to validate.

        Returns:
            Validated link string.

        Raises:
            JobValidationError: If link is empty or invalid.
        """
        if not v or not v.strip():
            raise JobValidationError("Job link cannot be empty")

        link = v.strip()
        if not link.startswith(("http://", "https://")):
            raise JobValidationError("Job link must start with http:// or https://")

        return link

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate that job description is not empty.

        Args:
            v: Description value to validate.

        Returns:
            Validated description string.

        Raises:
            JobValidationError: If description is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise JobValidationError("Job description cannot be empty")
        return v.strip()

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate that job location is not empty.

        Args:
            v: Location value to validate.

        Returns:
            Validated location string.

        Raises:
            JobValidationError: If location is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise JobValidationError("Job location cannot be empty")
        return v.strip()

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, v: str) -> str:
        """Validate that content hash is not empty.

        Args:
            v: Content hash value to validate.

        Returns:
            Validated content hash string.

        Raises:
            JobValidationError: If content hash is empty or invalid.
        """
        if not v or not v.strip():
            raise JobValidationError("Content hash cannot be empty")

        # MD5 hashes should be 32 characters long
        if len(v) != 32:
            raise JobValidationError(
                "Content hash must be a valid MD5 hash (32 characters)"
            )

        return v

    @field_validator("posted_date", mode="before")
    @classmethod
    def ensure_datetime_timezone_aware(cls, v) -> datetime | None:
        """Ensure datetime fields are timezone-aware (UTC)."""
        return ensure_timezone_aware(v)

    @field_validator("salary", mode="before")
    @classmethod
    def validate_salary_tuple(cls, v) -> tuple[int | None, int | None]:
        """Validate and normalize salary tuple.

        Args:
            v: Salary value to validate.

        Returns:
            Validated salary tuple.
        """
        if v is None:
            return (None, None)

        if isinstance(v, (list, tuple)) and len(v) == 2:
            min_sal, max_sal = v
            if min_sal is not None and max_sal is not None and min_sal > max_sal:
                # Swap if in wrong order
                return (max_sal, min_sal)
            return (min_sal, max_sal)

        return (None, None)

    model_config = ConfigDict(
        from_attributes=True,  # Enable SQLModel object conversion
    )
