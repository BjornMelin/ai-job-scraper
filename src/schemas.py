"""Pydantic schemas (DTOs) for API responses and UI layer data transfer.

This module contains Pydantic models that mirror the SQLModel database models
but are designed for data transfer between the service layer and UI components.
These DTOs solve the DetachedInstanceError by providing clean data objects
that don't maintain database session relationships.

The schemas include:
- Company: Company information without relationships
- Job: Job posting data with resolved company name

All schemas include validation, JSON encoding configuration, and proper
type hints for safe data transfer across application layers.
"""

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, field_validator


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
            ValueError: If name is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Company name cannot be empty")
        return v.strip()

    class Config:
        """Pydantic configuration for Company DTO.

        Enables conversion from SQLModel objects and provides custom JSON encoding
        for datetime objects using ISO format.
        """

        from_attributes = True  # Enable SQLModel object conversion
        json_encoders: ClassVar = {datetime: lambda v: v.isoformat() if v else None}


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

    # Backward compatibility alias
    @property
    def status(self) -> str:
        """Backward compatibility alias for application_status.

        Returns:
            Current application status string.

        Note:
            This property is maintained for backward compatibility.
            New code should use application_status directly.
        """
        return self.application_status

    @field_validator("link")
    @classmethod
    def validate_link(cls, v: str) -> str:
        """Validate that job link is not empty.

        Args:
            v: Link value to validate.

        Returns:
            Validated link string.

        Raises:
            ValueError: If link is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Job link cannot be empty")
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
            ValueError: If title is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Job title cannot be empty")
        return v.strip()

    class Config:
        """Pydantic configuration for Job DTO.

        Enables conversion from SQLModel objects and provides custom JSON encoding
        for datetime objects using ISO format.
        """

        from_attributes = True  # Enable SQLModel object conversion
        json_encoders: ClassVar = {datetime: lambda v: v.isoformat() if v else None}
