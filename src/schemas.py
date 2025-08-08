"""Pydantic schemas (DTOs) for API responses and UI layer data transfer.

This module contains Pydantic models that mirror the SQLModel database models
but are designed for data transfer between the service layer and UI components.
These DTOs solve the DetachedInstanceError by providing clean data objects
that don't maintain database session relationships.
"""

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel


class Company(BaseModel):
    """Pydantic DTO for Company data transfer.

    Mirrors CompanySQL fields but without SQLModel relationships,
    enabling safe data transfer across layers without session dependencies.
    """

    id: int | None = None
    name: str
    url: str
    active: bool = True
    last_scraped: datetime | None = None
    scrape_count: int = 0
    success_rate: float = 1.0

    class Config:
        """Pydantic configuration for Company DTO."""

        from_attributes = True  # Enable SQLModel object conversion
        json_encoders: ClassVar = {datetime: lambda v: v.isoformat() if v else None}


class Job(BaseModel):
    """Pydantic DTO for Job data transfer.

    Mirrors JobSQL fields but replaces company relationship with company name string,
    enabling safe data transfer across layers without session dependencies.
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
        """Backward compatibility alias for application_status."""
        return self.application_status

    class Config:
        """Pydantic configuration for Job DTO."""

        from_attributes = True  # Enable SQLModel object conversion
        json_encoders: ClassVar = {datetime: lambda v: v.isoformat() if v else None}
