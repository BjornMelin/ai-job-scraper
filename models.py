"""Database models for the AI Job Scraper application.

This module contains SQLAlchemy ORM models and Pydantic validation models
for companies and job postings used throughout the application.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CompanySQL(Base):
    """SQLAlchemy model for companies table.

    Represents companies that are scraped for job postings.

    Attributes:
        id (int): Primary key identifier.
        name (str): Company name, must be unique and not null.
        url (str): Company careers page URL, not null.
        active (bool): Whether to include company in scraping runs.

    """

    __tablename__ = "companies"
    __table_args__ = (Index("ix_companies_active", "active"),)

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    url = Column(String, nullable=False)
    active = Column(Boolean, default=True, nullable=False)


class JobSQL(Base):
    """SQLAlchemy model for jobs table.

    Represents job postings scraped from company websites.

    Attributes:
        id (int): Primary key identifier.
        company (str): Company name, not null.
        title (str): Job title, not null.
        description (str): Job description text, not null.
        link (str): Unique URL to job posting, not null.
        location (str): Job location.
        posted_date (datetime): When job was posted.
        hash (str): Content hash for change detection.
        last_seen (datetime): Last time job was found during scraping.
        favorite (bool): User-marked favorite status.
        status (str): Application status (New, Applied, etc.).
        notes (str): User notes about the job.

    """

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_company", "company"),
        Index("ix_jobs_title", "title"),
        Index("ix_jobs_posted_date", "posted_date"),
    )

    id = Column(Integer, primary_key=True)
    company = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    link = Column(String, unique=True, nullable=False)
    location = Column(String)
    posted_date = Column(DateTime)
    hash = Column(String)
    last_seen = Column(DateTime)
    favorite = Column(Boolean, default=False, nullable=False)
    status = Column(String, default="New", nullable=False)
    notes = Column(Text, default="", nullable=False)


class JobPydantic(BaseModel):
    """Pydantic model for job validation and serialization.

    Used for validating job data during scraping and API operations.
    Uses Pydantic v2 ConfigDict and field validators for robust validation.

    Attributes:
        company (str): Company name, 1-100 characters.
        title (str): Job title, 3-200 characters.
        description (str): Job description text, 10-1000 characters.
        link (str): Job posting URL, must be valid HTTP(S) URL.
        location (str, optional): Job location, defaults to "Unknown".
        posted_date (datetime, optional): When job was posted.
        hash (str, optional): Content hash for change detection.
        last_seen (datetime, optional): Last scraping timestamp.
        favorite (bool): User favorite status, defaults to False.
        status (str): Application status, defaults to "New".
        notes (str): User notes, defaults to empty string.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,  # Strip whitespace from string fields
        validate_default=True,  # Validate default values
        extra="forbid",  # Prevent extra fields
    )

    company: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=1000)
    link: str
    location: str | None = "Unknown"
    posted_date: datetime | None = None
    hash: str | None = None
    last_seen: datetime | None = None
    favorite: bool = False
    status: str = "New"
    notes: str = ""

    @field_validator("link")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that link is a proper HTTP/HTTPS URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        # Basic URL validation - avoid complex regex for KISS principle
        if len(v) < 10 or len(v) > 500:
            raise ValueError("URL length must be between 10 and 500 characters")
        return v
