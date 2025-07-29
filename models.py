"""Database models for the AI Job Scraper application.

This module contains SQLAlchemy ORM models and Pydantic validation models
for companies and job postings used throughout the application.
"""

from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CompanySQL(Base):
    """SQLAlchemy model for companies table.

    Represents companies that are scraped for job postings.

    Attributes:
        id (int): Primary key identifier.
        name (str): Company name, must be unique.
        url (str): Company careers page URL.
        active (bool): Whether to include company in scraping runs.

    """

    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    url = Column(String)
    active = Column(Boolean, default=True)


class JobSQL(Base):
    """SQLAlchemy model for jobs table.

    Represents job postings scraped from company websites.

    Attributes:
        id (int): Primary key identifier.
        company (str): Company name.
        title (str): Job title.
        description (str): Job description text.
        link (str): Unique URL to job posting.
        location (str): Job location.
        posted_date (datetime): When job was posted.
        hash (str): Content hash for change detection.
        last_seen (datetime): Last time job was found during scraping.
        favorite (bool): User-marked favorite status.
        status (str): Application status (New, Applied, etc.).
        notes (str): User notes about the job.

    """

    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    company = Column(String)
    title = Column(String)
    description = Column(Text)
    link = Column(String, unique=True)
    location = Column(String)
    posted_date = Column(DateTime)
    hash = Column(String)
    last_seen = Column(DateTime)
    favorite = Column(Boolean, default=False)
    status = Column(String, default="New")
    notes = Column(Text, default="")


class JobPydantic(BaseModel):
    """Pydantic model for job validation and serialization.

    Used for validating job data during scraping and API operations.

    Attributes:
        company (str): Company name.
        title (str): Job title.
        description (str): Job description text.
        link (str): Job posting URL, must be valid HTTP(S) URL.
        location (str, optional): Job location, defaults to "Unknown".
        posted_date (datetime, optional): When job was posted.
        hash (str, optional): Content hash for change detection.
        last_seen (datetime, optional): Last scraping timestamp.
        favorite (bool): User favorite status, defaults to False.
        status (str): Application status, defaults to "New".
        notes (str): User notes, defaults to empty string.

    """

    company: str
    title: str
    description: str
    link: str = Field(pattern=r"^https?://")
    location: str | None = "Unknown"
    posted_date: datetime | None = None
    hash: str | None = None
    last_seen: datetime | None = None
    favorite: bool = False
    status: str = "New"
    notes: str = ""
