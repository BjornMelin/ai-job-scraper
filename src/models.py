"""Database models for companies and jobs in the AI Job Scraper."""

import re

from datetime import datetime

from pydantic import computed_field, field_validator
from sqlalchemy.types import JSON
from sqlmodel import Column, Field, Relationship, SQLModel


class CompanySQL(SQLModel, table=True):
    """SQLModel for company records.

    Attributes:
        id: Primary key identifier.
        name: Company name.
        url: Company careers URL.
        active: Flag indicating if the company is active for scraping.
        last_scraped: Timestamp of the last successful scrape.
        scrape_count: Total number of scrapes performed for this company.
        success_rate: Success rate of scraping attempts (0.0 to 1.0).
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)  # Explicit index for name
    url: str
    active: bool = Field(default=True, index=True)  # Index for active status filtering
    last_scraped: datetime | None = Field(
        default=None, index=True
    )  # Index for scraping recency
    scrape_count: int = Field(default=0)
    success_rate: float = Field(default=1.0)

    # Relationships
    jobs: list["JobSQL"] = Relationship(back_populates="company_relation")


class JobSQL(SQLModel, table=True):
    """SQLModel for job records.

    Attributes:
        id: Primary key identifier.
        company_id: Foreign key reference to CompanySQL.
        title: Job title.
        description: Job description.
        link: Application link.
        location: Job location.
        posted_date: Date the job was posted.
        salary: Tuple of (min, max) salary values.
        favorite: Flag if the job is favorited.
        notes: User notes for the job.
        content_hash: Hash of job content for duplicate detection.
        application_status: Current status of the job application.
        application_date: Date when application was submitted.
        archived: Flag indicating if the job is archived (soft delete).
    """

    id: int | None = Field(default=None, primary_key=True)
    company_id: int | None = Field(default=None, foreign_key="companysql.id")
    title: str
    description: str
    link: str = Field(unique=True)
    location: str
    posted_date: datetime | None = None
    salary: tuple[int | None, int | None] = Field(
        default=(None, None), sa_column=Column(JSON)
    )
    favorite: bool = False
    notes: str = ""
    content_hash: str = Field(index=True)
    application_status: str = Field(default="New", index=True)
    application_date: datetime | None = None
    archived: bool = Field(default=False, index=True)
    last_seen: datetime | None = Field(
        default=None, index=True
    )  # Index for stale job queries

    # Relationships
    company_relation: "CompanySQL" = Relationship(back_populates="jobs")

    @computed_field  # type: ignore[misc]
    @property
    def company(self) -> str:
        """Get company name from relationship or return unknown.

        Returns:
            str: Company name or 'Unknown' if not found.
        """
        return self.company_relation.name if self.company_relation else "Unknown"

    @computed_field  # type: ignore[misc]
    @property
    def status(self) -> str:
        """Backward compatibility alias for application_status.

        Returns:
            str: Current application status.
        """
        return self.application_status

    @field_validator("salary", mode="before")
    @classmethod
    def parse_salary(
        cls, value: str | tuple[int | None, int | None] | None
    ) -> tuple[int | None, int | None]:
        """Parse salary string into (min, max) tuple.

        Handles formats like "$100k-150k", "£80,000 - £120,000", ignoring currencies.

        Args:
            value: Salary input as string, tuple, or None.

        Returns:
            tuple[int | None, int | None]: Parsed (min, max) salaries.
        """
        if isinstance(value, tuple) and len(value) == 2:
            return value

        if value is None or not isinstance(value, str) or value.strip() == "":
            return (None, None)

        # Remove currency symbols, commas, and extra text
        cleaned = re.sub(r"[£$€,]", "", value).lower()
        # Remove common phrases like "from", "a year", "+"
        cleaned = re.sub(r"\b(from|a year|\+)\b", "", cleaned).strip()

        # Find numbers with optional 'k' suffix
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(k)?", cleaned)

        if not numbers:
            return (None, None)

        parsed_nums = []
        for num, k in numbers:
            multiplier = 1000 if k else 1
            parsed_nums.append(int(float(num) * multiplier))

        if len(parsed_nums) == 1:
            return (parsed_nums[0], None)
        if len(parsed_nums) >= 2:
            return (min(parsed_nums), max(parsed_nums))
        return (None, None)
