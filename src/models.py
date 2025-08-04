"""Database models for companies and jobs in the AI Job Scraper."""

import re

from datetime import datetime

from pydantic import field_validator
from sqlalchemy.types import JSON
from sqlmodel import Column, Field, SQLModel


class CompanySQL(SQLModel, table=True):
    """SQLModel for company records.

    Attributes:
        id: Primary key identifier.
        name: Company name.
        url: Company careers URL.
        active: Flag indicating if the company is active for scraping.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    url: str
    active: bool = True


class JobSQL(SQLModel, table=True):
    """SQLModel for job records.

    Attributes:
        id: Primary key identifier.
        company: Company name.
        title: Job title.
        description: Job description.
        link: Application link.
        location: Job location.
        posted_date: Date the job was posted.
        salary: Tuple of (min, max) salary values.
        favorite: Flag if the job is favorited.
        notes: User notes for the job.
    """

    id: int | None = Field(default=None, primary_key=True)
    company: str
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
        elif len(parsed_nums) >= 2:
            return (min(parsed_nums), max(parsed_nums))
        return (None, None)
