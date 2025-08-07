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
    def parse_salary(  # noqa: PLR0911
        cls, value: str | tuple[int | None, int | None] | None
    ) -> tuple[int | None, int | None]:
        """Parse salary string into (min, max) tuple.

        Handles various salary formats including:
        - Range formats: "$100k-150k", "£80,000 - £120,000", "110k to 150k"
        - Single values: "$120k", "150000", "up to $150k", "from $110k"
        - Currency symbols: $, £, €
        - Suffixes: k, K (for thousands)
        - Common phrases: "per year", "per annum", "up to", "from", "starting at"

        Args:
            value: Salary input as string, tuple, or None.

        Returns:
            tuple[int | None, int | None]: Parsed (min, max) salaries.
                For ranges: (min_salary, max_salary)
                For single values: (salary, salary) for exact matches,
                                  (salary, None) for "from" patterns,
                                  (None, salary) for "up to" patterns
        """
        if isinstance(value, tuple) and len(value) == 2:
            return value

        if value is None or not isinstance(value, str) or value.strip() == "":
            return (None, None)

        original = value.strip()

        # Check for "up to" patterns first to handle them differently
        up_to_pattern = r"\b(?:up\s+to|maximum\s+of|max\s+of|not\s+more\s+than)\b"
        is_up_to = bool(re.search(up_to_pattern, original, re.IGNORECASE))

        # Check for "from" patterns
        from_pattern = r"\b(?:from|starting\s+at|minimum\s+of|min\s+of|at\s+least)\b"
        is_from = bool(re.search(from_pattern, original, re.IGNORECASE))

        # Remove currency symbols and normalize
        cleaned = re.sub(r"[£$€¥¢₹]", "", original)

        # Remove common phrases and normalize spacing
        phrases_to_remove = [
            r"\b(?:per\s+year|per\s+annum|annually|yearly|p\.?a\.?|/year|/yr)\b",
            r"\b(?:per\s+hour|hourly|/hour|/hr)\b",
            r"\b(?:per\s+month|monthly|/month|/mo)\b",
            r"\b(?:gross|net|before\s+tax|after\s+tax)\b",
            r"\b(?:plus\s+benefits?|\+\s*benefits?)\b",
            r"\b(?:negotiable|neg\.?|ono|o\.?n\.?o\.?)\b",
            r"\b(?:depending\s+on\s+experience|doe)\b",
            up_to_pattern,
            from_pattern,
        ]

        for pattern in phrases_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Handle commas in numbers first (e.g., "100,000" -> "100000")
        cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

        # Normalize spacing and remove extra punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"[^\d\s.k-]+", "", cleaned, flags=re.IGNORECASE)

        # Enhanced number extraction with 'k' suffix support
        # First check for patterns like "100-120k" where 'k' applies to both numbers
        range_k_pattern = r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*([kK])"
        range_k_match = re.search(range_k_pattern, cleaned)

        if range_k_match:
            # Handle range with shared 'k' suffix
            num1, num2, k_suffix = range_k_match.groups()
            try:
                multiplier = 1000 if k_suffix.lower() == "k" else 1
                val1 = int(float(num1) * multiplier)
                val2 = int(float(num2) * multiplier)
                return (min(val1, val2), max(val1, val2))
            except (ValueError, TypeError):
                pass

        # Standard number extraction
        number_pattern = r"(\d+(?:\.\d+)?)\s*([kK])?"
        numbers = re.findall(number_pattern, cleaned)

        if not numbers:
            return (None, None)

        parsed_nums = []
        for num_str, k_suffix in numbers:
            try:
                num = float(num_str)
                # Apply 'k' multiplier if present
                if k_suffix.lower() == "k":
                    num *= 1000
                parsed_nums.append(int(num))
            except (ValueError, TypeError):
                continue

        if not parsed_nums:
            return (None, None)

        # Handle different patterns based on context
        if len(parsed_nums) == 1:
            single_value = parsed_nums[0]
            if is_up_to:
                return (None, single_value)
            if is_from:
                return (single_value, None)
            # For single values without context, return as both min and max
            return (single_value, single_value)

        if len(parsed_nums) >= 2:
            # For multiple numbers, return as range (min, max)
            return (min(parsed_nums), max(parsed_nums))

        return (None, None)
