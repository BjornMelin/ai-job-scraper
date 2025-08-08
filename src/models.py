"""Database models for companies and jobs in the AI Job Scraper."""

import re

from datetime import datetime

from pydantic import computed_field, field_validator
from sqlalchemy.types import JSON
from sqlmodel import Column, Field, Relationship, SQLModel

# Compiled regex patterns for salary parsing
_UP_TO_PATTERN = re.compile(
    r"\b(?:up\s+to|maximum\s+of|max\s+of|not\s+more\s+than)\b", re.IGNORECASE
)
_FROM_PATTERN = re.compile(
    r"\b(?:from|starting\s+at|minimum\s+of|min\s+of|at\s+least)\b", re.IGNORECASE
)
_CURRENCY_PATTERN = re.compile(r"[£$€¥¢₹]")
# Pattern for shared k suffix at end: "100-120k"
_RANGE_K_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*([kK])")
# Pattern for both numbers with k: "100k-150k"
_BOTH_K_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([kK])\s*-\s*(\d+(?:\.\d+)?)([kK])")
# Pattern for one-sided k: "100k-120" (k on first number only)
_ONE_SIDED_K_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)([kK])\s*-\s*(\d+(?:\.\d+)?)(?!\s*[kK])"
)
_NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([kK])?")
_HOURLY_PATTERN = re.compile(r"\b(?:per\s+hour|hourly|/hour|/hr)\b", re.IGNORECASE)
_MONTHLY_PATTERN = re.compile(r"\b(?:per\s+month|monthly|/month|/mo)\b", re.IGNORECASE)

_PHRASES_TO_REMOVE = [
    r"\b(?:per\s+year|per\s+annum|annually|yearly|p\.?a\.?|/year|/yr)\b",
    r"\b(?:gross|net|before\s+tax|after\s+tax)\b",
    r"\b(?:plus\s+benefits?|\+\s*benefits?)\b",
    r"\b(?:negotiable|neg\.?|ono|o\.?n\.?o\.?)\b",
    r"\b(?:depending\s+on\s+experience|doe)\b",
]


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

    @classmethod
    def _detect_context(cls, text: str) -> tuple[bool, bool, bool, bool]:
        """Detect contextual patterns in salary text.

        Args:
            text: Original salary text

        Returns:
            tuple[bool, bool, bool, bool]: (is_up_to, is_from, is_hourly,
                is_monthly) flags
        """
        is_up_to = bool(_UP_TO_PATTERN.search(text))
        is_from = bool(_FROM_PATTERN.search(text))
        is_hourly = bool(_HOURLY_PATTERN.search(text))
        is_monthly = bool(_MONTHLY_PATTERN.search(text))
        return is_up_to, is_from, is_hourly, is_monthly

    @classmethod
    def _normalize_salary_string(cls, text: str) -> str:
        """Normalize salary string by removing currency symbols and common phrases.

        Args:
            text: Raw salary text

        Returns:
            str: Cleaned and normalized text
        """
        # Remove currency symbols
        cleaned = _CURRENCY_PATTERN.sub("", text)

        # Remove common phrases (but preserve hourly/monthly for conversion)
        all_patterns = [
            *_PHRASES_TO_REMOVE,
            _UP_TO_PATTERN.pattern,
            _FROM_PATTERN.pattern,
            _HOURLY_PATTERN.pattern,
            _MONTHLY_PATTERN.pattern,
        ]

        for pattern in all_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove all commas (thousands separators)
        cleaned = re.sub(r",", "", cleaned)

        # Normalize spacing and remove extra punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return re.sub(r"[^\d\s.k-]+", "", cleaned, flags=re.IGNORECASE)

    @classmethod
    def _convert_to_value(cls, num_str: str, k_suffix: str | None = None) -> int | None:
        """Convert a numeric string with optional k suffix to integer value.

        Args:
            num_str: Numeric string to convert
            k_suffix: Optional 'k' or 'K' suffix

        Returns:
            int | None: Converted value or None if conversion fails
        """
        try:
            multiplier = 1000 if k_suffix and k_suffix.lower() == "k" else 1
            return int(float(num_str) * multiplier)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _parse_shared_k_range(cls, text: str) -> tuple[int, int] | None:
        """Parse ranges with k suffix patterns like '100-120k', '100k-150k', '100k-120'.

        Args:
            text: Normalized salary text

        Returns:
            tuple[int, int] | None: (min, max) values or None if not found
        """
        # Try both-k pattern (e.g., "100k-150k")
        if match := _BOTH_K_PATTERN.search(text):
            num1, k1_suffix, num2, k2_suffix = match.groups()
            val1 = cls._convert_to_value(num1, k1_suffix)
            val2 = cls._convert_to_value(num2, k2_suffix)
            if val1 is not None and val2 is not None:
                return (min(val1, val2), max(val1, val2))

        # Try one-sided k pattern (e.g., "100k-120")
        if match := _ONE_SIDED_K_PATTERN.search(text):
            num1, k_suffix, num2 = match.groups()
            # Apply k to both numbers when only first has k
            val1 = cls._convert_to_value(num1, k_suffix)
            val2 = cls._convert_to_value(num2, k_suffix)
            if val1 is not None and val2 is not None:
                return (min(val1, val2), max(val1, val2))

        # Try shared k pattern (e.g., "100-120k")
        if match := _RANGE_K_PATTERN.search(text):
            num1, num2, k_suffix = match.groups()
            val1 = cls._convert_to_value(num1, k_suffix)
            val2 = cls._convert_to_value(num2, k_suffix)
            if val1 is not None and val2 is not None:
                return (min(val1, val2), max(val1, val2))

        return None

    @classmethod
    def _extract_numbers(cls, text: str) -> list[int]:
        """Extract and convert numeric values from text.

        Args:
            text: Normalized salary text

        Returns:
            list[int]: List of parsed numeric values
        """
        numbers = _NUMBER_PATTERN.findall(text)
        parsed_nums = []

        for num_str, k_suffix in numbers:
            if value := cls._convert_to_value(num_str, k_suffix):
                parsed_nums.append(value)

        return parsed_nums

    @classmethod
    def _convert_time_based_salary(
        cls, values: list[int], is_hourly: bool, is_monthly: bool
    ) -> list[int]:
        """Convert hourly or monthly rates to annual equivalents.

        Args:
            values: List of salary values
            is_hourly: True if values are hourly rates
            is_monthly: True if values are monthly rates

        Returns:
            list[int]: Converted annual salary values
        """
        if is_hourly:
            # Convert hourly to annual: hourly * 40 hours/week * 52 weeks/year
            return [int(val * 40 * 52) for val in values]
        if is_monthly:
            # Convert monthly to annual: monthly * 12 months/year
            return [int(val * 12) for val in values]
        return values

    @field_validator("salary", mode="before")
    @classmethod
    def parse_salary(  # noqa: PLR0911
        cls, value: str | tuple[int | None, int | None] | None
    ) -> tuple[int | None, int | None]:
        """Parse salary string into (min, max) tuple.

        Handles various salary formats including:
        - Range formats: "$100k-150k", "£80,000 - £120,000", "110k to 150k"
        - Single values: "$120k", "150000", "up to $150k", "from $110k"
        - Currency symbols: $, £, €, ¥, ¢, ₹
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
        # Handle tuple inputs directly
        if isinstance(value, tuple) and len(value) == 2:
            return value

        # Handle None or empty string inputs
        if value is None or not isinstance(value, str) or value.strip() == "":
            return (None, None)

        original = value.strip()

        # Detect contextual patterns
        is_up_to, is_from, is_hourly, is_monthly = cls._detect_context(original)

        # Normalize the string
        cleaned = cls._normalize_salary_string(original)

        # First try to parse shared-k range patterns
        if shared_k_range := cls._parse_shared_k_range(cleaned):
            # Convert time-based rates to annual equivalents
            min_val, max_val = shared_k_range
            converted_values = cls._convert_time_based_salary(
                [min_val, max_val], is_hourly, is_monthly
            )
            return (converted_values[0], converted_values[1])

        # Extract individual numbers
        parsed_nums = cls._extract_numbers(cleaned)

        if not parsed_nums:
            return (None, None)

        # Convert time-based rates to annual equivalents
        parsed_nums = cls._convert_time_based_salary(parsed_nums, is_hourly, is_monthly)

        # Handle different patterns based on context and number count
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
