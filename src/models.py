"""Database models for companies and jobs in the AI Job Scraper.

This module contains SQLModel classes representing database entities:
- CompanySQL: Company information with scraping statistics
- JobSQL: Job postings with application tracking and salary parsing

The module also includes salary parsing functionality with comprehensive
regex patterns for handling various salary formats from job boards.
"""

import hashlib
import re

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import computed_field, field_validator, model_validator
from sqlalchemy.types import JSON
from sqlmodel import Column, Field, Relationship, SQLModel

# Type aliases for better readability
type SalaryTuple = tuple[int | None, int | None]


@dataclass(frozen=True)
class SalaryContext:
    """Context flags for salary parsing."""

    is_up_to: bool = False
    is_from: bool = False
    is_hourly: bool = False
    is_monthly: bool = False


# Compiled regex patterns for salary parsing
_UP_TO_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:up\s+to|maximum\s+of|max\s+of|not\s+more\s+than)\b", re.IGNORECASE
)
_FROM_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:from|starting\s+at|minimum\s+of|min\s+of|at\s+least)\b", re.IGNORECASE
)
_CURRENCY_PATTERN: re.Pattern[str] = re.compile(r"[£$€¥¢₹]")
# Pattern for shared k suffix at end: "100-120k"
_RANGE_K_PATTERN: re.Pattern[str] = re.compile(
    r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*([kK])"
)
# Pattern for both numbers with k: "100k-150k"
_BOTH_K_PATTERN: re.Pattern[str] = re.compile(
    r"(\d+(?:\.\d+)?)([kK])\s*-\s*(\d+(?:\.\d+)?)([kK])"
)
# Pattern for one-sided k: "100k-120" (k on first number only)
_ONE_SIDED_K_PATTERN: re.Pattern[str] = re.compile(
    r"(\d+(?:\.\d+)?)([kK])\s*-\s*(\d+(?:\.\d+)?)(?!\s*[kK])"
)
_NUMBER_PATTERN: re.Pattern[str] = re.compile(r"(\d+(?:\.\d+)?)\s*([kK])?")
_HOURLY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:per\s+hour|hourly|/hour|/hr)\b", re.IGNORECASE
)
_MONTHLY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:per\s+month|monthly|/month|/mo)\b", re.IGNORECASE
)

_PHRASES_TO_REMOVE: list[str] = [
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

    model_config = {"validate_assignment": True}

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
    content_hash: str = Field(default="", index=True)
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
    def _detect_context(cls, text: str) -> SalaryContext:
        """Detect contextual patterns in salary text for parsing.

        Analyzes the input text to identify salary range indicators ("up to", "from"),
        and time-based qualifiers ("hourly", "monthly") to aid in accurate parsing.

        Args:
            text: Original salary text to analyze for context patterns.

        Returns:
            SalaryContext with boolean flags indicating detected patterns:
            - is_up_to: True if text contains upper bound indicators
            - is_from: True if text contains lower bound indicators
            - is_hourly: True if text contains hourly rate indicators
            - is_monthly: True if text contains monthly rate indicators

        Examples:
            >>> JobSQL._detect_context("up to $120k annually")
            (True, False, False, False)
            >>> JobSQL._detect_context("from $25/hour")
            (False, True, True, False)
        """
        return SalaryContext(
            is_up_to=bool(_UP_TO_PATTERN.search(text)),
            is_from=bool(_FROM_PATTERN.search(text)),
            is_hourly=bool(_HOURLY_PATTERN.search(text)),
            is_monthly=bool(_MONTHLY_PATTERN.search(text)),
        )

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

        Handles conversion of salary numbers with thousand multipliers,
        ensuring proper scaling when 'k' or 'K' suffixes are present.

        Args:
            num_str: Numeric string to convert (e.g., "120", "150.5").
            k_suffix: Optional 'k' or 'K' suffix indicating thousands multiplier.

        Returns:
            Converted integer value (multiplied by 1000 if k_suffix present),
            or None if conversion fails due to invalid input.

        Examples:
            >>> JobSQL._convert_to_value("120", "k")
            120000
            >>> JobSQL._convert_to_value("75", None)
            75
            >>> JobSQL._convert_to_value("invalid", "k")
            None
        """
        try:
            multiplier = 1000 if k_suffix and k_suffix.lower() == "k" else 1
            return int(float(num_str) * multiplier)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _parse_shared_k_range(cls, text: str) -> tuple[int, int] | None:
        """Parse salary ranges with various k suffix patterns.

        Handles complex k-suffix patterns including shared suffixes ("100-120k"),
        both-sided suffixes ("100k-150k"), and one-sided suffixes ("100k-120").

        Args:
            text: Normalized salary text to parse for range patterns.

        Returns:
            Tuple of (minimum, maximum) salary values if a valid range pattern
            is found, with values properly ordered. None if no valid pattern
            is detected.

        Examples:
            >>> JobSQL._parse_shared_k_range("100-120k")
            (100000, 120000)
            >>> JobSQL._parse_shared_k_range("100k-150k")
            (100000, 150000)
            >>> JobSQL._parse_shared_k_range("no range here")
            None
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
        """Extract and convert all numeric values from salary text.

        Finds all numeric patterns in the text and converts them to integers,
        applying thousand multipliers where k-suffixes are present.

        Args:
            text: Normalized salary text containing numeric values.

        Returns:
            List of all successfully parsed integer values found in the text,
            in order of appearance. Empty list if no valid numbers found.

        Examples:
            >>> JobSQL._extract_numbers("120k to 150k")
            [120000, 150000]
            >>> JobSQL._extract_numbers("salary range 80000-90000")
            [80000, 90000]
        """
        numbers = _NUMBER_PATTERN.findall(text)
        parsed_nums = []

        for num_str, k_suffix in numbers:
            if value := cls._convert_to_value(num_str, k_suffix):
                parsed_nums.append(value)

        return parsed_nums

    @classmethod
    def _convert_time_based_salary(
        cls, values: Sequence[int], is_hourly: bool, is_monthly: bool
    ) -> list[int]:
        """Convert time-based salary rates to standardized annual values.

        Standardizes salary values by converting hourly and monthly rates
        to annual equivalents using standard work hour assumptions.

        Args:
            values: Sequence of salary values to convert.
            is_hourly: True if values represent hourly rates requiring conversion.
            is_monthly: True if values represent monthly rates requiring conversion.

        Returns:
            List of annual salary values. Hourly rates are converted using
            40 hours/week x 52 weeks/year. Monthly rates use 12 months/year.
            Values are returned unchanged if neither hourly nor monthly.

        Examples:
            >>> JobSQL._convert_time_based_salary([25, 30], True, False)
            [52000, 62400]  # 25*40*52, 30*40*52
            >>> JobSQL._convert_time_based_salary([5000, 6000], False, True)
            [60000, 72000]  # 5000*12, 6000*12
        """
        if is_hourly:
            # Convert hourly to annual: hourly * 40 hours/week * 52 weeks/year
            return [int(val * 40 * 52) for val in values]
        if is_monthly:
            # Convert monthly to annual: monthly * 12 months/year
            return [int(val * 12) for val in values]
        return values

    @model_validator(mode="before")
    @classmethod
    def generate_content_hash(cls, data):
        """Auto-generate content hash from job content if not provided.

        Creates a deterministic hash from title, description, and link
        to enable duplicate detection and content fingerprinting.
        """
        # Convert object to dict if needed
        if not isinstance(data, dict):
            return data

        # Only generate if content_hash is not provided or is empty
        if not data.get("content_hash"):
            title = data.get("title", "")
            description = data.get("description", "")
            link = data.get("link", "")

            # Create deterministic content string from key job fields
            content = f"{title}|{description}|{link}"

            # Generate MD5 hash (acceptable for non-cryptographic fingerprinting)
            generated_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            data["content_hash"] = generated_hash

        return data

    @field_validator("salary", mode="before")
    @classmethod
    def parse_salary(cls, value: str | SalaryTuple | None) -> SalaryTuple:
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
        # Handle early exit cases
        result = cls._handle_early_cases(value)
        if result is not None:
            return result

        original = value.strip()

        # Detect contextual patterns
        context = cls._detect_context(original)

        # Normalize the string
        cleaned = cls._normalize_salary_string(original)

        # Try parsing shared-k range patterns first
        result = cls._try_shared_k_range(cleaned, context)
        if result is not None:
            return result

        # Extract and process individual numbers
        return cls._process_parsed_numbers(cleaned, context)

    @classmethod
    def _handle_early_cases(cls, value) -> tuple[int | None, int | None] | None:
        """Handle early return cases for parse_salary."""
        # Handle tuple inputs directly
        if isinstance(value, tuple) and len(value) == 2:
            return value

        # Handle None or empty string inputs
        if value is None or not isinstance(value, str) or value.strip() == "":
            return (None, None)

        return None

    @classmethod
    def _try_shared_k_range(
        cls, cleaned: str, context: SalaryContext
    ) -> tuple[int | None, int | None] | None:
        """Try to parse shared-k range patterns."""
        if shared_k_range := cls._parse_shared_k_range(cleaned):
            # Convert time-based rates to annual equivalents
            min_val, max_val = shared_k_range
            converted_values = cls._convert_time_based_salary(
                [min_val, max_val], context.is_hourly, context.is_monthly
            )
            return (converted_values[0], converted_values[1])
        return None

    @classmethod
    def _process_parsed_numbers(
        cls,
        cleaned: str,
        context: SalaryContext,
    ) -> tuple[int | None, int | None]:
        """Process extracted numbers based on context."""
        # Extract individual numbers
        parsed_nums = cls._extract_numbers(cleaned)

        if not parsed_nums:
            return (None, None)

        # Convert time-based rates to annual equivalents
        parsed_nums = cls._convert_time_based_salary(
            parsed_nums, context.is_hourly, context.is_monthly
        )

        # Handle single vs multiple numbers
        if len(parsed_nums) == 1:
            single_value = parsed_nums[0]
            if context.is_up_to:
                return (None, single_value)
            if context.is_from:
                return (single_value, None)
            # For single values without context, return as both min and max
            return (single_value, single_value)

        if len(parsed_nums) >= 2:
            # For multiple numbers, return as range (min, max)
            return (min(parsed_nums), max(parsed_nums))

        return (None, None)

    @classmethod
    def create_validated(cls, **data) -> "JobSQL":
        """Create a JobSQL instance with proper Pydantic validation.

        This factory method ensures that Pydantic validators (including model_validator)
        are executed properly, working around the SQLAlchemy + Pydantic v2 integration
        issue.

        Args:
            **data: Job data to validate and create instance from.

        Returns:
            JobSQL: Validated JobSQL instance with content_hash generated.

        Example:
            job = JobSQL.create_validated(
                title="Software Engineer",
                description="Great role...",
                link="https://example.com/job/123"
            )
        """
        # Step 1: Use Pydantic's validation on the raw data
        validated_data = cls.model_validate(data)

        # Step 2: Extract the validated data and create SQLModel instance
        clean_data = validated_data.model_dump()

        # Step 3: Create the actual table instance (bypasses validation but uses
        # clean data)
        return cls(**clean_data)
