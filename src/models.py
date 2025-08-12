"""Database models for companies and jobs in the AI Job Scraper.

This module contains SQLModel classes representing database entities:
- CompanySQL: Company information with scraping statistics
- JobSQL: Job postings with application tracking and salary parsing

The module also includes salary parsing functionality with comprehensive
regex patterns for handling various salary formats from job boards.
"""

import hashlib
import logging
import re

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from babel.numbers import NumberFormatError, parse_decimal, parse_number
from price_parser import Price
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

# Logger for salary parsing operations
logger = logging.getLogger(__name__)


class LibrarySalaryParser:
    """Library-first salary parser using price-parser and babel.

    This class implements a modern approach to salary parsing by leveraging:
    - price-parser: For currency extraction and basic price parsing
    - babel: For locale-aware decimal parsing
    - Custom logic: Only for salary-specific patterns (k-suffix, ranges, context)

    This replaces ~200 lines of regex-based parsing with library-first implementation.
    """

    @staticmethod
    def parse_salary_text(text: str) -> SalaryTuple:
        """Parse salary text using library-first approach.

        Args:
            text: Raw salary text to parse

        Returns:
            tuple[int | None, int | None]: Parsed (min, max) salary values
        """
        if not text or not text.strip():
            return (None, None)

        original_text = text.strip()
        logger.debug("Parsing salary text: %s", original_text)

        # Detect contextual patterns first
        context = LibrarySalaryParser._detect_context(original_text)

        # Try range parsing first (most specific)
        result = LibrarySalaryParser._parse_salary_range(original_text, context)
        if result != (None, None):
            return result

        # Try single value parsing
        result = LibrarySalaryParser._parse_single_salary(original_text, context)
        if result != (None, None):
            return result

        logger.debug("Could not parse salary: %s", original_text)
        return (None, None)

    @staticmethod
    def _detect_context(text: str) -> SalaryContext:
        """Detect contextual patterns for salary parsing."""
        return SalaryContext(
            is_up_to=bool(_UP_TO_PATTERN.search(text)),
            is_from=bool(_FROM_PATTERN.search(text)),
            is_hourly=bool(_HOURLY_PATTERN.search(text)),
            is_monthly=bool(_MONTHLY_PATTERN.search(text)),
        )

    @staticmethod
    def _parse_salary_range(text: str, context: SalaryContext) -> SalaryTuple:
        """Parse salary ranges using k-suffix patterns and price-parser."""
        # Handle k-suffix ranges first (most salary-specific)
        k_range = LibrarySalaryParser._parse_k_suffix_ranges(text)
        if k_range:
            min_val, max_val = k_range
            converted_values = LibrarySalaryParser._convert_time_based_salary(
                [min_val, max_val], context.is_hourly, context.is_monthly
            )
            return (converted_values[0], converted_values[1])

        # Try extracting multiple price objects for ranges
        prices = LibrarySalaryParser._extract_multiple_prices(text)
        if len(prices) >= 2:
            # Convert to annual if needed and return range
            values = [int(price.amount) for price in prices if price.amount]
            if values:
                converted_values = LibrarySalaryParser._convert_time_based_salary(
                    values, context.is_hourly, context.is_monthly
                )
                return (min(converted_values), max(converted_values))

        return (None, None)

    @staticmethod
    def _parse_single_salary(text: str, context: SalaryContext) -> SalaryTuple:
        """Parse single salary values using price-parser."""
        # First check for k-suffix patterns and handle them specially
        k_match = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", text)
        if k_match:
            try:
                base_value = float(k_match.group(1))
                value = int(base_value * 1000)

                # Convert time-based to annual and apply context
                converted_values = LibrarySalaryParser._convert_time_based_salary(
                    [value], context.is_hourly, context.is_monthly
                )
                final_value = converted_values[0]
                return LibrarySalaryParser._apply_context_logic(final_value, context)
            except (ValueError, TypeError):
                pass

        # Try price-parser for non-k-suffix cases
        try:
            price = Price.fromstring(text)
            if price.amount:
                value = int(price.amount)

                # Convert time-based to annual and apply context
                converted_values = LibrarySalaryParser._convert_time_based_salary(
                    [value], context.is_hourly, context.is_monthly
                )
                final_value = converted_values[0]
                return LibrarySalaryParser._apply_context_logic(final_value, context)
        except Exception as e:
            logger.debug("Price parser failed for '%s': %s", text, e)

        # Fallback to babel-based number extraction
        return LibrarySalaryParser._parse_with_babel_fallback(text, context)

    @staticmethod
    def _extract_multiple_prices(text: str) -> list[Price]:
        """Extract multiple price objects from text for range detection."""
        prices = []

        # Split text on common range separators and try parsing each part
        parts = re.split(r"\s*[-\u2013\u2014]\s*|\s+to\s+", text, flags=re.IGNORECASE)
        for raw_part in parts:
            part = raw_part.strip()

            # Handle k-suffix parts specially
            k_match = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", part)
            if k_match:
                try:
                    base_value = float(k_match.group(1))
                    amount = base_value * 1000
                    # Create a mock Price object
                    price = type("Price", (), {"amount": Decimal(str(amount))})()
                    prices.append(price)
                    continue
                except (ValueError, TypeError):
                    pass

            # Try normal price parsing
            try:
                price = Price.fromstring(part)
                if price.amount:
                    prices.append(price)
            except Exception as e:
                logger.debug("Failed to parse price from part '%s': %s", part, e)
                continue

        return prices

    @staticmethod
    def _parse_k_suffix_ranges(text: str) -> tuple[int, int] | None:
        """Parse k-suffix ranges like '100-120k', '100k-150k', '110k to 150k'."""
        # Try "to" patterns with k-suffix first
        to_pattern = re.search(
            r"(\d+(?:\.\d+)?)\s*[kK]\s+to\s+(\d+(?:\.\d+)?)\s*[kK]", text, re.IGNORECASE
        )
        if to_pattern:
            try:
                val1 = int(float(to_pattern.group(1)) * 1000)
                val2 = int(float(to_pattern.group(2)) * 1000)
                return (min(val1, val2), max(val1, val2))
            except (ValueError, TypeError):
                pass

        # Try different k-suffix patterns
        patterns = [
            _RANGE_K_PATTERN,  # 100-120k
            _BOTH_K_PATTERN,  # 100k-150k
            _ONE_SIDED_K_PATTERN,  # 100k-120
        ]

        for pattern in patterns:
            if match := pattern.search(text):
                groups = match.groups()

                if pattern == _RANGE_K_PATTERN:  # 100-120k
                    num1, num2, _k_suffix = groups
                    val1 = LibrarySalaryParser._safe_decimal_to_int(num1) * 1000
                    val2 = LibrarySalaryParser._safe_decimal_to_int(num2) * 1000
                elif pattern == _BOTH_K_PATTERN:  # 100k-150k
                    num1, _k1, num2, _k2 = groups
                    val1 = LibrarySalaryParser._safe_decimal_to_int(num1) * 1000
                    val2 = LibrarySalaryParser._safe_decimal_to_int(num2) * 1000
                elif pattern == _ONE_SIDED_K_PATTERN:  # 100k-120
                    num1, _k_suffix, num2 = groups
                    val1 = LibrarySalaryParser._safe_decimal_to_int(num1) * 1000
                    val2 = (
                        LibrarySalaryParser._safe_decimal_to_int(num2) * 1000
                    )  # Apply k to both

                if val1 and val2:
                    return (min(val1, val2), max(val1, val2))

        return None

    @staticmethod
    def _apply_k_suffix_multiplication(text: str, value: int) -> int:
        """Apply k-suffix multiplication if present."""
        # Check for k/K suffix in the text
        if re.search(r"\d+(?:\.\d+)?\s*[kK]\b", text):
            return value * 1000
        return value

    @staticmethod
    def _apply_context_logic(final_value: int, context: SalaryContext) -> SalaryTuple:
        """Apply context-based logic to determine salary tuple."""
        if context.is_up_to:
            return (None, final_value)
        if context.is_from:
            return (final_value, None)
        return (final_value, final_value)

    @staticmethod
    def _parse_with_babel_fallback(text: str, context: SalaryContext) -> SalaryTuple:
        """Fallback parsing using babel's number parsing."""
        try:
            # Clean text for babel parsing
            cleaned = LibrarySalaryParser._clean_text_for_babel(text)

            # Try parsing as decimal first
            try:
                amount = parse_decimal(cleaned, locale="en_US")
                value = int(amount)

                # Apply k-suffix if present
                value = LibrarySalaryParser._apply_k_suffix_multiplication(text, value)

                # Convert time-based and apply context
                converted_values = LibrarySalaryParser._convert_time_based_salary(
                    [value], context.is_hourly, context.is_monthly
                )
                final_value = converted_values[0]
                return LibrarySalaryParser._apply_context_logic(final_value, context)

            except NumberFormatError:
                # Try as integer
                amount = parse_number(cleaned, locale="en_US")
                value = int(amount)

                value = LibrarySalaryParser._apply_k_suffix_multiplication(text, value)

                converted_values = LibrarySalaryParser._convert_time_based_salary(
                    [value], context.is_hourly, context.is_monthly
                )
                final_value = converted_values[0]
                return LibrarySalaryParser._apply_context_logic(final_value, context)

        except (NumberFormatError, ValueError) as e:
            logger.debug("Babel parsing failed for '%s': %s", text, e)

        return (None, None)

    @staticmethod
    def _clean_text_for_babel(text: str) -> str:
        """Clean text for babel number parsing."""
        # Remove currency symbols (babel doesn't need them)
        cleaned = _CURRENCY_PATTERN.sub("", text)

        # Remove common phrases but keep numeric content
        for pattern in _PHRASES_TO_REMOVE:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove context patterns
        cleaned = _UP_TO_PATTERN.sub("", cleaned)
        cleaned = _FROM_PATTERN.sub("", cleaned)
        cleaned = _HOURLY_PATTERN.sub("", cleaned)
        cleaned = _MONTHLY_PATTERN.sub("", cleaned)

        # Clean up spacing and extract first numeric-looking part
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Extract first number-like sequence for babel
        match = re.search(r"[\d,./]+", cleaned)
        if match:
            return match.group().strip()

        return cleaned

    @staticmethod
    def _safe_decimal_to_int(value_str: str) -> int | None:
        """Safely convert decimal string to int using babel."""
        try:
            decimal_val = parse_decimal(value_str, locale="en_US")
            return int(decimal_val)
        except (NumberFormatError, ValueError, TypeError):
            return None

    @staticmethod
    def _convert_time_based_salary(
        values: Sequence[int], is_hourly: bool, is_monthly: bool
    ) -> list[int]:
        """Convert time-based salary rates to annual values."""
        if is_hourly:
            # Convert hourly to annual: hourly * 40 hours/week * 52 weeks/year
            return [int(val * 40 * 52) for val in values]
        if is_monthly:
            # Convert monthly to annual: monthly * 12 months/year
            return [int(val * 12) for val in values]
        return list(values)


class CompanySQL(SQLModel, table=True, extend_existing=True):
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


class JobSQL(SQLModel, table=True, extend_existing=True):
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
        """Parse salary string into (min, max) tuple using library-first approach.

        This method uses price-parser and babel libraries for robust parsing,
        with custom logic only for salary-specific patterns.

        Handles various salary formats including:
        - Range formats: "$100k-150k", "£80,000 - £120,000", "110k to 150k"
        - Single values: "$120k", "150000", "up to $150k", "from $110k"
        - Currency symbols: $, £, €, ¥, ¢, ₹
        - Suffixes: k, K (for thousands)
        - Common phrases: "per year", "per annum", "up to", "from", "starting at"
        - Time-based rates: "$50 per hour", "£5000 per month"

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

        # Use the new library-first parser
        return LibrarySalaryParser.parse_salary_text(value.strip())

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
