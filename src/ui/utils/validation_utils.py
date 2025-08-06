"""Validation utilities for type-safe data processing.

This module provides library-first validation utilities using Pydantic patterns
for robust type conversion and error handling throughout the application.

Key features:
- Safe integer conversion with comprehensive error handling
- Type-safe data validation using modern Python patterns
- Reusable validation functions following DRY principles
"""

import logging

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class SafeIntValidator(BaseModel):
    """Pydantic model for safe integer validation."""

    value: int = Field(ge=0, description="Non-negative integer value")

    @field_validator("value", mode="before")
    @classmethod
    def convert_to_safe_int(cls, v: Any) -> int:
        """Convert various input types to safe non-negative integers.

        Args:
            v: Input value of any type

        Returns:
            Non-negative integer

        Raises:
            ValueError: If value cannot be safely converted to non-negative integer
        """
        if v is None:
            return 0

        # Handle string inputs
        if isinstance(v, str):
            # Remove whitespace and handle empty strings
            v = v.strip()
            if not v:
                return 0

            # Try to convert string to number
            try:
                # Handle float strings by converting to float first
                v = float(v) if "." in v else int(v)
            except ValueError as e:
                error_msg = f"Cannot convert string '{v}' to integer: {e}"
                raise ValueError(error_msg) from e

        # Handle float inputs - round to nearest integer
        if isinstance(v, float):
            if not (-1e15 <= v <= 1e15):  # Prevent overflow
                error_msg = f"Float value {v} is too large to convert to integer"
                raise ValueError(error_msg)
            v = round(v)

        # Handle boolean inputs
        if isinstance(v, bool):
            v = int(v)

        # Final integer conversion and validation
        try:
            result = int(v)
        except (ValueError, TypeError) as e:
            error_msg = f"Cannot convert {type(v).__name__} value {v} to integer: {e}"
            raise ValueError(error_msg) from e

        # Ensure non-negative
        if result < 0:
            logger.warning("Negative value %d converted to 0 for safety", result)
            result = 0

        return result


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert any value to a non-negative integer.

    This function provides robust type conversion with comprehensive error handling,
    following library-first principles using Pydantic validation.

    Args:
        value: Input value of any type
        default: Default value to return if conversion fails (default: 0)

    Returns:
        Non-negative integer value

    Examples:
        >>> safe_int("123")
        123
        >>> safe_int("45.7")
        46
        >>> safe_int(None)
        0
        >>> safe_int("invalid", default=10)
        10
        >>> safe_int(-5)
        0
    """
    try:
        validator = SafeIntValidator(value=value)
    except ValidationError as e:
        logger.warning("Failed to convert %s to safe integer: %s", value, e)
        return max(0, default)  # Ensure default is also non-negative
    except Exception:
        logger.exception("Unexpected error converting %s to safe integer", value)
        return max(0, default)
    else:
        return validator.value


def safe_job_count(value: Any, company_name: str = "unknown") -> int:
    """Safely convert job count values with context-aware logging.

    Specialized function for converting job counts with additional context
    for better error tracking and debugging.

    Args:
        value: Job count value to convert
        company_name: Company name for context in error messages

    Returns:
        Non-negative integer representing job count
    """
    try:
        result = safe_int(value)
    except Exception as e:
        logger.warning(
            "Failed to convert job count for %s: %s (%s)", company_name, value, e
        )
        return 0
    else:
        if value != result and value is not None:
            logger.info(
                "Converted job count for %s: %s -> %s", company_name, value, result
            )
        return result


# Type aliases for better code documentation
JobCount = int
SafeInteger = int
