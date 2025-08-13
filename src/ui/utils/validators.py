"""Comprehensive validation utilities for safe data conversion and validation.

This module provides robust validation functions designed for handling potentially
unreliable input data, particularly from web scraping and user interfaces. It leverages
modern Pydantic 2.x patterns while maintaining backward compatibility.

Key Features:
- Safe integer conversion with multiple fallback strategies
- Context-aware job count validation with logging
- Modern Annotated type definitions for reusable validation
- Backward-compatible function interfaces
- Comprehensive error handling and logging

The validation logic handles edge cases including:
- None values, empty strings, and malformed data
- Boolean-to-integer conversion (respecting bool/int hierarchy)
- String parsing with regex extraction for embedded numbers
- Infinite/NaN float handling with math.isfinite checks
- Graceful fallbacks to default values on validation failures

Usage:
    # Modern Annotated types (recommended)
    from typing import Annotated
    from pydantic import BaseModel
    from .validators import SafeInt, JobCount

    class MyModel(BaseModel):
        count: SafeInt  # Automatically converts to safe non-negative int
        job_total: JobCount  # Includes context logging

    # Legacy function interface (backward compatibility)
    from .validators import safe_int, safe_job_count

    result = safe_int("42.7")  # Returns 42
    job_count = safe_job_count("5 jobs", company_name="Example Corp")
"""

import logging
import math
import re

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CORE VALIDATOR FUNCTIONS
# =============================================================================


def safe_int_converter(value: Any) -> int:
    """Convert various input types to safe non-negative integers.

    This is the core conversion logic that handles multiple input types
    with comprehensive fallback strategies. Designed for use with Pydantic
    validators and direct function calls.

    Args:
        value: Input value of any type to convert to safe integer

    Returns:
        Non-negative integer (>= 0)

    Conversion Strategy:
        - None -> 0
        - bool -> int(bool) (checked before int since bool subclasses int)
        - int -> max(0, int) (ensure non-negative)
        - float -> max(0, int(float)) if finite, else 0
        - str -> parse as number, with regex fallback for embedded numbers
        - Other types -> 0 (with exception handling)
    """
    try:
        if value is None:
            return 0

        # Check bool before int since bool is subclass of int in Python
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int | float):
            if isinstance(value, int):
                return max(0, value)
            if isinstance(value, float) and math.isfinite(value):
                return max(0, int(value))
            return 0  # Handle inf, -inf, nan

        if isinstance(value, str):
            value = value.strip()
            if value:
                try:
                    # Try direct conversion first
                    return max(0, int(float(value)))
                except (ValueError, TypeError):
                    # Extract first number from string using regex
                    match = re.search(r"-?\d+(?:\.\d+)?", value)
                    if match:
                        return max(0, int(float(match.group())))
            return 0

    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        # Comprehensive exception handling for edge cases
        logger.debug("Exception during safe_int_converter: %s", e)

    # Fallback for unhandled types or exceptions
    return 0


def job_count_converter_with_context(value: Any, company_name: str = "unknown") -> int:
    """Convert job count values with context-aware logging.

    This function provides the same safe integer conversion as safe_int_converter
    but adds logging for data quality monitoring in job scraping contexts.

    Args:
        value: Input value to convert to job count
        company_name: Company context for logging (default: "unknown")

    Returns:
        Non-negative integer representing job count

    Side Effects:
        - Logs warnings for conversion failures
        - Logs info for successful conversions when input differs from output
    """
    try:
        result = safe_int_converter(value)

        # Log data quality information when conversion occurs
        if value != result and value is not None:
            logger.info(
                "Converted job count for %s: %s -> %s", company_name, value, result
            )

    except Exception as e:
        logger.warning(
            "Failed to convert job count for %s: %s (%s)",
            company_name,
            value,
            e,
        )
        return 0
    else:
        return result


# =============================================================================
# MODERN ANNOTATED TYPE DEFINITIONS (PYDANTIC 2.X)
# =============================================================================


# Reusable Annotated types leveraging modern Pydantic patterns
SafeInt = Annotated[
    int,
    BeforeValidator(safe_int_converter),
    Field(ge=0, description="Non-negative integer with safe conversion"),
]

# Note: JobCount with context logging requires function interface
# since Annotated validators can't easily access additional context
JobCount = Annotated[
    int,
    BeforeValidator(safe_int_converter),
    Field(ge=0, description="Job count with safe conversion"),
]


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert any value to a non-negative integer.

    Legacy function interface providing backward compatibility.
    Uses modern validator internally while maintaining original signature.

    Args:
        value: Value to convert to safe integer
        default: Default value if conversion fails (will be made non-negative)

    Returns:
        Non-negative integer

    Example:
        >>> safe_int("42")
        42
        >>> safe_int("invalid")
        0
        >>> safe_int("-5")
        0
        >>> safe_int(None, default=10)
        0
    """
    try:
        return safe_int_converter(value)
    except Exception as e:
        logger.warning("Failed to convert %s to safe integer: %s", value, e)
        return max(0, default)


def safe_job_count(value: Any, company_name: str = "unknown") -> int:
    """Safely convert job count values with context-aware logging.

    Legacy function interface for job count conversion with company context.
    Provides enhanced logging for data quality monitoring in scraping workflows.

    Args:
        value: Value to convert to job count
        company_name: Company name for logging context

    Returns:
        Non-negative integer representing job count

    Example:
        >>> safe_job_count("5", "Example Corp")
        5
        >>> safe_job_count("5 jobs available", "Tech Co")
        5  # Extracts number from text
    """
    return job_count_converter_with_context(value, company_name)


# =============================================================================
# BACKWARD COMPATIBILITY - PYDANTIC MODEL
# =============================================================================


class SafeIntValidator(BaseModel):
    """Pydantic model for safe integer validation.

    Maintained for backward compatibility with existing code that expects
    this class interface. New code should prefer the Annotated types above.

    Attributes:
        value: Non-negative integer value with safe conversion validation
    """

    value: int = Field(ge=0, description="Non-negative integer value")

    @field_validator("value", mode="before")
    @classmethod
    def convert_to_safe_int(cls, v: Any) -> int:
        """Convert various input types to safe non-negative integers.

        This method preserves the original class-based validator interface
        while using the modernized core conversion logic.
        """
        return safe_int_converter(v)


# =============================================================================
# TYPE ALIASES AND EXPORTS
# =============================================================================


# Type aliases for cleaner type hints and backward compatibility
SafeInteger = int  # Alias for type annotations
JobCountType = int  # Alias for job count type annotations

# Comprehensive export list for clean module interface
__all__ = [
    "JobCount",
    "JobCountType",
    "SafeInt",
    "SafeIntValidator",
    "SafeInteger",
    "job_count_converter_with_context",
    "safe_int",
    "safe_int_converter",
    "safe_job_count",
]
