"""Data cleaning utilities for handling messy external data sources.

This module provides utilities for cleaning and validating data from external
sources like web scrapers, APIs, and CSV files. It handles common data quality
issues like NaN values, null strings, and malformed data.
"""

from typing import Any


def clean_string_field(value: Any) -> str:
    """Clean string fields to handle NaN, None, and invalid values.

    This utility handles common data quality issues from external sources:
    - NaN values from pandas/numpy operations
    - None/null values from APIs
    - String representations of null values

    Args:
        value: Input value that should be cleaned to a string.

    Returns:
        Clean string value, empty string for invalid inputs.

    Examples:
        >>> clean_string_field("valid string")
        'valid string'
        >>> clean_string_field(None)
        ''
        >>> clean_string_field("NaN")
        ''
        >>> clean_string_field("null")
        ''
    """
    if value is None or str(value).lower() in ["nan", "none", "null"]:
        return ""
    return str(value).strip()


def clean_numeric_field(
    value: Any, default: int | float | None = None
) -> int | float | None:
    """Clean numeric fields to handle NaN and invalid values.

    Args:
        value: Input value that should be cleaned to a number.
        default: Default value to return for invalid inputs.

    Returns:
        Clean numeric value or default for invalid inputs.
    """
    if value is None:
        return default

    value_str = str(value).lower()
    if value_str in ["nan", "none", "null", ""]:
        return default

    try:
        # Try integer first, then float
        if isinstance(value, int | float) and str(value).lower() not in [
            "nan",
            "inf",
            "-inf",
        ]:
            return value
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
