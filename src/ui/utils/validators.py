"""Input validation utilities with Pydantic integration.

This module provides validation functions and Pydantic types for:
- Safe integer conversion with non-negative constraints
- Job count validation with context-aware logging
- Custom validator factories for different default values

All validators handle edge cases gracefully and provide meaningful error recovery.
"""

import logging

from typing import Annotated, Any

from pydantic import BeforeValidator, validate_call

logger = logging.getLogger(__name__)


def ensure_non_negative_int(value: Any) -> int:
    """Ensure value is a non-negative integer.

    This validator handles various input types and safely converts them:
    - None -> 0
    - bool -> int conversion (True=1, False=0)
    - int/float -> max(0, int(value)) for finite values
    - str -> parse number, extract first number if mixed, use 0 if invalid
    - other types -> 0
    """
    try:
        if value is None:
            return 0
        if isinstance(
            value, bool
        ):  # Check bool before int since bool is subclass of int
            return int(value)
        if isinstance(value, int | float):
            import math

            return max(
                0,
                int(value)
                if isinstance(value, int)
                else (
                    int(value)
                    if isinstance(value, float) and math.isfinite(value)
                    else 0
                ),
            )
        if isinstance(value, str):
            value = value.strip()
            if value:
                try:
                    return max(0, int(float(value)))
                except (ValueError, TypeError):
                    # Extract first number from string
                    import re

                    match = re.search(r"-?\d+(?:\.\d+)?", value)
                    return max(0, int(float(match.group()))) if match else 0
            else:
                # Empty string becomes 0
                return 0
    except (ValueError, TypeError, AttributeError):
        logger.warning("Failed to convert %s to non-negative integer, using 0", value)
        return 0

    # Fallback for unhandled types
    return 0


def ensure_non_negative_int_with_default(default: int = 0):
    """Create a validator that ensures non-negative int with custom default.

    Args:
        default: Default value to use for invalid inputs (will be made non-negative)

    Returns:
        Validator function that converts to non-negative int
    """
    safe_default = max(0, default)

    def validator(value: Any) -> int:
        """Ensure value is a non-negative integer with custom default."""
        try:
            if value is None:
                return 0
            if isinstance(
                value, bool
            ):  # Check bool before int since bool is subclass of int
                return int(value)
            if isinstance(value, int | float):
                import math

                return max(
                    0,
                    int(value)
                    if isinstance(value, int)
                    else (
                        int(value)
                        if isinstance(value, float) and math.isfinite(value)
                        else 0
                    ),
                )
            if isinstance(value, str):
                value = value.strip()
                if value:
                    try:
                        return max(0, int(float(value)))
                    except (ValueError, TypeError):
                        # Extract first number from string
                        import re

                        match = re.search(r"-?\d+(?:\.\d+)?", value)
                        return (
                            max(0, int(float(match.group()))) if match else safe_default
                        )
                else:
                    # Empty string should use default
                    return safe_default
        except (ValueError, TypeError, AttributeError):
            logger.warning(
                "Failed to convert %s to non-negative integer, using default %s",
                value,
                safe_default,
            )
            return safe_default

        # Fallback for unhandled types (use default)
        return safe_default

    return validator


# Pydantic Annotated types for reuse across the codebase
SafeInt = Annotated[int, BeforeValidator(ensure_non_negative_int)]
JobCount = Annotated[int, BeforeValidator(ensure_non_negative_int)]


# Legacy functions for backward compatibility (deprecated)
@validate_call
def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert any value to a non-negative integer.

    DEPRECATED: Use SafeInt Annotated type instead.
    This function is kept for backward compatibility.
    """
    validator = ensure_non_negative_int_with_default(default)
    return validator(value)


@validate_call
def safe_job_count(value: Any, company_name: str = "unknown") -> int:
    """Safely convert job count values with context-aware logging.

    DEPRECATED: Use JobCount Annotated type instead.
    This function is kept for backward compatibility.
    """
    try:
        result = ensure_non_negative_int(value)
    except Exception as e:
        logger.warning(
            "Failed to convert job count for %s: %s (%s)",
            company_name,
            value,
            e,
        )
        return 0

    if value != result and value is not None:
        logger.info("Converted job count for %s: %s -> %s", company_name, value, result)
    return result
