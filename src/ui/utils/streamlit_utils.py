"""Streamlit-specific utilities and type aliases.

This module provides utilities specifically for Streamlit applications:
- Context detection functions
- Type aliases for common data structures
- Streamlit-specific helper functions

Focused on keeping Streamlit dependencies isolated and testable.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.utils.validators import SafeInt

# Type aliases for backward compatibility and clarity
type SalaryTuple = tuple[int | None, int | None]

# Import type aliases from validators to maintain interface
if not TYPE_CHECKING:
    from src.ui.utils.validators import SafeInt

SafeInteger = SafeInt  # Alias for backward compatibility


def is_streamlit_context() -> bool:
    """Check if we're running in a proper Streamlit context.

    Returns:
        True if running in Streamlit, False otherwise
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        return False
