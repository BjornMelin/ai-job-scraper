"""Utilities package for AI Job Scraper.

This package contains utility modules including:
- startup_helpers: Application startup optimizations
"""

# Export functions from the parent utils.py module
import sys

from pathlib import Path

# Add parent directory to path to access utils.py
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import from utils.py (the module file, not this package)
try:
    from utils import (
        get_extraction_model,
        get_llm_client,
        get_proxy,
        random_delay,
        random_user_agent,
        resolve_jobspy_proxies,
    )

    __all__ = [
        "get_extraction_model",
        "get_llm_client",
        "get_proxy",
        "random_delay",
        "random_user_agent",
        "resolve_jobspy_proxies",
    ]
except ImportError:
    # Fallback: create dummy functions if utils.py is not available
    def random_delay(min_sec: float = 1.0, max_sec: float = 5.0) -> None:
        """Dummy random_delay function."""

    def resolve_jobspy_proxies(_=None) -> list[str] | None:
        """Dummy resolve_jobspy_proxies function."""
        return None

    def get_extraction_model() -> str:
        """Dummy get_extraction_model function."""
        return "gpt-3.5-turbo"

    def get_llm_client():
        """Dummy get_llm_client function."""
        return

    def get_proxy() -> str | None:
        """Dummy get_proxy function."""
        return None

    def random_user_agent() -> str:
        """Dummy random_user_agent function."""
        return "Mozilla/5.0"

    __all__ = [
        "get_extraction_model",
        "get_llm_client",
        "get_proxy",
        "random_delay",
        "random_user_agent",
        "resolve_jobspy_proxies",
    ]
