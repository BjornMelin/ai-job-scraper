"""Utilities package for AI Job Scraper.

This package contains utility modules including:
- startup_helpers: Application startup optimizations
"""

# Re-export functions from the sibling utils.py module
import sys

from pathlib import Path

# Add the src directory to sys.path to enable proper imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    # Import core_utils.py as a module from src directory
    import core_utils as utils_module

    # Re-export available functions
    get_proxy = utils_module.get_proxy
    random_delay = utils_module.random_delay
    random_user_agent = utils_module.random_user_agent
    resolve_jobspy_proxies = utils_module.resolve_jobspy_proxies
    settings = utils_module.settings

    # Export modules for testing
    random = utils_module.random
    time = utils_module.time

    __all__ = [
        "get_proxy",
        "random",
        "random_delay",
        "random_user_agent",
        "resolve_jobspy_proxies",
        "settings",
        "time",
    ]

except ImportError as e:
    # Fallback implementations if core_utils.py can't be loaded
    print(f"Warning: Could not import core_utils.py: {e}")

    def get_proxy() -> str | None:
        """Get a proxy URL.

        Returns:
            str | None: Fallback implementation returns None.
        """
        return None

    def random_delay(min_sec: float = 1.0, max_sec: float = 5.0) -> None:
        """Simulate a random delay.

        Args:
            min_sec: Minimum delay in seconds.
            max_sec: Maximum delay in seconds.
        """

    def random_user_agent() -> str:
        """Generate a random user agent.

        Returns:
            str: A default user agent string.
        """
        return "Mozilla/5.0"

    def resolve_jobspy_proxies(_=None) -> list[str] | None:
        """Resolve proxies for JobSpy.

        Args:
            _: Optional argument (unused in fallback implementation).

        Returns:
            list[str] | None: Fallback implementation returns None.
        """
        return None

    from config import Settings

    settings = Settings()

    import random
    import time

    __all__ = [
        "get_proxy",
        "random",
        "random_delay",
        "random_user_agent",
        "resolve_jobspy_proxies",
        "settings",
        "time",
    ]
