"""Utility module for the AI Job Scraper application.

This module provides helper functions for web scraping evasion, random user
agent generation, delays, and proxy management. The AI client functionality
has been moved to src.ai_client for better separation of concerns.

Functions:
    get_proxy: Returns a random proxy if enabled.
    random_user_agent: Generates a random browser user agent string.
    random_delay: Pauses execution for a random duration.
    resolve_jobspy_proxies: Resolves proxy configuration for JobSpy.
    ensure_timezone_aware: Ensures datetime objects are timezone-aware.
"""

from __future__ import annotations

import random
import secrets
import sys
import time

from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings

# Add src directory to path if not already there
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

settings = Settings()


def get_proxy() -> str | None:
    """Get a random proxy URL from the configured pool if proxies are enabled.

    This function checks if proxy usage is enabled and if there are proxies
    available in the pool. If so, it selects and returns a random proxy URL;
    otherwise, it returns None.

    Returns:
        str | None: A proxy URL if available and enabled, otherwise None.
    """
    if not settings.use_proxies or not settings.proxy_pool:
        return None
    return secrets.choice(settings.proxy_pool)


def random_user_agent() -> str:
    """Generate a random user agent string to mimic browser headers.

    This function maintains a list of common user agent strings from various
    browsers and devices, and randomly selects one to help in evading
    detection during web scraping by simulating different user environments.

    Returns:
        str: A randomly selected user agent string.
    """
    user_agents = [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.114 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) "
            "Gecko/20100101 Firefox/89.0"
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) "
            "Gecko/20100101 Firefox/89.0"
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.864.48 Safari/537.36 "
            "Edg/91.0.864.48"
        ),
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 "
            "Mobile/15E148 Safari/604.1"
        ),
        (
            "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36"
        ),
    ]
    return secrets.choice(user_agents)


def random_delay(min_sec: float = 1.0, max_sec: float = 5.0) -> None:
    """Introduce a random delay to simulate human-like interaction timing.

    This function generates a random float value between the specified
    minimum and maximum seconds and pauses execution for that duration,
    which helps in avoiding rate limits and detection during automated
    scraping by mimicking natural user behavior.

    Args:
        min_sec: The minimum delay duration in seconds (default is 1.0).
        max_sec: The maximum delay duration in seconds (default is 5.0).
    """
    # Use random.uniform for performance, secrets.randbelow for non-cryptographic use
    time.sleep(random.uniform(min_sec, max_sec))


def resolve_jobspy_proxies(settings_obj=None) -> list[str] | None:
    """Resolve proxies configuration for JobSpy based on settings.

    Args:
        settings_obj: Settings instance to use. If None, uses global settings.

    Returns:
        Proxy pool list when enabled, empty list when enabled but pool empty,
        or None when proxies are disabled.
    """
    if settings_obj is None:
        settings_obj = settings

    if not settings_obj.use_proxies:
        return None
    return list(settings_obj.proxy_pool) if settings_obj.proxy_pool else []


def ensure_timezone_aware(v) -> datetime | None:
    """Ensure datetime is timezone-aware (UTC) - shared utility function.

    This function takes a datetime value which can be a string, datetime object,
    or None, and ensures it's timezone-aware with UTC timezone. This is used
    across models and schemas to maintain consistent timezone handling.

    Args:
        v: Input value that can be None, str, or datetime object.

    Returns:
        datetime | None: Timezone-aware datetime in UTC, or None if input is None or
            invalid.
    """
    if v is None:
        return None
    if isinstance(v, str):
        try:
            parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            try:
                parsed = datetime.strptime(v, "%Y-%m-%d")  # noqa: DTZ007
                return parsed.replace(tzinfo=UTC)
            except ValueError:
                return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    return None
