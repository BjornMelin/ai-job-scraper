"""Utility module for the AI Job Scraper application.

This module provides helper functions to manage LLM clients (OpenAI or Groq),
select extraction models, handle proxy rotation for scraping evasion, generate
random user agents to mimic browser behavior, and introduce random delays to
simulate human-like interactions. These utilities support hybrid LLM usage for
optimization and enhance scraping reliability by avoiding detection.

Functions:
    get_llm_client: Returns the appropriate LLM client based on settings.
    get_extraction_model: Returns the model name for extraction tasks.
    get_proxy: Returns a random proxy if enabled.
    random_user_agent: Generates a random browser user agent string.
    random_delay: Pauses execution for a random duration.
"""

import random
import time

from groq import Groq
from openai import OpenAI

from config import Settings

settings = Settings()


def get_llm_client() -> OpenAI | Groq:
    """Get the LLM client based on the application configuration.

    This function checks the settings to determine whether to use the Groq
    or OpenAI provider and returns the corresponding client instance
    initialized with the appropriate API key.

    Returns:
        Union[OpenAI, Groq]: The API client instance for the selected LLM provider.
    """
    if settings.use_groq:
        return Groq(api_key=settings.groq_api_key)
    return OpenAI(api_key=settings.openai_api_key)


def get_extraction_model() -> str:
    """Get the model name for extraction tasks based on the provider.

    This function selects the appropriate model name depending on whether
    Groq or OpenAI is being used, ensuring compatibility with the chosen
    LLM provider for tasks like data extraction.

    Returns:
        str: The name of the model suitable for extraction tasks.
    """
    if settings.use_groq:
        return "llama-3.3-70b-versatile"
    return settings.extraction_model


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
    return random.choice(settings.proxy_pool)


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
    return random.choice(user_agents)


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
    time.sleep(random.uniform(min_sec, max_sec))
