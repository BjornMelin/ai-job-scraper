"""VCR configuration for HTTP request mocking in tests.

This module provides VCR (Video Cassette Recorder) configurations for recording
and replaying HTTP requests in tests, enabling deterministic testing of external
API interactions without making actual network calls.
"""

import os

from pathlib import Path

import vcr

from vcr import VCR
from vcr.serializers import jsonserializer

# Base configuration for test VCR instances
BASE_VCR_CONFIG = {
    "record_mode": "once",  # Record once, then replay
    "match_on": ["uri", "method", "body"],
    "decode_compressed_response": True,
    "filter_headers": [
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
        "user-agent",
    ],
    "filter_query_parameters": [
        "api_key",
        "apikey",
        "key",
        "token",
    ],
}


def get_cassettes_dir() -> Path:
    """Get the directory where VCR cassettes are stored.

    Returns:
        Path to the cassettes directory
    """
    tests_dir = Path(__file__).parent.parent
    cassettes_dir = tests_dir / "cassettes"
    cassettes_dir.mkdir(exist_ok=True)
    return cassettes_dir


def create_vcr(
    cassette_name: str | None = None,
    record_mode: str = "once",
    match_on: list[str] | None = None,
    **kwargs,
) -> VCR:
    """Create a configured VCR instance for testing.

    Args:
        cassette_name: Name of the cassette file (without .yaml extension)
        record_mode: VCR record mode ("once", "new_episodes", "none", "all")
        match_on: List of request attributes to match on
        **kwargs: Additional VCR configuration options

    Returns:
        Configured VCR instance
    """
    config = BASE_VCR_CONFIG.copy()
    config.update(kwargs)

    if record_mode:
        config["record_mode"] = record_mode
    if match_on:
        config["match_on"] = match_on

    # Set cassette library directory
    config["cassette_library_dir"] = str(get_cassettes_dir())

    return vcr.VCR(**config)


def create_api_vcr(service_name: str = "general") -> VCR:
    """Create a VCR instance configured for API testing.

    Args:
        service_name: Name of the service being tested (for organization)

    Returns:
        VCR instance configured for API testing
    """
    cassettes_dir = get_cassettes_dir() / service_name
    cassettes_dir.mkdir(exist_ok=True)

    return vcr.VCR(
        cassette_library_dir=str(cassettes_dir),
        record_mode="once",
        match_on=["uri", "method", "body"],
        decode_compressed_response=True,
        serializer="json",
        filter_headers=[
            "authorization",
            "x-api-key",
            "openai-api-key",
            "groq-api-key",
            "cookie",
            "set-cookie",
            "user-agent",
        ],
        filter_query_parameters=[
            "api_key",
            "apikey",
            "key",
            "token",
        ],
        # Custom matchers for better API request matching
        custom_patches=[
            (
                jsonserializer,
                "dumps",
                lambda data: jsonserializer.dumps(data, sort_keys=True),
            )
        ],
    )


def create_scraping_vcr() -> VCR:
    """Create a VCR instance configured for web scraping tests.

    Returns:
        VCR instance configured for web scraping
    """
    cassettes_dir = get_cassettes_dir() / "scraping"
    cassettes_dir.mkdir(exist_ok=True)

    return vcr.VCR(
        cassette_library_dir=str(cassettes_dir),
        record_mode="once",
        match_on=["uri", "method"],
        decode_compressed_response=True,
        filter_headers=[
            "authorization",
            "cookie",
            "set-cookie",
            "user-agent",  # Filter dynamic user agents
        ],
        # Don't filter proxy headers for scraping tests
        ignore_localhost=True,
        ignore_hosts=["127.0.0.1", "localhost"],
    )


def create_llm_vcr() -> VCR:
    """Create a VCR instance specifically configured for LLM API testing.

    Returns:
        VCR instance configured for LLM APIs (OpenAI, Groq, etc.)
    """
    cassettes_dir = get_cassettes_dir() / "llm"
    cassettes_dir.mkdir(exist_ok=True)

    return vcr.VCR(
        cassette_library_dir=str(cassettes_dir),
        record_mode="once",
        match_on=["uri", "method", "body"],
        decode_compressed_response=True,
        serializer="json",
        filter_headers=[
            "authorization",
            "x-api-key",
            "openai-api-key",
            "groq-api-key",
            "x-groq-api-key",
            "cookie",
            "set-cookie",
            "user-agent",
        ],
        filter_post_data_parameters=[
            "api_key",
            "apikey",
            "key",
        ],
        # Custom request/response filtering for LLM APIs
        before_record_request=lambda request: sanitize_llm_request(request),
        before_record_response=lambda response: sanitize_llm_response(response),
    )


def sanitize_llm_request(request):
    """Sanitize LLM API requests before recording.

    Args:
        request: VCR request object

    Returns:
        Sanitized request object
    """
    # Remove sensitive data from request body
    if hasattr(request, "body") and request.body:
        try:
            import json

            body_data = json.loads(request.body.decode("utf-8"))

            # Replace API keys in body
            if "api_key" in body_data:
                body_data["api_key"] = "REDACTED"
            if "key" in body_data:
                body_data["key"] = "REDACTED"

            request.body = json.dumps(body_data).encode("utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError):  # noqa: S110
            pass  # Keep original body if parsing fails

    return request


def sanitize_llm_response(response):
    """Sanitize LLM API responses before recording.

    Args:
        response: VCR response object

    Returns:
        Sanitized response object
    """
    # Keep response as-is but could add filtering here if needed
    return response


# Convenience VCR instances for common use cases
api_vcr = create_api_vcr()
scraping_vcr = create_scraping_vcr()
llm_vcr = create_llm_vcr()


def clean_cassettes(service_name: str | None = None, older_than_days: int = 30) -> int:
    """Clean old cassette files.

    Args:
        service_name: Specific service to clean (if None, clean all)
        older_than_days: Remove cassettes older than this many days

    Returns:
        Number of cassettes removed
    """
    import time

    cassettes_dir = get_cassettes_dir()
    if service_name:
        cassettes_dir = cassettes_dir / service_name

    if not cassettes_dir.exists():
        return 0

    cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
    removed_count = 0

    for cassette_file in cassettes_dir.rglob("*.yaml"):
        if cassette_file.stat().st_mtime < cutoff_time:
            cassette_file.unlink()
            removed_count += 1

    return removed_count


def reset_cassettes(service_name: str | None = None) -> int:
    """Remove all cassettes to force re-recording.

    Args:
        service_name: Specific service to reset (if None, reset all)

    Returns:
        Number of cassettes removed
    """
    cassettes_dir = get_cassettes_dir()
    if service_name:
        cassettes_dir = cassettes_dir / service_name

    if not cassettes_dir.exists():
        return 0

    removed_count = 0
    for cassette_file in cassettes_dir.rglob("*.yaml"):
        cassette_file.unlink()
        removed_count += 1

    return removed_count


# Environment-based configuration
def is_recording_enabled() -> bool:
    """Check if VCR recording is enabled via environment variable.

    Returns:
        True if recording should happen, False if playback only
    """
    return os.getenv("VCR_RECORD", "false").lower() in ("true", "1", "yes")


def get_record_mode() -> str:
    """Get VCR record mode from environment.

    Returns:
        VCR record mode string
    """
    if is_recording_enabled():
        return os.getenv("VCR_RECORD_MODE", "once")
    return "none"  # Playback only
