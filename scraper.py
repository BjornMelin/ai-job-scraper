"""Web scraper for AI job postings from company websites.

This module handles the automated scraping of job postings from configured
company websites, filtering for relevant AI/ML positions, and updating
the local database with new and updated job information.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import threading
import time

from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx
import pandas as pd
import typer

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from dateutil.parser import parse as date_parse
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Settings
from database import SessionLocal
from models import CompanySQL, JobPydantic, JobSQL

settings = Settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RELEVANT_KEYWORDS = re.compile(r"(AI|Machine Learning|MLOps|AI Agent).*Engineer", re.I)

# Cache directory setup (configurable via settings)
CACHE_DIR = Path(settings.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)

# Optimized LLM schema and settings
# Company-specific extraction schemas (CSS-first approach)
COMPANY_SCHEMAS = {
    "anthropic": {
        "name": "anthropic",
        "job_selector": ".posting, .job-listing, [class*='job']",
        "fields": {
            "title": ".posting-name, .job-title, h3, h4",
            "description": ".posting-description, .job-description, p",
            "link": "a@href, .apply-link@href",
            "location": ".posting-location, .location",
            "posted_date": ".posting-date, .date",
        },
        "pagination": {
            "type": "page_param",
            "param": "page",
            "start": 1,
            "increment": 1,
        },
    },
    "openai": {
        "name": "openai",
        "job_selector": "[data-testid='job-card'], .job-card, .role-card",
        "fields": {
            "title": "[data-testid='job-title'], .job-title, h3",
            "description": "[data-testid='job-description'], .job-description",
            "link": "a@href",
            "location": "[data-testid='job-location'], .location",
            "posted_date": ".date, .posted",
        },
        "pagination": {
            "type": "page_param",
            "param": "page",
            "start": 1,
            "increment": 1,
        },
    },
    "microsoft": {
        "name": "microsoft",
        "job_selector": ".jobs-item, .ms-List-cell, [data-automation-id*='job']",
        "fields": {
            "title": ".jobs-title, h3, [data-automation-id*='title']",
            "description": ".jobs-description, .description",
            "link": "a@href",
            "location": ".jobs-location, [data-automation-id*='location']",
            "posted_date": ".jobs-date, .date",
        },
        "pagination": {
            "type": "page_param",
            "param": "pg",
            "start": 1,
            "increment": 1,
            "max_empty_pages": 2,
        },
    },
    "nvidia": {
        "name": "nvidia",
        "job_selector": "[data-automation-id='jobTitle'], .css-1q2dra3, .gwt-Label",
        "fields": {
            "title": "[data-automation-id='jobTitle'], .css-ur9034",
            "description": ".css-1t92pv, .jobdescription",
            "link": "a@href",
            "location": "[data-automation-id='locations']",
            "posted_date": ".css-129m7dg",
        },
        "pagination": {
            "type": "workday",
            "offset_param": "offset",
            "limit_param": "limit",
            "limit": 20,
        },
    },
    "meta": {
        "name": "meta",
        "job_selector": "._8ykd, .x1xzczfo, ._8isa",
        "fields": {
            "title": "._8ykg, h4, .x1f6kntn",
            "description": "._8ykf, .x193iq5w",
            "link": "a@href",
            "location": "._8ykh, .x1i10hfl",
            "posted_date": ".date, ._8yk8",
        },
        "pagination": {
            "type": "load_more_button",
            "button_selector": (
                "button[aria-label*='Load more'], .load-more, [data-testid='load-more']"
            ),
            "max_clicks": 10,
        },
    },
    "deepmind": {
        "name": "deepmind",
        "job_selector": ".glue-job, .mdc-card, .job-card",
        "fields": {
            "title": ".glue-job__title, h3, .job-title",
            "description": ".glue-job__description, .description",
            "link": "a@href",
            "location": ".glue-job__location, .location",
            "posted_date": ".date",
        },
        "pagination": {
            "type": "page_param",
            "param": "page",
            "start": 1,
            "increment": 1,
        },
    },
    "xai": {
        "name": "xai",
        "job_selector": ".job-posting, .position, .role-card",
        "fields": {
            "title": ".job-title, h3, h4",
            "description": ".job-description, .description",
            "link": "a@href",
            "location": ".location",
            "posted_date": ".date, .posted",
        },
        "pagination": {
            "type": "page_param",
            "param": "page",
            "start": 1,
            "increment": 1,
        },
    },
}

# Pagination detection patterns
PAGINATION_PATTERNS = {
    "next_button": [
        "a[aria-label*='Next'], button[aria-label*='Next']",
        "a:contains('Next'), button:contains('Next')",
        ".next, .pagination-next, [class*='next']",
        "a[rel='next'], link[rel='next']",
    ],
    "load_more_button": [
        "button[aria-label*='Load more'], button[aria-label*='Show more']",
        "button:contains('Load more'), button:contains('Show more')",
        ".load-more, .show-more, [class*='load-more']",
    ],
    "page_numbers": [
        ".pagination a, .pager a",
        "nav[aria-label*='pagination'] a",
        "[class*='pagination'] a:not(.active):not(.current)",
    ],
}

# Fallback LLM schema and instructions (used only when CSS fails)
LLM_SCHEMA = {
    "jobs": [
        {
            "title": "Job title (exact text)",
            "description": "Brief summary (50 words max)",
            "link": "Application URL",
            "location": "Location or Remote",
            "posted_date": "When posted",
        }
    ]
}

LLM_INSTRUCTIONS = """
Extract ONLY job postings from this page. 
Skip: company info, news, descriptions, alerts.
Return: title, summary, application link, location, date.
Keep descriptions under 50 words.
"""


def get_company_specific_instructions(company: str) -> str:
    """Generate company-specific extraction instructions."""
    company_lower = company.lower().strip()

    base_instructions = """
    Extract job postings as structured data. Focus on:
    1. Job title (exact text from the listing)
    2. Brief description (first 2-3 sentences or summary)
    3. Direct application link (full URL)
    4. Location (city, state/country or "Remote")
    5. Posted date (if available)
    
    Rules:
    - Only extract actual job openings, not company descriptions
    - Each job must have at least title, description, and link
    - Preserve exact job titles without modification
    - Keep descriptions concise (under 100 words)
    """

    # Company-specific additions
    company_hints = {
        "anthropic": "Look for job cards with class 'posting' or similar.",
        "openai": "Extract from elements with data-testid attributes or role cards.",
        "microsoft": "Focus on .jobs-item or data-automation-id elements.",
        "nvidia": "Look for Workday-style job listings with css classes "
        "starting with 'css-'.",
        "meta": "Extract from elements with obfuscated class names like '_8ykd'.",
        "deepmind": "Look for Google-style job cards with 'glue-' or 'mdc-' prefixes.",
    }

    hint = company_hints.get(company_lower, "")
    return (
        f"{base_instructions}\n\nCompany-specific hint: {hint}"
        if hint
        else base_instructions
    )


def validate_extraction_quality(jobs: list[dict], company: str) -> bool:
    """Validate that LLM extraction meets quality standards."""
    if not jobs or not isinstance(jobs, list):
        logger.warning(f"No jobs or invalid format for {company}")
        return False

    if len(jobs) == 0:
        logger.warning(f"Empty job list for {company}")
        return False

    valid_count = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue

        # Check required fields
        required_fields = ["title", "description", "link"]
        if not all(job.get(field, "").strip() for field in required_fields):
            logger.debug(f"Job missing required fields for {company}: {job}")
            continue

        # Basic sanity checks
        title = job.get("title", "")
        desc = job.get("description", "")
        link = job.get("link", "")

        if len(title) < 5 or len(title) > 200:
            continue
        if len(desc) < 10 or len(desc) > 500:
            continue
        if not link.startswith(("http://", "https://")):
            continue

        valid_count += 1

    # Require at least 50% valid jobs
    validity_rate = valid_count / len(jobs) if jobs else 0
    if validity_rate < 0.5:
        logger.warning(f"Low validity rate ({validity_rate:.1%}) for {company}")
        return False

    return True


# Company-specific rate limits
COMPANY_DELAYS = {
    "nvidia": 3.0,  # Slower for NVIDIA (complex site)
    "meta": 2.0,  # Slower for Meta
    "microsoft": 2.5,  # Slower for Microsoft
    "default": 1.0,  # Default delay
}


# Thread-safe session statistics
class SessionStats:
    """Thread-safe session statistics tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {
            "start_time": None,
            "companies_processed": 0,
            "jobs_found": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "errors": 0,
        }

    def increment(self, key: str, value: int = 1):
        """Thread-safe increment operation."""
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + value

    def set(self, key: str, value):
        """Thread-safe set operation."""
        with self._lock:
            self._stats[key] = value

    def get(self, key: str):
        """Thread-safe get operation."""
        with self._lock:
            return self._stats[key]

    def get_all(self):
        """Thread-safe get all stats operation."""
        with self._lock:
            return self._stats.copy()


session_stats = SessionStats()


def update_url_with_pagination(url: str, pagination_type: str, **kwargs) -> str:
    """Update URL with pagination parameters.

    Args:
        url: Base URL to update
        pagination_type: Type of pagination (page_param, offset_limit, workday)
        **kwargs: Pagination parameters (page, offset, limit, param names)

    Returns:
        Updated URL with pagination parameters
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if pagination_type == "page_param":
        param_name = kwargs.get("param", "page")
        page_value = kwargs.get("page", 1)
        params[param_name] = [str(page_value)]

    elif pagination_type in {"offset_limit", "workday"}:
        offset_param = kwargs.get("offset_param", "offset")
        limit_param = kwargs.get("limit_param", "limit")
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 20)
        params[offset_param] = [str(offset)]
        params[limit_param] = [str(limit)]

    # Rebuild URL with updated params
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


async def detect_pagination_elements(page_content: str, company: str) -> dict | None:
    """Detect pagination elements on a page.

    Args:
        page_content: HTML content of the page
        company: Company name for logging

    Returns:
        Dict with pagination info if detected, None otherwise
    """
    try:
        # Try to detect pagination patterns using CSS selectors
        async with AsyncWebCrawler() as crawler:
            # Check for next button
            for pattern in PAGINATION_PATTERNS["next_button"]:
                next_button_check = {
                    "next_button": {
                        "selector": pattern,
                        "fields": {"href": "@href", "text": "text"},
                    }
                }
                strategy = JsonCssExtractionStrategy(next_button_check)
                result = await crawler.arun(
                    html=page_content, extraction_strategy=strategy, bypass_cache=True
                )
                if result.success and result.extracted_content:
                    data = json.loads(result.extracted_content)
                    if data.get("next_button"):
                        logger.info(f"Found next button pagination for {company}")
                        return {"type": "next_button", "selector": pattern}

            # Check for load more button
            for pattern in PAGINATION_PATTERNS["load_more_button"]:
                load_more_check = {
                    "button": {
                        "selector": pattern,
                        "fields": {"text": "text", "onclick": "@onclick"},
                    }
                }
                strategy = JsonCssExtractionStrategy(load_more_check)
                result = await crawler.arun(
                    html=page_content, extraction_strategy=strategy, bypass_cache=True
                )
                if result.success and result.extracted_content:
                    data = json.loads(result.extracted_content)
                    if data.get("button"):
                        logger.info(f"Found load more button for {company}")
                        return {"type": "load_more", "selector": pattern}

            # Check for page numbers
            for pattern in PAGINATION_PATTERNS["page_numbers"]:
                page_check = {
                    "pages": {
                        "selector": pattern,
                        "fields": {"href": "@href", "page": "text"},
                    }
                }
                strategy = JsonCssExtractionStrategy(page_check)
                result = await crawler.arun(
                    html=page_content, extraction_strategy=strategy, bypass_cache=True
                )
                if result.success and result.extracted_content:
                    data = json.loads(result.extracted_content)
                    if data.get("pages") and len(data["pages"]) > 0:
                        logger.info(f"Found page number pagination for {company}")
                        return {"type": "page_numbers", "selector": pattern}

    except Exception as e:
        logger.debug(f"Pagination detection failed for {company}: {e}")

    return None


def get_cached_schema(company: str, ttl_hours: int = 168) -> dict | None:
    """Get cached extraction schema for company with TTL validation.

    Args:
        company: Company name for the cache file
        ttl_hours: Time-to-live in hours (default: 168 = 1 week)

    Returns:
        Cached schema dict if valid and within TTL, None otherwise
    """
    cache_file = CACHE_DIR / f"{company.lower()}.json"
    if not cache_file.exists():
        return None

    try:
        # Check file age
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if file_age_hours > ttl_hours:
            logger.info(f"Cache expired for {company} (age: {file_age_hours:.1f}h)")
            cache_file.unlink()  # Remove expired cache
            return None

        cached_data = json.loads(cache_file.read_text())

        # Check for schema version (future-proofing)
        if "schema_version" not in cached_data:
            logger.info(f"Cache lacks version for {company}, invalidating")
            cache_file.unlink()
            return None

        return cached_data.get("schema")
    except Exception as e:
        logger.warning(f"Cache read failed for {company}: {e}")
        with contextlib.suppress(Exception):
            cache_file.unlink()  # Remove corrupted cache
        return None


def save_schema_cache(company: str, schema: dict) -> None:
    """Save successful extraction schema with metadata."""
    cache_file = CACHE_DIR / f"{company.lower()}.json"
    cache_data = {
        "schema": schema,
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "company": company.lower(),
    }
    cache_file.write_text(json.dumps(cache_data, indent=2))


def normalize_job_data(job: dict, company: str) -> dict:  # noqa: ARG001
    """Normalize and clean job data to handle format variations.

    Args:
        job: Raw job dictionary
        company: Company name

    Returns:
        Normalized job dictionary
    """
    # Handle various title field names
    title = (
        job.get("title")
        or job.get("jobTitle")
        or job.get("position")
        or job.get("role")
        or job.get("job_title")
        or ""
    ).strip()

    # Handle various description field names
    description = (
        job.get("description")
        or job.get("jobDescription")
        or job.get("summary")
        or job.get("job_description")
        or job.get("details")
        or ""
    ).strip()

    # Handle various link field names
    link = (
        job.get("link")
        or job.get("url")
        or job.get("applyUrl")
        or job.get("apply_url")
        or job.get("job_url")
        or ""
    ).strip()

    # Handle various location field names
    location = (
        job.get("location")
        or job.get("jobLocation")
        or job.get("office")
        or job.get("workplace")
        or job.get("job_location")
        or "Unknown"
    ).strip()

    # Handle various date field names
    posted_date = (
        job.get("posted_date")
        or job.get("postedDate")
        or job.get("datePosted")
        or job.get("date")
        or job.get("publishedAt")
        or None
    )

    return {
        "title": title,
        "description": description,
        "link": link,
        "location": location,
        "posted_date": posted_date,
    }


def is_valid_job(job: dict, company: str) -> bool:
    """Job validation for a specific company with enhanced format handling.

    Args:
        job: Job dictionary to validate
        company: Company name (reserved for future company-specific validation)

    Returns:
        bool: True if job passes all validation checks, False otherwise
    """
    # Normalize job data first
    normalized = normalize_job_data(job, company)

    required = ["title", "description", "link"]

    # Check required fields exist and have content
    if not all(normalized.get(field, "").strip() for field in required):
        return False

    # Check reasonable lengths
    title = normalized["title"].strip()
    desc = normalized["description"].strip()
    link = normalized["link"].strip()

    if len(title) < 3 or len(title) > 500:  # Increased max length for flexibility
        return False

    if (
        len(desc) < 10 or len(desc) > 5000
    ):  # Increased max length for detailed descriptions
        return False

    # Check if link is valid and return True or False
    if not link.startswith(("http://", "https://")):
        # Try to fix relative URLs
        if link.startswith("/"):
            # This is a relative URL, we'll need the base URL to fix it
            return False  # For now, reject relative URLs
        return False

    # Update the original job dict with normalized data
    job.update(normalized)
    return True


def log_session_summary():
    """Print simple session summary."""
    stats = session_stats.get_all()
    duration = time.time() - stats["start_time"]
    companies_processed = stats["companies_processed"]

    if companies_processed == 0:
        cache_rate_str = "N/A (no companies processed)"
    else:
        cache_rate = stats["cache_hits"] / companies_processed
        cache_rate_str = f"{cache_rate:.1%}"

    logger.info("📊 Session Summary:")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info(f"  Companies: {companies_processed}")
    logger.info(f"  Jobs found: {stats['jobs_found']}")
    logger.info(f"  Cache hit rate: {cache_rate_str}")
    logger.info(f"  LLM calls: {stats['llm_calls']}")
    logger.info(f"  Errors: {stats['errors']}")


async def extract_jobs_safe(url: str, company: str) -> list[dict]:
    """Safe wrapper with retries."""
    max_retries = 2

    for attempt in range(max_retries):
        try:
            return await extract_jobs(url, company)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {company}: {e}")
            session_stats.increment("errors")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(f"❌ All attempts failed for {company}")
                return []


async def try_css_extraction(url: str, company: str, schema: dict) -> list[dict]:
    """Try CSS-based extraction with given schema."""
    try:
        async with AsyncWebCrawler() as crawler:
            strategy = JsonCssExtractionStrategy(schema)
            result = await crawler.arun(url=url, extraction_strategy=strategy)

            if result.success and result.extracted_content:
                jobs_data = json.loads(result.extracted_content)
                jobs = (
                    jobs_data.get("jobs", [])
                    if isinstance(jobs_data, dict)
                    else jobs_data
                )
                if jobs and len(jobs) > 0:
                    logger.info(
                        f"✅ CSS extraction worked for {company} - "
                        f"found {len(jobs)} jobs"
                    )
                    return jobs
    except Exception as e:
        logger.debug(f"CSS extraction failed for {company}: {e}")

    return []


async def try_llm_extraction(url: str, company: str) -> list[dict]:
    """LLM-based extraction with robust error handling and retries."""
    if (
        not settings.openai_api_key
        or settings.openai_api_key == "your_openai_api_key_here"
    ):
        logger.warning(f"No valid OpenAI API key for LLM extraction: {company}")
        return []

    max_retries = 3
    backoff_base = 2

    for attempt in range(max_retries):
        try:
            async with AsyncWebCrawler() as crawler:
                logger.info(
                    f"🤖 Using LLM extraction for {company} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                session_stats.increment("llm_calls")

                strategy = LLMExtractionStrategy(
                    llm_config=LLMConfig(
                        provider="openai/gpt-4o-mini",
                        api_token=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                    ),
                    extraction_schema=LLM_SCHEMA,
                    instructions=get_company_specific_instructions(company),
                    extraction_type="schema",
                    apply_chunking=True,
                    chunk_token_threshold=2000,  # Increased from 1000
                    overlap_rate=0.15,  # Increased from 0.02 for better context
                    input_format="fit_markdown",  # Better for structured extraction
                    extra_args={
                        "temperature": 0.1,  # Lower for consistency
                        "max_tokens": 1500,
                        "response_format": {"type": "json_object"},  # Force JSON
                    },
                )

                config = CrawlerRunConfig(
                    extraction_strategy=strategy,
                    page_timeout=30000,
                    # Wait for job elements
                    wait_for="css:.job-listing, css:[class*='job']",
                    # Scroll to load all jobs
                    js_code="window.scrollTo(0, document.body.scrollHeight);",
                )

                result = await crawler.arun(url=url, config=config)

                if not result.success:
                    raise Exception(f"Crawl failed: {result.error_message}")

                if not result.extracted_content:
                    raise Exception("No content extracted")

                # Parse and validate extracted content
                try:
                    extracted = json.loads(result.extracted_content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {company}: {e}")
                    logger.debug(f"Raw content: {result.extracted_content[:500]}")
                    raise

                jobs = extracted.get("jobs", []) if isinstance(extracted, dict) else []

                # Validate extraction quality
                if not validate_extraction_quality(jobs, company):
                    raise Exception(
                        "Poor quality extraction - retrying with different strategy"
                    )

                logger.info(
                    f"✅ LLM extraction successful for {company} - "
                    f"found {len(jobs)} jobs"
                )
                return jobs

        except Exception as e:
            error_msg = str(e)
            logger.warning(
                f"LLM extraction attempt {attempt + 1} failed for {company}: "
                f"{error_msg}"
            )

            # Check for specific errors that shouldn't be retried
            if "rate limit" in error_msg.lower():
                wait_time = backoff_base**attempt * 5  # Longer wait for rate limits
                logger.info(f"Rate limit hit, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            elif "token limit" in error_msg.lower():
                # Try with smaller chunks on next attempt
                logger.info("Token limit exceeded, will retry with smaller chunks")
                continue
            elif attempt < max_retries - 1:
                wait_time = backoff_base**attempt
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All LLM extraction attempts failed for {company}")
                return []

    return []


async def try_basic_fallback(url: str, company: str) -> list[dict]:
    """Basic HTTP fallback when all else fails."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            if response.status_code == 200:
                logger.info(f"🔧 Using basic HTTP fallback for {company}")
                return [
                    {
                        "title": f"AI Engineer opportunities at {company}",
                        "description": (
                            f"Visit {company} careers page for current AI and "
                            f"Machine Learning opportunities"
                        ),
                        "link": url,
                        "location": "Various",
                        "posted_date": None,
                    }
                ]
    except Exception as e:
        logger.warning(f"Basic HTTP fallback failed for {company}: {e}")

    return []


async def extract_jobs_from_page(
    url: str, company: str, schema: dict | None = None
) -> list[dict]:
    """Extract jobs from a single page using the provided schema."""
    jobs = []

    # Try CSS extraction with provided schema
    if schema:
        jobs = await try_css_extraction(url, company, schema)

    # If no jobs found and no schema provided, try generic patterns
    if not jobs and not schema:
        generic_selectors = [
            # Common job board selectors
            ".job-listing, .job-item, .position, [class*='job'], [class*='position']",
            ".career-item, .opening, .role, .vacancy",
            "article, .card, .entry, .posting",
            # Table-based layouts
            "tr[class*='job'], tbody tr, .job-row",
            # List-based layouts
            "li[class*='job'], ul.jobs li, .job-list-item",
            # Div-based layouts with data attributes
            "[data-job], [data-position], [data-role]",
            # Section-based layouts
            "section.job, .job-section, .career-section",
        ]

        for selector in generic_selectors:
            test_schema = {
                "jobs": {
                    "selector": selector,
                    "fields": {
                        "title": (
                            ".title, .job-title, h3, h4, h2, .position-title, "
                            ".role-title, [class*='title'], [data-testid*='title']"
                        ),
                        "description": (
                            ".description, .summary, .job-summary, .details, "
                            ".job-details, p, .content, [class*='description']"
                        ),
                        "link": (
                            "a@href, .apply-link@href, .job-link@href, "
                            ".view-job@href, [class*='link']@href, [data-link]@href"
                        ),
                        "location": (
                            ".location, .job-location, .office, .workplace, "
                            ".city, [class*='location'], [data-location]"
                        ),
                        "posted_date": (
                            ".date, .posted, .job-date, .posted-date, time, "
                            ".timestamp, [class*='date'], [datetime]"
                        ),
                    },
                }
            }
            jobs = await try_css_extraction(url, company, test_schema)
            if jobs:
                logger.info(f"✅ Generic CSS worked for {company} with: {selector}")
                # Cache successful schema for future use
                save_schema_cache(company, test_schema)
                break

    # Try LLM extraction as fallback
    if not jobs:
        jobs = await try_llm_extraction(url, company)

    return jobs


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_jobs(url: str, company: str) -> list[dict]:
    """Extract jobs with CSS-first hybrid approach and pagination support."""
    # Apply company-specific rate limiting
    normalized_company = (
        company.lower().strip().replace(" ", "").replace("_", "").replace("-", "")
    )
    delay = COMPANY_DELAYS.get(normalized_company, COMPANY_DELAYS["default"])
    await asyncio.sleep(delay)

    all_jobs = []
    seen_job_links = set()  # Track unique jobs across pages

    # Get company-specific configuration
    company_config = COMPANY_SCHEMAS.get(normalized_company, {})
    pagination_config = company_config.get("pagination", {})

    # Prepare extraction schema
    schema = None
    if company_config:
        schema = {
            "jobs": {
                "selector": company_config["job_selector"],
                "fields": company_config["fields"],
            }
        }
    else:
        # Try cached schema
        cached_schema = get_cached_schema(company)
        if cached_schema:
            schema = cached_schema
            logger.info(f"✅ Using cached schema for {company}")
            session_stats.increment("cache_hits")

    # Handle different pagination types
    pagination_type = pagination_config.get("type", "none")

    if pagination_type == "page_param":
        # Handle page parameter pagination
        param = pagination_config.get("param", "page")
        start_page = pagination_config.get("start", 1)
        increment = pagination_config.get("increment", 1)
        max_empty_pages = pagination_config.get("max_empty_pages", 2)
        empty_page_count = 0
        current_page = start_page

        while empty_page_count < max_empty_pages:
            page_url = update_url_with_pagination(
                url, "page_param", param=param, page=current_page
            )
            logger.info(f"🔍 Scraping {company} page {current_page}: {page_url}")

            page_jobs = await extract_jobs_from_page(page_url, company, schema)

            # Filter out duplicates
            new_jobs = []
            for job in page_jobs:
                if job.get("link") and job["link"] not in seen_job_links:
                    seen_job_links.add(job["link"])
                    new_jobs.append(job)

            if new_jobs:
                all_jobs.extend(new_jobs)
                empty_page_count = 0
            else:
                empty_page_count += 1

            current_page += increment
            await asyncio.sleep(delay)  # Rate limiting between pages

    elif pagination_type in ["offset_limit", "workday"]:
        # Handle offset/limit pagination
        offset_param = pagination_config.get("offset_param", "offset")
        limit_param = pagination_config.get("limit_param", "limit")
        limit = pagination_config.get("limit", 20)
        max_results = pagination_config.get("max_results", 200)
        current_offset = 0

        while current_offset < max_results:
            page_url = update_url_with_pagination(
                url,
                pagination_type,
                offset_param=offset_param,
                limit_param=limit_param,
                offset=current_offset,
                limit=limit,
            )
            logger.info(f"🔍 Scraping {company} offset {current_offset}: {page_url}")

            page_jobs = await extract_jobs_from_page(page_url, company, schema)

            # Filter out duplicates
            new_jobs = []
            for job in page_jobs:
                if job.get("link") and job["link"] not in seen_job_links:
                    seen_job_links.add(job["link"])
                    new_jobs.append(job)

            if not new_jobs:
                break  # No more results

            all_jobs.extend(new_jobs)
            current_offset += limit
            await asyncio.sleep(delay)  # Rate limiting between pages

    elif pagination_type == "load_more_button":
        # For load more buttons, we'd need browser automation
        # For now, just get the first page
        logger.info(
            f"⚠️ Load more pagination detected for {company}, getting first page only"
        )
        all_jobs = await extract_jobs_from_page(url, company, schema)

    else:
        # No pagination or unknown type - just scrape single page
        all_jobs = await extract_jobs_from_page(url, company, schema)

    # If still no jobs, try basic fallback
    if not all_jobs:
        all_jobs = await try_basic_fallback(url, company)

    # Process and validate jobs
    for job in all_jobs:
        try:
            job["posted_date"] = (
                date_parse(job.get("posted_date", ""), fuzzy=True)
                if job.get("posted_date")
                else None
            )
        except Exception:
            job["posted_date"] = None
        job["location"] = job.get("location", "Unknown")

    # Validate and clean jobs
    valid_jobs = []
    for job_item in all_jobs:
        try:
            if is_valid_job(job_item, company):
                valid_jobs.append(job_item)
        except Exception as ve:
            logger.debug(f"Job validation error for {company}: {ve} - job: {job_item}")
            continue

    logger.info(f"📊 {company}: {len(valid_jobs)}/{len(all_jobs)} valid jobs")
    return [{"company": company, **job} for job in valid_jobs]


def is_relevant(job: dict) -> bool:
    """Check if a job posting is relevant to AI/ML engineering roles.

    Uses regex pattern matching against job titles to identify
    relevant positions like AI Engineer, Machine Learning Engineer, etc.

    Args:
        job (dict): Job dictionary containing at least a 'title' field.

    Returns:
        bool: True if job title matches relevant keywords, False otherwise.

    """
    title = job.get("title", "")
    if not title:
        return False
    return bool(RELEVANT_KEYWORDS.search(title))


async def validate_link(link: str) -> str | None:
    """Validate that a job posting URL is accessible.

    Makes an HTTP HEAD request to verify the link returns a 200 status code.
    Used to filter out broken or inaccessible job posting links.

    Args:
        link (str): Job posting URL to validate.

    Returns:
        str | None: Original link if valid and accessible, None otherwise.

    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.head(link, follow_redirects=True)
            if resp.status_code == 200:
                return link
    except Exception as e:
        logger.error(f"Validation failed {link}: {e}")
    return None


def update_db(jobs_df: pd.DataFrame) -> None:
    """Update database with scraped job data from Pandas DataFrame.

    Performs full CRUD operations: validates jobs with Pydantic, adds new jobs,
    updates existing jobs when content changes (via hash comparison), and removes
    jobs that are no longer found. Preserves user edits (favorite, status, notes)
    when updating existing jobs.

    Args:
        jobs_df (pd.DataFrame): DataFrame containing scraped job data with columns:
            company, title, description, link, location, posted_date.

    Note:
        Uses database transactions with rollback on error to maintain consistency.
        Invalid jobs are logged and skipped rather than failing the entire operation.

    """
    session = SessionLocal()
    try:
        existing = {j.link: j for j in session.query(JobSQL).all()}
        validated_jobs = []
        for _, row in jobs_df.iterrows():
            job_dict = row.to_dict()
            try:
                JobPydantic(**job_dict)
            except Exception as ve:
                logger.warning(
                    f"Validation failed for job {job_dict.get('title')}: {ve}"
                )
                continue
            if "link" not in job_dict:
                logger.warning(
                    f"Job missing link field: {job_dict.get('title', 'Unknown')}"
                )
                continue
            valid_link = asyncio.run(validate_link(job_dict["link"]))
            if not valid_link:
                continue
            job_dict["link"] = valid_link
            job_hash = hashlib.sha256(job_dict["description"].encode()).hexdigest()
            if job_dict["link"] in existing:
                ex = existing[job_dict["link"]]
                if ex.hash != job_hash:
                    ex.title = job_dict["title"]
                    ex.description = job_dict["description"]
                    ex.location = job_dict["location"]
                    ex.posted_date = job_dict["posted_date"]
                    ex.hash = job_hash
                    ex.last_seen = datetime.now()
            else:
                new_job = JobSQL(
                    **job_dict,
                    hash=job_hash,
                    last_seen=datetime.now(),
                    favorite=False,
                    status="New",
                    notes="",
                )
                validated_jobs.append(new_job)
        session.add_all(validated_jobs)
        current_links = set(jobs_df["link"])
        for link in list(existing.keys()):
            if link not in current_links:
                session.delete(existing[link])
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"DB update failed: {e}")
    finally:
        session.close()


async def scrape_all() -> pd.DataFrame:
    """Scrape job postings from all active company websites.

    Retrieves active companies from database, scrapes their careers pages in
    parallel using asyncio, filters for relevant AI/ML positions, and returns
    consolidated results as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing all relevant scraped jobs with columns:
            company, title, description, link, location, posted_date. Empty DataFrame
            if no jobs found or all scraping attempts failed.

    Note:
        Individual company scraping failures are logged but don't stop the overall
        process. Only jobs with valid titles are included.

    """
    session = SessionLocal()
    active_companies = session.query(CompanySQL).filter_by(active=True).all()
    session.close()

    session_stats.set("companies_processed", len(active_companies))

    tasks = [extract_jobs_safe(c.url, c.name) for c in active_companies]
    all_jobs = []
    for task in asyncio.as_completed(tasks):
        try:
            jobs = await task
            relevant = [j for j in jobs if is_relevant(j)]
            all_jobs.extend(relevant)
        except Exception as e:
            logger.error(f"Scrape failed: {e}")
            session_stats.increment("errors")
    df = pd.DataFrame(all_jobs)
    if df.empty or "title" not in df.columns:
        return pd.DataFrame()
    return df[df["title"].notna()]


def main() -> None:
    """Command-line interface entry point for the job scraper.

    Orchestrates the complete scraping workflow: scrapes all active companies,
    updates the database, and logs the results. Handles top-level exceptions and
    provides user feedback via logging.

    Note:
        Designed to be run via CLI: `python scraper.py` or `uv run python scraper.py`
    """
    session_stats.set("start_time", time.time())

    try:
        jobs_df = asyncio.run(scrape_all())
        update_db(jobs_df)
        session_stats.set("jobs_found", len(jobs_df))
        logger.info(f"Scraped {len(jobs_df)} jobs.")
        log_session_summary()
    except Exception as e:
        logger.error(f"Main failed: {e}")
        session_stats.increment("errors")
        log_session_summary()


if __name__ == "__main__":
    typer.run(main)
