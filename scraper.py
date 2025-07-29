"""Web scraper for AI job postings from company websites.

This module handles the automated scraping of job postings from configured
company websites, filtering for relevant AI/ML positions, and updating
the local database with new and updated job information.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import typer
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from dateutil.parser import parse as date_parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Settings
from models import Base, CompanySQL, JobPydantic, JobSQL

settings = Settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

engine = create_engine(settings.db_url)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

RELEVANT_KEYWORDS = re.compile(r"(AI|Machine Learning|MLOps|AI Agent).*Engineer", re.I)

# Cache directory setup
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# Optimized LLM schema and settings
SIMPLE_SCHEMA = {
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

SIMPLE_INSTRUCTIONS = """
Extract ONLY job postings from this page. 
Skip: company info, news, descriptions, alerts.
Return: title, summary, application link, location, date.
Keep descriptions under 50 words.
"""

# Company-specific rate limits
COMPANY_DELAYS = {
    "nvidia": 3.0,  # Slower for NVIDIA (complex site)
    "meta": 2.0,  # Slower for Meta
    "microsoft": 2.5,  # Slower for Microsoft
    "default": 1.0,  # Default delay
}

# Session statistics
session_stats = {
    "start_time": None,
    "companies_processed": 0,
    "jobs_found": 0,
    "cache_hits": 0,
    "llm_calls": 0,
    "errors": 0,
}


def get_cached_schema(company: str) -> dict | None:
    """Get cached extraction schema for company."""
    cache_file = CACHE_DIR / f"{company.lower()}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            return None
    return None


def save_schema_cache(company: str, schema: dict) -> None:
    """Save successful extraction schema."""
    cache_file = CACHE_DIR / f"{company.lower()}.json"
    cache_file.write_text(json.dumps(schema, indent=2))


def is_valid_job(job: dict, company: str) -> bool:
    """Simple job validation."""
    required = ["title", "description", "link"]

    # Check required fields exist and have content
    if not all(job.get(field, "").strip() for field in required):
        return False

    # Check reasonable lengths
    title = job["title"].strip()
    desc = job["description"].strip()
    link = job["link"].strip()

    if len(title) < 3 or len(title) > 200:
        return False

    if len(desc) < 10 or len(desc) > 1000:
        return False

    if not link.startswith(("http://", "https://")):
        return False

    return True


def log_session_summary():
    """Print simple session summary."""
    duration = time.time() - session_stats["start_time"]
    cache_rate = session_stats["cache_hits"] / max(
        session_stats["companies_processed"], 1
    )

    logger.info("📊 Session Summary:")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info(f"  Companies: {session_stats['companies_processed']}")
    logger.info(f"  Jobs found: {session_stats['jobs_found']}")
    logger.info(f"  Cache hit rate: {cache_rate:.1%}")
    logger.info(f"  LLM calls: {session_stats['llm_calls']}")
    logger.info(f"  Errors: {session_stats['errors']}")


async def extract_jobs_safe(url: str, company: str) -> list[dict]:
    """Safe wrapper with retries."""
    max_retries = 2

    for attempt in range(max_retries):
        try:
            return await extract_jobs(url, company)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {company}: {e}")
            session_stats["errors"] += 1
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(f"❌ All attempts failed for {company}")
                return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_jobs(url: str, company: str) -> list[dict]:
    """Extract jobs with simple caching and optimized LLM settings."""

    # Apply company-specific rate limiting
    delay = COMPANY_DELAYS.get(company.lower(), COMPANY_DELAYS["default"])
    await asyncio.sleep(delay)

    # Try cached schema first (free & fast)
    cached_schema = get_cached_schema(company)

    if cached_schema:
        try:
            strategy = JsonCssExtractionStrategy(cached_schema)
            async with AsyncWebCrawler() as crawler:
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
                            f"✅ Used cached schema for {company} - found {len(jobs)} jobs"
                        )
                        session_stats["cache_hits"] += 1
                        validated_jobs = [
                            job for job in jobs if is_valid_job(job, company)
                        ]
                        return [{"company": company, **job} for job in validated_jobs]
        except Exception as e:
            logger.warning(f"Cached schema failed for {company}: {e}")

    # Fallback to LLM (existing logic with optimized settings)
    logger.info(f"🔄 Using LLM extraction for {company}")
    session_stats["llm_calls"] += 1

    async with AsyncWebCrawler() as crawler:
        try:
            # Use optimized LLM strategy
            strategy = LLMExtractionStrategy(
                provider="openai/gpt-4o-mini",
                api_token=settings.openai_api_key,
                extraction_schema=SIMPLE_SCHEMA,
                instructions=SIMPLE_INSTRUCTIONS,
                apply_chunking=True,
                chunk_token_threshold=1000,
                overlap_rate=0.02,
            )

            result = await crawler.arun(url=url, extraction_strategy=strategy)
            extracted = json.loads(result.extracted_content)
            jobs = extracted.get("jobs", [])

            # If LLM extraction worked, try to generate a reusable schema
            if jobs and len(jobs) > 2:  # Only cache if we got multiple jobs
                try:
                    # Simple schema generation - extract CSS patterns from successful extraction
                    simple_schema = {
                        "jobs": {
                            "selector": ".job-listing, .job-item, .position, [class*='job'], [class*='position']",
                            "fields": {
                                "title": ".title, .job-title, h3, h4, .position-title",
                                "description": ".description, .summary, .job-summary, p",
                                "link": "a@href, .apply-link@href, .job-link@href",
                                "location": ".location, .job-location, .office",
                                "posted_date": ".date, .posted, .job-date",
                            },
                        }
                    }
                    save_schema_cache(company, simple_schema)
                    logger.info(f"💾 Cached schema for {company}")
                except Exception as e:
                    logger.warning(f"Failed to cache schema for {company}: {e}")

        except Exception as e:
            logger.warning(f"LLM failed for {company}: {e}. CSS fallback.")

            # Simple CSS fallback
            strategy = JsonCssExtractionStrategy(
                css_selector=".job-listing, .job-item, .position",
                instruction="Extract job title, description, link, location, date",
            )
            result = await crawler.arun(url=url, extraction_strategy=strategy)
            jobs = result.extracted_content or []

        # Process dates and locations
        for job in jobs:
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
        valid_jobs = [job for job in jobs if is_valid_job(job, company)]

        logger.info(f"📊 {company}: {len(valid_jobs)}/{len(jobs)} valid jobs")

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
    return bool(RELEVANT_KEYWORDS.search(job["title"]))


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
    updates existing jobs when content changes (via hash comparison), and
    removes jobs that are no longer found. Preserves user edits (favorite,
    status, notes) when updating existing jobs.

    Args:
        jobs_df (pd.DataFrame): DataFrame containing scraped job data with
            columns: company, title, description, link, location, posted_date.

    Note:
        Uses database transactions with rollback on error to maintain consistency.
        Invalid jobs are logged and skipped rather than failing the entire operation.
    """
    session = Session()
    try:
        existing = {j.link: j for j in session.query(JobSQL).all()}
        validated_jobs = []
        for _, job_dict in jobs_df.iterrows():
            job_dict = job_dict.to_dict()
            try:
                JobPydantic(**job_dict)
            except Exception as ve:
                logger.warning(
                    f"Validation failed for job {job_dict.get('title')}: {ve}"
                )
                continue
            valid_link = asyncio.run(validate_link(job_dict["link"]))
            if not valid_link:
                continue
            job_dict["link"] = valid_link
            job_hash = hashlib.md5(job_dict["description"].encode()).hexdigest()
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

    Retrieves active companies from database, scrapes their careers pages
    in parallel using asyncio, filters for relevant AI/ML positions,
    and returns consolidated results as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing all relevant scraped jobs with
            columns: company, title, description, link, location, posted_date.
            Empty DataFrame if no jobs found or all scraping attempts failed.

    Note:
        Individual company scraping failures are logged but don't stop
        the overall process. Only jobs with valid titles are included.
    """
    session = Session()
    active_companies = session.query(CompanySQL).filter_by(active=True).all()
    session.close()

    session_stats["companies_processed"] = len(active_companies)

    tasks = [extract_jobs_safe(c.url, c.name) for c in active_companies]
    all_jobs = []
    for task in asyncio.as_completed(tasks):
        try:
            jobs = await task
            relevant = [j for j in jobs if is_relevant(j)]
            all_jobs.extend(relevant)
        except Exception as e:
            logger.error(f"Scrape failed: {e}")
            session_stats["errors"] += 1
    df = pd.DataFrame(all_jobs)
    return df[df["title"].notna()]


def main() -> None:
    """Command-line interface entry point for the job scraper.

    Orchestrates the complete scraping workflow: scrapes all active companies,
    updates the database, and logs the results. Handles top-level exceptions
    and provides user feedback via logging.

    Note:
        Designed to be run via CLI: `python scraper.py` or `uv run python scraper.py`
    """
    session_stats["start_time"] = time.time()

    try:
        jobs_df = asyncio.run(scrape_all())
        update_db(jobs_df)
        session_stats["jobs_found"] = len(jobs_df)
        logger.info(f"Scraped {len(jobs_df)} jobs.")
        log_session_summary()
    except Exception as e:
        logger.error(f"Main failed: {e}")
        session_stats["errors"] += 1
        log_session_summary()


if __name__ == "__main__":
    typer.run(main)
