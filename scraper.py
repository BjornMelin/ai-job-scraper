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
from datetime import datetime

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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_jobs(url: str, company: str) -> list[dict]:
    """Extract job postings from a company's careers page with LLM/CSS fallback.

    Uses Crawl4AI to scrape job listings, first attempting LLM extraction
    with OpenAI, falling back to CSS selector strategy if that fails.

    Args:
        url (str): Company careers page URL to scrape.
        company (str): Company name for job attribution.

    Returns:
        list[dict]: List of job dictionaries with company, title, description,
            link, location, and posted_date fields.

    Raises:
        Exception: Re-raises exceptions after retry attempts are exhausted.
    """
    async with AsyncWebCrawler() as crawler:
        try:
            strategy = LLMExtractionStrategy(
                provider="openai/gpt-4o-mini",
                api_token=settings.openai_api_key,
                extraction_schema={
                    "jobs": [
                        {
                            "title": "str",
                            "description": "str",
                            "link": "str",
                            "location": "str",
                            "posted_date": "str",
                        }
                    ]
                },
                instructions="Extract jobs: title, desc, link, location, posted date (any format).",
            )
            result = await crawler.arun(url=url, extraction_strategy=strategy)
            extracted = json.loads(result.extracted_content)
            jobs = extracted.get("jobs", [])
        except Exception as e:
            logger.warning(f"LLM failed for {company}: {e}. CSS fallback.")
            strategy = JsonCssExtractionStrategy(
                css_selector=".job-listing",
                instruction="Extract title, desc, link, location, date",
            )
            result = await crawler.arun(url=url, extraction_strategy=strategy)
            jobs = result.extracted_content or []

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

        return [{"company": company, **job} for job in jobs if "title" in job]


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
    tasks = [extract_jobs(c.url, c.name) for c in active_companies]
    all_jobs = []
    for task in asyncio.as_completed(tasks):
        try:
            jobs = await task
            relevant = [j for j in jobs if is_relevant(j)]
            all_jobs.extend(relevant)
        except Exception as e:
            logger.error(f"Scrape failed: {e}")
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
    try:
        jobs_df = asyncio.run(scrape_all())
        update_db(jobs_df)
        logger.info(f"Scraped {len(jobs_df)} jobs.")
    except Exception as e:
        logger.error(f"Main failed: {e}")


if __name__ == "__main__":
    typer.run(main)
