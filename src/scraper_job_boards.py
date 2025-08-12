"""Scraper module for structured job boards in the AI Job Scraper application.

This module uses the python-jobspy library to scrape job listings from sites like
LinkedIn and Indeed. It supports keyword and location-based searches, proxy rotation,
random delays for evasion, filtering for AI/ML-related roles, and normalization of
job data into a format suitable for database insertion.
"""

import logging

from typing import Any

import pandas as pd

from jobspy import Site, scrape_jobs

from .config import Settings
from .constants import AI_REGEX
from .utils import random_delay, resolve_jobspy_proxies

settings = Settings()
logger = logging.getLogger(__name__)


def scrape_job_boards(
    keywords: list[str], locations: list[str]
) -> list[dict[str, Any]]:
    """Scrape job listings from structured job boards using JobSpy.

    This function iterates over provided locations, scrapes jobs for the combined
    keywords from LinkedIn and Indeed, applies random delays, uses proxies if enabled,
    filters results for AI/ML-related titles using regex, removes duplicates, and
    returns the normalized job data as a list of dictionaries. The data includes
    pre-parsed salary fields from JobSpy for easy database insertion.

    Args:
        keywords: List of search keywords to combine with 'OR'.
        locations: List of locations to search in.

    Returns:
        List of dictionaries, each representing a job with fields like 'title',
        'company', 'location', 'description', 'job_url', 'min_amount', 'max_amount',
        etc.
    """
    logger.info("=" * 50)
    logger.info("🔍 STARTING JOB BOARDS SCRAPING")
    logger.info("=" * 50)
    logger.info("Search keywords: %s", ", ".join(keywords))
    logger.info("Locations to search: %s", ", ".join(locations))
    logger.info("Target sites: LinkedIn, Indeed")

    all_dfs: list[pd.DataFrame] = []
    search_term = " OR ".join(keywords)
    location_results = {}

    for i, location in enumerate(locations, 1):
        logger.info("📍 Scraping location %d/%d: %s", i, len(locations), location)
        random_delay()
        try:
            jobs: pd.DataFrame = scrape_jobs(
                site_name=[Site.LINKEDIN, Site.INDEED],
                search_term=search_term,
                location=location,
                results_wanted=100,
                proxies=resolve_jobspy_proxies(settings),
            )
            job_count = len(jobs) if jobs is not None and not jobs.empty else 0
            location_results[location] = job_count
            logger.info("  ✅ Found %d jobs in %s", job_count, location)

            if job_count > 0:
                all_dfs.append(jobs)
        except Exception:
            location_results[location] = 0
            logger.exception("  ❌ Error scraping jobs for location '%s'", location)

    if not all_dfs:
        logger.warning("No jobs found from any location")
        return []

    logger.info("-" * 50)
    logger.info("🔄 Processing and filtering job board results")

    all_jobs = pd.concat(all_dfs, ignore_index=True)
    initial_count = len(all_jobs)
    logger.info("Total jobs before deduplication: %d", initial_count)

    all_jobs = all_jobs.drop_duplicates(subset=["job_url"])
    after_dedup = len(all_jobs)
    logger.info(
        "Jobs after removing duplicates: %d (removed %d duplicates)",
        after_dedup,
        initial_count - after_dedup,
    )

    filtered_jobs = all_jobs[all_jobs["title"].str.contains(AI_REGEX, na=False)]
    final_count = len(filtered_jobs)
    logger.info(
        "Jobs after AI/ML keyword filtering: %d (removed %d non-AI jobs)",
        final_count,
        after_dedup - final_count,
    )

    # Log location summary
    logger.info("-" * 50)
    logger.info("📊 Job boards scraping summary:")
    logger.info("  • Total locations processed: %d", len(locations))
    logger.info(
        "  • Successful locations: %d",
        sum(1 for count in location_results.values() if count > 0),
    )
    for location, count in location_results.items():
        logger.info("    - %s: %d jobs", location, count)
    logger.info("  • Final filtered jobs returned: %d", final_count)
    logger.info("=" * 50)

    return filtered_jobs.to_dict(orient="records")
