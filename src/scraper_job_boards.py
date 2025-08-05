"""Scraper module for structured job boards in the AI Job Scraper application.

This module uses the python-jobspy library to scrape job listings from sites like
LinkedIn and Indeed. It supports keyword and location-based searches, proxy rotation,
random delays for evasion, filtering for AI/ML-related roles, and normalization of
job data into a format suitable for database insertion.
"""

from typing import Any

import pandas as pd

from jobspy import Site, scrape_jobs

from .config import Settings
from .constants import AI_REGEX
from .utils import random_delay

settings = Settings()


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
    all_dfs: list[pd.DataFrame] = []
    search_term = " OR ".join(keywords)

    for location in locations:
        random_delay()
        try:
            jobs: pd.DataFrame = scrape_jobs(
                site_name=[Site.LINKEDIN, Site.INDEED],
                search_term=search_term,
                location=location,
                results_wanted=100,
                proxies=settings.proxy_pool if settings.use_proxies else None,
            )
            all_dfs.append(jobs)
        except Exception as e:
            print(f"Error scraping jobs for location '{location}': {e}")

    if not all_dfs:
        return []

    all_jobs = pd.concat(all_dfs, ignore_index=True)
    all_jobs = all_jobs.drop_duplicates(subset=["job_url"])

    filtered_jobs = all_jobs[all_jobs["title"].str.contains(AI_REGEX, na=False)]

    return filtered_jobs.to_dict(orient="records")
