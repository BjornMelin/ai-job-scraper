"""Optimized job scraper: CSS-first, LLM-last approach.

This replaces the complex Crawl4AI + LLM approach with a simple, fast
tiered extraction system that prioritizes speed and cost-effectiveness.
"""

import asyncio
import json
import logging
import time

from pathlib import Path
from urllib.parse import urljoin

import httpx
import pandas as pd

from bs4 import BeautifulSoup

from config import Settings
from database import SessionLocal
from models import CompanySQL

# Import functions from existing scraper for LLM fallback and DB updates
try:
    from scraper import try_llm_extraction, update_db
except ImportError:
    # Handle missing imports gracefully
    try_llm_extraction = None
    update_db = None

settings = Settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cache directory for successful schemas
CACHE_DIR = Path(settings.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)

# Optimized company-specific schemas (CSS selectors)
FAST_COMPANY_SCHEMAS = {
    "openai": {
        "job_selectors": [
            "[data-testid='job-card']",
            ".job-card",
            ".role-card",
            "[class*='job']",
        ],
        "fields": {
            "title": "[data-testid='job-title'], .job-title, h3, h4",
            "description": (
                "[data-testid='job-description'], .job-description, .summary"
            ),
            "link": "a@href",
            "location": "[data-testid='job-location'], .location, .office",
            "posted_date": ".date, .posted, time@datetime",
        },
    },
    "anthropic": {
        "job_selectors": [".posting", ".job-listing", "[class*='job']", ".career-item"],
        "fields": {
            "title": ".posting-name, .job-title, h3, h4",
            "description": ".posting-description, .job-description",
            "link": "a@href",
            "location": ".posting-location, .location",
            "posted_date": ".posting-date, .date, time@datetime",
        },
    },
    "microsoft": {
        "job_selectors": [
            ".jobs-item",
            ".ms-List-cell",
            "[data-automation-id*='job']",
            "[class*='job']",
        ],
        "fields": {
            "title": ".jobs-title, h3, [data-automation-id*='title']",
            "description": ".jobs-description, .description",
            "link": "a@href",
            "location": ".jobs-location, [data-automation-id*='location']",
            "posted_date": ".jobs-date, .date, time@datetime",
        },
    },
    "meta": {
        "job_selectors": ["._8ykd", ".x1xzczfo", "._8isa", "[class*='job']"],
        "fields": {
            "title": "._8ykg, h4, .x1f6kntn",
            "description": "._8ykf, .x193iq5w",
            "link": "a@href",
            "location": "._8ykh, .x1i10hfl",
            "posted_date": ".date, ._8yk8, time@datetime",
        },
    },
}

# Generic patterns that work across many job sites
GENERIC_JOB_PATTERNS = [
    {
        "job_selectors": [".job", ".job-item", ".job-listing", ".job-card"],
        "fields": {
            "title": ".title, .job-title, h3, h4, h2",
            "description": ".description, .summary, .job-description, p",
            "link": "a@href",
            "location": ".location, .job-location, .office",
            "posted_date": ".date, .posted, time@datetime, .timestamp",
        },
    },
    {
        "job_selectors": [".posting", ".position", ".role", ".career"],
        "fields": {
            "title": ".posting-title, .position-title, .role-title, h3",
            "description": (
                ".posting-description, .position-description, .role-description"
            ),
            "link": "a@href",
            "location": ".posting-location, .position-location, .role-location",
            "posted_date": ".posting-date, .position-date, .role-date, time@datetime",
        },
    },
    {
        "job_selectors": ["article", ".entry", "li[class*='job']", "tr[class*='job']"],
        "fields": {
            "title": "h1, h2, h3, h4, .headline, .entry-title",
            "description": ".content, .entry-content, .summary, p",
            "link": "a@href",
            "location": ".meta, .info, .details",
            "posted_date": ".date, time@datetime, .published",
        },
    },
]


class OptimizedJobScraper:
    """Fast, cost-effective job scraper using CSS-first approach."""

    def __init__(self):
        self.client = None
        self.stats = {
            "companies_processed": 0,
            "jobs_found": 0,
            "css_successes": 0,
            "llm_fallbacks": 0,
            "cache_hits": 0,
            "total_time": 0,
        }

    async def __aenter__(self):
        """Async context manager setup."""
        # Realistic browser headers to avoid 403 errors
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }

        self.client = httpx.AsyncClient(
            timeout=30.0,  # Increased timeout
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers=headers,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        if self.client:
            await self.client.aclose()

    def get_cached_schema(self, company: str) -> dict | None:
        """Get cached successful schema for company."""
        cache_file = CACHE_DIR / f"{company.lower()}_schema.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Cache read error for {company}: {e}")
        return None

    def cache_schema(self, company: str, schema: dict) -> None:
        """Cache successful schema for future use."""
        cache_file = CACHE_DIR / f"{company.lower()}_schema.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Cached successful schema for {company}")
        except Exception as e:
            logger.debug(f"Cache write error for {company}: {e}")

    async def fetch_page(self, url: str, retry_count: int = 3) -> str | None:
        """Fetch page content with error handling and retries."""
        for attempt in range(retry_count):
            try:
                # Add small delay to avoid overwhelming servers
                if attempt > 0:
                    delay = 2**attempt  # Exponential backoff
                    logger.info(f"Retry {attempt + 1} for {url} after {delay}s delay")
                    await asyncio.sleep(delay)

                response = await self.client.get(url, follow_redirects=True)
                response.raise_for_status()
                logger.debug(
                    f"✅ Successfully fetched {url} ({len(response.text)} chars)"
                )
                return response.text

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = 10 + (attempt * 5)
                    logger.warning(f"Rate limited for {url}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code in [403, 401]:
                    logger.warning(
                        f"Access denied ({e.response.status_code}) for {url}"
                    )
                    return None  # Don't retry auth errors
                elif e.response.status_code == 404:
                    logger.warning(f"Page not found for {url}")
                    return None  # Don't retry 404s
                else:
                    logger.error(f"HTTP {e.response.status_code} for {url}")
                    if attempt < retry_count - 1:
                        continue
                    return None

            except Exception as e:
                logger.error(f"Fetch error for {url} (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    continue
                return None

        return None

    def extract_with_css(
        self, html: str, schema: dict, base_url: str = ""
    ) -> list[dict]:
        """Fast CSS extraction using lxml parser."""
        soup = BeautifulSoup(html, "lxml")  # 10x faster than default parser
        jobs = []

        # Try each job selector until we find jobs
        for job_selector in schema["job_selectors"]:
            job_elements = soup.select(job_selector)
            if job_elements:
                logger.debug(
                    f"Found {len(job_elements)} elements with '{job_selector}'"
                )
                break
        else:
            return []  # No job elements found

        for job_elem in job_elements:
            job = {}

            # Extract each field
            for field, selector in schema["fields"].items():
                value = None

                # Handle attribute extraction (e.g., "a@href")
                if "@" in selector:
                    selectors, attr = selector.split("@", 1)
                    for sel in selectors.split(","):
                        elem = job_elem.select_one(sel.strip())
                        if elem:
                            value = elem.get(attr.strip())
                            if value:
                                break
                else:
                    # Handle text extraction
                    for sel in selector.split(","):
                        elem = job_elem.select_one(sel.strip())
                        if elem:
                            value = elem.get_text(strip=True)
                            if value:
                                break

                job[field] = value

            # Basic validation - must have title and link
            if job.get("title") and job.get("link"):
                # Convert relative URLs to absolute
                if job["link"] and not job["link"].startswith("http") and base_url:
                    job["link"] = urljoin(base_url, job["link"])

                # Clean up description
                if job.get("description"):
                    job["description"] = job["description"][:500]  # Limit length

                jobs.append(job)

        return jobs

    async def try_css_extraction(self, url: str, company: str) -> list[dict]:
        """Try CSS extraction with company-specific and generic patterns."""
        html = await self.fetch_page(url)
        if not html:
            return []

        company_key = company.lower().replace(" ", "").replace("-", "").replace("_", "")

        # Tier 1: Cached schema (fastest)
        if cached_schema := self.get_cached_schema(company):
            jobs = self.extract_with_css(html, cached_schema, url)
            if jobs:
                self.stats["cache_hits"] += 1
                logger.info(f"✅ Cache hit for {company}: {len(jobs)} jobs")
                return jobs

        # Tier 2: Company-specific schema
        if company_key in FAST_COMPANY_SCHEMAS:
            schema = FAST_COMPANY_SCHEMAS[company_key]
            jobs = self.extract_with_css(html, schema, url)
            if jobs:
                self.cache_schema(company, schema)
                self.stats["css_successes"] += 1
                logger.info(f"✅ Company schema for {company}: {len(jobs)} jobs")
                return jobs

        # Tier 3: Generic patterns
        for i, generic_schema in enumerate(GENERIC_JOB_PATTERNS):
            jobs = self.extract_with_css(html, generic_schema, url)
            if jobs:
                self.cache_schema(company, generic_schema)
                self.stats["css_successes"] += 1
                logger.info(
                    f"✅ Generic pattern {i + 1} for {company}: {len(jobs)} jobs"
                )
                return jobs

        logger.warning(f"❌ CSS extraction failed for {company}")
        return []

    async def try_llm_fallback(self, url: str, company: str) -> list[dict]:
        """LLM fallback for difficult sites (expensive, use sparingly)."""
        self.stats["llm_fallbacks"] += 1
        logger.warning(f"🤖 Using expensive LLM fallback for {company}")

        # Use imported LLM extraction function
        if try_llm_extraction:
            try:
                return await try_llm_extraction(url, company)
            except Exception as e:
                logger.error(f"LLM fallback failed for {company}: {e}")
                return []
        else:
            logger.warning(f"LLM fallback not available for {company}")
            return []

    async def extract_jobs(self, url: str, company: str) -> list[dict]:
        """Extract jobs using tiered approach: CSS first, LLM last."""
        start_time = time.time()

        # Try CSS extraction first (fast, free)
        jobs = await self.try_css_extraction(url, company)

        # Only use LLM if CSS completely failed and it's a priority company
        if not jobs and len(company) > 3:  # Avoid LLM for unknown/test companies
            jobs = await self.try_llm_fallback(url, company)

        # Add company name to each job
        for job in jobs:
            job["company"] = company

        self.stats["companies_processed"] += 1
        self.stats["jobs_found"] += len(jobs)
        self.stats["total_time"] += time.time() - start_time

        logger.info(
            f"📊 {company}: {len(jobs)} jobs in {time.time() - start_time:.1f}s"
        )
        return jobs

    async def scrape_all_companies(self) -> list[dict]:
        """Scrape all active companies concurrently."""
        session = SessionLocal()
        try:
            active_companies = session.query(CompanySQL).filter_by(active=True).all()
        finally:
            session.close()

        if not active_companies:
            logger.warning("No active companies found")
            return []

        # Process companies concurrently with semaphore
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def scrape_with_semaphore(company):
            async with semaphore:
                return await self.extract_jobs(company.url, company.name)

        logger.info(
            f"🚀 Starting optimized scraping of {len(active_companies)} companies"
        )
        start_time = time.time()

        tasks = [scrape_with_semaphore(company) for company in active_companies]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        all_jobs = []
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            else:
                logger.error(f"Task failed: {result}")

        # Print performance stats
        total_time = time.time() - start_time
        self.stats["total_time"] = total_time
        self.print_stats()

        return all_jobs

    def print_stats(self):
        """Print performance statistics."""
        stats = self.stats
        avg_time = stats["total_time"] / max(stats["companies_processed"], 1)
        css_rate = stats["css_successes"] / max(stats["companies_processed"], 1) * 100
        cache_rate = stats["cache_hits"] / max(stats["companies_processed"], 1) * 100

        logger.info("📊 Optimized Scraper Performance:")
        logger.info(f"  Total time: {stats['total_time']:.1f}s")
        logger.info(f"  Companies: {stats['companies_processed']}")
        logger.info(f"  Jobs found: {stats['jobs_found']}")
        logger.info(f"  Avg time/company: {avg_time:.1f}s")
        logger.info(f"  CSS success rate: {css_rate:.1f}%")
        logger.info(f"  Cache hit rate: {cache_rate:.1f}%")
        logger.info(f"  LLM fallbacks: {stats['llm_fallbacks']}")


def update_db_optimized(jobs: list[dict]) -> None:
    """Update database with scraped jobs (optimized version)."""
    if not jobs:
        logger.info("No jobs to update in database")
        return

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(jobs)

    # Use imported update_db function
    if update_db:
        update_db(df)
    else:
        logger.error("update_db function not available")

    logger.info(f"💾 Updated database with {len(jobs)} jobs")


async def main():
    """Main function for optimized scraper."""
    logger.info("🚀 Starting Optimized Job Scraper")

    async with OptimizedJobScraper() as scraper:
        jobs = await scraper.scrape_all_companies()

        if jobs:
            update_db_optimized(jobs)
            logger.info(f"✅ Successfully scraped {len(jobs)} jobs")
        else:
            logger.warning("❌ No jobs found")


if __name__ == "__main__":
    asyncio.run(main())
