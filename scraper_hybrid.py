"""Hybrid job scraper: Static CSS + Crawl4AI + LLM fallback.

This combines the best of all approaches:
1. Fast static CSS extraction for simple sites
2. Crawl4AI JsonCssExtractionStrategy for JavaScript sites (no LLM cost)
3. Crawl4AI LLM extraction as expensive last resort
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
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    LLMConfig,
    LLMExtractionStrategy,
)

from config import Settings
from database import SessionLocal
from models import CompanySQL

# Import functions from existing scraper for DB updates
from scraper import update_db

settings = Settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cache directory for successful schemas
CACHE_DIR = Path(settings.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)

# Universal job schemas optimized for Crawl4AI JsonCssExtractionStrategy
CRAWL4AI_JOB_SCHEMA = {
    "name": "Universal Job Listings",
    "baseSelector": ".job-card, [data-testid='job-listing'], .job-item, .result, .posting, .position, .career-item, .job-opening, .opportunity, .role",
    "fields": [
        {
            "name": "title",
            "selector": "h1, h2, h3, h4, .job-title, [data-testid='job-title'], [data-testid='job-title-link'], .title, .position-title, .role-title",
            "type": "text",
        },
        {
            "name": "company",
            "selector": ".company, [data-testid='company-name'], .company-name, .employer, .organization",
            "type": "text",
        },
        {
            "name": "location",
            "selector": ".location, [data-testid='job-location'], .job-location, .office, .city, .geographic",
            "type": "text",
        },
        {
            "name": "salary",
            "selector": ".salary, [data-testid='salary'], .salary-range, .compensation, .pay, .wage",
            "type": "text",
        },
        {
            "name": "summary",
            "selector": ".summary, .job-summary, [data-testid='job-snippet'], .description-preview, .excerpt",
            "type": "text",
        },
        {
            "name": "job_url",
            "selector": "a[href*='/job/'], a[href*='/jobs/'], a[href*='/position/'], a[href*='/career/'], a",
            "type": "attribute",
            "attribute": "href",
        },
        {
            "name": "posted_date",
            "selector": ".posted-date, time, [data-testid='posted-date'], .date, .timestamp",
            "type": "text",
        },
    ],
}

# Static CSS patterns for fast extraction (from optimized scraper)
STATIC_CSS_PATTERNS = [
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
            "description": ".posting-description, .position-description, .role-description",
            "link": "a@href",
            "location": ".posting-location, .position-location, .role-location",
            "posted_date": ".posting-date, .position-date, .role-date, time@datetime",
        },
    },
]


class HybridJobScraper:
    """Hybrid scraper combining static CSS, Crawl4AI, and LLM approaches."""

    def __init__(self):
        self.client = None
        self.crawl4ai_available = AsyncWebCrawler is not None
        self.stats = {
            "companies_processed": 0,
            "jobs_found": 0,
            "static_css_successes": 0,
            "crawl4ai_css_successes": 0,
            "llm_fallbacks": 0,
            "cache_hits": 0,
            "total_time": 0,
            "403_errors": 0,
            "js_sites_detected": 0,
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
            timeout=30.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers=headers,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        if self.client:
            await self.client.aclose()

    async def fetch_page_static(self, url: str, retry_count: int = 3) -> str | None:
        """Fetch page content for static CSS extraction."""
        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    delay = 2**attempt
                    logger.info(f"Retry {attempt + 1} for {url} after {delay}s delay")
                    await asyncio.sleep(delay)

                response = await self.client.get(url, follow_redirects=True)
                response.raise_for_status()
                logger.debug(f"✅ Static fetch: {url} ({len(response.text)} chars)")
                return response.text

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [403, 401]:
                    self.stats["403_errors"] += 1
                    logger.warning(
                        f"🚫 Access denied ({e.response.status_code}) for {url} - will try Crawl4AI"
                    )
                    return None
                elif e.response.status_code == 429:
                    wait_time = 10 + (attempt * 5)
                    logger.warning(
                        f"⏰ Rate limited for {url}, waiting {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 404:
                    logger.warning(f"❌ Page not found: {url}")
                    return None
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

    def extract_with_static_css(self, html: str, base_url: str = "") -> list[dict]:
        """Fast static CSS extraction."""
        soup = BeautifulSoup(html, "lxml")
        jobs = []

        for pattern in STATIC_CSS_PATTERNS:
            # Try each job selector pattern
            for job_selector in pattern["job_selectors"]:
                job_elements = soup.select(job_selector)
                if job_elements:
                    logger.debug(
                        f"🎯 Found {len(job_elements)} jobs with '{job_selector}'"
                    )

                    for job_elem in job_elements:
                        job = {}

                        # Extract each field
                        for field, selector in pattern["fields"].items():
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

                        # Basic validation and URL conversion
                        if job.get("title") and job.get("link"):
                            if (
                                job["link"]
                                and not job["link"].startswith("http")
                                and base_url
                            ):
                                job["link"] = urljoin(base_url, job["link"])

                            if job.get("description"):
                                job["description"] = job["description"][:500]

                            jobs.append(job)

                    if jobs:  # If we found jobs with this pattern, return them
                        return jobs

        return jobs

    def detect_javascript_site(self, html: str) -> bool:
        """Detect if site likely needs JavaScript rendering."""
        if not html:
            return True

        # Check for indicators that content is loaded via JavaScript
        js_indicators = [
            "React",
            "Vue",
            "Angular",
            "Next.js",
            "nuxt",
            "data-reactroot",
            "ng-app",
            "v-app",
            "__NEXT_DATA__",
            "__NUXT__",
            "window.React",
            'document.addEventListener("DOMContentLoaded"',
            "window.onload",
            "jQuery",
            "$(",
            "Loading...",
            "Please wait",
            "spinner",
            "data-loading",
            "loading-content",
        ]

        html_lower = html.lower()
        js_score = sum(
            1 for indicator in js_indicators if indicator.lower() in html_lower
        )

        # Also check if page has very little content
        soup = BeautifulSoup(html, "lxml")
        text_content = soup.get_text(strip=True)
        content_ratio = len(text_content) / max(len(html), 1)

        is_js_site = js_score >= 3 or content_ratio < 0.02
        if is_js_site:
            logger.info(
                f"🔬 JavaScript site detected (score: {js_score}, content ratio: {content_ratio:.3f})"
            )
            self.stats["js_sites_detected"] += 1

        return is_js_site

    async def try_crawl4ai_css(self, url: str, company: str) -> list[dict]:
        """Try Crawl4AI with CSS extraction (no LLM cost)."""
        if not self.crawl4ai_available:
            logger.warning("⚠️ Crawl4AI not available")
            return []

        try:
            strategy = JsonCssExtractionStrategy(CRAWL4AI_JOB_SCHEMA, verbose=True)
            config = CrawlerRunConfig(
                extraction_strategy=strategy,
                cache_mode=CacheMode.BYPASS,
                delay_before_return_html=3.0,  # Wait for content to load
                js_code="window.scrollTo(0, document.body.scrollHeight/2);",  # Trigger lazy loading
            )

            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)

                if result.success and result.extracted_content:
                    try:
                        jobs_data = json.loads(result.extracted_content)
                        jobs = jobs_data if isinstance(jobs_data, list) else [jobs_data]

                        # Clean and validate jobs
                        valid_jobs = []
                        for job in jobs:
                            if isinstance(job, dict) and job.get("title"):
                                # Add company name
                                job["company"] = company
                                # Convert relative URLs
                                if job.get("job_url") and not job["job_url"].startswith(
                                    "http"
                                ):
                                    job["job_url"] = urljoin(url, job["job_url"])
                                valid_jobs.append(job)

                        if valid_jobs:
                            logger.info(
                                f"✅ Crawl4AI CSS found {len(valid_jobs)} jobs for {company}"
                            )
                            self.stats["crawl4ai_css_successes"] += 1
                            return valid_jobs

                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Crawl4AI JSON decode error: {e}")

                logger.warning(f"⚠️ Crawl4AI CSS extraction failed for {company}")

        except Exception as e:
            logger.error(f"❌ Crawl4AI CSS error for {company}: {e}")

        return []

    async def try_crawl4ai_llm(self, url: str, company: str) -> list[dict]:
        """Try Crawl4AI with LLM extraction (expensive fallback)."""
        if not self.crawl4ai_available:
            return []

        try:
            self.stats["llm_fallbacks"] += 1
            logger.warning(f"💰 Using expensive LLM fallback for {company}")

            extraction_schema = {
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "company": {"type": "string"},
                                "location": {"type": "string"},
                                "salary": {"type": "string"},
                                "summary": {"type": "string"},
                                "job_url": {"type": "string"},
                                "posted_date": {"type": "string"},
                            },
                            "required": ["title"],
                        },
                    }
                },
                "required": ["jobs"],
            }

            # Prefer GROQ over OpenAI for better performance and reliability
            if settings.groq_api_key:
                llm_config = LLMConfig(
                    provider="groq/llama3-70b-8192",
                    api_token=settings.groq_api_key,
                )
                logger.info(f"Using GROQ for LLM extraction: {company}")
            elif (
                settings.openai_api_key
                and settings.openai_api_key != "your_openai_api_key_here"
            ):
                llm_config = LLMConfig(
                    provider="openai/gpt-4o-mini",
                    api_token=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                )
                logger.info(f"Using OpenAI for LLM extraction: {company}")
            else:
                logger.warning(f"No valid API keys for LLM extraction: {company}")
                return []

            strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                schema=extraction_schema,
                extraction_type="schema",
                instruction=f"""
                Extract job postings from this {company} careers page.
                Focus on AI, Machine Learning, Engineering, and Data Science roles.
                Include job title, location, summary, and application link.
                """,
            )

            config = CrawlerRunConfig(
                extraction_strategy=strategy,
                cache_mode=CacheMode.BYPASS,
                delay_before_return_html=3.0,
            )

            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)

                if result.success and result.extracted_content:
                    try:
                        data = json.loads(result.extracted_content)

                        # Handle different response formats
                        if isinstance(data, list):
                            jobs = data
                        elif isinstance(data, dict):
                            jobs = data.get("jobs", [])
                        else:
                            jobs = []

                        # Add company name and clean data, filtering out error objects
                        valid_jobs = []
                        for job in jobs:
                            if isinstance(job, dict) and not job.get("error"):
                                # Must have at least title to be a valid job
                                if job.get("title"):
                                    job["company"] = company
                                    # Normalize field names from different sources
                                    if "job_url" in job:
                                        job["link"] = job.pop("job_url")
                                    if "summary" in job:
                                        job["description"] = job.pop("summary")

                                    # Fix relative URLs
                                    if job.get("link") and not job["link"].startswith(
                                        "http"
                                    ):
                                        job["link"] = urljoin(url, job["link"])

                                    valid_jobs.append(job)

                        if valid_jobs:
                            logger.info(
                                f"✅ Crawl4AI LLM found {len(valid_jobs)} jobs for {company}"
                            )
                            return valid_jobs

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ LLM extraction JSON error: {e}")

        except Exception as e:
            logger.error(f"❌ Crawl4AI LLM error for {company}: {e}")

        return []

    async def extract_jobs(self, url: str, company: str) -> list[dict]:
        """Extract jobs using hybrid tiered approach."""
        start_time = time.time()

        logger.info(f"🔍 Starting hybrid extraction for {company}: {url}")

        # Tier 1: Try static CSS extraction first (fastest)
        html = await self.fetch_page_static(url)

        if html:
            # Check if this looks like a JavaScript-heavy site
            is_js_site = self.detect_javascript_site(html)

            if not is_js_site:
                jobs = self.extract_with_static_css(html, url)
                if jobs:
                    self.stats["static_css_successes"] += 1
                    logger.info(f"✅ Static CSS found {len(jobs)} jobs for {company}")
                    for job in jobs:
                        job["company"] = company

                    extraction_time = time.time() - start_time
                    self.stats["total_time"] += extraction_time
                    return jobs

        # Tier 2: Try Crawl4AI CSS extraction (JavaScript support, no LLM cost)
        if self.crawl4ai_available:
            jobs = await self.try_crawl4ai_css(url, company)
            if jobs:
                extraction_time = time.time() - start_time
                self.stats["total_time"] += extraction_time
                return jobs

        # Tier 3: Try Crawl4AI LLM extraction (expensive, last resort)
        if self.crawl4ai_available and len(company) > 3:  # Only for real companies
            jobs = await self.try_crawl4ai_llm(url, company)
            if jobs:
                extraction_time = time.time() - start_time
                self.stats["total_time"] += extraction_time
                return jobs

        # All tiers failed
        logger.warning(f"❌ All extraction methods failed for {company}")
        extraction_time = time.time() - start_time
        self.stats["total_time"] += extraction_time
        return []

    async def scrape_all_companies(self) -> list[dict]:
        """Scrape all active companies with hybrid approach."""
        session = SessionLocal()
        try:
            active_companies = session.query(CompanySQL).filter_by(active=True).all()
        finally:
            session.close()

        if not active_companies:
            logger.warning("No active companies found")
            return []

        # Process companies with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Lower concurrency to avoid 403s

        async def scrape_with_semaphore(company):
            async with semaphore:
                await asyncio.sleep(1)  # Rate limiting between requests
                return await self.extract_jobs(company.url, company.name)

        logger.info(f"🚀 Starting hybrid scraping of {len(active_companies)} companies")
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

        # Update stats
        total_time = time.time() - start_time
        self.stats["companies_processed"] = len(active_companies)
        self.stats["jobs_found"] = len(all_jobs)
        self.stats["total_time"] = total_time

        self.print_stats()
        return all_jobs

    def print_stats(self):
        """Print performance statistics."""
        stats = self.stats
        total_companies = max(stats["companies_processed"], 1)

        logger.info("📊 Hybrid Scraper Performance:")
        logger.info(f"  Total time: {stats['total_time']:.1f}s")
        logger.info(f"  Companies processed: {stats['companies_processed']}")
        logger.info(f"  Jobs found: {stats['jobs_found']}")
        logger.info(f"  Avg time/company: {stats['total_time'] / total_companies:.1f}s")
        logger.info(
            f"  Static CSS successes: {stats['static_css_successes']} ({stats['static_css_successes'] / total_companies * 100:.1f}%)"
        )
        logger.info(
            f"  Crawl4AI CSS successes: {stats['crawl4ai_css_successes']} ({stats['crawl4ai_css_successes'] / total_companies * 100:.1f}%)"
        )
        logger.info(
            f"  LLM fallbacks: {stats['llm_fallbacks']} ({stats['llm_fallbacks'] / total_companies * 100:.1f}%)"
        )
        logger.info(f"  403 errors: {stats['403_errors']}")
        logger.info(f"  JS sites detected: {stats['js_sites_detected']}")


def update_db_hybrid(jobs: list[dict]) -> None:
    """Update database with scraped jobs (hybrid version)."""
    if not jobs:
        logger.info("No jobs to update in database")
        return

    # AI-related keywords to filter jobs
    ai_keywords = [
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml engineer",
        "mlops",
        "agentic",
        "agent",
        "ai agent",
        "cuda",
        "llm",
        "genai",
        "generative ai",
        "deep learning",
        "dl",
        "neural",
        "data scientist",
        "computer vision",
        "nlp",
        "natural language",
        "ai engineer",
        "ml infrastructure",
        "ai research",
        "ai/ml",
    ]

    # Filter out invalid jobs and non-AI jobs before database update
    valid_jobs = []
    ai_filtered_count = 0

    for job in jobs:
        # Must be dict and have required fields
        if isinstance(job, dict) and not job.get("error"):
            # Check for minimum required fields
            if job.get("title") and job.get("company"):
                # Check if job title contains AI-related keywords
                title_lower = job.get("title", "").lower()
                is_ai_job = any(keyword in title_lower for keyword in ai_keywords)

                if not is_ai_job:
                    ai_filtered_count += 1
                    logger.debug(
                        f"Filtering out non-AI job: {job.get('title')} at {job.get('company')}"
                    )
                    continue
                # Add default values for missing fields and clean data
                job.setdefault("description", "No description available")
                job.setdefault("link", "")
                job.setdefault("location", "")
                job.setdefault("posted_date", None)

                # Remove extra fields that cause validation errors
                job.pop("error", None)
                job.pop("salary", None)  # Remove NaN salary fields

                # Ensure description meets minimum length requirement
                if len(job.get("description", "")) < 10:
                    job["description"] = "No description available"

                # Fix posted_date validation issues
                if job.get("posted_date") == "":
                    job["posted_date"] = None

                valid_jobs.append(job)
            else:
                logger.debug(f"Skipping invalid job (missing title/company): {job}")
        else:
            logger.debug(f"Skipping error object: {job}")

    if not valid_jobs:
        logger.warning("No valid jobs to update in database")
        if ai_filtered_count > 0:
            logger.info(f"🎯 Filtered out {ai_filtered_count} non-AI jobs")
        return

    # Log AI filtering info
    logger.info(
        f"🤖 Found {len(valid_jobs)} AI-related jobs out of {len(jobs)} total jobs scraped"
    )
    if ai_filtered_count > 0:
        logger.info(f"🎯 Filtered out {ai_filtered_count} non-AI jobs")

    # Create our own synchronous database update to avoid asyncio conflicts
    try:
        from models import JobPydantic, JobSQL

        session = SessionLocal()

        try:
            existing = {j.link: j for j in session.query(JobSQL).all()}
            validated_jobs = []

            for job_dict in valid_jobs:
                try:
                    # Fix link validation issues BEFORE Pydantic validation
                    if not job_dict.get("link") or job_dict["link"] == "":
                        # Generate unique link using company + title + timestamp
                        unique_id = f"{job_dict.get('company', '')}-{job_dict.get('title', '')}-{time.time()}"
                        job_dict["link"] = (
                            f"https://example.com/job/{abs(hash(unique_id))}"
                        )
                    elif not job_dict["link"].startswith(("http://", "https://")):
                        # Fix relative URLs or invalid URLs
                        unique_id = f"{job_dict.get('company', '')}-{job_dict.get('title', '')}-{time.time()}"
                        job_dict["link"] = (
                            f"https://example.com/job/{abs(hash(unique_id))}"
                        )

                    # Fix posted_date validation issues BEFORE Pydantic validation
                    posted_date = job_dict.get("posted_date")
                    if posted_date and isinstance(posted_date, str):
                        # Replace common invalid date strings with None
                        invalid_dates = [
                            "today",
                            "yesterday",
                            "recent",
                            "new",
                            "",
                            "posted today",
                            "posted yesterday",
                        ]
                        if posted_date.lower().strip() in invalid_dates:
                            job_dict["posted_date"] = None

                    # Now validate with Pydantic
                    JobPydantic(**job_dict)
                    validated_jobs.append(job_dict)
                except Exception as ve:
                    logger.warning(
                        f"Validation failed for job {job_dict.get('title')}: {ve}"
                    )
                    continue

            # Add or update jobs
            added_count = 0
            updated_count = 0

            for job_dict in validated_jobs:
                link = job_dict["link"]
                job_hash = hash(str(job_dict))

                # Check if job already exists by link
                existing_job = session.query(JobSQL).filter_by(link=link).first()

                if existing_job:
                    # Update existing job if hash changed
                    if existing_job.hash != job_hash:
                        for key, value in job_dict.items():
                            if key not in [
                                "favorite",
                                "status",
                                "notes",
                            ]:  # Preserve user edits
                                setattr(existing_job, key, value)
                        existing_job.hash = job_hash
                        updated_count += 1
                        logger.debug(f"Updated existing job: {job_dict['title']}")
                else:
                    # Add new job - check if similar job exists by title and company
                    similar_job = (
                        session.query(JobSQL)
                        .filter_by(
                            title=job_dict.get("title"), company=job_dict.get("company")
                        )
                        .first()
                    )

                    if similar_job:
                        # Update the existing similar job with new link
                        for key, value in job_dict.items():
                            if key not in [
                                "favorite",
                                "status",
                                "notes",
                            ]:  # Preserve user edits
                                setattr(similar_job, key, value)
                        similar_job.hash = job_hash
                        updated_count += 1
                        logger.debug(f"Updated similar job: {job_dict['title']}")
                    else:
                        # Really is a new job
                        new_job = JobSQL(**job_dict, hash=job_hash)
                        session.add(new_job)
                        added_count += 1
                        logger.debug(f"Added new job: {job_dict['title']}")

            # Remove jobs that are no longer found (optional - might be too aggressive)
            current_links = {job["link"] for job in validated_jobs}
            for link, _job in existing.items():
                if link not in current_links:
                    # Only remove if it's older than a few days to avoid false removals
                    pass  # Skip removal for now

            session.commit()
            logger.info(
                f"💾 Successfully updated database: {added_count} added, {updated_count} updated"
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Database update failed: {e}")
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Failed to import database models: {e}")
        # Fallback to original update_db if available
        try:
            df = pd.DataFrame(valid_jobs)
            update_db(df)
        except Exception as fallback_error:
            logger.error(f"Fallback update_db also failed: {fallback_error}")


async def main():
    """Main function for hybrid scraper."""
    logger.info("🚀 Starting Hybrid Job Scraper")

    async with HybridJobScraper() as scraper:
        jobs = await scraper.scrape_all_companies()

        if jobs:
            update_db_hybrid(jobs)
            logger.info(f"✅ Successfully scraped {len(jobs)} jobs")
        else:
            logger.warning("❌ No jobs found")


if __name__ == "__main__":
    asyncio.run(main())
