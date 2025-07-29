# ADR-001: Scraping and Extraction Framework

## Title

Selection and Implementation of Web Scraping and Data Extraction

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

The core of the app involves scraping job postings from dynamic company websites (e.g., Anthropic, OpenAI, NVIDIA), handling JS rendering, extracting structured data (title, description, link, location, posted_date), and ensuring robustness with retries, fallbacks, and validation. Tools must be efficient, support async operations, and work offline where possible.

## Related Requirements

- Asynchronous scraping from configurable URLs.
- Structured extraction using LLM schemas, with CSS fallbacks for failures.
- Retries for transient errors, link validation to skip broken URLs.
- Integration with filtering and storage post-extraction.

## Alternatives

- Scrapy + Splash: Powerful for crawling but heavy setup and no built-in LLM extraction.
- BeautifulSoup + httpx: Lightweight and fast for static sites but lacks JS handling and structured LLM output.
- Selenium/Playwright standalone: Good for dynamic content but requires custom parsing and error handling.

## Decision

Adopt Crawl4AI v0.7.2 as the primary framework for its LLM-friendly structured extraction, async crawling, and integrated Playwright for JS rendering. Implement retries with Tenacity v9.1.2 (3 attempts, exponential backoff), fallbacks to CSSExtractionStrategy on LLM errors, and link validation using httpx v0.28.1 (HEAD request with timeout=5, status=200 check). Use dateutil for posted_date parsing with fuzzy=True and defaults like "Unknown" for location on failures.

## Related Decisions

- ADR-002 (Filtering applied post-extraction).
- ADR-003 (Extracted data fed into persistence layer).
- ADR-006 (Error handling complements retries/fallbacks here).

## Design

- **Framework Setup**: In scraper.py, use AsyncWebCrawler with LLMExtractionStrategy (provider="openai/gpt-4o-mini", schema for jobs including location/posted_date). Fallback: try LLM except use CSSExtractionStrategy(css=".job-listing").
- **Retries and Validation**: Decorate extract_jobs with @retry(stop=3, wait=exp). In update_db, async validate_link with httpx.head; Skip if not 200.
- **Post-Processing**: For each job: job["posted_date"] = date_parse(...) or None; job["location"] = ... or "Unknown".
- **Integration**: Gather tasks = [extract_jobs(url, company) for company in active SITES]; Process results with filtering before DB update.
- **Implementation Notes**: Respect robots.txt via delays (asyncio.sleep(1) between tasks). Use OPENAI_API_KEY from env; Fallback if unset.
- **Testing**: @pytest.mark.asyncio async def test_extract_with_fallback(): Simulate LLM fail, assert CSS used and jobs extracted. def test_validation_retry(): Mock failures, assert retries and valid links only processed.

## Consequences

- Robust and efficient scraping (async, structured, fallbacks ensure uptime).
- Handles dynamic sites and errors gracefully (retries/validation prevent bad data).
- Flexible for future extensions (e.g., more schemas).

- Dependency on OpenAI for optimal extraction (mitigated by CSS fallback).
- Minor overhead from retries/delays (necessary for reliability/politeness).

**Changelog:**  

- 1.0 (July 29, 2025): Consolidated scraping tool selection, retries, fallbacks, validation, and post-processing into a single ADR for thematic coherence.
