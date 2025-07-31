"""Alternative scraper using Firecrawl for faster LLM extraction."""

import asyncio
import json

# You'll need: pip install firecrawl-py
from firecrawl import FirecrawlApp

from config import Settings

settings = Settings()


class FirecrawlJobScraper:
    """Fast job scraper using Firecrawl's LLM extraction."""

    def __init__(self):
        if not hasattr(settings, "firecrawl_api_key"):
            raise ValueError("Add FIRECRAWL_API_KEY to your .env file")
        self.app = FirecrawlApp(api_key=settings.firecrawl_api_key)

    async def extract_jobs(self, url: str, company: str) -> list[dict]:
        """Extract jobs using Firecrawl's built-in LLM extraction."""
        # Define extraction schema
        schema = {
            "type": "object",
            "properties": {
                "jobs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "link": {"type": "string"},
                            "location": {"type": "string"},
                            "posted_date": {"type": "string"},
                        },
                        "required": ["title", "description", "link"],
                    },
                }
            },
        }

        try:
            # Firecrawl extract with LLM - much faster than Crawl4AI
            result = self.app.scrape_url(
                url,
                params={
                    "formats": ["extract"],
                    "extract": {
                        "schema": schema,
                        "prompt": f"""
                        Extract job postings from this {company} careers page.
                        Focus on AI, Machine Learning, and Engineering roles.
                        Include the job title, description summary, application link,
                        location, and posted date.
                        """,
                    },
                },
            )

            if result.get("success") and result.get("extract"):
                jobs = result["extract"].get("jobs", [])
                print(f"🚀 Firecrawl extracted {len(jobs)} jobs from {company}")
                return [{"company": company, **job} for job in jobs]
            else:
                print(f"❌ Firecrawl extraction failed for {company}")
                return []

        except Exception as e:
            print(f"❌ Error extracting {company}: {e}")
            return []

    async def scrape_companies(self, companies: list[dict[str, str]]) -> list[dict]:
        """Scrape multiple companies in parallel."""
        tasks = [
            self.extract_jobs(company["url"], company["name"]) for company in companies
        ]

        # Run in parallel for speed
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs = []
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            else:
                print(f"Task failed: {result}")

        return all_jobs


# Example usage
async def main():
    """Run example firecrawl scraping."""
    scraper = FirecrawlJobScraper()

    companies = [
        {"name": "OpenAI", "url": "https://openai.com/careers/"},
        {"name": "Anthropic", "url": "https://www.anthropic.com/careers"},
        {"name": "Meta", "url": "https://www.metacareers.com/jobs/"},
    ]

    jobs = await scraper.scrape_companies(companies)
    print(f"Total jobs found: {len(jobs)}")

    # Save to JSON for testing
    with open("firecrawl_jobs.json", "w") as f:
        json.dump(jobs, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
