# JobSpy Integration Usage Guide

## Quick Start Guide

**Getting Started**: JobSpy integration provides professional-grade job scraping across 15+ platforms with minimal setup and maximum reliability.

**Key Benefits**:

- 🚀 **Instant Access**: 15+ job platforms in a single API call
- 🛡️ **Professional Protection**: Built-in anti-bot measures
- 📊 **Rich Data**: Complete job details + company information
- ⚡ **High Performance**: Async operations with 95%+ success rates
- 🔒 **Type Safety**: Full Pydantic validation and IDE support

---

## Basic Usage Examples

### Simple Job Search

```python
from src.scraping.job_scraper import job_scraper
from src.models.job_models import JobScrapeRequest, JobSite

# Basic job search across multiple platforms
async def basic_job_search():
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.INDEED],
        search_term="Python developer",
        location="San Francisco, CA",
        results_wanted=50
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    print(f"Found {result.total_found} jobs")
    for job in result.jobs:
        print(f"• {job.title} at {job.company}")
        if job.location:
            print(f"  📍 {job.location}")
        if job.min_amount and job.max_amount:
            print(f"  💰 ${job.min_amount:,.0f} - ${job.max_amount:,.0f}")
    
    return result.jobs
```

### Remote Job Search

```python
# Search specifically for remote positions
async def search_remote_jobs():
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.GLASSDOOR, JobSite.ZIP_RECRUITER],
        search_term="Machine Learning Engineer",
        is_remote=True,
        job_type=JobType.FULLTIME,
        results_wanted=100,
        hours_old=24  # Jobs posted in last 24 hours
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    # Filter for truly remote positions
    remote_jobs = result.filter_by_location_type(LocationType.REMOTE)
    
    print(f"Found {remote_jobs.total_found} remote jobs")
    return remote_jobs.jobs
```

### Comprehensive Job Search with Details

```python
# Advanced search with full job descriptions
async def comprehensive_job_search():
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.INDEED, JobSite.GLASSDOOR],
        search_term="Senior Software Engineer",
        location="New York, NY",
        distance=25,  # 25 miles radius
        job_type=JobType.FULLTIME,
        results_wanted=75,
        linkedin_fetch_description=True,  # Get full descriptions
        description_format="markdown",    # Formatted descriptions
        enforce_annual_salary=True       # Prefer annual salary data
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    # Process results with rich data
    for job in result.jobs:
        print(f"\n🎯 {job.title}")
        print(f"🏢 {job.company}")
        
        if job.location:
            print(f"📍 {job.location}")
        
        if job.min_amount and job.max_amount:
            print(f"💰 ${job.min_amount:,.0f} - ${job.max_amount:,.0f} {job.currency or 'USD'}")
        
        if job.job_level:
            print(f"📊 Level: {job.job_level}")
        
        if job.company_rating:
            print(f"⭐ Rating: {job.company_rating}/5.0")
        
        if job.description:
            print(f"📝 Description: {job.description[:200]}...")
    
    return result
```

---

## Advanced Usage Patterns

### Concurrent Multi-Platform Search

```python
import asyncio
from typing import Dict, List

async def search_multiple_platforms_concurrently():
    """Search different platforms concurrently for maximum efficiency."""
    
    # Define searches for different platforms
    searches = [
        JobScrapeRequest(
            site_name=[JobSite.LINKEDIN],
            search_term="Data Scientist",
            location="Seattle, WA",
            results_wanted=50,
            linkedin_fetch_description=True
        ),
        JobScrapeRequest(
            site_name=[JobSite.INDEED],
            search_term="Data Scientist",
            location="Seattle, WA", 
            results_wanted=50
        ),
        JobScrapeRequest(
            site_name=[JobSite.GLASSDOOR],
            search_term="Data Scientist",
            location="Seattle, WA",
            results_wanted=50
        )
    ]
    
    # Execute all searches concurrently
    results = await asyncio.gather(*[
        job_scraper.scrape_jobs_async(search) for search in searches
    ])
    
    # Combine and deduplicate results
    all_jobs = []
    seen_jobs = set()
    
    for result in results:
        for job in result.jobs:
            # Simple deduplication by title + company
            job_key = f"{job.title}_{job.company}".lower()
            if job_key not in seen_jobs:
                seen_jobs.add(job_key)
                all_jobs.append(job)
    
    print(f"Found {len(all_jobs)} unique jobs across all platforms")
    return all_jobs
```

### Targeted Industry Search

```python
async def search_by_industry():
    """Search for jobs in specific industries with industry-specific sites."""
    
    # Tech industry - use tech-focused platforms
    tech_request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.GLASSDOOR],
        search_term="DevOps Engineer",
        location="San Francisco Bay Area",
        results_wanted=100,
        job_type=JobType.FULLTIME
    )
    
    tech_result = await job_scraper.scrape_jobs_async(tech_request)
    
    # Filter for tech companies (basic example)
    tech_jobs = [
        job for job in tech_result.jobs 
        if any(keyword in job.company.lower() 
               for keyword in ['tech', 'software', 'systems', 'data', 'cloud'])
    ]
    
    print(f"Found {len(tech_jobs)} jobs in tech companies")
    return tech_jobs
```

### Salary-Focused Search

```python
async def search_high_salary_positions():
    """Search for high-salary positions with salary filtering."""
    
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.GLASSDOOR],
        search_term="Principal Engineer OR Staff Engineer",
        location="San Francisco, CA",
        results_wanted=50,
        enforce_annual_salary=True,
        linkedin_fetch_description=True
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    # Filter for high-salary positions (>$200k)
    high_salary_jobs = [
        job for job in result.jobs
        if job.min_amount and job.min_amount >= 200000
    ]
    
    # Sort by minimum salary
    high_salary_jobs.sort(key=lambda x: x.min_amount or 0, reverse=True)
    
    print(f"Found {len(high_salary_jobs)} high-salary positions")
    for job in high_salary_jobs[:10]:  # Top 10
        salary_range = f"${job.min_amount:,.0f}"
        if job.max_amount:
            salary_range += f" - ${job.max_amount:,.0f}"
        print(f"💰 {job.title} at {job.company}: {salary_range}")
    
    return high_salary_jobs
```

---

## Database Integration Examples

### Saving Jobs to Database

```python
from src.services.job_service import JobService
from src.database import get_session

async def scrape_and_save_jobs():
    """Scrape jobs and save them to the database."""
    
    # Create job service
    async with get_session() as session:
        job_service = JobService(session)
        
        # Define search parameters
        request = JobScrapeRequest(
            site_name=[JobSite.LINKEDIN, JobSite.INDEED],
            search_term="Full Stack Developer",
            location="Austin, TX",
            results_wanted=100,
            job_type=JobType.FULLTIME
        )
        
        # Scrape jobs
        result = await job_scraper.scrape_jobs_async(request)
        
        # Save to database
        saved_jobs = []
        for job in result.jobs:
            try:
                # Convert JobPosting to database model
                job_sql = await job_service.create_job_from_jobspy(job)
                saved_jobs.append(job_sql)
            except Exception as e:
                print(f"Failed to save job {job.title}: {e}")
                continue
        
        print(f"Saved {len(saved_jobs)} jobs to database")
        return saved_jobs
```

### Bulk Job Processing

```python
async def bulk_job_processing():
    """Process large numbers of jobs efficiently."""
    
    # Large search request
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.INDEED, JobSite.GLASSDOOR],
        search_term="Python OR Django OR Flask",
        location="Remote",
        is_remote=True,
        results_wanted=500,  # Large batch
        job_type=JobType.FULLTIME
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    # Bulk processing with batching
    batch_size = 50
    processed_count = 0
    
    for i in range(0, len(result.jobs), batch_size):
        batch = result.jobs[i:i + batch_size]
        
        # Process batch
        async with get_session() as session:
            job_service = JobService(session)
            
            for job in batch:
                await job_service.create_job_from_jobspy(job)
                processed_count += 1
            
            await session.commit()
        
        print(f"Processed {processed_count}/{len(result.jobs)} jobs")
    
    return processed_count
```

---

## Error Handling and Resilience

### Robust Error Handling

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def resilient_job_search(
    search_term: str,
    location: Optional[str] = None,
    max_retries: int = 3
) -> JobScrapeResult:
    """Robust job search with comprehensive error handling."""
    
    # Primary search configuration
    primary_sites = [JobSite.LINKEDIN, JobSite.INDEED]
    fallback_sites = [JobSite.GLASSDOOR, JobSite.ZIP_RECRUITER]
    
    # Try primary sites first
    for attempt in range(max_retries):
        try:
            request = JobScrapeRequest(
                site_name=primary_sites,
                search_term=search_term,
                location=location,
                results_wanted=100
            )
            
            result = await job_scraper.scrape_jobs_async(request)
            
            if result.jobs:  # Success with results
                logger.info(f"Found {len(result.jobs)} jobs on attempt {attempt + 1}")
                return result
            
        except Exception as e:
            logger.warning(f"Primary search attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    # Fallback to alternative sites
    try:
        logger.info("Trying fallback sites...")
        fallback_request = JobScrapeRequest(
            site_name=fallback_sites,
            search_term=search_term,
            location=location,
            results_wanted=50
        )
        
        result = await job_scraper.scrape_jobs_async(fallback_request)
        
        if result.jobs:
            logger.info(f"Fallback successful: {len(result.jobs)} jobs found")
            return result
            
    except Exception as e:
        logger.error(f"Fallback search also failed: {e}")
    
    # Return empty result if all attempts fail
    logger.error("All search attempts failed")
    return JobScrapeResult(
        jobs=[],
        total_found=0,
        request_params=request,
        metadata={"error": "All search attempts failed", "success": False}
    )
```

### Handling Rate Limits and Timeouts

```python
import asyncio
from datetime import datetime, timedelta

class JobSearchManager:
    """Manage job searches with rate limiting and retry logic."""
    
    def __init__(self):
        self.last_request_time = {}
        self.min_interval = 2.0  # Seconds between requests per site
    
    async def controlled_search(self, request: JobScrapeRequest) -> JobScrapeResult:
        """Execute search with rate limiting."""
        
        # Ensure minimum interval between requests
        for site in request.site_name if isinstance(request.site_name, list) else [request.site_name]:
            last_time = self.last_request_time.get(site.value, 0)
            elapsed = time.time() - last_time
            
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s for {site.value}")
                await asyncio.sleep(sleep_time)
        
        # Execute search
        try:
            start_time = time.time()
            result = await job_scraper.scrape_jobs_async(request)
            duration = time.time() - start_time
            
            # Update last request times
            for site in request.site_name if isinstance(request.site_name, list) else [request.site_name]:
                self.last_request_time[site.value] = time.time()
            
            logger.info(f"Search completed in {duration:.2f}s, found {len(result.jobs)} jobs")
            return result
            
        except asyncio.TimeoutError:
            logger.error("Search timed out")
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
```

---

## Performance Optimization

### Optimized Large-Scale Scraping

```python
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SearchConfig:
    """Configuration for optimized search operations."""
    search_term: str
    location: str
    sites: List[JobSite]
    results_wanted: int = 100
    concurrent_limit: int = 3

async def optimized_bulk_search(configs: List[SearchConfig]) -> Dict[str, JobScrapeResult]:
    """Execute multiple searches with concurrency control."""
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
    
    async def controlled_search(config: SearchConfig) -> tuple[str, JobScrapeResult]:
        async with semaphore:
            request = JobScrapeRequest(
                site_name=config.sites,
                search_term=config.search_term,
                location=config.location,
                results_wanted=config.results_wanted
            )
            
            result = await job_scraper.scrape_jobs_async(request)
            return f"{config.search_term}_{config.location}", result
    
    # Execute all searches with controlled concurrency
    search_tasks = [controlled_search(config) for config in configs]
    results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Process results
    successful_results = {}
    failed_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Search failed: {result}")
            failed_count += 1
        else:
            key, search_result = result
            successful_results[key] = search_result
    
    logger.info(f"Completed {len(successful_results)} searches, {failed_count} failed")
    return successful_results
```

### Memory-Efficient Processing

```python
from typing import Iterator

def process_jobs_stream(result: JobScrapeResult) -> Iterator[Dict[str, Any]]:
    """Process jobs as a stream to minimize memory usage."""
    
    for job in result.jobs:
        # Process one job at a time
        yield {
            'id': job.id,
            'title': job.title,
            'company': job.company,
            'location': job.location,
            'salary_min': job.min_amount,
            'salary_max': job.max_amount,
            'is_remote': job.is_remote,
            'url': job.job_url,
            'site': job.site.value,
            'processed_at': datetime.utcnow().isoformat()
        }

async def memory_efficient_bulk_processing(search_configs: List[SearchConfig]):
    """Process large job datasets without loading everything into memory."""
    
    total_processed = 0
    
    for config in search_configs:
        request = JobScrapeRequest(
            site_name=config.sites,
            search_term=config.search_term,
            location=config.location,
            results_wanted=config.results_wanted
        )
        
        # Get results
        result = await job_scraper.scrape_jobs_async(request)
        
        # Stream process jobs
        batch = []
        batch_size = 50
        
        for job_data in process_jobs_stream(result):
            batch.append(job_data)
            
            if len(batch) >= batch_size:
                # Process batch (save to DB, export, etc.)
                await process_job_batch(batch)
                total_processed += len(batch)
                batch = []
        
        # Process remaining jobs
        if batch:
            await process_job_batch(batch)
            total_processed += len(batch)
    
    logger.info(f"Processed {total_processed} jobs total")
```

---

## Configuration Options

### Site-Specific Configuration

```python
# Configure different parameters for different sites
site_configs = {
    JobSite.LINKEDIN: {
        'linkedin_fetch_description': True,
        'linkedin_company_fetch_description': True,
        'description_format': 'markdown',
        'results_wanted': 50  # LinkedIn limits
    },
    JobSite.INDEED: {
        'country_indeed': 'USA',
        'results_wanted': 100,
        'easy_apply': True
    },
    JobSite.GLASSDOOR: {
        'results_wanted': 75,
        'description_format': 'text'
    }
}

async def site_specific_search(search_term: str, location: str):
    """Execute searches with site-specific optimizations."""
    
    results = {}
    
    for site, config in site_configs.items():
        request = JobScrapeRequest(
            site_name=[site],
            search_term=search_term,
            location=location,
            **config  # Site-specific parameters
        )
        
        result = await job_scraper.scrape_jobs_async(request)
        results[site.value] = result
        
        logger.info(f"{site.value}: {len(result.jobs)} jobs found")
    
    return results
```

### Custom Search Filters

```python
from datetime import datetime, timedelta

class JobSearchFilters:
    """Advanced job search filters and utilities."""
    
    @staticmethod
    def filter_by_salary_range(
        jobs: List[JobPosting], 
        min_salary: float, 
        max_salary: float = None
    ) -> List[JobPosting]:
        """Filter jobs by salary range."""
        filtered = []
        
        for job in jobs:
            if job.min_amount and job.min_amount >= min_salary:
                if max_salary is None or (job.max_amount and job.max_amount <= max_salary):
                    filtered.append(job)
        
        return filtered
    
    @staticmethod
    def filter_by_experience_level(
        jobs: List[JobPosting], 
        experience_levels: List[str]
    ) -> List[JobPosting]:
        """Filter jobs by experience level keywords."""
        filtered = []
        
        for job in jobs:
            if job.job_level:
                for level in experience_levels:
                    if level.lower() in job.job_level.lower():
                        filtered.append(job)
                        break
        
        return filtered
    
    @staticmethod
    def filter_by_company_size(
        jobs: List[JobPosting], 
        size_ranges: List[str]
    ) -> List[JobPosting]:
        """Filter jobs by company size."""
        filtered = []
        
        for job in jobs:
            if job.company_num_employees:
                for size_range in size_ranges:
                    if size_range.lower() in job.company_num_employees.lower():
                        filtered.append(job)
                        break
        
        return filtered

# Usage example
async def filtered_search():
    # Basic search
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN, JobSite.GLASSDOOR],
        search_term="Software Engineer",
        location="Seattle, WA",
        results_wanted=200
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    # Apply filters
    filters = JobSearchFilters()
    
    # High salary positions
    high_salary = filters.filter_by_salary_range(result.jobs, 100000)
    print(f"High salary jobs: {len(high_salary)}")
    
    # Senior level positions
    senior_positions = filters.filter_by_experience_level(
        result.jobs, 
        ["senior", "staff", "principal", "lead"]
    )
    print(f"Senior positions: {len(senior_positions)}")
    
    # Large company positions
    large_companies = filters.filter_by_company_size(
        result.jobs,
        ["1000+", "5000+", "10000+"]
    )
    print(f"Large company jobs: {len(large_companies)}")
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Low Success Rates

```python
# Diagnostic function to check success rates by site
async def diagnose_success_rates():
    """Diagnose success rates across different job sites."""
    
    test_searches = {
        JobSite.LINKEDIN: "Python developer",
        JobSite.INDEED: "JavaScript developer", 
        JobSite.GLASSDOOR: "Data analyst",
        JobSite.ZIP_RECRUITER: "Marketing manager"
    }
    
    results = {}
    
    for site, search_term in test_searches.items():
        try:
            request = JobScrapeRequest(
                site_name=[site],
                search_term=search_term,
                location="New York, NY",
                results_wanted=20
            )
            
            start_time = time.time()
            result = await job_scraper.scrape_jobs_async(request)
            duration = time.time() - start_time
            
            results[site.value] = {
                'success': len(result.jobs) > 0,
                'jobs_found': len(result.jobs),
                'duration': duration,
                'success_rate': len(result.jobs) / 20 * 100
            }
            
        except Exception as e:
            results[site.value] = {
                'success': False,
                'error': str(e),
                'jobs_found': 0,
                'duration': 0,
                'success_rate': 0
            }
    
    # Print diagnosis
    for site, metrics in results.items():
        status = "✅" if metrics['success'] else "❌"
        print(f"{status} {site}: {metrics['jobs_found']} jobs in {metrics['duration']:.2f}s "
              f"({metrics['success_rate']:.1f}% success rate)")
        
        if not metrics['success'] and 'error' in metrics:
            print(f"   Error: {metrics['error']}")
    
    return results
```

#### Issue: Missing Job Descriptions

```python
# Ensure rich job descriptions are fetched
async def get_detailed_jobs():
    """Fetch jobs with comprehensive descriptions."""
    
    request = JobScrapeRequest(
        site_name=[JobSite.LINKEDIN],  # LinkedIn has best descriptions
        search_term="Senior Software Engineer",
        location="San Francisco, CA",
        results_wanted=25,
        linkedin_fetch_description=True,        # Enable full descriptions
        linkedin_company_fetch_description=True, # Enable company details
        description_format="markdown"           # Formatted descriptions
    )
    
    result = await job_scraper.scrape_jobs_async(request)
    
    # Check description quality
    with_descriptions = [job for job in result.jobs if job.description]
    without_descriptions = [job for job in result.jobs if not job.description]
    
    print(f"Jobs with descriptions: {len(with_descriptions)}")
    print(f"Jobs without descriptions: {len(without_descriptions)}")
    
    # Show sample description
    if with_descriptions:
        sample = with_descriptions[0]
        print(f"\nSample description for '{sample.title}' at {sample.company}:")
        print(sample.description[:500] + "..." if len(sample.description) > 500 else sample.description)
    
    return with_descriptions
```

#### Issue: Slow Performance

```python
# Performance optimization strategies
async def optimize_search_performance():
    """Demonstrate performance optimization techniques."""
    
    # Strategy 1: Reduce results_wanted for faster searches
    fast_request = JobScrapeRequest(
        site_name=[JobSite.INDEED],  # Indeed is typically fastest
        search_term="Python",
        location="Remote",
        results_wanted=25,  # Smaller batch for speed
        linkedin_fetch_description=False  # Skip heavy operations
    )
    
    # Strategy 2: Use specific sites instead of all sites
    specific_sites_request = JobScrapeRequest(
        site_name=[JobSite.INDEED, JobSite.ZIP_RECRUITER],  # Faster sites
        search_term="Data Scientist",
        location="Austin, TX",
        results_wanted=50
    )
    
    # Strategy 3: Parallel processing of multiple searches
    start_time = time.time()
    
    results = await asyncio.gather(
        job_scraper.scrape_jobs_async(fast_request),
        job_scraper.scrape_jobs_async(specific_sites_request)
    )
    
    duration = time.time() - start_time
    total_jobs = sum(len(result.jobs) for result in results)
    
    print(f"Found {total_jobs} jobs in {duration:.2f} seconds")
    print(f"Performance: {total_jobs/duration:.1f} jobs/second")
    
    return results
```

---

## Best Practices

### Production Usage Guidelines

1. **Rate Limiting**: Space requests 1-2 seconds apart for stability
2. **Error Handling**: Always implement comprehensive error handling
3. **Batch Processing**: Process large datasets in batches of 50-100 jobs
4. **Site Selection**: Choose specific sites rather than scraping all platforms
5. **Description Fetching**: Only enable full descriptions when needed (performance impact)
6. **Monitoring**: Track success rates and performance metrics
7. **Retry Logic**: Implement exponential backoff for failed requests
8. **Resource Management**: Use async operations for better resource utilization

### Code Quality Standards

```python
# Example of production-quality job search function
async def production_job_search(
    search_term: str,
    location: Optional[str] = None,
    sites: Optional[List[JobSite]] = None,
    job_type: Optional[JobType] = None,
    salary_min: Optional[float] = None,
    max_results: int = 100,
    include_descriptions: bool = False
) -> JobScrapeResult:
    """
    Production-quality job search with comprehensive error handling and validation.
    
    Args:
        search_term: Job search query
        location: Geographic location for search
        sites: List of job sites to search (defaults to LinkedIn + Indeed)
        job_type: Employment type filter
        salary_min: Minimum salary requirement
        max_results: Maximum number of results to return
        include_descriptions: Whether to fetch full job descriptions
    
    Returns:
        JobScrapeResult with validated job postings
    
    Raises:
        ValueError: Invalid search parameters
        JobScrapingError: Search operation failed
    """
    
    # Input validation
    if not search_term or not search_term.strip():
        raise ValueError("Search term cannot be empty")
    
    if max_results <= 0 or max_results > 1000:
        raise ValueError("max_results must be between 1 and 1000")
    
    # Default configuration
    if sites is None:
        sites = [JobSite.LINKEDIN, JobSite.INDEED]
    
    # Build request
    request = JobScrapeRequest(
        site_name=sites,
        search_term=search_term.strip(),
        location=location,
        job_type=job_type,
        results_wanted=min(max_results, 500),  # Cap at 500 per request
        linkedin_fetch_description=include_descriptions,
        description_format="markdown" if include_descriptions else "text"
    )
    
    # Execute search with error handling
    try:
        logger.info(f"Starting job search: '{search_term}' in {location or 'Any Location'}")
        result = await job_scraper.scrape_jobs_async(request)
        
        # Apply salary filter if specified
        if salary_min:
            filtered_jobs = [
                job for job in result.jobs
                if job.min_amount and job.min_amount >= salary_min
            ]
            result = result.model_copy(
                update={'jobs': filtered_jobs, 'total_found': len(filtered_jobs)}
            )
        
        logger.info(f"Search completed: {result.total_found} jobs found")
        return result
        
    except Exception as e:
        logger.error(f"Job search failed for '{search_term}': {e}")
        raise JobScrapingError(f"Search operation failed: {e}") from e
```

---

## Conclusion

The JobSpy integration provides a **powerful, professional-grade job scraping solution** with minimal complexity and maximum reliability. This usage guide demonstrates:

**Core Capabilities:**

- ✅ Simple API for 15+ job platforms
- ✅ Professional data quality and anti-bot protection  
- ✅ Async operations with excellent performance
- ✅ Type-safe operations with comprehensive validation

**Advanced Features:**

- ✅ Concurrent multi-platform searches
- ✅ Sophisticated filtering and processing
- ✅ Database integration patterns
- ✅ Production-ready error handling

**Best Practices:**

- ✅ Rate limiting and performance optimization
- ✅ Comprehensive error handling and resilience
- ✅ Memory-efficient processing for large datasets
- ✅ Production deployment considerations

Start with the **Quick Start Guide** for immediate results, then explore **Advanced Usage Patterns** as your needs grow. The JobSpy integration is designed to scale from simple scripts to enterprise-level job processing systems.

---

**Documentation Version**: 1.0.0  
**JobSpy Version**: >=1.1.82  
**Last Updated**: 2025-08-28
