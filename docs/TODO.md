# AI Job Scraper Optimization TODO

## Executive Summary

Based on comprehensive analysis of Crawl4AI v0.7.2 capabilities, we've identified significant optimization opportunities that can deliver:

- **5-10x speed improvement** (50ms vs 2-10s per extraction)

- **90-95% cost reduction** in LLM API usage

- **Higher reliability** with intelligent fallback strategies

- **Better data quality** with enhanced validation

## Priority Matrix

### 🔴 **URGENT & IMPORTANT** (Do First)

Items that provide immediate impact with minimal risk

### 🟠 **URGENT, NOT IMPORTANT** (Do Next)

Quick wins that improve user experience

### 🟡 **NOT URGENT, IMPORTANT** (Schedule)

Strategic improvements for long-term success

### 🟢 **NOT URGENT, NOT IMPORTANT** (Nice to Have)

Advanced features for optimization

---

## 🔴 PHASE 1: IMMEDIATE WINS (1-2 Days)

### 1.1 Schema Caching System 🔴

**Priority**: Critical | **Effort**: Medium | **Impact**: High

**Problem**: Currently making expensive LLM calls for every company on every scrape run

**Solution**: Generate company-specific extraction schemas once, cache and reuse

**Implementation**:

```python

# Create new file: extraction_cache.py
from pathlib import Path
import json
from typing import Optional, Dict
from datetime import datetime, timedelta

class SchemaCache:
    def __init__(self, cache_dir: str = "./cache/schemas"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=30)  # Schemas expire after 30 days
    
    def get_schema(self, company: str) -> Optional[Dict]:
        """Get cached schema for company if valid"""
        schema_path = self.cache_dir / f"{company.lower()}_schema.json"
        
        if not schema_path.exists():
            return None
            
        try:
            data = json.loads(schema_path.read_text())
            cached_time = datetime.fromisoformat(data['cached_at'])
            
            if datetime.now() - cached_time > self.cache_ttl:
                schema_path.unlink()  # Remove expired cache
                return None
                
            return data['schema']
        except Exception:
            return None
    
    def save_schema(self, company: str, schema: Dict) -> None:
        """Save schema to cache with timestamp"""
        schema_path = self.cache_dir / f"{company.lower()}_schema.json"
        
        cache_data = {
            'company': company,
            'schema': schema,
            'cached_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        schema_path.write_text(json.dumps(cache_data, indent=2))
```

**Integration Steps**:

1. Create `extraction_cache.py` file with SchemaCache class
2. Add cache directory to `.gitignore`: `cache/`
3. Modify `scraper.py` to use schema cache in `extract_jobs()` function
4. Add cache status logging to track hit/miss rates

**Expected Results**: 90% reduction in LLM calls for established companies

---

### 1.2 Optimize LLM Configuration 🔴

**Priority**: Critical | **Effort**: Low | **Impact**: High

**Problem**: Current LLM settings not optimized for job extraction performance/cost

**Solution**: Fine-tune chunking, instructions, and schema for job posting pages

**Implementation**:

```python

# Update in scraper.py - extract_jobs function
OPTIMIZED_JOB_SCHEMA = {
    "jobs": [
        {
            "title": "Job title as posted (3-200 chars)",
            "description": "Brief job summary (50-500 words)",
            "link": "Direct application URL (full URL starting with https://)",
            "location": "Job location or 'Remote' (max 100 chars)", 
            "posted_date": "Posted date in any format",
            "job_type": "Employment type if mentioned (Full-time, Contract, etc.)",
            "experience_level": "Required level if mentioned (Junior, Senior, etc.)"
        }
    ]
}

OPTIMIZED_INSTRUCTIONS = """
Extract job postings from this career page. Focus ONLY on actual job openings.

Requirements:

- Job title: Exact text as posted

- Description: Concise summary (50-100 words max)

- Link: Full application URL

- Location: City/state, country, or "Remote"

- Posted date: Any format (will be parsed later)

SKIP: Company info, news, promotional content, job alerts, general descriptions.
RETURN: Only actual job postings with application links.
"""

# Updated LLM strategy configuration
optimized_strategy = LLMExtractionStrategy(
    provider="openai/gpt-4o-mini",  # Most cost-effective
    api_token=settings.openai_api_key,
    extraction_schema=OPTIMIZED_JOB_SCHEMA,
    instructions=OPTIMIZED_INSTRUCTIONS,
    apply_chunking=True,
    chunk_token_threshold=1200,  # Smaller chunks for job pages
    overlap_rate=0.05,  # Minimal overlap to reduce costs
    input_format="fit_markdown"  # Cleaner input format
)
```

**Integration Steps**:

1. Replace current schema and instructions in `extract_jobs()`
2. Update LLM strategy configuration with optimized parameters
3. Add validation to ensure schema compliance
4. Test with 2-3 companies to verify improved extraction

**Expected Results**: 30-50% reduction in token usage, better extraction quality

---

### 1.3 Basic Extraction Validation 🔴

**Priority**: Critical | **Effort**: Low | **Impact**: Medium

**Problem**: No quality checks on extracted data, allowing poor/empty results

**Solution**: Validate extraction quality before accepting results

**Implementation**:

```python

# Add to scraper.py
def validate_job_extraction(jobs: list, company: str) -> bool:
    """Validate that extraction meets minimum quality standards"""
    if not jobs or len(jobs) < 1:
        logger.warning(f"No jobs extracted for {company}")
        return False
    
    valid_jobs = 0
    required_fields = ['title', 'description', 'link']
    
    for job in jobs:
        # Check required fields exist and have content
        if not all(field in job and str(job[field]).strip() for field in required_fields):
            continue
            
        # Basic quality checks
        title_len = len(str(job['title']).strip())
        desc_len = len(str(job['description']).strip())
        link = str(job['link']).strip()
        
        if (title_len >= 5 and title_len <= 200 and
            desc_len >= 20 and desc_len <= 5000 and
            link.startswith(('http://', 'https://')) and
            company.lower().replace(' ', '') in link.lower().replace(' ', '')):
            valid_jobs += 1
    
    success_rate = valid_jobs / len(jobs)
    min_success_rate = 0.7  # At least 70% of jobs must be valid
    
    if success_rate < min_success_rate:
        logger.warning(f"Low quality extraction for {company}: {valid_jobs}/{len(jobs)} valid ({success_rate:.1%})")
        return False
    
    logger.info(f"✅ Good extraction for {company}: {valid_jobs}/{len(jobs)} valid ({success_rate:.1%})")
    return True

# Update extract_jobs to use validation
async def extract_jobs(url: str, company: str) -> list[dict]:
    # ... existing extraction logic ...
    
    jobs = [{"company": company, **job} for job in jobs if "title" in job]
    
    # Add validation before returning
    if not validate_job_extraction(jobs, company):
        logger.error(f"❌ Extraction validation failed for {company}")
        return []  # Return empty list if validation fails
    
    return jobs
```

**Integration Steps**:

1. Add `validate_job_extraction()` function to `scraper.py`
2. Integrate validation into `extract_jobs()` function
3. Add logging for validation results
4. Monitor validation success rates

**Expected Results**: Eliminate poor quality extractions, improve data reliability

---

### 1.4 Performance Tracking 🟠

**Priority**: High | **Effort**: Low | **Impact**: Medium

**Problem**: No visibility into extraction performance, costs, or success rates

**Solution**: Add comprehensive metrics tracking

**Implementation**:

```python

# Create new file: metrics.py
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path

class ExtractionMetrics:
    def __init__(self, metrics_file: str = "./cache/metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_metrics = {
            'extractions': [],
            'session_start': datetime.now().isoformat(),
            'total_jobs_found': 0,
            'total_companies_processed': 0,
            'strategy_usage': defaultdict(int),
            'success_rates': defaultdict(list),
            'processing_times': defaultdict(list),
            'estimated_costs': 0.0
        }
    
    def track_extraction(self, company: str, strategy: str, job_count: int, 
                        success: bool, processing_time: float) -> None:
        """Track individual extraction attempt"""
        extraction_record = {
            'timestamp': datetime.now().isoformat(),
            'company': company,
            'strategy': strategy,
            'job_count': job_count,
            'success': success,
            'processing_time': processing_time
        }
        
        self.session_metrics['extractions'].append(extraction_record)
        self.session_metrics['strategy_usage'][strategy] += 1
        self.session_metrics['success_rates'][strategy].append(success)
        self.session_metrics['processing_times'][strategy].append(processing_time)
        
        if success:
            self.session_metrics['total_jobs_found'] += job_count
            
        # Estimate costs for LLM strategies
        if 'llm' in strategy.lower():
            estimated_cost = job_count * 0.002  # Rough estimate per job
            self.session_metrics['estimated_costs'] += estimated_cost
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session metrics"""
        total_extractions = len(self.session_metrics['extractions'])
        successful_extractions = sum(1 for e in self.session_metrics['extractions'] if e['success'])
        
        return {
            'session_duration': (datetime.now() - datetime.fromisoformat(self.session_metrics['session_start'])).total_seconds(),
            'total_extractions': total_extractions,
            'successful_extractions': successful_extractions,
            'success_rate': successful_extractions / max(total_extractions, 1),
            'total_jobs_found': self.session_metrics['total_jobs_found'],
            'strategy_breakdown': dict(self.session_metrics['strategy_usage']),
            'estimated_session_cost': self.session_metrics['estimated_costs'],
            'avg_processing_time': sum(e['processing_time'] for e in self.session_metrics['extractions']) / max(total_extractions, 1)
        }
    
    def save_session_metrics(self) -> None:
        """Save session metrics to file"""
        try:
            # Load existing metrics if file exists
            if self.metrics_file.exists():
                historical_data = json.loads(self.metrics_file.read_text())
            else:
                historical_data = {'sessions': []}
            
            # Add current session
            historical_data['sessions'].append(self.session_metrics)
            
            # Keep only last 50 sessions
            historical_data['sessions'] = historical_data['sessions'][-50:]
            
            self.metrics_file.write_text(json.dumps(historical_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

# Usage in scraper.py
metrics = ExtractionMetrics()

async def extract_jobs(url: str, company: str) -> list[dict]:
    start_time = time.time()
    strategy_used = "unknown"
    
    try:
        # ... extraction logic ...
        strategy_used = "llm_primary"  # or "css_fallback", etc.
        processing_time = time.time() - start_time
        
        success = len(jobs) > 0
        metrics.track_extraction(company, strategy_used, len(jobs), success, processing_time)
        
        return jobs
    except Exception as e:
        processing_time = time.time() - start_time
        metrics.track_extraction(company, strategy_used, 0, False, processing_time)
        raise

# Add to main() function
def main():
    try:
        jobs_df = asyncio.run(scrape_all())
        update_db(jobs_df)
        
        # Print session summary
        summary = metrics.get_session_summary()
        logger.info(f"Scraping session completed:")
        logger.info(f"  - Jobs found: {summary['total_jobs_found']}")
        logger.info(f"  - Success rate: {summary['success_rate']:.1%}")
        logger.info(f"  - Estimated cost: ${summary['estimated_session_cost']:.4f}")
        logger.info(f"  - Avg processing time: {summary['avg_processing_time']:.2f}s")
        
        metrics.save_session_metrics()
        
    except Exception as e:
        logger.error(f"Main failed: {e}")
        metrics.save_session_metrics()
```

**Integration Steps**:

1. Create `metrics.py` file with ExtractionMetrics class
2. Add metrics tracking to `extract_jobs()` and `main()` functions
3. Add session summary logging to main()
4. Create cache directory structure

**Expected Results**: Full visibility into performance, costs, and optimization opportunities

---

## 🟠 PHASE 2: STRATEGIC IMPROVEMENTS (3-5 Days)

### 2.1 Multi-Tier Extraction Strategy 🟠

**Priority**: High | **Effort**: High | **Impact**: Very High

**Problem**: Single extraction strategy with basic fallback

**Solution**: Intelligent multi-tier system: cache → generated schema → LLM → regex

**Implementation**:

```python

# Create new file: extraction_strategies.py
from enum import Enum
from typing import Optional, List, Dict, Tuple
import time
import json
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    RegexExtractionStrategy
)

class ExtractionStrategy(Enum):
    CACHED_SCHEMA = "cached_schema"
    GENERATED_SCHEMA = "generated_schema"
    OPTIMIZED_LLM = "optimized_llm"
    REGEX_PATTERNS = "regex_patterns"

class MultiTierExtractor:
    def __init__(self, schema_cache, metrics):
        self.cache = schema_cache
        self.metrics = metrics
        
    async def extract_with_strategy_hierarchy(self, url: str, company: str, crawler) -> Tuple[List[Dict], str]:
        """Try extraction strategies in order of efficiency"""
        
        strategies = [
            (ExtractionStrategy.CACHED_SCHEMA, self._try_cached_schema),
            (ExtractionStrategy.GENERATED_SCHEMA, self._try_generate_schema),
            (ExtractionStrategy.OPTIMIZED_LLM, self._try_optimized_llm),
            (ExtractionStrategy.REGEX_PATTERNS, self._try_regex_fallback)
        ]
        
        for strategy_enum, strategy_func in strategies:
            start_time = time.time()
            strategy_name = strategy_enum.value
            
            try:
                logger.info(f"🔄 Trying {strategy_name} for {company}")
                jobs = await strategy_func(url, company, crawler)
                processing_time = time.time() - start_time
                
                if self._validate_extraction_quality(jobs, company):
                    logger.info(f"✅ Success with {strategy_name} for {company} ({len(jobs)} jobs)")
                    self.metrics.track_extraction(company, strategy_name, len(jobs), True, processing_time)
                    return jobs, strategy_name
                else:
                    logger.warning(f"⚠️ Poor quality from {strategy_name}, trying next strategy")
                    self.metrics.track_extraction(company, strategy_name, len(jobs), False, processing_time)
                    
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"❌ {strategy_name} failed for {company}: {e}")
                self.metrics.track_extraction(company, strategy_name, 0, False, processing_time)
                continue
        
        logger.error(f"❌ All extraction strategies failed for {company}")
        return [], "failed"
    
    async def _try_cached_schema(self, url: str, company: str, crawler) -> List[Dict]:
        """Try using cached company-specific schema (fastest, free)"""
        cached_schema = self.cache.get_schema(company)
        
        if not cached_schema:
            raise Exception("No cached schema available")
        
        strategy = JsonCssExtractionStrategy(cached_schema)
        
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(extraction_strategy=strategy)
        )
        
        if not result.success or not result.extracted_content:
            raise Exception("Cached schema extraction failed")
        
        jobs = json.loads(result.extracted_content)
        return jobs.get('jobs', []) if isinstance(jobs, dict) else jobs
    
    async def _try_generate_schema(self, url: str, company: str, crawler) -> List[Dict]:
        """Generate new schema for company and cache it"""
        
        # First, get page content for schema generation
        result = await crawler.arun(url=url)
        
        if not result.success:
            raise Exception("Failed to fetch page for schema generation")
        
        # Generate schema using LLM (one-time cost)
        schema = await self._generate_company_schema(result.fit_html, company)
        
        if not schema:
            raise Exception("Schema generation failed")
        
        # Cache the generated schema
        self.cache.save_schema(company, schema)
        
        # Use the generated schema for extraction
        strategy = JsonCssExtractionStrategy(schema)
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(extraction_strategy=strategy)
        )
        
        if not result.success:
            raise Exception("Generated schema extraction failed")
        
        jobs = json.loads(result.extracted_content)
        return jobs.get('jobs', []) if isinstance(jobs, dict) else jobs
    
    async def _try_optimized_llm(self, url: str, company: str, crawler) -> List[Dict]:
        """Fallback to optimized LLM extraction"""
        strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o-mini",
            api_token=settings.openai_api_key,
            extraction_schema=OPTIMIZED_JOB_SCHEMA,
            instructions=OPTIMIZED_INSTRUCTIONS,
            apply_chunking=True,
            chunk_token_threshold=1200,
            overlap_rate=0.05,
            input_format="fit_markdown"
        )
        
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(extraction_strategy=strategy)
        )
        
        if not result.success:
            raise Exception("LLM extraction failed")
        
        extracted = json.loads(result.extracted_content)
        return extracted.get("jobs", [])
    
    async def _try_regex_fallback(self, url: str, company: str, crawler) -> List[Dict]:
        """Last resort: regex pattern matching for common job page elements"""
        
        # Get raw page content
        result = await crawler.arun(url=url)
        
        if not result.success:
            raise Exception("Failed to fetch page content")
        
        # Basic regex patterns for job information
        patterns = {
            'job_titles': r'(?i)<[^>]*(?:class|id)[^>]*(?:job|position|role|title)[^>]*>([^<]+)<',
            'locations': r'(?i)(?:location|office|remote|work from home)[:\s]*([^<\n]{5,50})',
            'links': r'(?i)<a[^>]*href=["\']([^"\']*(?:job|career|apply|position)[^"\']*)["\'][^>]*>',
            'dates': r'(?i)(?:posted|published|added)[:\s]*([^<\n]{5,30})'
        }
        
        jobs = []
        html_content = result.html
        
        # Extract potential job information using regex
        import re
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, html_content)
            logger.info(f"Found {len(matches)} potential {pattern_name} for {company}")
        
        # This is a basic fallback - would need more sophisticated logic
        # to combine regex matches into job objects
        
        return jobs  # Would return constructed job objects
    
    async def _generate_company_schema(self, html_content: str, company: str) -> Optional[Dict]:
        """Generate extraction schema for specific company using LLM"""
        
        schema_generation_prompt = f"""
        Analyze this {company} career page and generate an optimal CSS selector schema for extracting job postings.
        
        Focus on:
        1. Job title selectors
        2. Job description/summary selectors  
        3. Application link selectors
        4. Location selectors
        5. Posted date selectors
        
        Return a JSON schema with CSS selectors that can reliably extract job posting data.
        
        Example format:
        {{
            "jobs": {{
                "selector": ".job-item",
                "fields": {{
                    "title": ".job-title",
                    "description": ".job-summary", 
                    "link": "a.apply-link@href",
                    "location": ".job-location",
                    "posted_date": ".job-date"
                }}
            }}
        }}
        """
        
        # Use LLM to generate schema (one-time cost per company)
        schema_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o-mini",
            api_token=settings.openai_api_key,
            instructions=schema_generation_prompt,
            extraction_type="schema"
        )
        
        # This would need actual implementation with Crawl4AI's schema generation features
        # For now, return None to fallback to next strategy
        return None
    
    def _validate_extraction_quality(self, jobs: List[Dict], company: str) -> bool:
        """Validate extraction quality (reuse from Phase 1)"""
        return validate_job_extraction(jobs, company)
```

**Integration Steps**:

1. Create `extraction_strategies.py` with MultiTierExtractor class
2. Refactor `extract_jobs()` to use multi-tier approach
3. Update imports and dependencies
4. Test with various company websites
5. Monitor strategy success rates and optimize order

**Expected Results**:

- 90% of extractions use fast/free cached schemas

- Automatic schema generation for new companies

- Intelligent fallback handling

---

### 2.2 RegexExtractionStrategy Integration 🟠

**Priority**: Medium | **Effort**: Medium | **Impact**: Medium

**Problem**: Missing fast extraction for common patterns (dates, salaries, locations)

**Solution**: Add regex-based extraction for structured data elements

**Implementation**:

```python

# Add to extraction_strategies.py
from crawl4ai.extraction_strategy import RegexExtractionStrategy

class JobRegexPatterns:
    """Common regex patterns for job posting data"""
    
    SALARY_PATTERNS = [
        r'\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s?-\s?\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?(?:\s?(?:per|/)\s?(?:year|yr|annum|annually))?',
        r'\$\s?\d{1,3}(?:,\d{3})*k?(?:\s?-\s?\$?\s?\d{1,3}(?:,\d{3})*k?)?(?:\s?(?:per|/)\s?(?:year|yr|hour|hr))?',
        r'(?:salary|compensation|pay):\s?\$?\d{1,3}(?:,\d{3})*(?:k|000)?(?:\s?-\s?\$?\d{1,3}(?:,\d{3})*(?:k|000)?)?'
    ]
    
    EXPERIENCE_PATTERNS = [
        r'(\d+)[-\s](\d+)?\s*(?:years?\s*)?(?:of\s*)?(?:experience|exp)',
        r'(?:minimum|min|at least)\s+(\d+)\s*(?:years?\s*)?(?:of\s*)?(?:experience|exp)',
        r'(?:experience|exp):\s*(\d+)[-\s](\d+)?\s*(?:years?)?'
    ]
    
    REMOTE_PATTERNS = [
        r'(?i)\b(?:remote|work from home|wfh|telecommute|distributed|anywhere)\b',
        r'(?i)(?:100%\s*)?remote(?:\s*work)?(?:\s*ok(?:ay)?)?',
        r'(?i)(?:fully\s*)?remote(?:\s*position)?'
    ]
    
    EMPLOYMENT_TYPE_PATTERNS = [
        r'(?i)\b(?:full[-\s]?time|fulltime|ft)\b',
        r'(?i)\b(?:part[-\s]?time|parttime|pt)\b', 
        r'(?i)\b(?:contract|contractor|consulting|freelance)\b',
        r'(?i)\b(?:intern|internship|co[-\s]?op)\b',
        r'(?i)\b(?:temporary|temp|seasonal)\b'
    ]
    
    EXPERIENCE_LEVEL_PATTERNS = [
        r'(?i)\b(?:senior|sr|lead|principal|staff|architect)\b',
        r'(?i)\b(?:junior|jr|entry[-\s]?level|associate|new grad)\b',
        r'(?i)\b(?:mid[-\s]?level|intermediate|experienced)\b'
    ]

def create_job_regex_strategy() -> RegexExtractionStrategy:
    """Create regex extraction strategy for job-specific patterns"""
    
    return RegexExtractionStrategy(
        patterns={
            'salaries': JobRegexPatterns.SALARY_PATTERNS,
            'experience_required': JobRegexPatterns.EXPERIENCE_PATTERNS,
            'remote_indicators': JobRegexPatterns.REMOTE_PATTERNS,
            'employment_types': JobRegexPatterns.EMPLOYMENT_TYPE_PATTERNS,
            'experience_levels': JobRegexPatterns.EXPERIENCE_LEVEL_PATTERNS,
            'emails': [RegexExtractionStrategy.Email],
            'urls': [RegexExtractionStrategy.Url],
            'dates': [RegexExtractionStrategy.DateIso, RegexExtractionStrategy.DateUS]
        }
    )

# Enhanced extraction with regex preprocessing
async def extract_with_regex_enhancement(self, url: str, company: str, crawler) -> List[Dict]:
    """Extract jobs with regex pattern enhancement"""
    
    # First extract basic job structure with primary strategy
    jobs = await self._try_primary_extraction(url, company, crawler)
    
    if not jobs:
        return jobs
    
    # Enhance with regex-extracted data
    regex_strategy = create_job_regex_strategy()
    
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(extraction_strategy=regex_strategy)
    )
    
    if result.success and result.extracted_content:
        regex_data = json.loads(result.extracted_content)
        
        # Enhance job objects with regex-extracted patterns
        for job in jobs:
            job['extracted_patterns'] = {}
            
            # Add salary information if found
            if regex_data.get('salaries'):
                job['extracted_patterns']['salary_mentioned'] = True
                job['salary_range'] = regex_data['salaries'][0]  # Take first match
            
            # Add remote work indicators
            if regex_data.get('remote_indicators'):
                job['extracted_patterns']['remote_friendly'] = True
                job['remote_work'] = True
                
            # Add experience level indicators
            if regex_data.get('experience_levels'):
                job['experience_level'] = regex_data['experience_levels'][0]
                
            # Add employment type
            if regex_data.get('employment_types'):
                job['job_type'] = regex_data['employment_types'][0]
    
    return jobs
```

**Integration Steps**:

1. Add JobRegexPatterns class to extraction_strategies.py
2. Create regex strategy factory function
3. Enhance main extraction flow with regex post-processing
4. Add extracted pattern fields to job validation
5. Test regex pattern accuracy with sample job pages

**Expected Results**: Enhanced job data with salary, remote work, and experience level indicators

---

### 2.3 Company-Specific Configuration 🟡

**Priority**: Medium | **Effort**: Medium | **Impact**: High

**Problem**: All companies treated the same despite different website structures

**Solution**: Company-specific extraction configurations and customizations

**Implementation**:

```python

# Create new file: company_configs.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ExtractionDifficulty(Enum):
    EASY = "easy"           # Standard job boards, clear structure
    MEDIUM = "medium"       # Some JavaScript, but accessible
    HARD = "hard"          # Heavy JavaScript, anti-bot measures
    EXPERT = "expert"      # Complex SPAs, authentication required

@dataclass
class CompanyConfig:
    name: str
    difficulty: ExtractionDifficulty
    preferred_strategy: str
    custom_selectors: Optional[Dict[str, str]] = None
    special_instructions: Optional[str] = None
    rate_limit_delay: float = 1.0
    requires_js: bool = False
    auth_required: bool = False
    custom_headers: Optional[Dict[str, str]] = None
    extraction_hints: Optional[List[str]] = None

# Company-specific configurations
COMPANY_CONFIGS = {
    "anthropic": CompanyConfig(
        name="Anthropic",
        difficulty=ExtractionDifficulty.EASY,
        preferred_strategy="llm",
        custom_selectors={
            "job_container": ".job-posting",
            "title": ".job-title",
            "description": ".job-description",
            "location": ".job-location",
            "apply_link": ".apply-button"
        },
        special_instructions="Focus on AI/ML engineering roles, research positions",
        extraction_hints=["Look for 'Research', 'Safety', 'Engineering' in titles"]
    ),
    
    "openai": CompanyConfig(
        name="OpenAI", 
        difficulty=ExtractionDifficulty.MEDIUM,
        preferred_strategy="generated_schema",
        rate_limit_delay=2.0,
        requires_js=True,
        special_instructions="Extract both technical and research positions",
        extraction_hints=["Include 'Applied AI', 'Research', 'Infrastructure' roles"]
    ),
    
    "nvidia": CompanyConfig(
        name="NVIDIA",
        difficulty=ExtractionDifficulty.HARD,
        preferred_strategy="llm",
        rate_limit_delay=3.0,
        requires_js=True,
        custom_headers={"User-Agent": "Mozilla/5.0 (compatible; JobScraper/1.0)"},
        special_instructions="Focus on AI, Deep Learning, CUDA, and Graphics roles",
        extraction_hints=["Filter for 'AI', 'Deep Learning', 'CUDA', 'Graphics', 'Omniverse'"]
    ),
    
    "deepmind": CompanyConfig(
        name="DeepMind",
        difficulty=ExtractionDifficulty.MEDIUM,
        preferred_strategy="cached_schema",
        special_instructions="Extract both research and engineering positions",
        extraction_hints=["Include 'Research Scientist', 'Research Engineer', 'Software Engineer'"]
    ),
    
    "meta": CompanyConfig(
        name="Meta",
        difficulty=ExtractionDifficulty.HARD,
        preferred_strategy="llm",
        rate_limit_delay=2.5,
        requires_js=True,
        special_instructions="Focus on AI, ML, and Reality Labs positions",
        extraction_hints=["Filter for 'AI', 'ML', 'Reality Labs', 'Research', 'Fundamental AI Research'"]
    )
}

class CompanyConfigManager:
    def __init__(self):
        self.configs = COMPANY_CONFIGS
    
    def get_config(self, company_name: str) -> CompanyConfig:
        """Get configuration for specific company"""
        company_key = company_name.lower().replace(" ", "").replace("-", "")
        
        # Try exact match first
        if company_key in self.configs:
            return self.configs[company_key]
        
        # Try partial matches
        for key, config in self.configs.items():
            if key in company_key or company_key in key:
                return config
        
        # Return default config for unknown companies
        return CompanyConfig(
            name=company_name,
            difficulty=ExtractionDifficulty.MEDIUM,
            preferred_strategy="llm",
            special_instructions=f"Extract all relevant job positions from {company_name}"
        )
    
    def should_skip_company(self, company_name: str) -> bool:
        """Check if company should be skipped due to difficulty/restrictions"""
        config = self.get_config(company_name)
        
        # Skip companies that require authentication for now
        return config.auth_required
    
    def get_extraction_delay(self, company_name: str) -> float:
        """Get rate limiting delay for company"""
        config = self.get_config(company_name)
        return config.rate_limit_delay
    
    def get_custom_instructions(self, company_name: str) -> str:
        """Get company-specific extraction instructions"""
        config = self.get_config(company_name)
        
        base_instructions = OPTIMIZED_INSTRUCTIONS
        
        if config.special_instructions:
            return f"{base_instructions}\n\nCompany-specific focus: {config.special_instructions}"
        
        return base_instructions

# Integration with extraction logic
async def extract_jobs_with_company_config(url: str, company: str) -> list[dict]:
    """Extract jobs using company-specific configuration"""
    
    config_manager = CompanyConfigManager()
    config = config_manager.get_config(company)
    
    # Skip if company is too difficult or restricted
    if config_manager.should_skip_company(company):
        logger.warning(f"Skipping {company} due to configuration restrictions")
        return []
    
    # Apply rate limiting
    await asyncio.sleep(config.rate_limit_delay)
    
    # Use company-specific extraction strategy preference
    if config.preferred_strategy == "cached_schema":
        # Try cached schema first
        pass
    elif config.preferred_strategy == "generated_schema":
        # Prefer generated schema
        pass
    elif config.preferred_strategy == "llm":
        # Prefer LLM extraction
        pass
    
    # Apply custom headers if specified
    crawler_config = {}
    if config.custom_headers:
        crawler_config['headers'] = config.custom_headers
    
    # Use custom instructions
    custom_instructions = config_manager.get_custom_instructions(company)
    
    # Continue with extraction using customized parameters...
    return []
```

**Integration Steps**:

1. Create `company_configs.py` with configuration classes
2. Add company-specific configurations for major AI companies
3. Integrate configuration manager into extraction flow
4. Add rate limiting and custom header support
5. Test with different company configurations

**Expected Results**: Higher success rates for difficult companies, respectful scraping practices

---

## 🟡 PHASE 3: ADVANCED FEATURES (1 Week)

### 3.1 Adaptive Schema Learning 🟡

**Priority**: Low | **Effort**: High | **Impact**: High

**Problem**: Static schemas don't adapt when company websites change

**Solution**: Machine learning-based schema adaptation and success tracking

**Implementation**:

```python

# Create new file: adaptive_learning.py
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

class SchemaPerformanceTracker:
    def __init__(self, performance_file: str = "./cache/schema_performance.json"):
        self.performance_file = Path(performance_file)
        self.performance_data = self._load_performance_data()
    
    def _load_performance_data(self) -> Dict:
        """Load historical schema performance data"""
        if self.performance_file.exists():
            return json.loads(self.performance_file.read_text())
        
        return {
            'schema_versions': {},
            'performance_history': {},
            'adaptation_events': []
        }
    
    def track_schema_performance(self, company: str, schema_version: str, 
                               success: bool, job_count: int, quality_score: float) -> None:
        """Track performance of specific schema version"""
        
        if company not in self.performance_data['performance_history']:
            self.performance_data['performance_history'][company] = []
        
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'schema_version': schema_version,
            'success': success,
            'job_count': job_count,
            'quality_score': quality_score
        }
        
        self.performance_data['performance_history'][company].append(performance_record)
        
        # Keep only last 100 records per company
        self.performance_data['performance_history'][company] = \
            self.performance_data['performance_history'][company][-100:]
    
    def should_regenerate_schema(self, company: str) -> bool:
        """Determine if schema should be regenerated based on performance"""
        
        if company not in self.performance_data['performance_history']:
            return False
        
        recent_records = self._get_recent_performance(company, days=7)
        
        if len(recent_records) < 3:  # Need minimum data points
            return False
        
        # Calculate recent success rate
        recent_success_rate = sum(1 for r in recent_records if r['success']) / len(recent_records)
        
        # Calculate recent quality score average
        quality_scores = [r['quality_score'] for r in recent_records if r['success']]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Regenerate if success rate < 60% or quality < 0.7
        should_regenerate = recent_success_rate < 0.6 or avg_quality < 0.7
        
        if should_regenerate:
            logger.info(f"Schema regeneration recommended for {company}: "
                       f"success_rate={recent_success_rate:.1%}, quality={avg_quality:.2f}")
        
        return should_regenerate
    
    def _get_recent_performance(self, company: str, days: int = 7) -> List[Dict]:
        """Get performance records from last N days"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            record for record in self.performance_data['performance_history'].get(company, [])
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]

class AdaptiveSchemaManager:
    def __init__(self, schema_cache, performance_tracker):
        self.cache = schema_cache
        self.tracker = performance_tracker
    
    async def get_optimal_schema(self, company: str, url: str) -> Tuple[Optional[Dict], str]:
        """Get the best schema for company, regenerating if needed"""
        
        # Check if current schema needs regeneration
        if self.tracker.should_regenerate_schema(company):
            logger.info(f"🔄 Regenerating schema for {company} due to poor performance")
            
            # Generate new schema version
            new_schema = await self._generate_improved_schema(company, url)
            
            if new_schema:
                # Version the schema
                version = datetime.now().strftime("%Y%m%d_%H%M")
                
                # Save with version info
                self.cache.save_schema(company, {
                    'schema': new_schema,
                    'version': version,
                    'generated_at': datetime.now().isoformat(),
                    'reason': 'performance_degradation'
                })
                
                return new_schema, version
        
        # Get current cached schema
        cached_data = self.cache.get_schema(company)
        
        if cached_data:
            if isinstance(cached_data, dict) and 'schema' in cached_data:
                return cached_data['schema'], cached_data.get('version', 'unknown')
            else:
                return cached_data, 'legacy'
        
        return None, 'none'
    
    async def _generate_improved_schema(self, company: str, url: str) -> Optional[Dict]:
        """Generate improved schema based on recent failures"""
        
        # Analyze recent failure patterns
        recent_records = self.tracker._get_recent_performance(company, days=14)
        failure_patterns = self._analyze_failure_patterns(recent_records)
        
        # Generate schema with failure analysis context
        improvement_context = f"""
        Previous schema for {company} has been failing. Common issues identified:
        {failure_patterns}
        
        Generate a more robust schema that addresses these specific issues.
        Focus on more specific CSS selectors and alternative extraction paths.
        """
        
        # Use LLM to generate improved schema
        # (Implementation would use Crawl4AI's schema generation with context)
        
        return None  # Placeholder for actual implementation
    
    def _analyze_failure_patterns(self, records: List[Dict]) -> str:
        """Analyze patterns in recent failures"""
        
        failures = [r for r in records if not r['success']]
        
        if not failures:
            return "No recent failures to analyze"
        
        # Analyze failure patterns (this is simplified)
        patterns = []
        
        if len(failures) > len([r for r in records if r['success']]):
            patterns.append("High failure rate indicates major website changes")
        
        low_quality = [r for r in records if r['success'] and r['quality_score'] < 0.5]
        if len(low_quality) > 0:
            patterns.append("Low quality extractions suggest partial website changes")
        
        return "; ".join(patterns) if patterns else "No clear failure patterns identified"

# Integration example
async def extract_with_adaptive_learning(url: str, company: str) -> List[Dict]:
    """Extract jobs with adaptive schema learning"""
    
    performance_tracker = SchemaPerformanceTracker()
    adaptive_manager = AdaptiveSchemaManager(schema_cache, performance_tracker)
    
    # Get optimal schema (may trigger regeneration)
    schema, version = await adaptive_manager.get_optimal_schema(company, url)
    
    if schema:
        # Try extraction with current/regenerated schema
        jobs = await extract_with_schema(url, company, schema)
        
        # Calculate quality score
        quality_score = calculate_extraction_quality(jobs, company)
        
        # Track performance
        performance_tracker.track_schema_performance(
            company, version, len(jobs) > 0, len(jobs), quality_score
        )
        
        return jobs
    
    # Fallback to other strategies
    return await extract_with_fallback_strategies(url, company)

def calculate_extraction_quality(jobs: List[Dict], company: str) -> float:
    """Calculate quality score for extraction (0.0 to 1.0)"""
    
    if not jobs:
        return 0.0
    
    quality_factors = []
    
    for job in jobs:
        job_quality = 0.0
        
        # Check required fields (40% of score)
        required_fields = ['title', 'description', 'link']
        field_score = sum(1 for field in required_fields if job.get(field)) / len(required_fields)
        job_quality += field_score * 0.4
        
        # Check field quality (40% of score)
        if job.get('title'):
            title_quality = min(len(job['title']) / 50, 1.0)  # Prefer longer titles
            job_quality += title_quality * 0.2
        
        if job.get('description'):
            desc_quality = min(len(job['description']) / 100, 1.0)  # Prefer longer descriptions
            job_quality += desc_quality * 0.2
        
        # Check link validity (20% of score)
        if job.get('link'):
            link = job['link']
            if link.startswith(('http://', 'https://')) and company.lower() in link.lower():
                job_quality += 0.2
        
        quality_factors.append(job_quality)
    
    return np.mean(quality_factors) if quality_factors else 0.0
```

**Integration Steps**:

1. Create `adaptive_learning.py` with performance tracking
2. Add quality scoring for extractions
3. Integrate adaptive manager into extraction flow
4. Add schema versioning and performance history
5. Implement failure pattern analysis
6. Test adaptation behavior with simulated failures

**Expected Results**: Self-improving extraction that adapts to website changes automatically

---

### 3.2 Advanced Validation & Quality Assurance 🟡

**Priority**: Low | **Effort**: Medium | **Impact**: Medium

**Problem**: Basic validation doesn't catch subtle quality issues

**Solution**: Multi-dimensional quality scoring and anomaly detection

**Implementation**:

```python

# Create new file: quality_assurance.py
from typing import List, Dict, Tuple
import re
from collections import Counter
from dataclasses import dataclass
import statistics

@dataclass
class QualityMetrics:
    completeness_score: float      # How complete are the required fields
    relevance_score: float         # How relevant are the jobs to AI/ML
    uniqueness_score: float        # How unique/non-duplicate are the jobs
    freshness_score: float         # How recent/fresh are the job postings
    link_validity_score: float     # How valid are the application links
    content_quality_score: float   # How good is the content quality
    overall_score: float           # Combined quality score

class AdvancedQualityValidator:
    def __init__(self):
        self.ai_ml_keywords = {
            'high_relevance': ['artificial intelligence', 'machine learning', 'deep learning', 
                              'neural network', 'computer vision', 'natural language processing',
                              'ai engineer', 'ml engineer', 'data scientist', 'research scientist'],
            'medium_relevance': ['python', 'tensorflow', 'pytorch', 'data analysis', 'algorithm',
                                'statistics', 'data mining', 'predictive modeling'],
            'low_relevance': ['software engineer', 'backend', 'frontend', 'fullstack', 'devops']
        }
        
        self.quality_thresholds = {
            'completeness': 0.8,    # 80% of fields should be complete
            'relevance': 0.6,       # 60% relevance to AI/ML
            'uniqueness': 0.9,      # 90% of jobs should be unique
            'freshness': 0.7,       # 70% should be recent postings
            'link_validity': 0.95,  # 95% of links should be valid
            'content_quality': 0.7, # 70% content quality score
            'overall': 0.75         # 75% overall quality
        }
    
    def validate_extraction_quality(self, jobs: List[Dict], company: str) -> Tuple[bool, QualityMetrics]:
        """Comprehensive quality validation of extracted jobs"""
        
        if not jobs:
            return False, QualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate individual quality metrics
        completeness = self._calculate_completeness_score(jobs)
        relevance = self._calculate_relevance_score(jobs)
        uniqueness = self._calculate_uniqueness_score(jobs)
        freshness = self._calculate_freshness_score(jobs)
        link_validity = self._calculate_link_validity_score(jobs, company)
        content_quality = self._calculate_content_quality_score(jobs)
        
        # Calculate overall score (weighted average)
        overall = (
            completeness * 0.2 +
            relevance * 0.25 +
            uniqueness * 0.15 +
            freshness * 0.1 +
            link_validity * 0.15 +
            content_quality * 0.15
        )
        
        metrics = QualityMetrics(
            completeness, relevance, uniqueness, freshness,
            link_validity, content_quality, overall
        )
        
        # Check if extraction meets quality thresholds
        quality_passed = (
            completeness >= self.quality_thresholds['completeness'] and
            relevance >= self.quality_thresholds['relevance'] and
            uniqueness >= self.quality_thresholds['uniqueness'] and
            link_validity >= self.quality_thresholds['link_validity'] and
            overall >= self.quality_thresholds['overall']
        )
        
        if not quality_passed:
            logger.warning(f"Quality validation failed for {company}:")
            logger.warning(f"  Completeness: {completeness:.2f} (need {self.quality_thresholds['completeness']})")
            logger.warning(f"  Relevance: {relevance:.2f} (need {self.quality_thresholds['relevance']})")
            logger.warning(f"  Uniqueness: {uniqueness:.2f} (need {self.quality_thresholds['uniqueness']})")
            logger.warning(f"  Link Validity: {link_validity:.2f} (need {self.quality_thresholds['link_validity']})")
            logger.warning(f"  Overall: {overall:.2f} (need {self.quality_thresholds['overall']})")
        
        return quality_passed, metrics
    
    def _calculate_completeness_score(self, jobs: List[Dict]) -> float:
        """Calculate how complete the required fields are"""
        
        required_fields = ['title', 'description', 'link']
        optional_fields = ['location', 'posted_date']
        
        completeness_scores = []
        
        for job in jobs:
            required_complete = sum(1 for field in required_fields if job.get(field, '').strip())
            optional_complete = sum(1 for field in optional_fields if job.get(field, '').strip())
            
            # Required fields worth 80%, optional worth 20%
            job_completeness = (
                (required_complete / len(required_fields)) * 0.8 +
                (optional_complete / len(optional_fields)) * 0.2
            )
            
            completeness_scores.append(job_completeness)
        
        return statistics.mean(completeness_scores)
    
    def _calculate_relevance_score(self, jobs: List[Dict]) -> float:
        """Calculate relevance to AI/ML positions"""
        
        relevance_scores = []
        
        for job in jobs:
            title = job.get('title', '').lower()
            description = job.get('description', '').lower()
            
            combined_text = f"{title} {description}"
            
            relevance_score = 0.0
            
            # High relevance keywords (worth more)
            for keyword in self.ai_ml_keywords['high_relevance']:
                if keyword in combined_text:
                    relevance_score += 0.3
            
            # Medium relevance keywords
            for keyword in self.ai_ml_keywords['medium_relevance']:
                if keyword in combined_text:
                    relevance_score += 0.15
            
            # Low relevance keywords (minimal points)
            for keyword in self.ai_ml_keywords['low_relevance']:
                if keyword in combined_text:
                    relevance_score += 0.05
            
            # Cap at 1.0
            relevance_scores.append(min(relevance_score, 1.0))
        
        return statistics.mean(relevance_scores) if relevance_scores else 0.0
    
    def _calculate_uniqueness_score(self, jobs: List[Dict]) -> float:
        """Calculate how unique the job postings are (detect duplicates)"""
        
        if len(jobs) <= 1:
            return 1.0
        
        # Create fingerprints for jobs
        fingerprints = []
        for job in jobs:
            title = job.get('title', '').lower().strip()
            # Create simple fingerprint from first 50 chars of title
            fingerprint = re.sub(r'[^a-z0-9]', '', title)[:50]
            fingerprints.append(fingerprint)
        
        # Count unique fingerprints
        unique_count = len(set(fingerprints))
        total_count = len(fingerprints)
        
        return unique_count / total_count
    
    def _calculate_freshness_score(self, jobs: List[Dict]) -> float:
        """Calculate how fresh/recent the job postings are"""
        
        fresh_indicators = ['today', 'yesterday', '1 day', '2 days', '3 days', 
                           'new', 'just posted', 'recently posted']
        
        old_indicators = ['30+ days', 'over a month', 'expired', '60 days', '90 days']
        
        freshness_scores = []
        
        for job in jobs:
            posted_date = job.get('posted_date', '').lower()
            
            if not posted_date:
                freshness_scores.append(0.5)  # Unknown = neutral
                continue
            
            if any(indicator in posted_date for indicator in fresh_indicators):
                freshness_scores.append(1.0)
            elif any(indicator in posted_date for indicator in old_indicators):
                freshness_scores.append(0.0)
            else:
                # Try to parse relative dates like "5 days ago", "2 weeks ago"
                if 'day' in posted_date:
                    days_match = re.search(r'(\d+)\s*days?', posted_date)
                    if days_match:
                        days = int(days_match.group(1))
                        freshness_scores.append(max(0, 1 - (days / 30)))  # Linear decay over 30 days
                    else:
                        freshness_scores.append(0.7)
                elif 'week' in posted_date:
                    weeks_match = re.search(r'(\d+)\s*weeks?', posted_date)
                    if weeks_match:
                        weeks = int(weeks_match.group(1))
                        freshness_scores.append(max(0, 1 - (weeks / 4)))  # Linear decay over 4 weeks
                    else:
                        freshness_scores.append(0.5)
                else:
                    freshness_scores.append(0.5)  # Unknown format = neutral
        
        return statistics.mean(freshness_scores) if freshness_scores else 0.5
    
    def _calculate_link_validity_score(self, jobs: List[Dict], company: str) -> float:
        """Calculate validity of application links"""
        
        valid_links = 0
        total_links = 0
        
        for job in jobs:
            link = job.get('link', '').strip()
            
            if not link:
                continue
            
            total_links += 1
            
            # Check basic URL format
            if not link.startswith(('http://', 'https://')):
                continue
            
            # Check if link contains company domain or is reasonable job application link
            company_in_link = company.lower().replace(' ', '') in link.lower().replace(' ', '')
            
            # Common job application domains/paths
            job_domains = ['greenhouse.io', 'lever.co', 'workday.com', 'bamboohr.com',
                          'smartrecruiters.com', 'jobvite.com', 'icims.com']
            job_paths = ['/jobs/', '/careers/', '/apply/', '/job/', '/position/']
            
            domain_match = any(domain in link.lower() for domain in job_domains)
            path_match = any(path in link.lower() for path in job_paths)
            
            if company_in_link or domain_match or path_match:
                valid_links += 1
        
        return valid_links / max(total_links, 1)
    
    def _calculate_content_quality_score(self, jobs: List[Dict]) -> float:
        """Calculate content quality based on text characteristics"""
        
        quality_scores = []
        
        for job in jobs:
            title = job.get('title', '').strip()
            description = job.get('description', '').strip()
            
            job_quality = 0.0
            
            # Title quality checks (40% of content quality)
            if title:
                # Good length (5-100 characters)
                title_length_score = min(len(title) / 50, 1.0) if len(title) >= 5 else 0
                job_quality += title_length_score * 0.2
                
                # No excessive capitalization
                if not title.isupper() and title[0].isupper():
                    job_quality += 0.1
                
                # Contains meaningful words (not just company name repeated)
                words = title.lower().split()
                if len(set(words)) >= len(words) * 0.7:  # 70% unique words
                    job_quality += 0.1
            
            # Description quality checks (60% of content quality)
            if description:
                # Good length (50-2000 characters)
                if 50 <= len(description) <= 2000:
                    job_quality += 0.3
                elif len(description) > 2000:
                    job_quality += 0.2  # Too long is still better than too short
                
                # Contains sentences (has periods)
                if '.' in description and len(description.split('.')) >= 2:
                    job_quality += 0.15
                
                # Not just repeated text
                words = description.lower().split()
                unique_words = set(words)
                if len(words) > 0 and len(unique_words) / len(words) > 0.5:
                    job_quality += 0.15
            
            quality_scores.append(job_quality)
        
        return statistics.mean(quality_scores) if quality_scores else 0.0

# Enhanced extraction with advanced validation
async def extract_with_quality_assurance(url: str, company: str) -> Tuple[List[Dict], QualityMetrics]:
    """Extract jobs with comprehensive quality validation"""
    
    validator = AdvancedQualityValidator()
    
    # Extract jobs using existing strategies
    jobs = await extract_jobs_basic(url, company)
    
    # Validate quality
    quality_passed, metrics = validator.validate_extraction_quality(jobs, company)
    
    if not quality_passed:
        logger.warning(f"❌ Quality validation failed for {company}")
        
        # Option 1: Try different extraction strategy
        # jobs = await try_alternative_extraction_strategy(url, company)
        
        # Option 2: Return empty list if quality is too poor
        # return [], metrics
        
        # Option 3: Continue with warnings (current approach)
        logger.warning(f"⚠️ Continuing with poor quality extraction for {company}")
    
    return jobs, metrics
```

**Integration Steps**:

1. Create `quality_assurance.py` with advanced validation
2. Add quality metrics tracking to existing extraction flow
3. Integrate quality thresholds and scoring
4. Add quality-based retry logic
5. Create quality reports and monitoring
6. Test quality validation with various extraction results

**Expected Results**: Higher data quality, automatic detection of extraction degradation

---

### 3.3 Performance Monitoring Dashboard 🟢

**Priority**: Low | **Effort**: Medium | **Impact**: Low

**Problem**: No easy way to monitor scraper performance and costs

**Solution**: Simple dashboard for tracking metrics and performance

**Implementation**:

```python

# Create new file: dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

class ScrapingDashboard:
    def __init__(self, metrics_file: str = "./cache/metrics.json"):
        self.metrics_file = Path(metrics_file)
    
    def load_metrics_data(self) -> Dict:
        """Load historical metrics data"""
        if self.metrics_file.exists():
            return json.loads(self.metrics_file.read_text())
        return {'sessions': []}
    
    def create_dashboard(self):
        """Create Streamlit dashboard for scraping metrics"""
        
        st.title("🤖 AI Job Scraper Performance Dashboard")
        
        # Load data
        data = self.load_metrics_data()
        sessions = data.get('sessions', [])
        
        if not sessions:
            st.warning("No scraping data available yet. Run the scraper first!")
            return
        
        # Create metrics overview
        self.create_overview_section(sessions)
        
        # Create performance charts
        self.create_performance_charts(sessions)
        
        # Create company-specific analysis
        self.create_company_analysis(sessions)
        
        # Create cost analysis
        self.create_cost_analysis(sessions)
    
    def create_overview_section(self, sessions: List[Dict]):
        """Create overview metrics section"""
        
        st.header("📊 Overview")
        
        # Latest session stats
        latest_session = sessions[-1] if sessions else {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_jobs = sum(s.get('total_jobs_found', 0) for s in sessions[-7:])  # Last 7 sessions
            st.metric("Jobs Found (Last 7 Runs)", total_jobs)
        
        with col2:
            latest_success_rate = latest_session.get('success_rate', 0)
            st.metric("Latest Success Rate", f"{latest_success_rate:.1%}")
        
        with col3:
            latest_cost = latest_session.get('estimated_session_cost', 0)
            st.metric("Latest Session Cost", f"${latest_cost:.4f}")
        
        with col4:
            avg_time = latest_session.get('avg_processing_time', 0)
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
    
    def create_performance_charts(self, sessions: List[Dict]):
        """Create performance trend charts"""
        
        st.header("📈 Performance Trends")
        
        # Prepare data for charts
        df_sessions = pd.DataFrame([
            {
                'session': i,
                'date': s.get('session_start', datetime.now().isoformat())[:10],
                'jobs_found': s.get('total_jobs_found', 0),
                'success_rate': s.get('success_rate', 0),
                'cost': s.get('estimated_session_cost', 0),
                'avg_time': s.get('avg_processing_time', 0)
            }
            for i, s in enumerate(sessions[-30:])  # Last 30 sessions
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Jobs found over time
            fig_jobs = px.line(df_sessions, x='date', y='jobs_found', 
                              title='Jobs Found Over Time',
                              labels={'jobs_found': 'Jobs Found', 'date': 'Date'})
            st.plotly_chart(fig_jobs, use_container_width=True)
        
        with col2:
            # Success rate over time
            fig_success = px.line(df_sessions, x='date', y='success_rate',
                                 title='Success Rate Over Time',
                                 labels={'success_rate': 'Success Rate', 'date': 'Date'})
            fig_success.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig_success, use_container_width=True)
        
        # Cost and time analysis
        col3, col4 = st.columns(2)
        
        with col3:
            fig_cost = px.bar(df_sessions, x='date', y='cost',
                             title='Cost Per Session',
                             labels={'cost': 'Cost ($)', 'date': 'Date'})
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col4:
            fig_time = px.line(df_sessions, x='date', y='avg_time',
                              title='Average Processing Time',
                              labels={'avg_time': 'Avg Time (seconds)', 'date': 'Date'})
            st.plotly_chart(fig_time, use_container_width=True)
    
    def create_company_analysis(self, sessions: List[Dict]):
        """Create company-specific performance analysis"""
        
        st.header("🏢 Company Performance Analysis")
        
        # Aggregate company data from all sessions
        company_stats = {}
        
        for session in sessions[-10:]:  # Last 10 sessions
            for extraction in session.get('extractions', []):
                company = extraction['company']
                
                if company not in company_stats:
                    company_stats[company] = {
                        'total_attempts': 0,
                        'successful_attempts': 0,
                        'total_jobs': 0,
                        'total_time': 0,
                        'strategies_used': []
                    }
                
                stats = company_stats[company]
                stats['total_attempts'] += 1
                
                if extraction['success']:
                    stats['successful_attempts'] += 1
                    stats['total_jobs'] += extraction['job_count']
                
                stats['total_time'] += extraction['processing_time']
                stats['strategies_used'].append(extraction.get('strategy', 'unknown'))
        
        # Create company performance DataFrame
        company_data = []
        for company, stats in company_stats.items():
            success_rate = stats['successful_attempts'] / max(stats['total_attempts'], 1)
            avg_time = stats['total_time'] / max(stats['total_attempts'], 1)
            avg_jobs = stats['total_jobs'] / max(stats['successful_attempts'], 1)
            
            company_data.append({
                'Company': company,
                'Success Rate': success_rate,
                'Avg Jobs Found': avg_jobs,
                'Avg Time (s)': avg_time,
                'Total Attempts': stats['total_attempts']
            })
        
        df_companies = pd.DataFrame(company_data)
        
        if not df_companies.empty:
            # Company success rate chart
            fig_company_success = px.bar(df_companies, x='Company', y='Success Rate',
                                        title='Success Rate by Company',
                                        color='Success Rate',
                                        color_continuous_scale='RdYlGn')
            fig_company_success.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig_company_success, use_container_width=True)
            
            # Company performance table
            st.subheader("Company Performance Summary")
            st.dataframe(df_companies.round(2), use_container_width=True)
    
    def create_cost_analysis(self, sessions: List[Dict]):
        """Create cost analysis section"""
        
        st.header("💰 Cost Analysis")
        
        # Strategy cost breakdown
        strategy_costs = {}
        total_extractions = 0
        
        for session in sessions[-10:]:  # Last 10 sessions
            for extraction in session.get('extractions', []):
                strategy = extraction.get('strategy', 'unknown')
                
                if strategy not in strategy_costs:
                    strategy_costs[strategy] = {'count': 0, 'estimated_cost': 0}
                
                strategy_costs[strategy]['count'] += 1
                
                # Estimate cost based on strategy
                if 'llm' in strategy.lower():
                    strategy_costs[strategy]['estimated_cost'] += extraction.get('job_count', 0) * 0.002
                
                total_extractions += 1
        
        # Create strategy breakdown chart
        if strategy_costs:
            strategy_data = []
            for strategy, data in strategy_costs.items():
                strategy_data.append({
                    'Strategy': strategy,
                    'Usage Count': data['count'],
                    'Usage %': data['count'] / max(total_extractions, 1) * 100,
                    'Estimated Cost': data['estimated_cost']
                })
            
            df_strategies = pd.DataFrame(strategy_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_strategy_usage = px.pie(df_strategies, values='Usage Count', names='Strategy',
                                           title='Strategy Usage Distribution')
                st.plotly_chart(fig_strategy_usage, use_container_width=True)
            
            with col2:
                fig_strategy_cost = px.bar(df_strategies, x='Strategy', y='Estimated Cost',
                                          title='Estimated Cost by Strategy')
                st.plotly_chart(fig_strategy_cost, use_container_width=True)
            
            # Cost savings calculation
            st.subheader("Cost Optimization Impact")
            
            total_cost = sum(data['estimated_cost'] for data in strategy_costs.values())
            llm_only_cost = total_extractions * 0.005  # Estimated cost if all LLM
            
            savings = llm_only_cost - total_cost
            savings_percent = (savings / max(llm_only_cost, 0.001)) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Actual Cost", f"${total_cost:.4f}")
            
            with col2:
                st.metric("LLM-Only Cost", f"${llm_only_cost:.4f}")
            
            with col3:
                st.metric("Savings", f"${savings:.4f} ({savings_percent:.1f}%)")

# Add to main application
def run_dashboard():
    """Run the Streamlit dashboard"""
    dashboard = ScrapingDashboard()
    dashboard.create_dashboard()

if __name__ == "__main__":
    run_dashboard()
```

**Integration Steps**:

1. Create `dashboard.py` with Streamlit dashboard
2. Add plotly and dashboard dependencies to pyproject.toml
3. Create dashboard launch script or add to main app
4. Add dashboard route to existing Streamlit app
5. Test dashboard with sample metrics data

**Expected Results**: Visual monitoring of scraper performance and cost optimization

---

## 🔧 IMPLEMENTATION GUIDELINES

### Dependencies to Add

```toml

# Add to pyproject.toml
dependencies = [
    # ... existing dependencies ...
    "plotly>=5.15.0",           # For dashboard charts
    "numpy>=1.24.0",            # For statistical calculations  
    "scikit-learn>=1.3.0",      # For adaptive learning (Phase 3)
]
```

### Directory Structure

```text
ai-job-scraper/
├── cache/
│   ├── schemas/               # Company-specific extraction schemas
│   ├── metrics.json          # Performance metrics history
│   └── schema_performance.json # Schema performance tracking
├── extraction_cache.py       # Schema caching system
├── extraction_strategies.py  # Multi-tier extraction strategies
├── metrics.py                # Performance tracking
├── company_configs.py        # Company-specific configurations
├── quality_assurance.py      # Advanced validation
├── adaptive_learning.py      # Schema adaptation (Phase 3)
├── dashboard.py              # Performance dashboard (Phase 3)
└── scraper.py                # Updated main scraper
```

### Testing Strategy

1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test complete extraction workflows
3. **Performance Tests**: Measure speed and cost improvements
4. **Quality Tests**: Validate extraction quality metrics
5. **Regression Tests**: Ensure changes don't break existing functionality

### Monitoring & Alerts

- Set up alerts for extraction success rate drops below 70%

- Monitor cost increases beyond expected thresholds

- Track schema regeneration frequency

- Alert on quality score degradation

### Rollback Plan

- Keep current `extract_jobs()` function as `extract_jobs_legacy()`

- Implement feature flags for gradual rollout

- Monitor extraction comparison between old and new methods

- Quick rollback option if issues detected

---

## 📅 IMPLEMENTATION TIMELINE

### Week 1: Foundation (Phase 1)

- Days 1-2: Schema caching system and LLM optimization

- Days 3-4: Basic validation and performance tracking  

- Day 5: Testing and integration

### Week 2: Strategic Improvements (Phase 2)

- Days 1-3: Multi-tier extraction strategy

- Days 4-5: Company-specific configurations and regex patterns

### Week 3: Advanced Features (Phase 3)

- Days 1-3: Adaptive learning system

- Days 4-5: Advanced validation and dashboard

### Week 4: Testing & Optimization

- Days 1-2: Comprehensive testing

- Days 3-4: Performance optimization

- Day 5: Documentation and deployment

---

## 🎯 SUCCESS METRICS

### Performance Targets

- **Speed**: 5-10x improvement for cached extractions

- **Cost**: 90%+ reduction in LLM API costs

- **Reliability**: 95%+ success rate across all companies

- **Quality**: 85%+ average quality score

### Key Performance Indicators (KPIs)

1. **Extraction Success Rate**: % of successful extractions per company
2. **Cost Per Job**: Average cost to extract one job posting
3. **Quality Score**: Multi-dimensional quality rating (0-1.0)
4. **Cache Hit Rate**: % of extractions using cached schemas
5. **Processing Time**: Average time per company extraction
6. **Schema Regeneration Rate**: How often schemas need updates

---

This comprehensive TODO provides detailed implementation guidance for transforming your AI Job Scraper into a highly optimized, cost-effective, and intelligent extraction system. Each task includes specific code examples, integration steps, and expected results to ensure successful implementation.
