# JobSpy Scraping Integration Compliance Guidelines

## Document Metadata

**Version:** 1.0  
**Date:** 2025-08-28  
**Phase:** 3 - JobSpy Integration  
**Status:** Active  
**Related ADR:** [ADR-015: Proxy Anti-Bot Integration](../../adrs/ADR-015-proxy-anti-bot-integration-2025.md)

## Overview

This document establishes comprehensive compliance guidelines for the JobSpy scraping integration, ensuring ethical and legal scraping practices while maintaining 95%+ success rates through strategic proxy usage. These guidelines implement the frameworks established in **ADR-015** and support the 2-tier scraping architecture.

## 1. JobSpy Built-in Compliance Features

### 1.1 Robots.txt Respect Mechanisms

JobSpy automatically respects robots.txt directives for supported sites:

```python
# Automatic robots.txt compliance (built-in)
from jobspy import scrape_jobs

jobs_df = scrape_jobs(
    site_name=["linkedin", "indeed", "glassdoor"],
    search_term="software engineer",
    respect_robots=True,  # Default: True (automatic)
    robots_txt_check=True  # Validates before scraping
)
```

**Implementation Requirements:**

- Never override `respect_robots=False` unless explicitly approved
- Monitor robots.txt changes through automated validation
- Log robots.txt violations for compliance auditing

### 1.2 Rate Limiting and Anti-Bot Protection

JobSpy includes built-in protective mechanisms:

```python
jobs_df = scrape_jobs(
    site_name=["linkedin", "indeed"],
    search_term="data scientist",
    random_delay=True,        # MANDATORY: 1-5 second delays
    max_workers=3,           # MANDATORY: Conservative concurrency
    delay_min=1,             # Minimum delay between requests
    delay_max=5,             # Maximum delay between requests
    timeout=30               # Request timeout protection
)
```

**Compliance Standards:**

- `random_delay=True` is **mandatory** for all scraping operations
- `max_workers` must not exceed 3 for stability and courtesy
- Never disable timeout protection
- Implement exponential backoff for rate limit responses

### 1.3 Professional User Agent Headers

JobSpy manages professional user agent rotation:

```python
# Professional user agent management (automatic)
jobs_df = scrape_jobs(
    site_name=["glassdoor"],
    search_term="machine learning",
    user_agent="rotate",     # Professional user agent rotation
    headers_rotation=True    # Vary request headers naturally
)
```

**Requirements:**

- Use only professional, realistic user agents
- Never use user agents that misrepresent identity
- Rotate headers to mimic natural browsing patterns

### 1.4 Session Management and Cookie Handling

```python
# Built-in session management
jobs_df = scrape_jobs(
    site_name=["indeed"],
    search_term="python developer",
    session_management=True,  # Maintain consistent sessions
    cookie_handling=True,     # Handle authentication cookies
    persist_session=False     # Don't persist across runs
)
```

**Compliance Notes:**

- Sessions are automatically managed per scraping operation
- Cookies are handled transparently without storing personal data
- No session persistence across different scraping operations

## 2. ADR-015 IPRoyal Proxy Integration

### 2.1 Monthly Budget Limits

**Budget Framework:**

- **Target Budget:** $15-25/month
- **Alert Threshold:** 80% of monthly budget ($12-20)
- **Hard Limit:** 100% of monthly budget (automatic cutoff)
- **Priority Allocation:** High-value companies always receive proxy protection

```python
# Cost-controlled proxy integration
class IPRoyalProxyManager:
    def __init__(self, monthly_budget: float = 20.0):
        self.monthly_budget = monthly_budget
        self.usage_tracker = ProxyUsageTracker()
        
    def should_use_proxy(self, company: str, attempt: int = 1) -> bool:
        """Strategic proxy usage with budget awareness."""
        monthly_usage = self.usage_tracker.get_monthly_cost()
        
        # High-priority companies always use proxy
        high_priority = ["google", "microsoft", "amazon", "apple", "tesla", "meta", "netflix"]
        if company.lower() in high_priority:
            return True
            
        # Standard companies use proxy after failure if budget allows
        if attempt > 1 and monthly_usage < self.monthly_budget * 0.8:
            return True
            
        return False
```

### 2.2 Residential Proxy Usage for Legitimate Access

**IPRoyal Configuration:**

- **Proxy Type:** Residential endpoints only (highest success rate)
- **Rotation:** Automatic endpoint rotation every 10 minutes
- **Geographic Distribution:** US, UK, CA for natural traffic patterns
- **Connection Pooling:** Maximum 3 concurrent connections

```env
# IPRoyal residential proxy configuration
IPROYAL_USERNAME="your-username-here"
IPROYAL_PASSWORD="your-password-here" 
IPROYAL_ENDPOINT="rotating-residential.iproyal.com:12321"
PROXY_POOL_SIZE=3
PROXY_ROTATION_INTERVAL=600  # 10 minutes
ALLOWED_COUNTRIES="US,UK,CA"
```

### 2.3 Usage Tracking and Budget Alerts

**Automated Monitoring:**

```python
class ProxyUsageTracker:
    def track_usage(self, company: str, proxy_used: str, cost: float, 
                   success: bool = True, response_time: float = None):
        """Track all proxy usage for cost control."""
        usage = ProxyUsage(
            company=company,
            proxy_endpoint=proxy_used,
            cost=cost,  # ~$0.05 per successful scrape
            timestamp=datetime.utcnow(),
            success=success,
            response_time=response_time
        )
        self.db_session.add(usage)
        self.db_session.commit()
        
        # Check budget threshold
        if self.get_monthly_cost() > self.monthly_budget * 0.8:
            self.send_budget_alert()
```

**Alert Configuration:**

- 80% budget threshold triggers warning notification
- 95% budget threshold triggers final warning
- 100% budget threshold disables proxy usage for standard companies

### 2.4 Priority-Based Allocation

**High-Value Company List:**

- Tier 1: Google, Microsoft, Amazon, Apple, Tesla, Meta, Netflix
- Tier 2: Configurable via `HIGH_PRIORITY_COMPANIES` environment variable
- Standard Companies: Proxy usage only after direct scraping failures

```python
# Priority-based proxy allocation
def get_company_priority(company: str) -> str:
    """Determine company priority for proxy allocation."""
    high_priority = os.getenv('HIGH_PRIORITY_COMPANIES', '').lower().split(',')
    
    if company.lower() in high_priority:
        return "high"
    elif company.lower() in FORTUNE_500_LIST:
        return "medium" 
    else:
        return "standard"
```

## 3. Legal Considerations Matrix

### 3.1 LinkedIn Professional Scraping Practices

**Compliance Requirements:**

- **Public Data Only:** Only scrape publicly available job postings
- **Rate Limiting:** Maximum 1 request per 3 seconds
- **Respect Terms:** Honor robots.txt and User Agent policies
- **No Authentication:** Never scrape behind login walls
- **Attribution:** Properly attribute data source in applications

```python
# LinkedIn-specific compliance settings
linkedin_jobs = scrape_jobs(
    site_name=["linkedin"],
    search_term="software engineer",
    random_delay=True,
    delay_min=3,           # Conservative 3-second minimum
    delay_max=8,           # Extended maximum delay
    max_workers=1,         # Single worker for LinkedIn
    results_wanted=25      # Conservative result limits
)
```

**Legal References:**

- [LinkedIn Terms of Service](https://www.linkedin.com/legal/user-agreement)
- Acceptable use for publicly available job data
- No automated account creation or credential usage

### 3.2 Indeed Rate Limiting Compliance

**Implementation Standards:**

- **Request Frequency:** Maximum 10 requests per minute
- **Robots.txt Adherence:** Automatic compliance with crawl-delay
- **User Agent:** Professional identification
- **Content Limits:** Respect pagination and result limits

```python
# Indeed-specific compliance
indeed_jobs = scrape_jobs(
    site_name=["indeed"],
    search_term="data analyst",
    random_delay=True,
    delay_min=6,           # Indeed crawl-delay compliance
    max_workers=2,         # Conservative concurrency
    results_wanted=50,     # Reasonable result limits
    country_indeed="USA"   # Explicit country specification
)
```

### 3.3 Glassdoor Conservative Approach

**Restricted Access Policy:**

- **Minimal Requests:** Only when explicitly requested by users
- **Extended Delays:** 5-10 second delays between requests
- **Single Worker:** No concurrent requests
- **Limited Results:** Maximum 20 results per search

```python
# Glassdoor conservative scraping
glassdoor_jobs = scrape_jobs(
    site_name=["glassdoor"],
    search_term="product manager",
    random_delay=True,
    delay_min=5,           # Extended courtesy delay
    delay_max=10,          # Conservative maximum
    max_workers=1,         # Single worker only
    results_wanted=20,     # Limited results
    glassdoor_warnings=False  # Handle warnings gracefully
)
```

### 3.4 Site-Specific Terms of Service Compliance

| Job Board | Terms of Service | Compliance Link | Scraping Policy |
|-----------|-----------------|-----------------|-----------------|
| LinkedIn | [User Agreement](https://www.linkedin.com/legal/user-agreement) | Section 8.2 - Automated processing | Public data only, rate limited |
| Indeed | [Terms of Service](https://www.indeed.com/legal/terms-of-service) | Section 6 - Prohibited conduct | Respect robots.txt, professional use |
| Glassdoor | [Community Guidelines](https://www.glassdoor.com/community-guidelines/) | Fair use policy | Conservative approach, minimal requests |
| ZipRecruiter | [Terms of Use](https://www.ziprecruiter.com/terms) | Section 4 - User conduct | Standard compliance, rate limiting |

## 4. Operational Controls

### 4.1 Default Safe Mode Settings

**Safe Mode Configuration:**

```python
# Default safe mode for all scraping operations
SAFE_MODE_DEFAULTS = {
    "random_delay": True,
    "delay_min": 2,
    "delay_max": 5, 
    "max_workers": 3,
    "respect_robots": True,
    "timeout": 30,
    "retry_attempts": 2,
    "backoff_factor": 2.0
}

def scrape_with_safe_defaults(site_name: list, search_term: str, **kwargs):
    """Apply safe mode defaults to all scraping operations."""
    config = SAFE_MODE_DEFAULTS.copy()
    config.update(kwargs)  # Allow overrides for approved cases
    
    return scrape_jobs(
        site_name=site_name,
        search_term=search_term,
        **config
    )
```

### 4.2 Environment Variables for Site-Specific Controls

**Configuration Management:**

```env
# Site-specific compliance controls
LINKEDIN_MAX_WORKERS=1
LINKEDIN_DELAY_MIN=3
LINKEDIN_DELAY_MAX=8
LINKEDIN_RESULTS_LIMIT=25

INDEED_MAX_WORKERS=2  
INDEED_DELAY_MIN=6
INDEED_RESULTS_LIMIT=50

GLASSDOOR_MAX_WORKERS=1
GLASSDOOR_DELAY_MIN=5
GLASSDOOR_DELAY_MAX=10
GLASSDOOR_RESULTS_LIMIT=20

# Global compliance settings
RESPECT_ROBOTS_TXT=true
ENABLE_RATE_LIMITING=true
PROFESSIONAL_USER_AGENTS=true
SESSION_PERSISTENCE=false
```

### 4.3 Explicit Opt-in for Aggressive Scraping

**Aggressive Mode Requirements:**

- Explicit user consent through UI confirmation
- Enhanced monitoring and logging
- Automatic fallback to safe mode on detection
- Compliance officer notification for enterprise use

```python
class AggressiveScrapingMode:
    def __init__(self, user_consent: bool = False):
        if not user_consent:
            raise ComplianceError("Aggressive mode requires explicit user consent")
        
        self.monitoring_enabled = True
        self.fallback_threshold = 3  # Failures before safe mode
        
    def scrape_aggressive(self, site_name: list, search_term: str):
        """Aggressive scraping with enhanced monitoring."""
        try:
            return scrape_jobs(
                site_name=site_name,
                search_term=search_term,
                random_delay=True,
                delay_min=1,        # Reduced delays
                delay_max=3,
                max_workers=5,      # Increased concurrency  
                results_wanted=100  # Higher result limits
            )
        except Exception as e:
            logger.warning(f"Aggressive mode failed: {e}")
            return self.fallback_to_safe_mode(site_name, search_term)
```

### 4.4 Automatic Throttling and Backoff Mechanisms

**Intelligent Response Handling:**

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry_error_callback=lambda retry_state: None
)
def scrape_with_backoff(site_name: list, search_term: str):
    """Scraping with automatic exponential backoff."""
    try:
        return scrape_jobs(
            site_name=site_name,
            search_term=search_term,
            **SAFE_MODE_DEFAULTS
        )
    except Exception as e:
        if "rate limit" in str(e).lower():
            # Extend delay for rate limit responses
            time.sleep(30)
            raise
        elif "blocked" in str(e).lower():
            # Switch to proxy mode if available
            return scrape_with_proxy_protection(site_name, search_term)
        else:
            raise
```

## 5. Runtime Policy Framework

### 5.1 Never Scrape Behind Authentication

**Authentication Policy:**

- **No Login Scraping:** Never attempt to scrape authenticated content
- **Public Data Only:** Limit scraping to publicly accessible job postings
- **Session Boundaries:** Respect login walls and subscription barriers
- **API Preference:** Use official APIs when available

```python
def validate_public_access(url: str) -> bool:
    """Ensure URL points to publicly accessible content."""
    protected_patterns = [
        "/login",
        "/signin", 
        "/account",
        "/premium",
        "/membership",
        "login_required=true"
    ]
    
    return not any(pattern in url.lower() for pattern in protected_patterns)

# Pre-scraping validation
def scrape_public_jobs_only(site_name: list, search_term: str):
    """Ensure only public job data is scraped."""
    if not validate_public_access_for_sites(site_name):
        raise ComplianceError("Cannot scrape authenticated content")
        
    return scrape_jobs(site_name=site_name, search_term=search_term)
```

### 5.2 Honor Robots.txt at All Times

**Robots.txt Implementation:**

```python
import robotparser

class RobotsTxtValidator:
    def __init__(self):
        self.robots_cache = {}
        
    def can_fetch(self, site_url: str, user_agent: str = "*") -> bool:
        """Check robots.txt permissions before scraping."""
        if site_url not in self.robots_cache:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{site_url}/robots.txt")
            try:
                rp.read()
                self.robots_cache[site_url] = rp
            except Exception:
                # Conservative approach: allow if robots.txt unavailable
                return True
        
        return self.robots_cache[site_url].can_fetch(user_agent, site_url)

# Integration with JobSpy
validator = RobotsTxtValidator()

def compliant_scrape_jobs(site_name: list, search_term: str):
    """Scrape jobs with robots.txt validation."""
    site_urls = {
        "linkedin": "https://linkedin.com",
        "indeed": "https://indeed.com", 
        "glassdoor": "https://glassdoor.com"
    }
    
    allowed_sites = [
        site for site in site_name 
        if validator.can_fetch(site_urls.get(site, ""))
    ]
    
    if not allowed_sites:
        raise ComplianceError("No sites allow scraping per robots.txt")
        
    return scrape_jobs(site_name=allowed_sites, search_term=search_term)
```

### 5.3 Prefer Official APIs/Feeds When Available

**API-First Strategy:**

```python
class JobDataSource:
    """Prioritize official APIs over scraping."""
    
    OFFICIAL_APIS = {
        "indeed": "https://indeed.com/api",
        "ziprecruiter": "https://api.ziprecruiter.com",
        "jobs2careers": "https://api.jobs2careers.com"
    }
    
    async def get_jobs_api_first(self, site: str, search_term: str):
        """Attempt API first, fallback to scraping."""
        if site in self.OFFICIAL_APIS:
            try:
                return await self.fetch_via_api(site, search_term)
            except APIUnavailableError:
                logger.info(f"API unavailable for {site}, falling back to scraping")
        
        return await self.scrape_jobs_compliant(site, search_term)
```

### 5.4 Employer Career Site Preference

**Direct Career Page Strategy:**

```python
class CareerSitePreference:
    """Prefer company career sites over job boards."""
    
    def get_scraping_strategy(self, company: str, search_term: str):
        """Determine optimal scraping approach."""
        career_url = self.find_company_career_page(company)
        
        if career_url and self.is_scrapable(career_url):
            # Prefer direct company career page
            return self.scrape_career_page(career_url, search_term)
        else:
            # Fallback to job boards
            return self.scrape_job_boards(company, search_term)
    
    def find_company_career_page(self, company: str) -> str | None:
        """Locate company's official career page."""
        common_patterns = [
            f"https://{company.lower()}.com/careers",
            f"https://{company.lower()}.com/jobs",
            f"https://careers.{company.lower()}.com"
        ]
        
        for url in common_patterns:
            if self.validate_career_page(url):
                return url
        
        return None
```

## 6. Compliance Checklist

### 6.1 Pre-Deployment Validation

- [ ] **Robots.txt Respect Enabled**
  - JobSpy `respect_robots=True` configured
  - Automatic robots.txt validation implemented
  - Fallback handling for unavailable robots.txt

- [ ] **Rate Limiting Configured**
  - `random_delay=True` for all scraping operations
  - Site-specific delay minimums configured
  - Conservative `max_workers` limits enforced

- [ ] **Proxy Budget Monitoring Active**
  - IPRoyal residential proxy integration tested
  - Budget tracking and alerts configured
  - Monthly cost limits enforced ($15-25/month)

- [ ] **Site-Specific Controls Documented**
  - LinkedIn: Conservative delays, single worker
  - Indeed: Robots.txt compliance, 6-second minimum delay
  - Glassdoor: Minimal requests, extended delays

- [ ] **Legal Approval for Restricted Sites**
  - Terms of Service compliance reviewed
  - Legal counsel approval for commercial use
  - Data usage and attribution policies established

### 6.2 Operational Readiness

- [ ] **Safe Mode Defaults**
  - Default configuration prioritizes compliance over performance
  - Aggressive mode requires explicit user consent
  - Automatic fallback to safe mode on detection issues

- [ ] **Environment Configuration**
  - All compliance environment variables configured
  - Site-specific limits properly set
  - Proxy credentials secured and rotated

- [ ] **Monitoring and Alerting**
  - Proxy usage tracking operational
  - Budget alert thresholds configured
  - Compliance violation logging enabled

- [ ] **Error Handling and Recovery**
  - Exponential backoff for rate limiting
  - Graceful degradation for blocked requests
  - Automatic retry with increased delays

### 6.3 Ongoing Compliance Maintenance

- [ ] **Monthly Compliance Review**
  - Proxy usage and costs reviewed
  - Site-specific compliance policies updated
  - Success rate and error pattern analysis

- [ ] **Quarterly Legal Review**
  - Terms of Service changes monitored
  - Robots.txt updates validated
  - Legal compliance documentation updated

- [ ] **Annual Compliance Audit**
  - Full scraping strategy review
  - Third-party compliance assessment
  - Policy updates and training refresh

## 7. Implementation Guidelines

### 7.1 Development Standards

```python
# Template for compliant JobSpy integration
class CompliantJobSpyScraper:
    """JobSpy scraper with full compliance integration."""
    
    def __init__(self):
        self.proxy_manager = IPRoyalProxyManager()
        self.robots_validator = RobotsTxtValidator()
        self.compliance_monitor = ComplianceMonitor()
    
    async def scrape_jobs_compliant(self, company: str, search_term: str):
        """Execute compliant job scraping with all safeguards."""
        # Pre-flight compliance checks
        self.validate_scraping_request(company, search_term)
        
        # Determine proxy usage based on budget and priority
        use_proxy = self.proxy_manager.should_use_proxy(company)
        
        # Configure site-specific compliance settings
        config = self.get_site_specific_config(company)
        
        # Execute scraping with monitoring
        try:
            result = await self.execute_scraping(
                search_term=search_term,
                config=config,
                use_proxy=use_proxy
            )
            
            # Log successful compliance
            self.compliance_monitor.log_success(company, result)
            return result
            
        except Exception as e:
            self.compliance_monitor.log_violation(company, e)
            return await self.handle_compliance_failure(company, search_term, e)
```

### 7.2 Testing and Validation

```python
# Compliance testing framework
class ComplianceTestSuite:
    """Test suite for scraping compliance validation."""
    
    def test_robots_txt_respect(self):
        """Verify robots.txt compliance across all supported sites."""
        for site in ["linkedin", "indeed", "glassdoor"]:
            assert self.scraper.validates_robots_txt(site)
    
    def test_rate_limiting_enforcement(self):
        """Ensure rate limiting is properly implemented."""
        start_time = time.time()
        self.scraper.scrape_jobs_compliant("TestCorp", "engineer")
        duration = time.time() - start_time
        
        # Should include appropriate delays
        assert duration >= MIN_SCRAPING_DURATION
    
    def test_proxy_budget_controls(self):
        """Validate proxy usage stays within budget."""
        monthly_cost = self.proxy_manager.get_monthly_cost()
        assert monthly_cost <= MONTHLY_PROXY_BUDGET
    
    def test_authentication_boundaries(self):
        """Ensure no authenticated content is accessed."""
        with pytest.raises(ComplianceError):
            self.scraper.scrape_authenticated_content("linkedin.com/login")
```

## 8. Risk Assessment and Mitigation

### 8.1 Legal Risk Mitigation

| Risk Category | Risk Level | Mitigation Strategy |
|---------------|------------|-------------------|
| Terms of Service Violation | Medium | Automated compliance monitoring, legal review |
| Rate Limiting Detection | Low | Built-in delays, exponential backoff |
| IP Blocking | Low | Residential proxy rotation, conservative limits |
| Robots.txt Violations | Very Low | Automatic validation, respect enforcement |
| Data Attribution Issues | Low | Proper source attribution, usage documentation |

### 8.2 Operational Risk Controls

- **Service Availability:** Monitor proxy service uptime and health
- **Cost Overruns:** Automatic budget controls and spending alerts
- **Performance Impact:** Balance compliance with scraping efficiency
- **Data Quality:** Validate scraped data meets quality standards

## 9. Compliance Contacts and Resources

### 9.1 Internal Contacts

- **Compliance Officer:** [TBD - Internal compliance contact]
- **Legal Counsel:** [TBD - Legal review contact]
- **Technical Lead:** [TBD - Implementation lead]

### 9.2 External Resources

- **IPRoyal Support:** [https://iproyal.com/support](https://iproyal.com/support)
- **JobSpy Documentation:** [https://pypi.org/project/jobspy/](https://pypi.org/project/jobspy/)
- **Robots.txt Specification:** [https://www.robotstxt.org/](https://www.robotstxt.org/)

---

## Document Approval

**Document Owner:** Technical Architecture Team  
**Legal Review:** [Pending]  
**Compliance Review:** [Pending]  
**Implementation Review:** [Pending]

**Next Review Date:** 2025-11-28 (Quarterly)

---

*This document ensures ethical, legal, and sustainable JobSpy scraping integration that respects website policies while maintaining operational effectiveness.*
