# ADR-013: Hybrid Scraping Strategy with Playwright

## Title

Hybrid Scraping Strategy: Playwright for Known Patterns, ScrapeGraphAI for Complex Sites

## Version/Date

1.0 / August 18, 2025

## Status

Proposed

## Context

Performance analysis revealed that ScrapeGraphAI, while excellent for adaptability, introduces latency (3s average) and LLM API costs for every scrape. Meanwhile, 70% of target sites use standard job board platforms (Greenhouse, Lever, Workday) with predictable HTML patterns. Playwright benchmarks show 10x speed improvement (290ms vs 3s) for known patterns. Need strategy that optimizes for speed while maintaining adaptability.

## Related Requirements

- Sub-second response for known patterns (<500ms target)
- Zero maintenance for site changes
- Cost optimization (reduce LLM calls by 70%)
- Handle both static and JavaScript-rendered content
- Maintain 95%+ extraction accuracy

## Alternatives

1. **ScrapeGraphAI Only**: Current approach, slow but adaptable
2. **Playwright Only**: Fast but high maintenance for selector changes
3. **Crawl4AI**: Good middle ground but requires additional integration
4. **Hybrid Strategy**: Playwright for known, ScrapeGraphAI for unknown (chosen)

## Decision

Implement hybrid scraping strategy:

- **Tier 1**: JobSpy for job boards (LinkedIn, Indeed, Glassdoor)
- **Tier 2**: Playwright for known patterns (Greenhouse, Lever, Workday)
- **Tier 3**: Crawl4AI for semi-structured sites (optional future enhancement)
- **Tier 4**: ScrapeGraphAI for complex/unknown sites (fallback)

## Related Decisions

- ADR-001: Keep JobSpy for board scraping
- ADR-003: Proxy integration remains unchanged
- ADR-011: Performance optimization via tiered approach

## Design

```python
class HybridScraper:
    KNOWN_PATTERNS = {
        "greenhouse.io": {"job": "div.opening", "title": "a.opening-link"},
        "lever.co": {"job": "div.posting", "title": "h5[data-qa='posting-name']"},
        "workday.com": {"job": "li[data-automation-id='job']", "title": "a[data-automation-id='jobTitle']"}
    }
    
    async def scrape(url: str, company: str):
        # Tier 1: Job boards
        if "linkedin.com" in url or "indeed.com" in url:
            return jobspy_scrape(company)
        
        # Tier 2: Known patterns
        for domain, selectors in KNOWN_PATTERNS.items():
            if domain in url:
                return playwright_scrape(url, selectors)
        
        # Tier 3: Generic patterns
        if jobs := try_generic_patterns(url):
            return jobs
        
        # Tier 4: AI fallback
        return scrapegraph_scrape(url)
```

## Consequences

### Positive

- **10x speed improvement** for 70% of scrapes (3s → 290ms)
- **70% reduction in LLM costs** ($0.10/scrape → $0.03/scrape average)
- **Graceful degradation**: Falls back to AI when patterns fail
- **Progressive enhancement**: Can add patterns without breaking existing

### Negative

- **Increased complexity**: Multiple scraping paths to maintain
- **Pattern maintenance**: Need to update selectors occasionally
- **Testing burden**: Must test each tier independently

### Mitigations

- **Pattern versioning**: Track selector success rates, auto-disable failing patterns
- **Automated testing**: CI pipeline tests known patterns weekly
- **Observability**: Log which tier handled each scrape for optimization

## Implementation Plan

1. **Phase 1** (Day 1): Add Playwright, implement known patterns
2. **Phase 2** (Day 2): Add fallback logic, performance tracking
3. **Phase 3** (Day 3): Testing, optimization, documentation
4. **Future**: Consider Crawl4AI for Tier 3 if needed

## Metrics

Track per-tier metrics:

- Success rate by tier
- Average response time by tier
- Cost per scrape by tier
- Fallback frequency

Target: 70% Tier 1-2 (fast path), 30% Tier 4 (AI fallback)
