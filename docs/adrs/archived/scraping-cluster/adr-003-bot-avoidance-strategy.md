# ADR-003: Bot Avoidance Strategy

## Title

Proxy Rotation with IPRoyal and Headers/Delays for Scraping Resilience

## Version/Date

1.0 / July 31, 2025

## Status

Accepted

## Context

Job sites block bots via IP/rate limits (e.g., LinkedIn caps 100 jobs/IP). Research showed residential rotating proxies (95% success) with headers/delays essential; IPRoyal best for budget/reliability ($5-15/mo, 32M+ IPs).

## Related Requirements

- 90%+ success on restricted pages.

- Low cost for personal use.

- Easy integration with libs.

## Alternatives

- No Proxies: 50-70% success but blocks frequent.

- Datacenter Proxies: Cheap but low success (70%).

- IPRoyal Residential: High success/low min.

## Decision

Use IPRoyal residential rotating proxies managed by proxies lib Pool; integrate with headers/delays. Toggle in config.

## Related Decisions

- ADR-001 (Proxy pool in utils).

- ADR-004 (Retries complement avoidance).

## Design

Mermaid for proxy flow:

```mermaid
graph TD
    A[Scrape Request] --> B[Get Proxy from Pool]
    B --> C[Rotate if Needed]
    C --> D[Add Headers/Delay]
    D --> E[Send via httpx/Graph Config]
    E --> F[Success] | G[Block? Retry with New Proxy]
```

- utils.py: get_proxy_pool() returns Pool; rotate_proxy() gets/validates.

- Integration: scraper.py passes to ScrapeGraphAI/JobSpy params, LangGraph nodes use httpx.Client(proxies=...).

## Consequences

- Positive: High resilience (95% success), low bans.

- Negative: $5-15/mo cost; minor latency from rotation.

- Mitigations: Config toggle for light use; log blocks.
