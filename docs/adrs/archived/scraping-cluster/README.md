# Archived Scraping & Bot Avoidance ADRs

**Archive Date:** August 19, 2025  
**Review Agent:** 001-scraping  
**Review Report:** `/docs/adrs/review_reports/001-scraping-review.md`

## Archived ADRs Summary

### ADR-001: Scraping Library Selection

- **Status:** SUPERSEDED by ADR-032 (Simplified Scraping Strategy)
- **Reason:** Complex multi-library approach replaced by Crawl4AI-primary strategy
- **Code Impact:** 75% reduction (400+ lines → 100 lines)
- **Key Change:** JobSpy + ScrapeGraphAI → Crawl4AI + JobSpy fallback

### ADR-003: Bot Avoidance Strategy  

- **Status:** SUPERSEDED by ADR-036 (Proxy and Anti-Bot Integration 2025)
- **Reason:** Enhanced with Crawl4AI native capabilities and improved cost optimization
- **Key Enhancement:** Basic IPRoyal integration → Advanced stealth + proxy integration
- **Cost Optimization:** $5-15/month → $20/month with smart allocation

### ADR-013: Hybrid Scraping Strategy with Playwright

- **Status:** PARTIALLY_INTEGRATED into ADR-032 and ADR-031
- **Reason:** 4-tier complexity simplified to 2-tier library-first approach
- **Preserved Concepts:** Performance optimization, cost reduction goals
- **Simplified Away:** Playwright patterns, complex orchestration logic

## Migration Summary

### What Was Preserved

✅ **Core Functionality:** All scraping and bot avoidance capabilities maintained  
✅ **Performance Goals:** Speed and cost optimization objectives carried forward  
✅ **Library Choices:** JobSpy and IPRoyal proxies retained where appropriate  
✅ **Technical Requirements:** 95% success rates, cost targets, parallel processing  

### What Was Simplified  

🔄 **Architecture:** Multi-tier complexity → Library-first simplicity  
🔄 **Code Volume:** 400+ lines → 100 lines implementation  
🔄 **Dependencies:** 3-4 libraries → 2 primary libraries  
🔄 **Maintenance:** Complex orchestration → Configuration-driven approach  

### Integration Quality

**Overall Score:** 9.2/10 - Excellent preservation of functionality with dramatic simplification

## References

- **Superseding ADRs:** ADR-032, ADR-036, ADR-031
- **Implementation Guide:** `/docs/planning/implementation-guide.md`
- **Architecture Overview:** `/docs/adrs/new/ARCHITECTURE_OVERVIEW.md`

---

*These ADRs remain archived for historical reference and to understand the architectural evolution from complex multi-library systems to modern library-first approaches.*
