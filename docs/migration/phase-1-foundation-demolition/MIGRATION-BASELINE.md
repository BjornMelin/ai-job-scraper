=== PRE-MIGRATION BASELINE ===
Date: 2025-08-27
Branch: TBD
Status: TEMPLATE - TO BE POPULATED BY DELETION AGENTS

## Code Volume Analysis

### Total Lines Assessment

```
Total lines in src/: TBD
Expected: ~26,289 lines (8,663% code bloat vs <300 target)
```

### Directory Structure (Pre-Deletion)

```
Major directories:
TBD - TO BE POPULATED

Expected targets for deletion:
- src/coordination/ (~3,500+ lines)
- src/ai/ complex files (~2,095+ lines) 
- src/ui/utils/fragment_* (~1,576+ lines)
- src/services/unified_scraper.py + company_service.py (~2,943+ lines)
- src/services/cache_manager.py (~506+ lines)
```

## File Structure Analysis

### Python Files Count

```
Python files count: TBD
Expected: 80+ files before deletion
Target: <20 files after cleanup
```

### Largest Files (Top 20)

```
TO BE POPULATED BY BASELINE SCRIPT

Expected largest violators:
1. unified_scraper.py (~979 lines)
2. fragment_orchestrator.py (~854 lines) 
3. hybrid_ai_router.py (~789 lines)
4. service_orchestrator.py (~751 lines)
5. cache_manager.py (~506 lines)
```

## Testing Metrics

### Test Files Analysis

```
Test files: TBD
Mock/patch references: TBD

Expected test cleanup needed:
- Remove tests for deleted orchestration
- Remove tests for deleted AI routing
- Remove tests for deleted fragment system
- Remove tests for deleted scraping services  
- Remove tests for deleted cache manager
```

## Current Dependencies

### From pyproject.toml

```
TO BE POPULATED - Dependencies before cleanup

Expected reductions:
- Remove enterprise pattern dependencies
- Eliminate custom framework requirements  
- Simplify to core library-first stack:
  * JobSpy + ScrapeGraphAI (scraping)
  * LiteLLM (AI routing)
  * Streamlit (UI with native fragments)
  * SQLModel + DuckDB (data + analytics)
```

## ADR-001 Compliance Violations

### Critical Over-Engineering Identified

```
1. ServiceOrchestrator (751 lines)
   - Violation: Enterprise workflow patterns for simple async operations
   - Replacement: Native threading.Thread + Streamlit components

2. HybridAIRouter (789 lines)  
   - Violation: Custom routing logic vs proven LiteLLM
   - Replacement: 50-line LiteLLM configuration

3. FragmentOrchestrator (854 lines)
   - Violation: Wrapper around native Streamlit @st.fragment
   - Replacement: Direct @st.fragment decorators

4. UnifiedScraper (979 lines)
   - Violation: Custom parsing vs proven JobSpy capabilities
   - Replacement: Direct JobSpy + ScrapeGraphAI usage

5. CacheManager (506 lines)
   - Violation: Wrapper around native @st.cache_data  
   - Replacement: Direct @st.cache_data decorators
```

### Library-First Principle Violations

- **Custom implementations:** 20,000+ lines where libraries exist
- **Enterprise patterns:** Inappropriate for personal job tracker
- **Maintenance burden:** 8,663% above sustainable level
- **Deployment blocker:** Preventing 1-week deployment target

## Success Criteria Definition

### Quantitative Targets

- **Minimum 70% reduction** in total lines (expect 80%+)
- **Zero import errors** in remaining modules
- **<300 lines total** final implementation  
- **Complete elimination** of 5 major over-engineered components

### Qualitative Validation  

- All functionality preserved through library replacements
- Improved reliability via battle-tested libraries
- Reduced maintenance overhead (20+ hours → <2 hours monthly)
- Clear path to 1-week deployment

## Baseline Commit Information

```
Commit: TBD
Author: TBD  
Message: TO BE POPULATED
```

---

*This baseline will be populated by deletion agents during SPEC-001 initialization*  
*Serves as definitive pre-migration reference for audit purposes*
