# Coverage Strategy Documentation

> **Strategic Analysis**: Path from 26% to 80% coverage with evidence-based prioritization and resource allocation for production readiness.

## Current Coverage Analysis (Measured)

### Overall Metrics

- **Line Coverage**: 26.03% (1,156 of 4,441 lines covered)
- **Branch Coverage**: 14.78% (162 of 1,096 branches covered)
- **Target Coverage**: 80% line coverage for production deployment
- **Coverage Gap**: 2,395 additional lines needed (+107% improvement)

## Module-Specific Coverage Breakdown (Evidence-Based)

### Core Infrastructure (High Coverage)

| Module | Lines | Covered | Coverage | Status | Priority |
|--------|-------|---------|----------|---------|----------|
| `src/__init__.py` | 6 | 6 | 100% | ✅ Complete | MAINTAIN |
| `src/constants.py` | 11 | 11 | 100% | ✅ Complete | MAINTAIN |
| `src/config.py` | 54 | 45 | 83% | ✅ Good | LOW |
| `src/database.py` | 100 | 59 | 59% | 🔶 Moderate | MEDIUM |

### Services Layer (Mixed Coverage)

| Module | Lines | Covered | Coverage | Status | Priority |
|--------|-------|---------|----------|---------|----------|
| `services/analytics_service.py` | 97 | 77 | 79% | ✅ Good | LOW |
| `services/company_service.py` | 158 | 129 | 82% | ✅ Good | LOW |
| `services/job_service.py` | 221 | 150 | 68% | 🔶 Moderate | MEDIUM |
| `services/search_service.py` | 114 | 89 | 78% | ✅ Good | LOW |
| `services/cost_monitor.py` | 89 | 60 | 67% | 🔶 Moderate | MEDIUM |
| `services/database_sync.py` | 54 | 27 | 50% | 🔶 Moderate | MEDIUM |

### UI Layer (Critical Gap - Major Focus Area)

| Module | Lines | Covered | Coverage | Status | Priority |
|--------|-------|---------|----------|---------|----------|
| **UI Pages** (Total: 1,142 lines) ||||
| `ui/pages/companies.py` | 342 | 189 | 55% | 🔶 Moderate | HIGH |
| `ui/pages/jobs.py` | 326 | 105 | 32% | ❌ Low | HIGH |
| `ui/pages/scraping.py` | 213 | 81 | 38% | ❌ Low | HIGH |
| `ui/pages/analytics.py` | 136 | 79 | 58% | 🔶 Moderate | HIGH |
| `ui/pages/settings.py` | 125 | 17 | 14% | ❌ Critical | HIGH |
| **UI Utils** (Total: 955 lines) ||||
| `ui/utils/background_helpers.py` | 223 | 121 | 54% | 🔶 Moderate | HIGH |
| `ui/utils/formatters.py` | 237 | 89 | 38% | ❌ Low | HIGH |
| `ui/utils/url_state.py` | 244 | 16 | 7% | ❌ Critical | HIGH |
| `ui/utils/database_helpers.py` | 147 | 29 | 20% | ❌ Low | HIGH |
| `ui/utils/validators.py` | 75 | 35 | 47% | ❌ Low | MEDIUM |
| `ui/ui_rendering.py` | 161 | 31 | 19% | ❌ Low | HIGH |

### AI/Scraping Layer (Zero Coverage - Immediate Action)

| Module | Lines | Covered | Coverage | Status | Priority |
|--------|-------|---------|----------|---------|----------|
| `ai_models.py` | 76 | 0 | 0% | ❌ Critical | HIGH |
| `data_cleaning.py` | 28 | 0 | 0% | ❌ Critical | HIGH |
| `scraper.py` | 97 | 0 | 0% | ❌ Critical | HIGH |
| `scraper_company_pages.py` | 123 | 0 | 0% | ❌ Critical | HIGH |
| `scraper_job_boards.py` | 159 | 0 | 0% | ❌ Critical | HIGH |

## Prioritization Matrix (Resource Allocation)

### Phase 1: Critical Infrastructure (Target: 4 weeks)

**Coverage Target**: 35% → 55% (+20 percentage points)
**Focus**: Zero-coverage modules that are operationally critical

| Priority | Module | Current | Target | Lines to Cover | Effort | Rationale |
|----------|--------|---------|---------|----------------|---------|-----------|
| **P0** | `ai_models.py` | 0% | 85% | +65 lines | 8 hours | Core AI functionality |
| **P0** | `scraper.py` | 0% | 80% | +78 lines | 12 hours | Primary business logic |
| **P0** | `scraper_job_boards.py` | 0% | 75% | +119 lines | 16 hours | Revenue-critical workflows |
| **P1** | `scraper_company_pages.py` | 0% | 70% | +86 lines | 12 hours | Company data integrity |

### Phase 2: UI Layer Focus (Target: 6 weeks)

**Coverage Target**: 55% → 70% (+15 percentage points)
**Focus**: User-facing workflows and state management

| Priority | Module | Current | Target | Lines to Cover | Effort | Rationale |
|----------|--------|---------|---------|----------------|---------|-----------|
| **P0** | `ui/utils/url_state.py` | 7% | 75% | +166 lines | 20 hours | Navigation foundation |
| **P0** | `ui/pages/settings.py` | 14% | 80% | +83 lines | 12 hours | Configuration critical |
| **P1** | `ui/pages/jobs.py` | 32% | 75% | +140 lines | 18 hours | Primary user workflow |
| **P1** | `ui/pages/scraping.py` | 38% | 75% | +79 lines | 12 hours | Operations interface |
| **P2** | `ui/utils/formatters.py` | 38% | 70% | +76 lines | 10 hours | Data presentation |

### Phase 3: Service Enhancement (Target: 4 weeks)

**Coverage Target**: 70% → 80% (+10 percentage points)
**Focus**: Business logic completeness and edge cases

| Priority | Module | Current | Target | Lines to Cover | Effort | Rationale |
|----------|--------|---------|---------|----------------|---------|-----------|
| **P1** | `services/database_sync.py` | 50% | 85% | +19 lines | 8 hours | Data consistency |
| **P2** | `services/cost_monitor.py` | 67% | 85% | +16 lines | 6 hours | Budget monitoring |
| **P2** | `services/job_service.py` | 68% | 85% | +38 lines | 10 hours | Job processing |

## Testing Strategy by Module Type

### AI/ML Components

**Pattern**: Mock external APIs, test logic in isolation

```python
# Test pattern for ai_models.py
@responses.activate
def test_openai_completion():
    responses.add(responses.POST, "https://api.openai.com/v1/chat/completions")
    # Test internal model logic without API calls
```

**Coverage Techniques**:

- Mock all external AI API calls (OpenAI, Groq, LiteLLM)
- Property-based testing for prompt templates
- Error handling for rate limits and API failures

### Scraping Components  

**Pattern**: VCR cassettes for HTTP recording, edge case testing

```python
# Test pattern for scraper_job_boards.py
@pytest.mark.vcr
def test_indeed_scraping():
    # Uses VCR cassettes for reproducible HTTP tests
    result = scrape_indeed_jobs("AI Engineer")
    assert len(result) > 0
```

**Coverage Techniques**:

- VCR cassettes for external site testing
- Mock anti-bot detection mechanisms
- Test malformed HTML handling

### UI Components (Streamlit)

**Pattern**: Session state mocking, component isolation

```python
# Test pattern for UI pages
def test_jobs_page_rendering(mock_streamlit_session):
    with patch("streamlit.session_state", mock_streamlit_session):
        jobs_page_main()
        # Verify component interactions
```

**Coverage Techniques**:

- Mock Streamlit session state and components
- Test user interaction workflows
- Validate state transitions and URL updates

## Coverage Quality Metrics

### Line Coverage Targets by Category

- **Critical Business Logic**: 90%+ (AI models, scrapers, services)
- **UI Layer**: 75%+ (pages, utils, components)  
- **Infrastructure**: 85%+ (database, config, constants)
- **Utilities**: 70%+ (formatters, helpers, validators)

### Branch Coverage Targets

- **Current Branch Coverage**: 14.78% (162/1,096 branches)
- **Target Branch Coverage**: 60%+ (656+ branches)
- **Focus Areas**: Error handling paths, conditional logic

### Mock Usage Guidelines

- **Current Mock Count**: ~189 mocks across test suite
- **Target Mock Limit**: <250 (maintainability threshold)
- **Mock Strategy**: External services only, never internal logic

## Coverage Collection Performance

### Current Performance (Measured)

```bash
# Coverage overhead analysis  
Base Test Execution: 8.7s (no coverage)
With Coverage Collection: 10.9s (+2.2s overhead, +25%)
Parallel Coverage: Supported (pytest-cov + xdist)
HTML Report Generation: 1.3s additional
```

### Optimization Strategies

- **Parallel Safe**: pytest-cov supports pytest-xdist execution
- **Incremental**: Only measure changed files during development
- **CI Optimization**: Cache coverage data between runs

## Technical Implementation Plan

### Week 1-2: Zero Coverage Elimination

**Goal**: Remove all 0% coverage modules
**Deliverables**:

- AI models test suite (65+ tests)
- Scraper integration tests (80+ tests)
- Basic HTTP mocking infrastructure

### Week 3-4: UI Testing Framework

**Goal**: Establish Streamlit testing patterns
**Deliverables**:

- Streamlit session state mocking
- UI component test utilities
- URL state management tests (100+ tests)

### Week 5-8: Comprehensive Coverage

**Goal**: Achieve 70% overall coverage
**Deliverables**:

- Complete page-level test coverage
- Service layer edge case testing
- Integration workflow tests

### Week 9-12: Quality & Maintenance

**Goal**: Reach 80% sustainable coverage
**Deliverables**:

- Property-based test integration
- Performance regression prevention
- Coverage maintenance automation

## Success Metrics & Monitoring

### Coverage Progression Tracking

```bash
# Weekly coverage targets
Week 2:  26% → 35% (+9 percentage points)
Week 4:  35% → 50% (+15 percentage points) 
Week 8:  50% → 70% (+20 percentage points)
Week 12: 70% → 80% (+10 percentage points)
```

### Quality Gates

- **Minimum Coverage**: 80% for production deployment
- **Branch Coverage**: 60% minimum
- **Mock Ratio**: <200 mocks per 1,000 tests
- **Test Execution**: <60s for full suite

### Long-term Maintenance

- **Coverage Regression Prevention**: CI fails below 78%
- **New Code Coverage**: 90%+ for new features
- **Technical Debt**: Monthly mock count audits
- **Performance**: Quarterly execution time analysis

---

**Next**: Test Patterns & Best Practices Guide - library-first testing patterns and practical examples.
