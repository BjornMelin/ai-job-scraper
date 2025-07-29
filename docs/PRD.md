# Product Requirements Document (PRD) for AI Job Scraper

📋 **TOC**  

1. Introduction  
2. Goals and Objectives  
3. User Personas  
4. Functional Requirements  
5. Non-Functional Requirements  
6. Technical Stack  
7. Risks and Mitigations  
8. Timeline and Milestones  
9. Acceptance Criteria  
10. Appendices  

## 1. Introduction

### 1.1 Purpose

This PRD outlines the requirements for AI Job Scraper, a local Python application that automates scraping, filtering, storage, and management of AI/ML job postings from top company websites. It enables users to track relevant roles efficiently while ensuring data privacy through offline processing. The document serves as a blueprint for development, aligning with architectural decisions (ADRs) for modularity and robustness.

### 1.2 Scope

In Scope: Scraping from configurable company sites using Crawl4AI, regex/LLM filtering for relevance, SQLite storage with update logic, Streamlit UI for viewing/editing/managing (tabs/views/search/sort/paginate/export/theme), Docker support, error handling/logging/retries/validation.

Out of Scope: Real-time notifications, integration with job boards (e.g., LinkedIn API), advanced analytics (e.g., salary trends), mobile-native app.

### 1.3 Target Audience

- Job Seekers: Individuals hunting for AI Engineer, MLOps, or similar roles, needing a tool to aggregate and track postings.
- Recruiters/Talent Scouts: Monitor openings across companies.
- Developers: Extend with new sites/filters or integrate LLMs.

### 1.4 Assumptions and Dependencies

- Users have Python 3.12+ or Docker installed.
- Optional: OpenAI API key for enhanced extraction (fallback to CSS).
- Company websites remain accessible and structured similarly.
- Dependencies: Crawl4AI, Streamlit, SQLAlchemy, etc. (pinned versions in pyproject.toml).

## 2. Goals and Objectives

- **Primary Goal**: Provide an efficient, privacy-focused tool for aggregating and managing AI job postings from key companies.
- **Objectives**:
  - Automate scraping and filtering to save user time (e.g., daily rescrapes in <1 min).
  - Enable intuitive tracking (e.g., favorites/applied tabs, edits).
  - Ensure reliability (99% uptime via retries/fallbacks).
  - Support extensibility (configurable companies, modular for LLM upgrades).

Success Metrics: User can scrape/view 100+ jobs in <5 min; 95% scrape success rate; Positive feedback on UI usability.

## 3. User Personas

- **Persona 1: Alex the Job Seeker** (Primary): Mid-level AI engineer, 28, tech-savvy but time-constrained. Needs quick scans of new roles, marking favorites, noting applications. Pain Point: Manual site checks tedious.
- **Persona 2: Jordan the Recruiter** (Secondary): HR professional, 35, manages talent pipelines. Wants aggregated listings with filters/search for outreach. Pain Point: Scattered sources.
- **Persona 3: Sam the Developer** (Tertiary): Open-source contributor, 32, extends tool with new features (e.g., LLM filter). Needs modular code/ADRs.

## 4. Functional Requirements

### 4.1 Core Features

- **Scraping**: Async fetch from active companies using Crawl4AI (LLM schema for title/desc/link/location/posted_date; CSS fallback; Retries with Tenacity).
- **Filtering**: Regex on titles for relevance; Modular for LLM upgrade.
- **Storage/Updates**: SQLite via SQLAlchemy; Add/update (hash check)/delete on rescrape, preserving user fields.
- **Company Management**: DB table; UI to add/remove/activate.

### 4.2 User Interface

- **Dashboard**: Tabs (All/Favorites/Applied using status/favorite); Views (list with data_editor for edits, card with HTML/CSS for visuals); Global filters (company/keyword/date) + per-tab search (filter DF by title/desc).
- **Interactions**: Rescrape button with spinner; Sort (selectbox on columns like Posted/Title); Paginate (buttons, 9/page for cards); Theme toggle (CSS injection); Export CSV per tab.
- **Edits**: Inline in list (checkbox/selectbox/text); Quick widgets in cards (toggle/select/area) with rerun.

### 4.3 Advanced Features

- **Validation**: Pydantic for data integrity (e.g., link patterns); httpx for link checks.
- **Error Handling**: Try/except with logging/st.error; Retries/fallbacks.
- **Docker**: Dockerfile/Compose for setup (volume for DB, Playwright deps).

## 5. Non-Functional Requirements

### 5.1 Performance

- Scrape time: <30s per site (async); UI load: <1s for 1000 jobs (pagination).
- Scalability: Handle 5000+ jobs via efficient queries/pagination.

### 5.2 Security and Privacy

- Local-only processing/DB; No external sends except optional OpenAI.
- Validate inputs (e.g., URLs in company add).

### 5.3 Usability

- Intuitive/responsive (mobile CSS, emojis in UI); Accessible (ARIA via Streamlit).
- Error feedback (st.error for failures).

### 5.4 Reliability

- 99% scrape success (retries/fallbacks); Graceful degradation (e.g., CSS if LLM fails).
- Testing: 80% coverage with pytest (unit/integration).

### 5.5 Maintainability

- Modular code (scraper.py/app.py/models.py); Ruff linted; ADRs/PRD for guidance.
- Principles: KISS/DRY/YAGNI, but value-adding features like search/views included.

## 6. Technical Stack

- **Core**: Python 3.12, Crawl4AI v0.7.2 (scraping), SQLAlchemy v2.0.42 (DB), Pandas v2.3.1 (data).
- **UI**: Streamlit v1.47.1.
- **Tools**: Tenacity v9.1.2 (retries), httpx v0.28.1 (validation), Pydantic v2.11.7 (models), logging (built-in).
- **Deployment**: Docker (python-slim base), uv for deps.
- **Testing**: Pytest.

## 7. Risks and Mitigations

- Risk: Site structure changes break scraping—Mitigation: Fallbacks/retries; Monitor/update ADRs.
- Risk: DB corruption on concurrent access—Mitigation: Session management/rollbacks.
- Risk: Dependency conflicts—Mitigation: Pinned versions/tests.
- Risk: Poor performance on large data—Mitigation: Pagination/sorting optimizations.

## 8. Timeline and Milestones

- **MVP (Week 1-2)**: Scraping/filtering/DB core (Complete per history).
- **v0.2 (Week 3-4)**: UI enhancements/Docker (Complete).
- **v1.0 (Q3 2025)**: Full search/views/validation, testing (Ready for release).
- **Post-v1.0**: LLM filter integration, more sites.

## 9. Acceptance Criteria

- Scrape: Successfully extracts/validates from all initial sites; Handles failures gracefully.
- UI: Tabs/views/search/sort/paginate work; Edits persist; Mobile no overlaps.
- Performance: <1min full rescrape; UI responsive.
- Testing: All tests pass; Manual verification on sample data.
- Documentation: ADRs/PRD/README complete.

## 10. Appendices

- **Architecture Diagram**:

```mermaid
graph TD
    A[User] -->|Interact| B[Streamlit UI: Tabs/Views/Search/Edit/Export]
    B -->|Rescrape/Manage| C[Scraper: Crawl4AI Async/Retries/Fallbacks/Validate]
    C -->|Filter/Update| D[SQLite DB: Jobs/Companies (SQLAlchemy ORM/Pydantic)]
    B -->|Query/Filter| D
    E[Logging/Errors] -->|All| A & C & B
    F[Docker] -->|Container| B & D
```

- **Developers**: See ADRs for decisions; tests/ for coverage.
