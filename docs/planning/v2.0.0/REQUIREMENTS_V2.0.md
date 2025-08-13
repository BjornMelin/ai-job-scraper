# AI Job Scraper - Requirements Document (V2.0 - "Boring Technology" Architecture)

## Introduction

This document defines the comprehensive requirements for **V2.0 "Boring Technology" Architecture**, a complete architectural redesign based on research-driven insights and adherence to KISS, DRY, YAGNI principles. This version represents a fundamental shift from over-engineered complexity to proven, simple solutions that perfectly match the application's actual scale and requirements.

**Design Philosophy**: Leverage battle-tested libraries exactly as designed, eliminate unnecessary abstractions, and build a tool that "just works" for years without maintenance.

---

## 1. System & Architecture Requirements (SYS)

### Core Architecture Principles

- **SYS-ARCH-01: Single File Application**: The entire application must be contained in a single `streamlit_app.py` file of less than 200 lines, eliminating architectural complexity and improving maintainability.

- **SYS-ARCH-02: Boring Technology Stack**: The application must use only proven, stable technologies: SQLite for data persistence, Pandas for data manipulation, and native Streamlit components for UI.

- **SYS-ARCH-03: No Custom Abstractions**: The application must avoid custom service layers, repository patterns, or ORM abstractions, instead using direct database queries and native library features.

- **SYS-ARCH-04: Library-First Implementation**: All functionality must leverage existing library capabilities to their fullest extent, with zero custom implementations where library solutions exist.

### Performance & Scalability

- **SYS-PERF-01: Optimal Scale Targeting**: The application must be optimized for its actual scale: 50-200 companies, 2K-10K jobs, 1-5 concurrent users, with daily/weekly scraping frequency.

- **SYS-PERF-02: Native Performance**: The application must rely on native library performance characteristics rather than custom optimization layers.

- **SYS-PERF-03: Minimal Memory Footprint**: The application must use minimal memory through direct data streaming and native component efficiency.

---

## 2. Database & Data Management Requirements (DB)

### Data Architecture

- **DB-ARCH-01: SQLite Direct Access**: The application must use SQLite directly via `sqlite3` module with Pandas integration, eliminating ORM overhead and complexity.

- **DB-ARCH-02: Simplified Schema**: The database must use a minimal schema with two tables: `companies` and `jobs`, with foreign key relationships maintained at the database level.

- **DB-ARCH-03: Pandas Integration**: All data operations must use Pandas DataFrames for consistency and performance, leveraging `pd.read_sql()` and `df.to_sql()` patterns.

### Data Operations

- **DB-OPS-01: Direct SQL Queries**: All database operations must use direct SQL queries through Pandas, avoiding query builders or ORM abstractions.

- **DB-OPS-02: Transaction Simplicity**: Database transactions must be handled through simple connection context managers, avoiding complex transaction management.

- **DB-OPS-03: Bulk Operations**: Data synchronization and updates must use efficient bulk operations via Pandas `to_sql()` method with appropriate parameters.

### Data Integrity

- **DB-INT-01: Referential Integrity**: Foreign key constraints must be enforced at the database level using SQLite's FOREIGN KEY pragma.

- **DB-INT-02: Content Hashing**: Job duplicate detection must use simple MD5 hashing of core fields for change detection.

- **DB-INT-03: Soft Deletion**: Job archiving must use a simple `archived` boolean flag for soft deletion functionality.

---

## 3. User Interface & Experience Requirements (UI)

### Component Architecture

- **UI-COMP-01: Native Streamlit Only**: The application must use only native Streamlit components (`st.dataframe`, `st.form`, `st.dialog`, etc.), eliminating third-party UI libraries.

- **UI-COMP-02: Built-in Data Display**: Job listings must use `st.dataframe` with native scrolling, searching, and sorting capabilities, eliminating custom pagination.

- **UI-COMP-03: Native Modals**: Job details must use `st.dialog()` for modal display, replacing custom modal implementations.

### Page Structure

- **UI-PAGE-01: Three-Page Maximum**: The application must consist of exactly three pages: Jobs List, Job Details (modal), and Settings, accessible through native Streamlit navigation.

- **UI-PAGE-02: Jobs List Primary**: The main page must display all jobs in a searchable, sortable `st.dataframe` with filtering controls in the sidebar.

- **UI-PAGE-03: Minimal Settings**: The settings page must provide only essential configuration options through native form components.

### User Experience

- **UI-UX-01: Instant Responsiveness**: All UI interactions must be instant for the target data scale, leveraging native component performance.

- **UI-UX-02: Zero Loading States**: The application must eliminate loading states and progress bars for operations on small datasets.

- **UI-UX-03: Natural Workflows**: User workflows must follow standard patterns with minimal cognitive load and obvious next actions.

---

## 4. Data Processing & Business Logic Requirements (BIZ)

### Salary Processing

- **BIZ-SAL-01: Library-First Parsing**: Salary parsing must continue to use the existing library-first implementation with price-parser and babel libraries (the only component rated 8.1/10 to retain).

- **BIZ-SAL-02: Simple Display**: Salary ranges must be displayed using simple formatting functions without computed properties.

### Job Management

- **BIZ-JOB-01: Simplified CRUD**: Job create, read, update, delete operations must use direct SQL operations through Pandas.

- **BIZ-JOB-02: Basic Filtering**: Job filtering must support basic text search, company selection, and date ranges through simple SQL WHERE clauses.

- **BIZ-JOB-03: Status Management**: Job application status must be managed through simple dropdown selections with direct database updates.

### Company Management

- **BIZ-COMP-01: Direct Management**: Company operations must use direct SQL operations for add, edit, delete, and activation status management.

- **BIZ-COMP-02: Simple Statistics**: Company statistics must be calculated through simple SQL aggregations without caching layers.

---

## 5. Caching & Performance Requirements (CACHE)

### Caching Strategy

- **CACHE-STRAT-01: Streamlit Native Only**: The application must use only `@st.cache_data` decorator for database query caching, eliminating custom caching layers.

- **CACHE-STRAT-02: Query-Level Caching**: Caching must be applied only at the database query level with appropriate TTL values (5-15 minutes).

- **CACHE-STRAT-03: No Persistence**: The application must not use persistent caching (disk-based) as data volumes don't justify the complexity.

### Performance Optimization

- **CACHE-PERF-01: Memory Efficiency**: Cache size must be naturally limited by the small data scale, requiring no cache eviction logic.

- **CACHE-PERF-02: Simple Invalidation**: Cache invalidation must rely on TTL expiration rather than complex invalidation logic.

---

## 6. Configuration & Settings Requirements (CFG)

### Configuration Management

- **CFG-MGMT-01: Environment Variables**: Configuration must use standard environment variables with sensible defaults, loaded through a simple Settings class.

- **CFG-MGMT-02: pydantic-settings**: Configuration must use `pydantic-settings` for type-safe configuration with validation.

- **CFG-MGMT-03: File-Based Config**: Settings must support `.env` file loading for development convenience.

### Configuration Scope

- **CFG-SCOPE-01: Minimal Settings**: The application must expose only essential settings: database path, scraping frequency, and basic display preferences.

- **CFG-SCOPE-02: No UI Configuration**: Complex configuration must be handled through environment variables rather than UI forms.

---

## 7. Scraping & Data Synchronization Requirements (SCRAPE)

### Scraping Architecture

- **SCRAPE-ARCH-01: Preserved Integration**: The existing scraping system must be preserved and integrated with the simplified architecture through direct database operations.

- **SCRAPE-ARCH-02: Simple Progress**: Scraping progress must use native `st.progress()` and `st.status()` components for user feedback.

### Data Synchronization

- **SCRAPE-SYNC-01: Bulk Processing**: Scraped data must be processed in bulk using Pandas operations for efficiency.

- **SCRAPE-SYNC-02: Simple Deduplication**: Job deduplication must use straightforward content hash comparison without complex merging logic.

---

## 8. Non-Functional Requirements (NFR)

### Code Quality

- **NFR-CODE-01: Extreme Simplicity**: The entire codebase must be simple enough for a new developer to understand completely in 30 minutes.

- **NFR-CODE-02: Self-Documenting**: Code must be self-documenting through clear variable names and simple logic flow, minimizing comment requirements.

- **NFR-CODE-03: Standard Formatting**: All code must follow standard Python formatting using `ruff format` and pass `ruff check` linting.

### Maintainability

- **NFR-MAINT-01: Zero Maintenance**: The application must require zero ongoing maintenance for normal operation.

- **NFR-MAINT-02: Obvious Extensions**: Future enhancements must be obvious and straightforward to implement.

- **NFR-MAINT-03: No Breaking Changes**: The architecture must be stable enough to avoid breaking changes for years.

### Performance

- **NFR-PERF-01: Sub-Second Response**: All user interactions must complete in under one second for the target data scale.

- **NFR-PERF-02: Minimal Resource Usage**: The application must use minimal CPU and memory resources appropriate for its scale.

- **NFR-PERF-03: Instant Startup**: The application must start instantly without complex initialization sequences.

### Reliability

- **NFR-REL-01: Crash Resistance**: The application must handle edge cases gracefully without crashes.

- **NFR-REL-02: Data Safety**: All data operations must be safe and recoverable, with automatic SQLite transaction handling.

- **NFR-REL-03: Graceful Degradation**: The application must degrade gracefully when optional features are unavailable.

---

## 9. Migration & Compatibility Requirements (MIG)

### Data Migration

- **MIG-DATA-01: Schema Migration**: The application must provide a one-time migration script to convert existing SQLModel data to the simplified schema.

- **MIG-DATA-02: Data Preservation**: All existing job and company data must be preserved during migration.

- **MIG-DATA-03: User Data Retention**: All user-entered data (favorites, notes, application status) must be retained during migration.

### Backward Compatibility

- **MIG-COMPAT-01: Settings Migration**: Existing configuration must be migrated to the new environment variable format.

- **MIG-COMPAT-02: Database Format**: The database format must remain SQLite for easy migration and continued compatibility.

---

## 10. Success Criteria & Metrics (SUCCESS)

### Quantitative Metrics

- **SUCCESS-CODE-01: Line Count Reduction**: The total application code must be reduced by 70% (from ~2000 lines to ~600 lines).

- **SUCCESS-CODE-02: File Reduction**: The number of Python files must be reduced by 80% (to less than 10 files total).

- **SUCCESS-CODE-03: Dependency Reduction**: Third-party dependencies must be reduced to essential libraries only (less than 10 total).

### Quality Metrics

- **SUCCESS-QUAL-01: Simplicity Score**: The application must achieve a 9.5/10 score on the KISS principle evaluation.

- **SUCCESS-QUAL-02: Maintenance Score**: The application must achieve a 10/10 score on maintenance burden evaluation.

- **SUCCESS-QUAL-03: Performance Score**: The application must achieve sub-second response times for all user interactions.

### User Experience Metrics

- **SUCCESS-UX-01: Onboarding Time**: New developers must be productive within 30 minutes of first seeing the code.

- **SUCCESS-UX-02: Feature Completeness**: All essential user workflows must be preserved or improved.

- **SUCCESS-UX-03: Reliability**: The application must run for months without intervention or issues.

---

## Conclusion

These requirements define a radical simplification that maintains all essential functionality while eliminating architectural complexity. The "Boring Technology" approach ensures the application will be reliable, maintainable, and fit-for-purpose for years to come, while providing an excellent foundation for future enhancements when truly needed.
