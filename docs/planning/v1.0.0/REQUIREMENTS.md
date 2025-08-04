# AI Job Scraper - Consolidated Requirements Document

## 1. System & Architecture Requirements (SYS)

- **SYS-ARCH-01: Component-Based Architecture**: The application must be built using a modular, component-based architecture, separating UI, services, state, and configuration into distinct directories (`src/ui`, `src/services`, etc.).
- **SYS-ARCH-02: Centralized State Management**: A centralized, singleton state manager (`StateManager`) must be implemented to handle global application state, ensuring predictable state transitions and UI updates.
- **SYS-ARCH-03: Multi-Page Navigation**: The application must support navigation between distinct pages (Dashboard, Jobs, Companies, etc.) without relying on browser reloads.
- **SYS-ARCH-04: Background Task Execution**: Long-running operations, specifically web scraping, must execute in a non-blocking background task to keep the UI responsive.
- **SYS-ARCH-05: Layered Configuration**: Application settings must be managed through a layered configuration system, separating UI, scraping, and data settings, and be persistable in the session state.

## 2. Database & Data Management Requirements (DB)

- **DB-SCHEMA-01: Relational Integrity**: The database must use a foreign key relationship to link `JobSQL` records to `CompanySQL` records (`JobSQL.company_id`).
- **DB-SCHEMA-02: Job Data Model**: The `JobSQL` model must include fields for core job data, user-editable data (`favorite`, `notes`), application tracking (`application_status`, `application_date`), and synchronization (`content_hash`, `created_at`, `updated_at`, `scraped_at`).
- **DB-SCHEMA-03: Company Data Model**: The `CompanySQL` model must include fields for company details (`name`, `url`) and scraping metrics (`last_scraped`, `scrape_count`, `success_rate`).
- **DB-SYNC-01: Intelligent Job Synchronization**: The system must intelligently synchronize scraped job data with the existing database, avoiding duplicates and preserving user data.
- **DB-SYNC-02: Content-Based Change Detection**: The system must use a content hash (e.g., MD5) of key job fields (title, description snippet, location) to detect changes in job postings.
- **DB-SYNC-03: User Data Preservation**: During a job record update, all user-editable fields (`favorite`, `notes`, `application_status`, etc.) must be preserved.
- **DB-SYNC-04: Soft Deletion (Archiving)**: Jobs that are no longer found on a company's website but have associated user data must be "soft-deleted" (e.g., marked as `archived = True`) instead of being permanently removed from the database.
- **DB-PERF-01: Strategic Indexing**: The database must have indexes on frequently queried columns (`company_id`, `posted_date`, `content_hash`, `location`, `title`) to ensure fast query performance.

## 3. Scraping & Background Task Requirements (SCR)

- **SCR-EXEC-01: Asynchronous Scraping**: The scraping process for multiple companies must be executed asynchronously to improve overall speed.
- **SCR-PROG-01: Real-Time Progress Reporting**: The background scraping task must provide real-time progress updates (e.g., per-company status, overall progress) to the UI via a callback mechanism.
- **SCR-CTRL-01: User Controls**: The UI must provide controls to start and stop the scraping process.

## 4. User Interface & Experience Requirements (UI)

- **UI-JOBS-01: Grid-Based Job Browser**: The primary job browsing interface must be a responsive, Pinterest-style grid of job cards.
- **UI-JOBS-02: Interactive Job Card**: Each job card must display key information (title, company, location) and provide interactive controls for favoriting and changing application status.
- **UI-JOBS-03: Job Details View**: Users must be able to view the full details of a job, including the full description and a place to add personal notes.
- **UI-JOBS-04: Filtering and Search**: The job browser must provide functionality to filter jobs by a text search term, company, and application status.
- **UI-COMP-01: Company Management**: The application must have a dedicated page for users to add, view, and activate/deactivate companies for scraping.
- **UI-COMP-02: Company Status Indicators**: The company management interface must visually indicate the health and status of each company's scraping configuration.
- **UI-SETT-01: Settings Configuration**: The application must have a settings page allowing users to manage API keys and switch between LLM providers.
- **UI-PROG-01: Scraping Dashboard**: A dedicated page must display the real-time progress of active scraping sessions.
- **UI-TRACK-01: Application Status Tracking**: The UI must allow users to set and update the status of their job applications for each job posting (e.g., "Applied", "Interviewing").
- **UI-ANALYTICS-01: Analytics Dashboard**: A dashboard must display visualizations of job market trends and personal application statistics.

## 5. Non-Functional Requirements (NFR)

- **NFR-PERF-01: UI Responsiveness**: The UI must remain fluid and responsive at all times, with filter and search operations completing in under 100ms.
- **NFR-PERF-02: Scalability**: The application must perform efficiently with a database of over 5,000 job records.
- **NFR-CODE-01: Code Quality**: All code must adhere to modern Python standards, including full type hinting, Google-style docstrings, and passing `ruff` linting and formatting checks.
- **NFR-MAINT-01: Maintainability**: The codebase must be modular and well-documented to facilitate future enhancements and maintenance.
- **NFR-TEST-01: Test Coverage**: The application must have a comprehensive test suite with a target of >80% code coverage.
