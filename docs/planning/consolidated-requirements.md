# AI Job Scraper - Consolidated Requirements Document

## 1.0 Introduction

This document provides a complete and consolidated set of requirements for the AI Job Scraper application. It merges the foundational requirements from the initial refactoring phase with the feature-based requirements of subsequent development versions (V1.0, V1.1, and V2.0). The purpose is to create a single, authoritative source for all system, architectural, data, user experience, and non-functional requirements for the entire project.

### 1.1 Intended Audience

This document is intended for all project stakeholders, including developers, project managers, and testers, to ensure a shared understanding of the project's scope and objectives.

### 1.2 Scope

The project's scope is to develop an AI-powered job scraping tool that allows users to aggregate job postings from various company websites, manage them in a central database, track application statuses, and gain insights into their job search. The system will feature a responsive user interface, background scraping processes, and robust data synchronization and management capabilities.

---

## 2.0 System & Architecture Requirements (SYS)

This section details the high-level architectural and systemic requirements that form the foundation of the application.

- **SYS-ARCH-01: Component-Based Architecture**: The application must be built using a modular, component-based architecture, separating UI, services, state, and configuration into distinct directories (`src/ui`, `src/services`, etc.).
- **SYS-ARCH-02: Centralized State Management**: A centralized, singleton state manager (`StateManager`) must be implemented to handle global application state, ensuring predictable state transitions and UI updates.
- **SYS-ARCH-03: Multi-Page Navigation**: The application must support navigation between distinct pages (e.g., Dashboard, Jobs, Companies) without relying on browser reloads.
- **SYS-ARCH-04: Background Task Execution**: Long-running operations, specifically web scraping, must execute in a non-blocking background task to keep the UI responsive.
- **SYS-ARCH-05: Layered Configuration**: Application settings must be managed through a layered configuration system, allowing users to configure API keys.

---

## 3.0 Database & Data Management Requirements (DB)

This section outlines the requirements for the database schema, data models, and data synchronization logic.

### 3.1 DB Schema & Data Models

- **DB-SCHEMA-01: Relational Integrity**: The database must use a foreign key relationship to link `JobSQL` records to `CompanySQL` records (`JobSQL.company_id`).
- **DB-SCHEMA-02: Job Data Model**: The `JobSQL` model must include fields for:
  - Core job data
  - User-editable data (`favorite`, `notes`)
  - Application tracking (`application_status`, `application_date`)
  - Synchronization (`content_hash`, `created_at`, `updated_at`, `scraped_at`)
  - Archiving (`archived`)
- **DB-SCHEMA-03: Company Data Model**: The `CompanySQL` model must include fields for company details (`name`, `url`) and scraping metrics (`last_scraped`, `scrape_count`, `success_rate`).

### 3.2 DB Synchronization & Auditing

- **DB-SYNC-01: Intelligent Job Synchronization**: The system must intelligently synchronize scraped job data with the existing database, avoiding duplicates and preserving user data.
- **DB-SYNC-02: Content-Based Change Detection**: The system must use a content hash (e.g., MD5) of key job fields (title, description snippet, location) to detect changes in job postings.
- **DB-SYNC-03: User Data Preservation**: During a job record update, all user-editable fields (`favorite`, `notes`, `application_status`, etc.) must be preserved.
- **DB-SYNC-04: Soft Deletion (Archiving)**: Jobs that are no longer found on a company's website but have associated user data must be "soft-deleted" (e.g., marked as `archived = True`) instead of being permanently removed.
- **DB-AUDIT-01: Synchronization Auditing**: The system must log every synchronization operation (insert, update, delete) to a dedicated `SyncLogSQL` table for auditing and debugging purposes.
- **DB-AUDIT-02: Field-Level Change History**: The system must track and store the history of changes for individual fields within a job posting (e.g., when a description or title changes) in a dedicated `JobChangeSQL` table.

---

## 4.0 Scraping & Background Task Requirements (SCR)

This section defines the requirements for the web scraping functionality and its execution as a background process.

- **SCR-EXEC-01: Asynchronous Scraping**: The scraping process for multiple companies must be executed asynchronously to improve overall speed.
- **SCR-PROG-01: Real-Time Progress Reporting**: The background scraping task must provide real-time progress updates (e.g., per-company status, overall progress) to the UI via a callback mechanism.
- **SCR-CTRL-01: User Controls**: The UI must provide controls to start the scraping process.

---

## 5.0 User Interface & Experience Requirements (UI)

This section describes the requirements for the application's user interface and the overall user experience.

### 5.1 Job Management

- **UI-JOBS-01: Grid-Based Job Browser**: The primary job browsing interface must be a responsive, Pinterest-style grid of job cards.
- **UI-JOBS-02: Interactive Job Card**: Each job card must display key information (title, company, location) and provide interactive controls for favoriting and changing application status.
- **UI-JOBS-03: Modal Job Details View**: The job details view, triggered from the job card, must be presented in a modal overlay window to provide a focused user experience. This view should include the full job description and a place for users to add personal notes.
- **UI-JOBS-04: Filtering and Search**: The job browser must provide functionality to filter jobs by a text search term, company, and application status.
- **UI-JOBS-05: Advanced Job Filtering**: The job browser's filtering capabilities must be extended to include filtering by salary range and date posted.
- **UI-TRACK-01: Application Status Tracking**: The UI must allow users to set and update the status of their job applications for each job posting (e.g., "New", "Interested", "Applied", "Rejected").

### 5.2 Company & Settings Management

- **UI-COMP-01: Company Management**: The application must have a dedicated page for users to add, view, and activate/deactivate companies for scraping.
- **UI-COMP-02: Company Status Indicators**: The company management interface must visually indicate the active/inactive status of each company.
- **UI-SETT-01: Settings Configuration**: The application must have a settings page allowing users to manage API keys for LLM providers.

### 5.3 Dashboards & Analytics

- **UI-PROG-01: Scraping Dashboard**: A dedicated page must display the real-time progress of active scraping sessions in a simple, text-based format.
- **UI-PROG-02: Enhanced Progress Dashboard**: The real-time scraping dashboard must be enhanced to display rich, calculated metrics, including scraping speed (jobs/minute) and an estimated time of arrival (ETA). The layout must be upgraded to a more organized card-based grid.
- **UI-ANALYTICS-01: Analytics & Insights Dashboard**: A dedicated dashboard page must be implemented to provide users with interactive data visualizations, including job posting trends over time and a funnel chart of their application statuses.

### 5.4 General UX Enhancements

- **UI-UX-01: Polished Micro-interactions**: The application must incorporate subtle animations and hover effects on all interactive elements to provide clear visual feedback and a professional feel.
- **UI-UX-02: Smooth Page Transitions**: Navigating between pages and opening modals should be accompanied by smooth, non-jarring transition animations.
- **UI-UX-03: Skeleton Loading States**: When loading data for the first time (e.g., the initial population of the job grid), the UI must display skeleton screens that mimic the final layout to improve perceived performance.

---

## 6.0 Non-Functional Requirements (NFR)

This section lists the non-functional requirements, which define the quality attributes and constraints of the system.

### 6.1 Code Quality & Maintainability

- **NFR-CODE-01: Code Quality**: All code must adhere to modern Python standards, including full type hinting, Google-style docstrings, and passing `ruff` linting and formatting checks.
- **NFR-MAINT-01: Maintainability**: The final codebase must be modular and well-documented to facilitate future enhancements and maintenance.

### 6.2 Performance & Scalability

- **NFR-PERF-01: UI Responsiveness**: The UI must remain fluid and responsive at all times, with filter and search operations completing in under 100ms.
- **NFR-PERF-02: Scalability**: The application must perform efficiently with a database of over 5,000 job records.

### 6.3 User Experience & Component Usage

- **NFR-UX-01: Modern Component Usage**: The application should leverage specialized third-party libraries (e.g., `streamlit-elements`) where they provide a significant UX improvement over native Streamlit components for features like modals.

### 6.4 Testing & Documentation

- **NFR-TEST-01: Comprehensive Test Coverage**: The application must have a comprehensive, automated test suite covering services, utilities, and UI components, with a target of achieving over 80% code coverage.
- **NFR-TEST-02: Unit Testing**: The test suite must include unit tests that validate the logic of individual functions and methods in isolation.
- **NFR-TEST-03: Integration Testing**: The test suite must include integration tests that validate the interactions between different components of the system.
- **NFR-DOCS-01: User Documentation**: The project must include a clear, concise `USER_GUIDE.md` that enables a non-technical user to install, configure, and use the application effectively.
- **NFR-DOCS-02: Developer Documentation**: The codebase must be sufficiently documented with docstrings and comments to allow a new developer to understand the architecture and contribute to the project.
