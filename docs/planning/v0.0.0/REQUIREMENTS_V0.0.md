# AI Job Scraper - Requirements Document (V0.0 - Foundational Refactoring)

## Introduction

This document lists the specific system, architectural, and data management requirements that will be fulfilled by completing the tasks in **Phase 0: Foundational Refactoring & Stabilization**. The goal of this phase is not to add new user-facing features, but to refactor the existing codebase into a stable, maintainable, and scalable platform. Fulfilling these requirements is a prerequisite for all future development.

---

## 1. System & Architecture Requirements (SYS)

- **SYS-ARCH-01: Component-Based Architecture**: The application must be built using a modular, component-based architecture, separating UI, services, state, and configuration into distinct directories (`src/ui`, `src/services`, etc.).
- **SYS-ARCH-02: Centralized State Management**: A centralized, singleton state manager (`StateManager`) must be implemented to handle global application state, ensuring predictable state transitions and UI updates.
- **SYS-ARCH-03: Multi-Page Navigation**: The application must support navigation between distinct pages (Dashboard, Jobs, Companies, etc.) without relying on browser reloads.

## 2. Database & Data Management Requirements (DB)

- **DB-SCHEMA-01: Relational Integrity**: The database must use a foreign key relationship to link `JobSQL` records to `CompanySQL` records (`JobSQL.company_id`).
- **DB-SCHEMA-02: Job Data Model**: The `JobSQL` model must include fields for core job data, user-editable data (`favorite`, `notes`), application tracking (`application_status`, `application_date`), synchronization (`content_hash`, `created_at`, `updated_at`, `scraped_at`), and archiving (`archived`).
- **DB-SCHEMA-03: Company Data Model**: The `CompanySQL` model must include fields for company details (`name`, `url`) and scraping metrics (`last_scraped`, `scrape_count`, `success_rate`).
- **DB-SYNC-01: Intelligent Job Synchronization**: The system must intelligently synchronize scraped job data with the existing database, avoiding duplicates and preserving user data.
- **DB-SYNC-02: Content-Based Change Detection**: The system must use a content hash (e.g., MD5) of key job fields (title, description snippet, location) to detect changes in job postings.
- **DB-SYNC-03: User Data Preservation**: During a job record update, all user-editable fields (`favorite`, `notes`, `application_status`, etc.) must be preserved.
- **DB-SYNC-04: Soft Deletion (Archiving)**: Jobs that are no longer found on a company's website but have associated user data must be "soft-deleted" (e.g., marked as `archived = True`) instead of being permanently removed from the database.

## 3. Non-Functional Requirements (NFR)

- **NFR-CODE-01: Code Quality**: All code produced during the refactoring must adhere to modern Python standards, including full type hinting, Google-style docstrings, and passing `ruff` linting and formatting checks.
- **NFR-MAINT-01: Maintainability**: The final codebase of this phase must be modular and well-documented to facilitate future enhancements and maintenance, resolving the issues of the monolithic `app.py`.
