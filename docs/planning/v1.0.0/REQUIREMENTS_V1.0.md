# AI Job Scraper - Requirements Document (V1.0 - Job Hunter's MVP)

## Introduction

This document lists the specific requirements that will be fulfilled by completing the tasks in the **V1.0 "Job Hunter's MVP" Release**. It assumes that all requirements outlined in `REQUIREMENTS_V0.0.md` have been met and the application has a stable, modular foundation. The goal of this phase is to build the core set of user-facing features needed for an effective job-hunting experience.

---

## 1. System & Architecture Requirements (SYS)

- **SYS-ARCH-04: Background Task Execution**: Long-running operations, specifically web scraping, must execute in a non-blocking background task to keep the UI responsive.
- **SYS-ARCH-05: Layered Configuration**: Application settings must be managed through a layered configuration system, allowing users to configure API keys.

## 2. Scraping & Background Task Requirements (SCR)

- **SCR-EXEC-01: Asynchronous Scraping**: The scraping process for multiple companies must be executed asynchronously to improve overall speed.
- **SCR-PROG-01: Real-Time Progress Reporting**: The background scraping task must provide real-time progress updates (e.g., per-company status, overall progress) to the UI via a callback mechanism.
- **SCR-CTRL-01: User Controls**: The UI must provide controls to start the scraping process.

## 3. User Interface & Experience Requirements (UI)

- **UI-JOBS-01: Grid-Based Job Browser**: The primary job browsing interface must be a responsive, Pinterest-style grid of job cards.
- **UI-JOBS-02: Interactive Job Card**: Each job card must display key information (title, company, location) and provide interactive controls for favoriting and changing application status.
- **UI-JOBS-03: Job Details View**: Users must be able to view the full details of a job, including the full description and a place to add personal notes, via an in-line expander view.
- **UI-JOBS-04: Filtering and Search**: The job browser must provide functionality to filter jobs by a text search term, company, and application status.
- **UI-COMP-01: Company Management**: The application must have a dedicated page for users to add, view, and activate/deactivate companies for scraping.
- **UI-COMP-02: Company Status Indicators**: The company management interface must visually indicate the active/inactive status of each company.
- **UI-SETT-01: Settings Configuration**: The application must have a settings page allowing users to manage API keys for LLM providers.
- **UI-PROG-01: Scraping Dashboard**: A dedicated page must display the real-time progress of active scraping sessions in a simple, text-based format.
- **UI-TRACK-01: Application Status Tracking**: The UI must allow users to set and update the status of their job applications for each job posting (e.g., "New", "Interested", "Applied", "Rejected").

## 4. Non-Functional Requirements (NFR)

- **NFR-PERF-01: UI Responsiveness**: The UI must remain fluid and responsive at all times, with filter and search operations completing in under 100ms.
- **NFR-PERF-02: Scalability**: The application must perform efficiently with a database of over 5,000 job records.
