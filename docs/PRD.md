# Product Requirements Document (PRD): AI Job Scraper

## 1. Introduction

### 1.1 Purpose

This document outlines the product requirements for the **AI Job Scraper**, a local-first, privacy-focused Python application designed to automate the scraping, filtering, and management of AI/ML job postings. It serves as the single source of truth for development, aligning business goals with technical implementation.

### 1.2 Scope

* **In Scope:**
  * Scraping from major job boards (LinkedIn, Indeed) and configurable company career pages.
  * Hybrid scraping strategy using specialized libraries (`JobSpy`) and agentic workflows (`ScrapeGraphAI`, `LangGraph`).
  * Intelligent, content-aware database synchronization (`SmartSyncEngine`) that preserves user data.
  * A rich, interactive Streamlit UI for job browsing, filtering, and application tracking.
  * Component-based, modular architecture for maintainability.
  * Robust background task management for a non-blocking user experience.
  * Production-ready deployment via Docker.

* **Out of Scope:**
  * Real-time push notifications.
  * Direct integration with third-party applicant tracking systems (ATS).
  * A hosted, multi-tenant SaaS version.
  * Mobile-native applications (the web UI will be mobile-responsive).

## 2. User Personas

* **Alex, the Job Seeker (Primary):** A mid-level AI engineer who is actively or passively looking for new opportunities. Alex is tech-savvy but time-constrained and needs an efficient way to aggregate relevant job postings without manually checking dozens of sites.

* **Sam, the Power User/Developer (Secondary):** An open-source contributor who wants to customize the tool, add new scraping sources, or integrate it into a larger workflow. Sam values clean, modular code and comprehensive documentation.

## 3. Functional Requirements

### 3.1 Scraping & Data Processing

* **FR-SCR-01: Hybrid Scraping:** The system must use `JobSpy` for structured job boards and `ScrapeGraphAI` for unstructured company career pages.

* **FR-SCR-02: Background Execution:** All scraping operations must run via threading.Thread with st.status for real-time progress display, allowing the user to continue interacting with the UI.

* **FR-SCR-03: Real-Time Progress:** The UI must display real-time progress using st.rerun() + session_state for non-blocking updates, including overall progress, per-company status, and jobs found.

* **FR-SCR-04: Bot Evasion:** The system must employ bot evasion strategies, including proxy rotation and user-agent randomization.

### 3.2 Database & Synchronization

* **FR-DB-01: Relational Integrity:** The database (`SQLModel`) must enforce a foreign key relationship between jobs and companies.

* **FR-DB-02: Smart Synchronization:** The system must use a `SmartSyncEngine` to update the database.
  * **FR-DB-02a (Change Detection):** Use a content hash of key job fields to detect changes.
  * **FR-DB-02b (User Data Preservation):** User-editable fields (`favorite`, `notes`, `application_status`) must be preserved during updates.
  * **FR-DB-02c (Smart Archiving):** Jobs no longer found on a source that have user data must be soft-deleted (archived) instead of permanently removed.

* **FR-DB-03 (V2.0): Auditing:** The system must log all sync operations and field-level changes to dedicated audit tables.

### 3.3 User Interface (UI)

* **FR-UI-01: Component-Based UI:** The UI must be built with a modular, component-based architecture.

* **FR-UI-02: Job Browser:** The primary interface must be a responsive, card-based grid of job postings.

* **FR-UI-03: Advanced Filtering:** Users must be able to filter jobs by text search, company, application status, salary range, and date posted.

* **FR-UI-04: Application Tracking:** Users must be able to set and update the status of their job applications (`New`, `Interested`, `Applied`, `Rejected`).

* **FR-UI-05: Job Details View:** Users must be able to view full job details and add personal notes. (Initially an expander, upgraded to a modal in V1.1).

* **FR-UI-06: Company Management:** A dedicated UI must exist for users to add, view, and activate/deactivate companies for scraping.

* **FR-UI-07: Settings Page:** A settings page must allow users to manage API keys and configure scraping limits.

* **FR-UI-08 (V1.1): Analytics Dashboard:** A dashboard must provide visualizations of job posting trends and application statuses.

* **FR-UI-09 (V2.0): UI Polish:** The application must incorporate micro-interactions, smooth transitions, and skeleton loading states to enhance perceived performance.

## 4. Non-Functional Requirements

* **NFR-PERF-01: Responsiveness:** UI filter and search operations must complete in under 100ms via st.cache_data (Streamlit native caching).

* **NFR-PERF-02: Scalability:** The application must perform efficiently with a database of over 5,000 job records using the current Pandas + SQLModel + SQLite stack.

* **NFR-SEC-01: Privacy:** All user data must be stored locally. No personal data should be transmitted to external services.

* **NFR-MAINT-01: Maintainability:** The codebase must be modular, well-documented, and adhere to modern Python standards (type hinting, linting).

* **NFR-TEST-01 (V2.0): Test Coverage:** The application must have a comprehensive, automated test suite with over 80% code coverage.

* **NFR-DOCS-01 (V2.0): Documentation:** The project must include clear user and developer documentation.

## 5. Technical Stack

### **Core Architecture**

* **Local LLM:** Qwen/Qwen3-4B-Instruct-2507-FP8 with FP8 quantization
* **Cloud LLM:** gpt-4o-mini for complex tasks (>8K tokens or high complexity)
* **Inference Engine:** vLLM >=0.6.2 with FP8 support on RTX 4090 Laptop GPU
* **Context Window:** 8192 tokens (optimal for job posts)
* **GPU Utilization:** 90% with aggressive memory optimization

### **Backend/Processing**

* **Language:** Python 3.12+ with uv package management  
* **Background Tasks:** threading.Thread with st.status for real-time progress display
* **Database:** SQLModel + SQLite (proven foundation), DuckDB analytics sidecar (metrics-driven evolution)
* **Caching:** st.cache_data (Streamlit native, session-based with TTL)
* **Scraping:** ScrapeGraphAI, JobSpy, LangGraph for hybrid approaches

### **UI Framework**

* **Primary:** Streamlit (latest stable) with built-in components
* **State Management:** st.session_state with auto-refresh fragments
* **Real-time Updates:** st.rerun() + threading for non-blocking UI
* **Caching Strategy:** st.cache_data for <100ms filter operations
* **Deployment:** Docker + Docker Compose

### **Hardware Requirements**

* **GPU:** RTX 4090 Laptop GPU with 16GB VRAM
* **Software:** CUDA >=12.1, vLLM >=0.6.2 for FP8 support
* **Memory:** Optimized for FP8 quantization efficiency

### **Development Tools**

* **Package Management:** uv (not pip)
* **Linting/Formatting:** ruff for Python
* **Testing:** pytest with >80% coverage target
* **Timeline:** 1-week deployment target achieved
