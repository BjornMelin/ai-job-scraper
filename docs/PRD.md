# Product Requirements Document (PRD): AI Job Scraper

## 1. Introduction

### 1.1 Purpose

This document outlines the product requirements for the **AI Job Scraper**, a local-first, privacy-focused Python application designed to automate the scraping, filtering, and management of AI/ML job postings. It serves as the single source of truth for development, aligning business goals with technical implementation.

### 1.2 Scope

* **In Scope:**
  * Scraping from major job boards (LinkedIn, Indeed, Glassdoor) and configurable company career pages
  * 2-tier scraping strategy using JobSpy (90% structured sources) and ScrapeGraphAI (10% unstructured sources)
  * Local AI processing using Qwen/Qwen3-4B-Instruct-2507-FP8 with vLLM inference
  * Content-aware database synchronization with SQLite + SQLModel foundation
  * SQLite FTS5 full-text search with porter stemming: 5-300ms response scaling from 1K-500K records
  * Rich, interactive Streamlit UI with modern card-based job browsing and filtering
  * Component-based, modular architecture for maintainability and extensibility
  * Background task management using Python threading.Thread with real-time progress tracking
  * Performance-triggered analytics scaling from SQLite to DuckDB (p95 >500ms threshold)
  * Production-ready deployment via Docker with GPU support for RTX 4090

* **Out of Scope:**
  * Multi-user or enterprise deployment patterns
  * Complex vector search or semantic similarity (over-engineered for personal use)
  * Real-time push notifications or external integrations
  * Custom UI frameworks beyond Streamlit native components
  * Production hosting or SaaS deployment

## 2. User Personas

* **Alex, the Job Seeker (Primary):** A mid-level AI engineer who is actively or passively looking for new opportunities. Alex is tech-savvy but time-constrained and needs an efficient way to aggregate relevant job postings without manually checking dozens of sites.

* **Sam, the Power User/Developer (Secondary):** An open-source contributor who wants to customize the tool, add new scraping sources, or integrate it into a larger workflow. Sam values clean, modular code and comprehensive documentation.

## 3. Functional Requirements

### 3.1 Scraping & Data Processing

* **FR-SCR-01: 2-Tier Scraping Strategy:** The system must use JobSpy for 90% coverage of structured job boards (LinkedIn, Indeed, Glassdoor) with native proxy support and ScrapeGraphAI for 10% coverage of unstructured company career pages with AI-powered extraction.

* **FR-SCR-02: Background Execution:** All scraping operations must run via threading.Thread with st.status for real-time progress display, allowing the user to continue interacting with the UI.

* **FR-SCR-03: Real-Time Progress:** The UI must display real-time progress using st.rerun() + session_state for non-blocking updates, including overall progress, per-company status, and jobs found.

* **FR-SCR-04: Structured Extraction:** AI-powered extraction must use LiteLLM + Instructor for structured outputs with reliable JSON parsing, eliminating custom parsing logic.

* **FR-SCR-05: Bot Evasion:** The system must employ bot evasion strategies, including IPRoyal residential proxy rotation, user-agent randomization, and respectful rate limiting.

### 3.2 Database & Synchronization

* **FR-DB-01: SQLModel Foundation:** The database must use SQLModel + SQLite with type-safe operations and automatic relationship handling for personal scale (tested capacity 500K+ jobs).

* **FR-DB-02: Database Synchronization:** The system must use content hash-based synchronization to preserve user data during updates.
  * **FR-DB-02a (Change Detection):** Use content hash of core job fields (title, company, description) to detect changes.
  * **FR-DB-02b (User Data Preservation):** Application status, favorites, and notes must be preserved across job updates.
  * **FR-DB-02c (Conflict Resolution):** Automatic merge of updated job content with existing user annotations.

* **FR-DB-03: Analytics Scaling:** The system must provide SQLite foundation with automatic DuckDB sqlite_scanner activation when p95 query latency exceeds 500ms threshold.

### 3.3 User Interface (UI)

* **FR-UI-01: Component-Based UI:** The UI must be built with a modular, component-based architecture using Streamlit native components.

* **FR-UI-02: Job Browser:** The primary interface must be a responsive, card-based grid of job postings with mobile-first design.

* **FR-UI-03: Advanced Filtering:** Users must be able to filter jobs by text search (SQLite FTS5 with porter stemming), company, application status, salary range, and date posted with <10ms response time.

* **FR-UI-04: Application Tracking:** Users must be able to set and update the status of their job applications (`New`, `Interested`, `Applied`, `Rejected`) with visual status indicators.

* **FR-UI-05: Job Details View:** Users must be able to view full job details and add personal notes using expandable interface components.

* **FR-UI-06: Real-time Updates:** The UI must provide non-blocking background task progress using threading with st.status components and st.rerun() for session state updates.

* **FR-UI-07: High-Performance Caching:** UI filter and search operations must complete in <100ms via st.cache_data (Streamlit native caching).

* **FR-UI-08: Analytics Dashboard:** The system must provide analytics with automatic method selection between SQLite (baseline) and DuckDB (high-performance) based on p95 query latency thresholds.

* **FR-UI-09: Company Management:** A dedicated UI must exist for users to add, view, and activate/deactivate companies for scraping.

* **FR-UI-10: Settings Management:** A settings page must allow users to manage API keys, configure AI processing thresholds, proxy usage, and database optimization settings.

### 3.4 Analytics & Monitoring (ADR-019)

* **FR-ANALYTICS-01: Automatic Method Selection:** The system must automatically select between SQLite and DuckDB analytics methods based on performance triggers (p95 latency >500ms).

* **FR-ANALYTICS-02: Streamlit Caching:** Performance tracking must use Streamlit native caching (st.cache_data) for optimized operations with minimal overhead.

* **FR-ANALYTICS-03: Cost Control Integration:** Real-time cost tracking must monitor the $50 monthly budget with automated alerts at 80% and 100% utilization.

* **FR-ANALYTICS-04: Visual Performance Indicators:** The analytics dashboard must display current method selection, performance metrics, and cost utilization with interactive visualizations.

### 3.5 Modern Card Interface (ADR-021)

* **FR-CARDS-01: Mobile-First Design:** Job cards must automatically adapt to screen size with responsive grid layouts (desktop: 3 columns, mobile: 1 column).

* **FR-CARDS-02: Status Visual Indicators:** Cards must display color-coded status indicators with icons for application progress (new, interested, applied, rejected).

* **FR-CARDS-03: Interactive Actions:** Each card must provide quick actions for status updates, apply links, and additional options menu.

* **FR-CARDS-04: Enhanced Information Display:** Cards must present job information with clear hierarchy: company/title header, salary/location body, status/actions footer.

## 4. Non-Functional Requirements

* **NFR-PERF-01: Search Performance:** SQLite FTS5 search operations must complete in <10ms for 1,000 jobs with porter stemming and BM25 relevance ranking.

* **NFR-PERF-02: AI Processing:** The system must route content <8K tokens to local Qwen3-4B processing and ≥8K tokens to GPT-4o-mini cloud fallback based on tiktoken measurement.

* **NFR-PERF-03: UI Responsiveness:** Job card rendering must complete in <200ms for 50+ jobs with full content using Streamlit native caching and optimization.

* **NFR-SCALE-01: Personal Scale Architecture:** The application must be architected for single user with tested capacity of 500K jobs (1.3GB database), with performance-based scaling triggers.

* **NFR-COST-01: Cost Optimization:** Monthly operational costs must remain under $50 budget ceiling with actual target of $25-30 through local/cloud processing allocation.

* **NFR-MAINT-01: Maintenance Requirements:** The system must operate with monthly dependency updates and quarterly library maintenance through library-first architecture.

* **NFR-PRIV-01: Local-First Privacy:** All job data must be processed and stored locally with only processing-level API usage for cloud AI fallback, no data retention by external services.

* **NFR-LIB-01: Library-First Implementation:** The codebase must leverage modern Python libraries and frameworks to minimize custom code and maintenance overhead while maximizing reliability and performance.

## 5. Technical Stack

### **Core Architecture - Library-Based Implementation**

* **AI Processing:** LiteLLM unified client + Instructor structured outputs + vLLM inference server
* **Local AI:** Qwen/Qwen3-4B-Instruct-2507-FP8 with FP8 quantization on RTX 4090
* **Token Routing:** 8K context window threshold measured via tiktoken for local/cloud processing routing  
* **Cloud AI:** GPT-4o-mini for complex tasks and fallback scenarios

### **Data Layer**

* **Database:** SQLModel + SQLite with WAL mode and type-safe operations
* **Search:** SQLite FTS5 with porter stemming and BM25 relevance ranking via sqlite-utils (ADR-018)
* **Analytics:** Automatic method selection - SQLite baseline with DuckDB sqlite_scanner scaling (ADR-019)
* **Performance Optimization:** Streamlit native caching for streamlined performance tracking
* **Cost Control:** Real-time $50 budget monitoring with automated service-level cost tracking
* **Synchronization:** Database sync with content hash detection and user data preservation

### **Scraping & Data Collection**

* **2-Tier Strategy:** JobSpy (90% structured job boards) + ScrapeGraphAI (10% company pages)
* **Proxy Integration:** IPRoyal residential proxies with JobSpy native compatibility
* **HTTP Resilience:** Native HTTPX transport retries (eliminates custom retry logic)
* **Background Tasks:** Python threading.Thread with Streamlit st.status integration

### **User Interface**

* **Framework:** Streamlit with native fragments, column configuration, and auto-refresh
* **Modern UI:** Card-based job browser with mobile-first responsive design (ADR-021)
* **Visual Design:** Status-coded cards for improved information scanning compared to table format
* **Interactive Elements:** Real-time status updates, quick actions, and hover effects
* **Search Integration:** Real-time FTS5 search with <10ms response time via sqlite-utils
* **Analytics Dashboard:** Analytics visualization with method selection indicators
* **State Management:** Native st.session_state with optimistic UI feedback

### **Hardware Requirements**

* **GPU:** RTX 4090 Laptop GPU with 16GB VRAM for local AI processing
* **Software:** CUDA >=12.1, Python 3.12+
* **Storage:** 2GB free disk space for models and database
* **Network:** Internet connection for proxy services and cloud AI fallback

### **Development & Deployment**

* **Package Management:** uv for Python dependency management
* **Code Quality:** ruff for linting and formatting
* **Testing:** pytest with library-first testing patterns
* **Containerization:** Docker + docker-compose for consistent development environment
* **Timeline:** 7-day implementation with 4-phase deployment strategy

### **Performance Characteristics**

* **Search:** <10ms FTS5 queries with porter stemming and BM25 ranking via sqlite-utils (ADR-018)
* **Analytics:** Automatic method selection triggers at p95 >500ms for DuckDB sqlite_scanner activation (ADR-019)
* **AI Processing:** <2s local vLLM inference with FP8 optimization, 98% local processing rate (ADR-011/012)
* **Performance Optimization:** Streamlit native caching provides streamlined operations vs custom monitoring (ADR-019)
* **UI Rendering:** Card-based interface with improved scanning efficiency, <200ms rendering for 50+ jobs (ADR-021)
* **GPU Utilization:** 90% efficiency with RTX 4090 FP8 quantization and continuous batching
* **Cost Control:** $50 monthly budget with real-time monitoring and automated alerts (ADR-019)
* **Success Rate:** JobSpy extraction reliability with native proxy integration and resilient HTTP transport
* **Library Integration:** Significant code reduction through library utilization (ADR-001)
