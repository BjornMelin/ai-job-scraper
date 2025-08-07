# 🚀 Scraping Integration Guide

This guide provides technical details on how the scraping engine is architected and integrated with the Streamlit UI and the database.

## 🏗️ Architecture Overview

The scraping system is designed to be modular and decoupled. The UI triggers a background task, which runs an orchestrator that calls specialized scraping modules. The results are then passed to a synchronization engine that safely updates the database.

```mermaid
graph TD
    A[UI: Scraping Page] -- Clicks "Start" --> B[background_tasks.py]
    B -- Spawns Thread --> C[scraper.py: scrape_all()]
    C -- Calls --> D[scraper_job_boards.py]
    C -- Calls --> E[scraper_company_pages.py]
    D & E -- Return Normalized Jobs --> C
    C -- Passes Jobs to --> F[SmartSyncEngine]
    F -- Updates --> G[Database]
    B -- Updates State --> A
```

## 📁 Key Implementation Files

* **`src/scraper.py` (Orchestrator):** The `scrape_all()` function is the main entry point. It calls the other scrapers, combines their results, filters, deduplicates, and finally calls the `SmartSyncEngine`.

* **`src/scraper_job_boards.py`:** Uses the `JobSpy` library to efficiently scrape structured data from sites like LinkedIn and Indeed.

* **`src/scraper_company_pages.py`:** Uses `ScrapeGraphAI` and `LangGraph` to create an agentic workflow for extracting data from unstructured company career pages.

* **`src/services/database_sync.py`:** Contains the `SmartSyncEngine`, which handles all the logic for safely writing data to the database.

* **`src/ui/utils/background_tasks.py`:** Manages running the `scrape_all` function in a separate thread so the UI remains responsive.

* **`src/ui/pages/scraping.py`:** The Streamlit page that provides the user interface for starting scrapes and viewing real-time progress.

## 🎯 Core Integration Points

### 1. UI to Background Task

* The "Start Scraping" button on the `scraping.py` page calls `start_background_scraping()` from `background_tasks.py`.

* This function sets a flag `st.session_state.scraping_active = True` and starts a new `threading.Thread`.

* The thread's target is a worker function that calls the main `scraper.scrape_all()` orchestrator.

### 2. Real-Time Progress Reporting

* The background worker function updates `st.session_state` at various stages of the process (e.g., updating per-company status, overall progress).

* The `scraping.py` page has a small section of code that checks if `scraping_active` is true. If it is, it calls `st.rerun()` on a throttled interval (e.g., every 2 seconds).

* This `rerun` causes the page to redraw, reading the latest progress from `st.session_state` and updating the progress bars and metrics.

### 3. Scraper to Database

* The scraping modules (`scraper_job_boards.py`, `scraper_company_pages.py`) are designed to be **read-only** from the perspective of the `jobs` table. Their only job is to fetch and normalize data into `JobSQL` objects.

* The `scrape_all()` orchestrator collects the lists of these objects.

* It then passes the final, deduplicated list to `SmartSyncEngine.sync_jobs()`.

* The `SmartSyncEngine` is the **only** component responsible for writing job data to the database, ensuring all data goes through the same safe, intelligent update logic. This decoupling is a key architectural principle of the application.
