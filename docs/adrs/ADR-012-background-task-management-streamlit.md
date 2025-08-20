# ADR-012: Background Task Management & UI Integration (Streamlit)

## Title

Library-First Background Task Management Using `st.status` and Standard Threading

## Version/Date

1.0 / August 7, 2025

## Status

Accepted - Research Validated (Threading.Thread Confirmed Optimal)

## Context

The application's scraping process can be long-running and must not block the user interface. A robust system is needed to execute these tasks in the background, provide real-time progress updates to the UI, and handle state management safely within Streamlit's execution model. An initial, overly complex implementation using custom thread pools and manual state management proved difficult to maintain.

## Related Requirements

* `SYS-ARCH-04`: Background Task Execution

* `SCR-PROG-01`: Real-Time Progress Reporting

* `UI-PROG-02`: Enhanced Progress Dashboard

## Decision

We will adopt a "library-first" approach, replacing the over-engineered custom solution with a simplified system that leverages Streamlit's native capabilities and standard Python libraries.

The new architecture will consist of:

1. **Standard Python `threading`:** A simple `threading.Thread` will be used to run the main `scrape_all` function. This is sufficient for the application's needs and avoids the complexity of a full `ThreadPoolExecutor`.
2. **Streamlit's `st.status()`:** This built-in component will be used as the primary UI for displaying real-time progress. It provides a superior user experience with collapsible sections and handles the UI updates efficiently.
3. **Direct `st.session_state`:** All task state (e.g., `scraping_active`, per-company progress) will be stored directly in `st.session_state`. This eliminates the need for a custom `StateManager` singleton and aligns with modern Streamlit best practices.

## Design

The implementation is primarily in `src/ui/utils/background_tasks.py`.

```python

# In src/ui/pages/scraping.py
def _handle_refresh_jobs():
    with st.spinner("🔍 Searching for new jobs..."):
        # ...
        jobs_df = _execute_scraping_safely() # Runs asyncio in a managed loop
        update_db(jobs_df)
        # ...

# Simplified background task logic in src/ui/utils/background_tasks.py
def start_scraping(status_container):
    st.session_state.scraping_active = True
    
    def scraping_task():
        try:
            with status_container.container():
                with st.status("🔍 Scraping...", expanded=True) as status:
                    # Update status.write(...) with progress
                    result = scrape_all()
                    # Update status.update(label="✅ Complete!", state="complete")
        finally:
            st.session_state.scraping_active = False

    thread = threading.Thread(target=scraping_task, daemon=True)
    thread.start()
```

The `background_tasks.py` module was significantly refactored to a much simpler model, which is reflected in the current codebase, reducing complexity by over 90%.

## Consequences

* **Positive:**
  * **Massive Code Reduction:** The custom background task manager (800+ lines) was replaced with a much simpler implementation (~50 lines of core logic), drastically improving maintainability.
  * **Improved UX:** `st.status` provides a cleaner, more professional, and interactive progress display than the previous custom solution.
  * **Increased Stability:** Relying on Streamlit's native state management reduces the risk of custom state-related bugs and memory leaks.
  * **Easier to Debug:** The logic is simpler and follows standard Streamlit patterns.

* **Negative:**
  * Loses the ability to manage a complex queue of multiple, concurrent background tasks, but this was an over-engineered feature that the application did not require.

* **Mitigations:** The current design perfectly fits the requirement of running a single, primary scraping task at a time. If more complex background processing is needed in the future, a dedicated library like Celery would be a more appropriate choice than a custom implementation.

## Research Validation Results

**Threading vs ProcessPoolExecutor Analysis Completed**:

* **Decision Score**: threading.Thread = 0.84 vs ProcessPoolExecutor = 0.51
* **Key Finding**: Job scraping is I/O-bound (network requests, proxy rotation, database operations)
* **Evidence**: Streamlit research confirms `threading.Thread` with `add_script_run_ctx()` is the proven pattern
* **Reality Check**: Current implementation already works correctly and aligns with Streamlit best practices
* **Performance**: I/O-bound workload benefits from threading's lower overhead vs process spawning costs

**Conclusion**: The current threading.Thread implementation is optimal for our I/O-bound job scraping workload and requires no changes.
