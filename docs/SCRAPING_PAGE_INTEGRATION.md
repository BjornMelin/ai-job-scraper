# Scraping Page Integration Guide

This document explains how to integrate the new scraping dashboard page into the AI Job Scraper Streamlit application.

## Files Created

### 1. Background Task Manager (`src/ui/utils/background_tasks.py`)

A comprehensive background task management system with:

- **StreamlitTaskManager**: Main class for managing scraping operations

- **CompanyProgress**: Dataclass for tracking individual company scraping status

- **ScrapeProgress**: Dataclass for overall scraping operation progress

- **Real-time progress updates** through session state

- **Background threading** with proper daemon thread management

- **Error handling and recovery** mechanisms

### 2. Scraping Page UI (`src/ui/pages/scraping.py`)

A complete scraping dashboard with:

- **Control buttons**: Start/Stop scraping, Reset progress

- **Real-time progress display**: Overall progress bar and company status

- **Status indicators**: Visual feedback for scraping state

- **Recent activity metrics**: Last run statistics

- **Debug information** panel for development

## Key Features Implemented

### ✅ Required Features

1. **"Start Scraping" button** that calls `StreamlitTaskManager.start_background_scraping`
2. **UI section visible when scraping is active** using `st.session_state.get("scraping_active", False)`
3. **Overall progress bar** using `st.progress` inside the active section
4. **Company status display** iterating through `st.session_state.progress_data` with `st.text`
5. **Real-time updates** without blocking the UI

### ✅ Additional Features

1. **Stop scraping functionality** for user control
2. **Progress reset capability** for dashboard cleanup
3. **Comprehensive error handling** with user-friendly messages
4. **Active company detection** from database
5. **Session state management** for persistent progress tracking
6. **Auto-refresh mechanism** for real-time updates

## Integration Steps

### Method 1: Add as a New Page (Recommended)

To add the scraping page as a separate page in the application:

1. **Update main.py** to support multiple pages:

```python
import streamlit as st
from src.ui.components.sidebar import render_sidebar
from src.ui.pages.jobs import render_jobs_page
from src.ui.pages.scraping import render_scraping_page  # New import
from src.ui.state.app_state import StateManager
from src.ui.styles.theme import load_theme

def main() -> None:
    st.set_page_config(
        page_title="AI Job Tracker",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "AI-powered job tracker for managing your job search efficiently."
        },
    )

    load_theme()
    StateManager()

    # Add page navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Jobs Dashboard", "Scraping Dashboard"],
        index=0
    )

    render_sidebar()

    # Render selected page
    if page == "Jobs Dashboard":
        render_jobs_page()
    elif page == "Scraping Dashboard":
        render_scraping_page()
```

### Method 2: Add as a Tab in Existing Page

To add scraping functionality as a tab in the existing jobs page:

```python

# In src/ui/pages/jobs.py, add a tab for scraping
def render_jobs_page() -> None:
    # ... existing header code ...
    
    # Create tabs
    tab1, tab2 = st.tabs(["Jobs", "Scraping"])
    
    with tab1:
        # ... existing jobs functionality ...
    
    with tab2:
        from src.ui.pages.scraping import render_scraping_page
        render_scraping_page()
```

### Method 3: Add to Sidebar

To add scraping controls to the sidebar:

```python

# In src/ui/components/sidebar.py, add a new section
def render_sidebar() -> None:
    state_manager = StateManager()

    with st.sidebar:
        _render_search_filters(state_manager)
        st.divider()
        _render_view_settings(state_manager)
        st.divider()
        _render_scraping_controls()  # New section
        st.divider()
        _render_company_management()

def _render_scraping_controls():
    """Render scraping controls in sidebar."""
    from src.ui.utils.background_tasks import StreamlitTaskManager
    
    st.markdown("### 🔍 Scraping")
    
    is_scraping = StreamlitTaskManager.is_scraping_active()
    
    if st.button("🚀 Start Scraping", disabled=is_scraping):
        StreamlitTaskManager.start_background_scraping()
        st.rerun()
    
    if st.button("⏹️ Stop Scraping", disabled=not is_scraping):
        StreamlitTaskManager.stop_scraping()
        st.rerun()
    
    if is_scraping:
        progress_data = StreamlitTaskManager.get_progress_data()
        st.progress(progress_data.overall_progress / 100.0)
        st.caption(progress_data.current_stage)
```

## Usage Examples

### Starting a Scraping Operation

```python
from src.ui.utils.background_tasks import StreamlitTaskManager

# Check if scraping is active
if StreamlitTaskManager.is_scraping_active():
    st.info("Scraping is already running...")
else:
    # Start scraping
    task_id = StreamlitTaskManager.start_background_scraping()
    st.success(f"Scraping started with task ID: {task_id}")
```

### Monitoring Progress

```python

# Get real-time progress data
progress_data = StreamlitTaskManager.get_progress_data()

# Display overall progress
st.progress(progress_data.overall_progress / 100.0)
st.text(f"Stage: {progress_data.current_stage}")

# Display company-specific progress
for company_name, company_progress in progress_data.companies.items():
    st.text(f"{company_name}: {company_progress.status}")
```

### Stopping and Resetting

```python

# Stop active scraping
if StreamlitTaskManager.stop_scraping():
    st.warning("Scraping stopped")

# Reset progress data
StreamlitTaskManager.reset_progress()
st.info("Progress data cleared")
```

## Technical Implementation Details

### Session State Management

The implementation uses Streamlit's session state to maintain:

- `scraping_active`: Boolean flag for scraping status

- `progress_data`: ScrapeProgress object with detailed progress information

- `task_id`: Unique identifier for tracking operations

- `scraping_thread`: Reference to background thread

### Background Threading

Uses Python's `threading.Thread` with:

- **Daemon threads** for automatic cleanup

- **Task ID verification** to prevent race conditions

- **Proper exception handling** with error propagation

- **Session state updates** for real-time UI refresh

### Real-time Updates

Achieves real-time updates through:

- **Auto-refresh mechanism** with `st.rerun()`

- **Progress polling** at regular intervals

- **Session state synchronization** between background thread and UI

- **Responsive UI** that remains interactive during operations

## Error Handling

The implementation includes comprehensive error handling for:

- **Database connection failures**

- **Scraping operation exceptions**

- **Thread management errors**

- **Session state corruption**

- **User interaction conflicts**

## Security Considerations

- **Daemon threads** prevent hanging processes

- **Task ID validation** prevents unauthorized progress updates

- **Error message sanitization** to prevent information leakage

- **Resource cleanup** on operation completion or failure

## Performance Optimizations

- **Efficient progress updates** with minimal UI redraws

- **Database connection pooling** through SessionLocal

- **Memory-efficient progress tracking** with structured data

- **Background operation isolation** from UI thread

## Testing the Implementation

1. **Start the application**: `streamlit run src/main.py`
2. **Navigate to scraping page** (after integration)
3. **Click "Start Scraping"** to begin operation
4. **Monitor real-time progress** in the dashboard
5. **Test stop functionality** during operation
6. **Verify reset capabilities** after completion

## Future Enhancements

Potential improvements for future versions:

1. **Progress persistence** across app restarts
2. **Scheduled scraping** with cron-like functionality
3. **Historical progress tracking** and analytics
4. **Email notifications** for completion/errors
5. **Multi-user support** with user-specific sessions
6. **Advanced filtering** for selective company scraping

## Dependencies

The implementation requires:

- `streamlit >= 1.47.0` (for latest session state features)

- `threading` (Python standard library)

- `dataclasses` (Python standard library)

- `time` (Python standard library)

- `logging` (Python standard library)

All dependencies are part of the existing project requirements.
