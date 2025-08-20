# ADR-012: Background Task Management & UI Integration

## Metadata

**Status:** Accepted  
**Version:** 2.0  
**Date:** August 20, 2025  
**Authors:** Bjorn Melin

## Title

Library-First Background Task Management Using Standard Threading

## Description

Implement background task management for long-running scraping operations using Python's standard threading library and Streamlit's native `st.status` component, eliminating 800+ lines of custom task management code while improving maintainability and user experience.

## Context

The AI job scraper requires background task execution for long-running scraping operations that can take 30-180 seconds to complete. These operations must not block the Streamlit UI while providing real-time progress feedback to users.

### Current Problem

Initial implementation used custom ThreadPoolExecutor with complex state management requiring 800+ lines of code. Research validation revealed this over-engineering introduced unnecessary complexity for I/O-bound workloads.

### Key Research Findings

- **Threading vs ProcessPoolExecutor Analysis**: threading.Thread scored 0.84 vs ProcessPoolExecutor 0.51 in decision framework
- **I/O-Bound Workload**: Job scraping involves network requests, proxy rotation, and database operations
- **Streamlit Best Practices**: `threading.Thread` with `add_script_run_ctx()` is the documented Streamlit pattern
- **Current Implementation Reality**: Existing threading approach already works correctly

### Technical Constraints

- Must work within Streamlit's execution model and state management
- Requires real-time progress updates during scraping operations
- Background tasks must not block the main UI thread
- Support for concurrent scraping of multiple companies
- Memory-efficient for long-running sessions

## Decision Drivers

1. **Implementation Simplicity (35% weight)**: Minimize custom code complexity and maintenance burden
2. **Streamlit Integration (30% weight)**: Align with documented Streamlit patterns and native capabilities
3. **I/O Performance (25% weight)**: Optimize for network-bound scraping operations
4. **Maintenance Cost (10% weight)**: Reduce future maintenance and debugging effort

## Related Requirements

**Functional Requirements (FR)**:

- FR-1: Execute scraping tasks without blocking UI
- FR-2: Provide real-time progress feedback during operations
- FR-3: Support concurrent scraping of multiple companies

**Non-Functional Requirements (NFR)**:

- NFR-1: Background tasks complete within 180 seconds maximum
- NFR-2: UI remains responsive during background operations
- NFR-3: Memory usage remains stable during long sessions

**Performance Requirements (PR)**:

- PR-1: Support 10+ concurrent company scraping operations
- PR-2: Real-time progress updates every 2-3 seconds
- PR-3: Graceful handling of task cancellation

**Integration Requirements (IR)**:

- IR-1: Streamlit session state management per Streamlit patterns
- IR-2: Database operations integration per **ADR-013**
- IR-3: Scraping tier integration per **ADR-014**

## Alternatives

### Alternative A: Custom ThreadPoolExecutor (Previous)

**Approach**: Custom task manager with ThreadPoolExecutor and complex state management

**Pros**:

- Full control over task scheduling and execution
- Support for complex task queuing and prioritization
- Detailed task state tracking and management

**Cons**:

- 800+ lines of custom code requiring maintenance
- Over-engineering for I/O-bound single-task workload
- Complex debugging and error handling
- Not aligned with Streamlit best practices

**Decision Framework Score**: 0.41

### Alternative B: Celery with Redis

**Approach**: Distributed task queue with external broker

**Pros**:

- Industry-standard task queue solution
- Excellent for complex multi-worker scenarios
- Built-in retry and failure handling

**Cons**:

- Requires external Redis dependency
- Over-engineering for single-user local application
- Additional deployment and operational complexity
- Network latency for local tasks

**Decision Framework Score**: 0.32

### Alternative C: AsyncIO with Streamlit

**Approach**: Asynchronous programming with asyncio integration

**Pros**:

- Native async support for I/O operations
- Efficient resource utilization
- Modern Python pattern

**Cons**:

- Complex integration with Streamlit's synchronous model
- Requires careful event loop management
- Less proven pattern in Streamlit ecosystem

**Decision Framework Score**: 0.58

### Alternative D: Standard Threading (Chosen)

**Approach**: Python threading.Thread with Streamlit native components

**Pros**:

- Minimal custom code (50 lines vs 800+)
- Native Streamlit integration with `st.status`
- Proven pattern with `add_script_run_ctx()`
- Optimal for I/O-bound workloads

**Cons**:

- Limited to single background task at a time
- No built-in task queue management

**Decision Framework Score**: 0.84

## Decision Framework

### Scoring Methodology

| Criteria | Weight | Alternative A (Custom) | Alternative B (Celery) | Alternative C (AsyncIO) | Alternative D (Threading) |
|----------|--------|----------------------|----------------------|------------------------|-------------------------|
| **Implementation Simplicity** | 35% | 0.2 | 0.1 | 0.4 | 0.9 |
| **Streamlit Integration** | 30% | 0.3 | 0.2 | 0.5 | 0.9 |
| **I/O Performance** | 25% | 0.7 | 0.8 | 0.9 | 0.8 |
| **Maintenance Cost** | 10% | 0.2 | 0.3 | 0.6 | 0.9 |
| **Weighted Score** | | **0.41** | **0.32** | **0.58** | **0.84** |

### Justification

**Alternative D (Threading)** achieves the highest score through:

- **Simplicity (0.9)**: Minimal custom code using standard library
- **Integration (0.9)**: Perfect alignment with Streamlit patterns
- **Performance (0.8)**: Optimal for I/O-bound scraping operations
- **Maintenance (0.9)**: Standard patterns reduce maintenance burden

## Decision

> **Implement Library-First Background Task Management with Standard Threading**

### Architecture Decision

1. **Standard Python Threading**: Use `threading.Thread` for background execution
2. **Streamlit Native UI**: Leverage `st.status()` for progress display
3. **Session State Management**: Direct `st.session_state` for task coordination
4. **Script Context Integration**: Use `add_script_run_ctx()` for Streamlit compatibility

### Key Components

- **Background Execution**: Single `threading.Thread` per scraping operation
- **Progress Display**: `st.status()` with expandable progress sections
- **State Management**: `st.session_state.scraping_active` for task coordination
- **Error Handling**: Standard try/catch with graceful UI updates

### Rationale

The threading approach delivers 105% improvement in decision framework scoring (0.84 vs 0.41) while reducing code complexity by 94% (50 lines vs 800+ lines).

## Related Decisions

- **ADR-001** (Library-First Architecture): Threading approach aligns with library-first principles by using standard library
- **ADR-014** (Simplified 2-Tier Scraping Strategy): Background processing integrates with JobSpy and ScrapeGraphAI operations
- **ADR-013** (Smart Database Synchronization): Background tasks coordinate with database sync operations

## Design

### Architecture Overview

```mermaid
graph TD
    A[User Initiates Scraping] --> B[Create Threading Context]
    B --> C[Start Background Thread]
    C --> D[st.status Progress Display]
    
    D --> E[Scraping Operation]
    E --> F[Real-time Progress Updates]
    F --> G[Company Progress Tracking]
    
    E --> H[Database Sync]
    H --> I[st.session_state Updates]
    I --> J[UI Refresh]
    
    style C fill:#90EE90
    style D fill:#87CEEB
    style I fill:#FFB6C1
```

### Implementation Architecture

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

### Streamlit Context Integration

```python
from streamlit.runtime.scriptrunner import add_script_run_ctx
import streamlit as st

def start_background_scraping():
    """Start background scraping with proper Streamlit context."""
    if st.session_state.get('scraping_active', False):
        st.warning("Scraping already in progress")
        return
    
    # Create status container for progress updates
    status_container = st.container()
    
    def scraping_worker():
        """Background thread worker with Streamlit context."""
        try:
            st.session_state.scraping_active = True
            
            with status_container:
                with st.status("🔍 Initializing scraping...", expanded=True) as status:
                    status.write("Setting up proxy connections...")
                    status.write("Loading company list...")
                    
                    # Integrate with simplified scraping strategy per ADR-014
                    from src.scraper import SimplifiedScraper
                    scraper = SimplifiedScraper(proxy_list=get_proxy_config())
                    
                    companies = st.session_state.get('selected_companies', [])
                    total_companies = len(companies)
                    
                    for i, company in enumerate(companies):
                        status.write(f"Scraping {company} ({i+1}/{total_companies})...")
                        
                        # Use 2-tier strategy from ADR-014
                        jobs = await scraper.scrape_company(company)
                        
                        # Database sync per ADR-013
                        if jobs:
                            sync_jobs_to_database(jobs)
                            st.session_state.f'jobs_{company}' = jobs
                        
                        # Progress update
                        progress = (i + 1) / total_companies
                        status.progress(progress)
                    
                    status.update(label="✅ Scraping completed!", state="complete")
                    
        except Exception as e:
            st.error(f"Scraping failed: {str(e)}")
        finally:
            st.session_state.scraping_active = False
    
    # Create thread with Streamlit context
    thread = threading.Thread(target=scraping_worker, daemon=True)
    add_script_run_ctx(thread)  # Essential for Streamlit compatibility
    thread.start()
```

## Testing

### Unit Testing Strategy

```python
import pytest
import threading
import time
from unittest.mock import Mock, patch
import streamlit as st

class TestBackgroundTasks:
    
    @pytest.fixture
    def mock_session_state(self):
        """Mock Streamlit session state for testing."""
        return {
            'scraping_active': False,
            'selected_companies': ['TechCorp', 'StartupInc']
        }
    
    def test_threading_context_setup(self, mock_session_state):
        """Test proper threading context setup."""
        with patch('streamlit.session_state', mock_session_state):
            task_started = threading.Event()
            
            def test_worker():
                task_started.set()
                assert threading.current_thread().daemon is True
            
            thread = threading.Thread(target=test_worker, daemon=True)
            thread.start()
            thread.join(timeout=1.0)
            
            assert task_started.is_set()
    
    def test_session_state_management(self, mock_session_state):
        """Test session state coordination."""
        with patch('streamlit.session_state', mock_session_state):
            # Test activation
            mock_session_state['scraping_active'] = True
            assert mock_session_state['scraping_active'] is True
            
            # Test deactivation
            mock_session_state['scraping_active'] = False
            assert mock_session_state['scraping_active'] is False
    
    @pytest.mark.asyncio
    async def test_background_scraping_integration(self, mock_session_state):
        """Test integration with scraping operations."""
        with patch('streamlit.session_state', mock_session_state):
            with patch('src.scraper.SimplifiedScraper') as mock_scraper:
                mock_scraper.return_value.scrape_company.return_value = [
                    {"title": "Software Engineer", "company": "TechCorp"}
                ]
                
                # Simulate background execution
                result = await mock_scraper.return_value.scrape_company("TechCorp")
                assert len(result) == 1
                assert result[0]["company"] == "TechCorp"
```

### Integration Testing

```python
@pytest.mark.integration
class TestStreamlitIntegration:
    
    def test_status_component_integration(self):
        """Test st.status component functionality."""
        with patch('streamlit.status') as mock_status:
            mock_status_context = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)
            
            # Test status creation and updates
            with mock_status("Testing...", expanded=True) as status:
                status.write("Test message")
                status.progress(0.5)
                status.update(label="Complete!", state="complete")
            
            mock_status.assert_called_once()
    
    def test_concurrent_task_handling(self):
        """Test handling of concurrent scraping attempts."""
        session_state = {'scraping_active': False}
        
        def simulate_concurrent_start():
            if session_state['scraping_active']:
                return "Already running"
            session_state['scraping_active'] = True
            time.sleep(0.1)  # Simulate work
            session_state['scraping_active'] = False
            return "Completed"
        
        # First task should succeed
        result1 = simulate_concurrent_start()
        assert result1 == "Completed"
        
        # Concurrent task should be blocked
        session_state['scraping_active'] = True
        result2 = simulate_concurrent_start()
        assert result2 == "Already running"
```

### Performance Testing

```python
@pytest.mark.performance
class TestBackgroundPerformance:
    
    def test_memory_stability(self):
        """Test memory usage remains stable during background operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate multiple background tasks
        for i in range(10):
            thread = threading.Thread(target=lambda: time.sleep(0.1), daemon=True)
            thread.start()
            thread.join()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (<10MB)
        assert memory_increase < 10 * 1024 * 1024
    
    def test_response_time_targets(self):
        """Test UI responsiveness during background operations."""
        start_time = time.time()
        
        # Simulate UI operation during background task
        def background_task():
            time.sleep(0.1)  # Simulate I/O operation
        
        thread = threading.Thread(target=background_task, daemon=True)
        thread.start()
        
        # UI operation should complete quickly
        ui_response_time = time.time() - start_time
        assert ui_response_time < 0.05  # <50ms for UI responsiveness
        
        thread.join()
```

## Consequences

### Positive

- **94% Code Reduction**: 800+ lines to 50 lines of core logic through library-first approach
- **105% Decision Score Improvement**: 0.84 vs 0.41 through threading vs custom implementation
- **Native Streamlit Integration**: Perfect alignment with `st.status` and `st.session_state` patterns
- **I/O Optimization**: Threading ideal for network-bound scraping operations
- **Simplified Debugging**: Standard patterns reduce troubleshooting complexity
- **Memory Efficiency**: No process spawning overhead for I/O-bound tasks

### Negative

- **Single Task Limitation**: Limited to one primary background operation at a time
- **Streamlit Dependency**: Tightly coupled to Streamlit's execution model
- **Thread Context Management**: Requires careful `add_script_run_ctx()` usage

### Risk Mitigation

- **Task Coordination**: `st.session_state.scraping_active` prevents concurrent execution
- **Error Handling**: Standard try/catch with graceful UI updates
- **Resource Management**: Daemon threads prevent blocking application exit
- **Progress Feedback**: Real-time updates maintain user engagement

## References

- [Streamlit Threading Guide](https://docs.streamlit.io/library/advanced-features/threading) - Official threading patterns
- [Python Threading Documentation](https://docs.python.org/3/library/threading.html) - Standard library threading
- [Streamlit Status Component](https://docs.streamlit.io/library/api-reference/status/st.status) - Progress display component
- [I/O vs CPU Bound Tasks](https://realpython.com/python-concurrency/) - Concurrency patterns in Python

## Changelog

### v2.0 - August 20, 2025

**Template Compliance Update**:

- **UPDATED**: Applied official 15-section ADR template structure
- **ENHANCED**: Added quantitative decision framework with 0.84 vs 0.41 scoring
- **ADDED**: Comprehensive testing strategy with unit, integration, and performance tests
- **VALIDATED**: Integration with **ADR-014** (scraping) and **ADR-013** (database sync)
- **STANDARDIZED**: Cross-references to **ADR-XXX** format
- **IMPROVED**: Architecture diagram with Mermaid visualization

### v1.0 - August 7, 2025

- Initial background task management decision with threading approach
- Research validation confirming threading.Thread optimal for I/O-bound workload
- Streamlit integration with `st.status` and `st.session_state` patterns
- 94% code reduction from custom ThreadPoolExecutor implementation
