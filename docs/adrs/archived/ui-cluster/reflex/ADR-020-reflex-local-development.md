# ADR-020: Reflex Local Development UI

## Title

Reflex Component Architecture for Local Development

## Version/Date

2.0 / August 19, 2025

## Status

**Accepted** - *Implementation in progress*

**Current Reality**: Active migration from Streamlit to Reflex UI framework. Local development environment being configured for Reflex component-based architecture as defined in this ADR.

## Description

Simple Reflex-based UI component architecture for local development. Leverages Reflex's native capabilities for real-time updates, component-based design, and state management without production complexity. Provides clean, maintainable UI components optimized for development workflow and iteration speed.

## Context

### Local Development Focus

This UI architecture prioritizes:

1. **Development Velocity**: Fast iteration with hot reload
2. **Simple Components**: Clean, understandable component structure
3. **Reflex Native**: Use framework capabilities without custom complexity
4. **Real-Time Updates**: Basic state synchronization for development
5. **Easy Debugging**: Clear component hierarchy and state management

### Framework Capabilities

- **Reflex Framework**: Python-native UI with React-like patterns
- **Component Architecture**: Reusable components with clean separation
- **State Management**: Native Reflex state with automatic updates
- **Real-Time**: Built-in WebSocket abstraction with yield patterns
- **Styling**: Chakra UI integration for consistent design

## Decision

**Use Simple Reflex Component Architecture** for local development:

### Basic Component Structure

```python
# src/components/base.py
import reflex as rx
from typing import List, Dict, Any

def app_header() -> rx.Component:
    """Simple application header."""
    return rx.hstack(
        rx.heading("AI Job Scraper", size="lg"),
        rx.spacer(),
        rx.hstack(
            rx.link("Jobs", href="/jobs"),
            rx.link("Companies", href="/companies"),
            rx.link("Settings", href="/settings"),
            spacing="4"
        ),
        width="100%",
        padding="4",
        bg="gray.50",
        border_bottom="1px solid",
        border_color="gray.200"
    )

def app_footer() -> rx.Component:
    """Simple application footer."""
    return rx.center(
        rx.text(
            "AI Job Scraper - Local Development",
            color="gray.500",
            font_size="sm"
        ),
        padding="4",
        border_top="1px solid",
        border_color="gray.200"
    )

def page_container(*children, title: str = "") -> rx.Component:
    """Standard page container."""
    return rx.vstack(
        app_header(),
        rx.container(
            rx.cond(
                title,
                rx.heading(title, size="xl", margin_bottom="4")
            ),
            *children,
            max_width="1200px",
            padding="4"
        ),
        app_footer(),
        min_height="100vh",
        width="100%",
        spacing="0"
    )
```

### Job Components

```python
# src/components/job_components.py
import reflex as rx
from typing import Dict, List

def job_card(job: Dict) -> rx.Component:
    """Simple job card component."""
    return rx.card(
        rx.vstack(
            # Header
            rx.hstack(
                rx.vstack(
                    rx.heading(job["title"], size="md"),
                    rx.text(job["company"], color="blue.500", font_weight="bold"),
                    align_items="start",
                    spacing="1"
                ),
                rx.spacer(),
                rx.icon_button(
                    rx.icon("heart"),
                    variant="ghost",
                    color="red.500" if job.get("is_favorited") else "gray.400"
                ),
                width="100%",
                align_items="start"
            ),
            
            # Content
            rx.text(job.get("location", "Location not specified"), color="gray.600"),
            rx.cond(
                job.get("salary_text"),
                rx.text(job["salary_text"], color="green.600", font_weight="medium")
            ),
            rx.text(
                job.get("description", "")[:200] + "..." if len(job.get("description", "")) > 200 else job.get("description", ""),
                color="gray.700"
            ),
            
            # Footer
            rx.hstack(
                rx.text(f"Posted: {job.get('scraped_at', 'Unknown')}", font_size="sm", color="gray.500"),
                rx.spacer(),
                rx.link(
                    rx.button("View Details", size="sm"),
                    href=job.get("url", "#"),
                    is_external=True
                ),
                width="100%"
            ),
            
            align_items="start",
            spacing="3"
        ),
        padding="4",
        margin="2",
        width="100%"
    )

def job_list(jobs: List[Dict]) -> rx.Component:
    """Simple job list component."""
    return rx.cond(
        jobs,
        rx.vstack(
            rx.foreach(jobs, job_card),
            spacing="2",
            width="100%"
        ),
        rx.center(
            rx.text("No jobs found", color="gray.500"),
            height="200px"
        )
    )

def job_search_bar(on_search) -> rx.Component:
    """Simple job search component."""
    return rx.hstack(
        rx.input(
            placeholder="Search jobs, companies, or locations...",
            width="100%"
        ),
        rx.button(
            "Search",
            on_click=on_search,
            bg="blue.500",
            color="white"
        ),
        spacing="2",
        width="100%"
    )

def job_filters() -> rx.Component:
    """Simple job filters."""
    return rx.hstack(
        rx.select(
            ["All Locations", "Remote", "San Francisco", "New York", "Seattle"],
            placeholder="Location",
            width="200px"
        ),
        rx.select(
            ["All Types", "Full-time", "Part-time", "Contract", "Internship"],
            placeholder="Job Type",
            width="200px"
        ),
        rx.checkbox("Favorites Only"),
        spacing="3",
        wrap="wrap"
    )
```

### Statistics Components

```python
# src/components/stats_components.py
import reflex as rx
from typing import Dict

def stats_card(title: str, value: str, color: str = "blue") -> rx.Component:
    """Simple statistics card."""
    return rx.card(
        rx.vstack(
            rx.text(title, font_size="sm", color="gray.600"),
            rx.text(value, font_size="2xl", font_weight="bold", color=f"{color}.500"),
            align_items="center",
            spacing="2"
        ),
        padding="4",
        text_align="center"
    )

def stats_overview(stats: Dict) -> rx.Component:
    """Statistics overview component."""
    return rx.hstack(
        stats_card("Total Jobs", str(stats.get("total_jobs", 0)), "blue"),
        stats_card("Active Jobs", str(stats.get("active_jobs", 0)), "green"),
        stats_card("Companies", str(stats.get("total_companies", 0)), "purple"),
        stats_card("Favorites", str(stats.get("favorited_jobs", 0)), "red"),
        spacing="4",
        width="100%",
        wrap="wrap"
    )

def progress_indicator(progress: float, message: str = "") -> rx.Component:
    """Simple progress indicator."""
    return rx.vstack(
        rx.progress(value=progress, width="100%"),
        rx.text(f"{progress:.1f}% {message}", font_size="sm", color="gray.600"),
        spacing="2",
        width="100%"
    )
```

### Page Components

```python
# src/pages/home.py
import reflex as rx
from src.components.base import page_container
from src.components.stats_components import stats_overview
from src.state.database_state import DatabaseState

def home_content() -> rx.Component:
    """Home page content."""
    return rx.vstack(
        # Welcome section
        rx.text(
            "Welcome to AI Job Scraper",
            font_size="xl",
            color="gray.700",
            margin_bottom="4"
        ),
        
        # Statistics
        stats_overview(DatabaseState.stats),
        
        # Quick actions
        rx.hstack(
            rx.button(
                "Start Scraping",
                bg="green.500",
                color="white",
                size="lg"
            ),
            rx.button(
                "View Jobs",
                bg="blue.500",
                color="white",
                size="lg"
            ),
            spacing="4"
        ),
        
        spacing="6",
        width="100%"
    )

def page() -> rx.Component:
    """Home page."""
    DatabaseState.load_stats()  # Load stats on page load
    
    return page_container(
        home_content(),
        title="Dashboard"
    )
```

```python
# src/pages/jobs.py
import reflex as rx
from src.components.base import page_container
from src.components.job_components import job_search_bar, job_filters, job_list
from src.state.database_state import DatabaseState

def jobs_content() -> rx.Component:
    """Jobs page content."""
    return rx.vstack(
        # Search and filters
        job_search_bar(DatabaseState.search_jobs),
        job_filters(),
        
        # Results
        rx.text(f"Found {len(DatabaseState.jobs)} jobs", color="gray.600"),
        job_list(DatabaseState.jobs),
        
        spacing="4",
        width="100%"
    )

def page() -> rx.Component:
    """Jobs page."""
    DatabaseState.load_jobs()  # Load jobs on page load
    
    return page_container(
        jobs_content(),
        title="Job Listings"
    )
```

### State Integration

```python
# src/state/ui_state.py
import reflex as rx
from typing import Dict, List

class UIState(rx.State):
    """UI-specific state management."""
    
    # Navigation
    current_page: str = "home"
    
    # Search
    search_query: str = ""
    selected_location: str = "All Locations"
    selected_job_type: str = "All Types"
    favorites_only: bool = False
    
    # UI feedback
    toast_message: str = ""
    toast_type: str = "info"  # info, success, warning, error
    
    def navigate_to(self, page: str):
        """Navigate to a page."""
        self.current_page = page
    
    def update_search(self, query: str):
        """Update search query."""
        self.search_query = query
    
    def show_toast(self, message: str, toast_type: str = "info"):
        """Show toast notification."""
        self.toast_message = message
        self.toast_type = toast_type
        # Could integrate with Reflex toast component
    
    def clear_filters(self):
        """Clear all filters."""
        self.search_query = ""
        self.selected_location = "All Locations"
        self.selected_job_type = "All Types"
        self.favorites_only = False
```

### Simple Real-Time Updates

```python
# src/components/realtime_components.py
import reflex as rx
import asyncio
from src.state.task_state import TaskState

def realtime_scraping_status() -> rx.Component:
    """Real-time scraping status component."""
    return rx.cond(
        TaskState.current_task_status == "running",
        rx.card(
            rx.vstack(
                rx.heading("Scraping in Progress", size="md"),
                rx.progress(value=TaskState.current_task_progress, width="100%"),
                rx.text(TaskState.current_task_message, color="gray.600"),
                rx.button(
                    "Cancel",
                    on_click=TaskState.cancel_current_task,
                    bg="red.500",
                    color="white"
                ),
                spacing="3"
            ),
            padding="4",
            bg="blue.50",
            border="1px solid",
            border_color="blue.200"
        )
    )

def live_job_counter(count: int) -> rx.Component:
    """Live job counter that updates in real-time."""
    return rx.hstack(
        rx.icon("briefcase"),
        rx.text(f"{count} jobs", font_weight="bold"),
        rx.cond(
            TaskState.current_task_status == "running",
            rx.spinner(size="sm")
        ),
        spacing="2",
        align_items="center"
    )
```

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive Outcomes

- **Simple Architecture**: Clean component structure easy to understand and modify
- **Reflex Native**: Uses framework capabilities without custom complexity
- **Development Focus**: Optimized for development iteration and debugging
- **Real-Time Updates**: Built-in state synchronization for live updates
- **Component Reuse**: Modular components for maintainability

### Negative Consequences

- **Development Scope**: Components designed for local development, not production scale
- **Limited Optimization**: Basic patterns without production performance tuning
- **Simple Styling**: Basic styling without advanced design system
- **No Advanced Features**: Missing complex UI patterns and interactions

### Risk Mitigation

- **Clear Structure**: Well-organized component hierarchy
- **Documentation**: Component usage examples and patterns
- **Upgrade Path**: Clear migration to production UI components
- **Consistent Patterns**: Reusable design patterns throughout

## Development Guidelines

### Component Design

- Keep components simple and focused on single responsibilities
- Use Reflex native patterns and avoid custom complexity
- Implement proper state integration with event handlers
- Include helpful debugging information and logging

### State Management

- Use Reflex state variables for component data
- Implement real-time updates with yield patterns
- Keep state operations simple and predictable
- Add proper error handling and user feedback

### Styling and Layout

- Use Chakra UI components for consistency
- Implement responsive design with simple patterns
- Focus on readability and development experience
- Use consistent spacing and color schemes

## Related ADRs

- **Would Support ADR-017**: Local Development Architecture (UI component integration)
- **Would Use ADR-018**: Local Database Setup (data integration)
- **Would Use ADR-023**: Background Job Processing with RQ/Redis (real-time updates)
- **Future Migration From**: Current Streamlit implementation

## Success Criteria

- [ ] Component architecture is clean and maintainable
- [ ] Real-time updates work correctly with Reflex state
- [ ] Navigation and routing function properly
- [ ] Basic responsive design works on different screen sizes
- [ ] Development workflow supports hot reload and iteration
- [ ] Components integrate properly with database and task states

---

*This ADR provides a simple, maintainable UI component architecture for local development using Reflex framework's native capabilities.*
