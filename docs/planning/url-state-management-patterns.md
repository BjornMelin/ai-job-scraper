# URL State Management Patterns for Reflex Implementation

## Overview

This document preserves valuable URL state management patterns discovered during Streamlit prototyping and provides guidance for implementing similar functionality in the Reflex architecture. These patterns support shareable job searches, bookmarkable views, and browser history support.

> **Source**: Adapted from legacy Streamlit implementation (`URL_STATE_IMPLEMENTATION.md`) for future Reflex integration per ADR-016 routing and navigation design.

## Core URL State Management Concepts

### 1. State Persistence Patterns

**Concept**: Sync application filter state with URL parameters for deep linking and sharing.

**Reflex Implementation Approach** (based on ADR-016):

```python
class URLStateManager(rx.State):
    """Manage application state in URLs for Reflex"""
    
    # URL parameters mapping
    url_params: dict = {}
    
    @rx.event
    def sync_filters_from_url(self):
        """Read filter parameters from router and update state"""
        params = self.router.page.params
        
        # Sync keyword search
        if "keyword" in params:
            self.search_query = params["keyword"]
        
        # Sync company filters (comma-separated)
        if "company" in params:
            companies = params["company"].split(",") if params["company"] else []
            self.company_filters = companies
        
        # Continue for other filter types...
    
    @rx.event
    def update_url_from_filters(self):
        """Update URL when filters change"""
        params = {}
        
        # Build clean URL parameters
        if self.search_query:
            params["keyword"] = self.search_query
        
        if self.company_filters:
            params["company"] = ",".join(self.company_filters)
        
        # Update URL without navigation using Reflex router
        return self.navigate_to(self.router.page.path, params)
```

### 2. Filter State Parameters

**Supported Parameters** (adaptable from Streamlit implementation):

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `keyword` | string | Search terms | Non-empty string |
| `company` | string | Comma-separated company list | Valid company names |
| `salary_min` | integer | Minimum salary filter | 0-500,000 range |
| `salary_max` | integer | Maximum salary filter | 0-500,000 range |
| `date_from` | string | Start date filter (ISO format) | Valid date, 2020-present |
| `date_to` | string | End date filter (ISO format) | Valid date, reasonable future |
| `tab` | string | Selected tab | "all", "favorites", "applied" |
| `selected` | string | Selected items (comma-separated IDs) | Valid integer IDs |

### 3. URL Parameter Validation Patterns

**Validation Strategy**:

```python
class URLValidation:
    """URL parameter validation for Reflex"""
    
    @staticmethod
    def validate_salary_range(min_val: str, max_val: str) -> dict[str, str]:
        """Validate salary parameters"""
        errors = {}
        
        try:
            salary_min = int(min_val)
            if not 0 <= salary_min <= 500000:
                errors["salary_min"] = "Must be between 0 and 500,000"
        except ValueError:
            errors["salary_min"] = "Must be a valid number"
        
        # Similar validation for max...
        return errors
    
    @staticmethod
    def validate_date_range(date_from: str, date_to: str) -> dict[str, str]:
        """Validate date parameters"""
        errors = {}
        
        try:
            from_date = datetime.fromisoformat(date_from)
            if from_date.year < 2020 or from_date > datetime.now():
                errors["date_from"] = "Date must be between 2020 and now"
        except ValueError:
            errors["date_from"] = "Must be in ISO format (YYYY-MM-DD)"
        
        return errors
```

### 4. Clean URL Generation

**Pattern**: Only include non-default parameters in URLs to keep them clean and readable.

**Implementation**:

```python
def build_clean_url(base_path: str, filters: dict) -> str:
    """Build clean URL with only non-default parameters"""
    params = {}
    
    # Only add non-default values
    if filters.get("keyword"):
        params["keyword"] = filters["keyword"]
    
    if filters.get("salary_min", DEFAULT_MIN) != DEFAULT_MIN:
        params["salary_min"] = str(filters["salary_min"])
    
    # Continue for other parameters...
    
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base_path}?{query_string}"
    
    return base_path
```

## Integration with Reflex Architecture

### 1. State Management Integration (ADR-013)

**Approach**: Integrate URL state with modular state classes:

```python
class JobState(AppState):
    """Job management state with URL integration"""
    
    # Filter state
    search_query: str = ""
    company_filters: list[str] = []
    salary_range: tuple[int, int] = (DEFAULT_MIN, DEFAULT_MAX)
    
    @rx.event
    def on_mount(self):
        """Initialize from URL on page load"""
        self.sync_from_url()
        return self.load_filtered_data()
    
    @rx.event
    def update_search(self, query: str):
        """Update search and sync to URL"""
        self.search_query = query
        self.update_url_state()
        return self.load_filtered_data()
    
    def sync_from_url(self):
        """Sync state from URL parameters"""
        # Implementation based on router.page.params
        pass
    
    def update_url_state(self):
        """Update URL to reflect current state"""
        # Implementation using Reflex navigation
        pass
```

### 2. Routing Integration (ADR-016)

**URL Patterns** for job filtering:

```text
/jobs                                    # Default job listing
/jobs?keyword=python                     # Search results
/jobs?keyword=python&tab=favorites      # Search + tab selection
/jobs?company=openai,anthropic           # Company filter
/jobs?salary_min=100000&salary_max=200000 # Salary range
```

**Reflex Page Implementation**:

```python
@rx.page(route="/jobs", on_load=JobState.on_mount)
def jobs_page():
    return app_layout(
        rx.vstack(
            filter_sidebar(),  # Updates JobState and URL
            job_listing(),     # Displays filtered results
            width="100%"
        )
    )
```

### 3. Component Integration

**Filter Sidebar with URL Sync**:

```python
def filter_sidebar():
    return rx.vstack(
        # Search input
        rx.input(
            placeholder="Search jobs...",
            value=JobState.search_query,
            on_change=JobState.update_search,  # Triggers URL update
        ),
        
        # Company filter
        rx.multi_select(
            options=CompanyState.company_options,
            value=JobState.company_filters,
            on_change=JobState.update_companies,  # Triggers URL update
        ),
        
        # Salary range
        rx.range_slider(
            min=0,
            max=500000,
            value=JobState.salary_range,
            on_change=JobState.update_salary_range,  # Triggers URL update
        ),
        
        # Clear filters button
        rx.button(
            "Clear Filters",
            on_click=JobState.clear_filters,  # Clears state and URL
        ),
        
        width="300px",
        spacing="4"
    )
```

## Advanced Features

### 1. Shareable URLs

**Pattern**: Generate complete URLs for sharing specific filter combinations.

```python
class ShareableURL(rx.State):
    """Generate shareable URLs for current view"""
    
    @rx.var
    def current_shareable_url(self) -> str:
        """Get complete shareable URL"""
        base_url = self.get_base_url()  # From config
        current_path = self.router.page.path
        params = self.build_current_params()
        
        return f"{base_url}{current_path}?{params}"
    
    @rx.event
    def copy_shareable_url(self):
        """Copy shareable URL to clipboard"""
        return rx.call_script(
            f"navigator.clipboard.writeText('{self.current_shareable_url}')"
        )
```

### 2. Browser History Support

**Pattern**: Each filter change creates a new browser history entry for back/forward navigation.

```python
@rx.event
def update_with_history(self, new_state: dict):
    """Update state and create history entry"""
    # Update internal state
    self.apply_filters(new_state)
    
    # Create new URL and push to history
    new_url = self.build_url_from_state()
    return rx.call_script(
        f"window.history.pushState(null, '', '{new_url}')"
    )
```

### 3. Deep Linking Support

**Pattern**: Support direct navigation to specific content with preserved context.

```text
/jobs/123?tab=applied&keyword=python  # Job detail with context
/companies/456?selected=1,2,3         # Company detail with selection
```

## Implementation Priority

### Phase 1: Basic URL State (High Priority)

- [ ] Implement URL parameter parsing in Reflex router
- [ ] Create URLStateManager component
- [ ] Add basic filter persistence (keyword, company, tab)
- [ ] Implement URL validation

### Phase 2: Advanced Filtering (Medium Priority)

- [ ] Add salary range URL parameters
- [ ] Implement date range persistence
- [ ] Add selection state persistence
- [ ] Create clean URL generation

### Phase 3: Enhanced Features (Low Priority)

- [ ] Implement shareable URL generation
- [ ] Add browser history management
- [ ] Create deep linking support
- [ ] Add URL state analytics

## Migration Considerations

### From Streamlit to Reflex

**Key Differences**:

1. **State Management**: Streamlit `st.session_state` → Reflex `rx.State`
2. **URL Parameters**: Streamlit `st.query_params` → Reflex `router.page.params`
3. **Navigation**: Streamlit page reloads → Reflex client-side routing
4. **Component Updates**: Streamlit automatic rerun → Reflex reactive updates

**Migration Strategy**:

1. Preserve URL parameter schema for backward compatibility
2. Implement gradual migration with feature flags
3. Maintain validation patterns for data integrity
4. Test deep linking thoroughly across migration

## Related Architecture

- **ADR-012**: Reflex UI Framework Decision - Foundation framework
- **ADR-013**: State Management Architecture - State integration patterns
- **ADR-016**: Routing and Navigation Design - URL handling implementation
- **ADR-020**: Reflex Local Development - Development-specific patterns

## References

- [Reflex Routing Documentation](https://reflex.dev/docs/pages/overview/)
- [Reflex State Management](https://reflex.dev/docs/state/overview/)
- [URL State Management Patterns](https://www.patterns.dev/posts/url-state/)
- Original Streamlit Implementation (archived)
