# ADR-013: State Management Architecture

## Status

**Accepted** - *Scope: Production architecture*

## Context

The AI Job Scraper requires sophisticated state management to handle:

- User session data across multiple pages
- Real-time scraping progress updates
- Complex filtering and search operations
- Application tracking workflows
- Cached data for performance
- Background task coordination

State management is critical for maintaining consistency, performance, and user experience across the application.

## Decision Drivers

1. **Reactivity**: Automatic UI updates when state changes
2. **Performance**: Efficient updates without full page reloads
3. **Scalability**: Handle growing complexity without performance degradation
4. **Type Safety**: Leverage Python type hints for reliability
5. **Developer Experience**: Intuitive patterns for state manipulation
6. **Real-time Support**: WebSocket integration for live updates
7. **Persistence**: Integration with database for permanent storage

## Considered Options

### Option 1: Centralized Monolithic State

Single AppState class containing all application state.

**Pros:**

- Simple to understand
- Single source of truth
- Easy to debug

**Cons:**

- Becomes unwieldy as app grows
- Poor separation of concerns
- Difficult to test in isolation
- Performance issues with large state objects

### Option 2: Redux-style Store Pattern

Implement Redux-like patterns with actions and reducers.

**Pros:**

- Predictable state updates
- Time-travel debugging
- Well-understood pattern

**Cons:**

- Verbose boilerplate code
- Overkill for our use case
- Not idiomatic Python
- Complex for simple updates

### Option 3: Hierarchical State with Inheritance

Use Reflex's state inheritance for logical grouping.

**Pros:**

- Logical organization
- Code reusability
- Clear relationships
- Native Reflex pattern

**Cons:**

- Deep inheritance can be confusing
- Potential for naming conflicts
- Coupling between state classes

### Option 4: Modular State Classes with Inheritance

Use inherited rx.State classes for logical grouping and separation of concerns.

**Pros:**

- Encapsulation of component logic
- Reusable components
- Clear separation of concerns
- Optimal performance
- Native Reflex patterns

**Cons:**

- Need to manage two state types
- Communication between states requires patterns
- Initial learning curve

## Decision

**We will use a Modular State Classes with Inheritance approach.**

## Detailed Design

### State Architecture

```python
# Global application state
class AppState(rx.State):
    """Root application state for shared data"""
    user_id: str
    session_token: str
    theme_config: dict
    notifications: list[Notification]

# Domain-specific state inheriting from AppState
class JobState(AppState):
    """Job management state"""
    jobs: list[Job]
    filters: JobFilters
    search_query: str
    
    @rx.var
    def filtered_jobs(self) -> list[Job]:
        # Computed property for filtered results
        pass

# Component-specific state (using regular functions)
def job_card(job: Job, expanded: bool = False):
    """Reusable job card component"""
    return rx.card(
        rx.vstack(
            rx.heading(job.title),
            rx.text(job.company.name),
            rx.cond(
                expanded,
                rx.text(job.description)
            ),
            rx.button(
                "Show More" if not expanded else "Show Less",
                on_click=lambda: JobState.toggle_job_expanded(job.id)
            )
        )
    )

# Real-time state for WebSocket updates
class ScrapingState(AppState):
    """Real-time scraping state"""
    progress: float
    current_source: str
    
    @rx.event(background=True)
    async def stream_updates(self):
        # Background task for streaming
        pass
```

### State Communication Patterns

```python
# Cross-state communication
class InterStateComm(AppState):
    @rx.event
    async def update_with_context(self):
        # Access another state
        job_state = await self.get_state(JobState)
        # Update based on job state
        self.process(job_state.selected_job)
```

## Rationale

The hybrid approach provides:

1. **Encapsulation**: Component state keeps UI logic isolated
2. **Reusability**: Components can be reused with their own state
3. **Performance**: Only relevant state triggers re-renders
4. **Organization**: Clear separation between UI and business logic
5. **Flexibility**: Choose appropriate state type per use case
6. **Reflex Native**: Uses framework's intended patterns

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive

- Clean separation of concerns
- Reusable component library
- Optimal rendering performance
- Easy to test components in isolation
- Scalable architecture
- Follows Reflex best practices

### Negative

- Initial complexity in understanding two state types
- Need patterns for state communication
- Potential state synchronization issues
- More files and classes to manage

### Neutral

- Team needs to learn both state patterns
- Documentation required for state conventions
- Refactoring existing code to new patterns

## Implementation Guidelines

### 1. State Organization

```text
state/
├── app.py          # Global AppState
├── auth.py         # Authentication state
├── jobs.py         # Job management state
├── scraping.py     # Scraping state
└── components/     # Component states
    ├── job_card.py
    └── filters.py
```

### 2. State Conventions

- Use `rx.State` for shared/global state
- Use functional components for reusable UI elements
- Prefix helper methods with underscore
- Use `@rx.var` for computed properties
- Use `@rx.cached_var` for expensive computations
- Use `yield` in event handlers for real-time UI updates

### 3. Performance Patterns

- Implement debouncing for search inputs
- Use pagination for large data sets
- Cache expensive computations
- Lazy load detailed data
- Use local storage for client-side caching

### 4. Testing Strategy

- Unit test state logic separately
- Mock WebSocket connections
- Test state transitions
- Verify computed properties
- Test background tasks

## Related ADRs

- **ADR-012**: Reflex UI Framework Decision - Foundation framework choice
- **ADR-014**: Real-time Updates Strategy - Implementation of real-time state patterns
- **ADR-015**: Component Library Selection - Component integration with state
- **ADR-016**: Routing and Navigation Design - URL state management patterns
- **ADR-020**: Reflex Local Development - Development-specific state usage

## References

- [Reflex State Management](https://reflex.dev/docs/state/overview/)
- [Background Tasks in Reflex](https://reflex.dev/docs/state/background-tasks/)
- [State Communication Patterns](https://reflex.dev/docs/state/overview/#client-states)
