# ADR-015: Component Library Selection

## Status

**Accepted** - *Scope: Production architecture*

## Context

The AI Job Scraper requires a comprehensive set of UI components to deliver a professional, modern user experience. Components needed include:

- Data tables with sorting and filtering
- Forms with validation
- Cards and lists for job display
- Navigation elements (sidebar, navbar, tabs)
- Modals and dialogs
- Charts and visualizations
- Progress indicators
- Notifications and toasts

The component library must integrate seamlessly with Reflex and provide consistent styling, accessibility, and behavior.

## Decision Drivers

1. **Reflex Compatibility**: Native integration with Reflex framework
2. **Component Coverage**: Comprehensive set of components for our needs
3. **Customization**: Ability to theme and customize appearance
4. **Accessibility**: WCAG 2.2 AA compliance
5. **Performance**: Lightweight and optimized
6. **Documentation**: Clear examples and API documentation
7. **Maintenance**: Active development and support

## Considered Options

### Option 1: Reflex Built-in Components

Use only Reflex's native component library.

**Pros:**

- No additional dependencies
- Guaranteed compatibility
- Consistent API
- Well-documented
- Actively maintained

**Cons:**

- Limited advanced components
- Less customization options
- May need custom components

**Component Coverage:**

- ✅ Basic form elements
- ✅ Layout components
- ✅ Navigation elements
- ✅ Data display
- ⚠️ Limited advanced components
- ⚠️ Basic charts only

### Option 2: Reflex + Radix UI Primitives

Combine Reflex components with Radix UI primitives.

**Pros:**

- Accessible by default
- Unstyled primitives
- Full customization control
- Production-ready
- Great DX

**Cons:**

- Need to style everything
- More initial work
- Learning curve

**Component Coverage:**

- ✅ All form primitives
- ✅ Advanced interactions
- ✅ Accessibility built-in
- ❌ No charts
- ❌ No styled components

### Option 3: Reflex + Ant Design Wrapper

Wrap Ant Design components for Reflex.

**Pros:**

- Comprehensive component set
- Professional design
- Extensive documentation
- Battle-tested

**Cons:**

- Requires wrapping work
- Potential compatibility issues
- Heavier bundle size
- Opinionated styling

### Option 4: Reflex + Recharts/Victory

Reflex components plus specialized chart library.

**Pros:**

- Best-in-class charts
- Highly customizable
- Good performance
- Reflex has Recharts support

**Cons:**

- Only solves visualization needs
- Additional dependency
- Learning curve for charts

### Option 5: Hybrid Approach

Reflex built-ins + Radix primitives + Recharts.

**Pros:**

- Best tool for each job
- Maximum flexibility
- Gradual adoption possible
- Leverages Reflex's strengths

**Cons:**

- Multiple libraries to manage
- Potential inconsistencies
- More complex setup

## Decision

**We will use a Hybrid Approach: Reflex built-in components as the foundation, with Radix UI primitives for advanced interactions and Recharts for data visualization.**

## Detailed Component Strategy

### Component Selection Matrix

| Component Type | Library Choice | Rationale |
|---------------|---------------|-----------|
| **Layout** | | |
| Grid/Flex | Reflex (`rx.box`, `rx.flex`) | Native, performant |
| Container | Reflex (`rx.container`) | Built-in responsiveness |
| Spacer | Reflex (`rx.spacer`) | Simple, effective |
| **Navigation** | | |
| Sidebar | Reflex (`rx.drawer`) | Native drawer component |
| Navbar | Reflex (`rx.hstack` + custom) | Composable |
| Tabs | Reflex (`rx.tabs`) | Built-in functionality |
| **Forms** | | |
| Input | Reflex (`rx.input`) | Native validation |
| Select | Reflex (`rx.select`) | Good enough for most cases |
| Checkbox | Reflex (`rx.checkbox`) | Standard implementation |
| Radio | Reflex (`rx.radio`) | Grouped options |
| Date Picker | Radix Primitives | Better UX needed |
| Autocomplete | Radix Primitives | Advanced interaction |
| **Data Display** | | |
| Table | Reflex (`rx.table`) | Native sorting/filtering |
| Cards | Reflex (`rx.card`) | Flexible container |
| List | Reflex (`rx.list`) | Simple lists |
| DataGrid | Custom with Reflex | Complex requirements |
| **Feedback** | | |
| Toast | Reflex (`rx.toast`) | Built-in notifications |
| Modal | Reflex (`rx.dialog`) | Native implementation |
| Popover | Reflex (`rx.popover`) | Contextual info |
| Tooltip | Reflex (`rx.tooltip`) | Hover information |
| Progress | Reflex (`rx.progress`) | Loading states |
| **Visualization** | | |
| Charts | Recharts via Reflex | `rx.recharts.*` |
| Sparklines | Recharts | Inline charts |
| Gauges | Recharts | Progress viz |

### Implementation Patterns

```python
# Standard Reflex components
def job_card(job: Job):
    return rx.card(
        rx.vstack(
            rx.heading(job.title),
            rx.text(job.company),
            rx.badge(job.status),
            spacing="2"
        ),
        width="100%"
    )

# Radix primitive for advanced interaction
class DateRangePicker(rx.Component):
    """Custom date range picker using Radix"""
    library = "@radix-ui/react-date-picker"
    tag = "DateRangePicker"
    # ... implementation

# Recharts for visualization
def job_trends_chart(data: list):
    return rx.recharts.line_chart(
        rx.recharts.line(
            data_key="jobs",
            stroke="#8884d8"
        ),
        rx.recharts.x_axis(data_key="date"),
        rx.recharts.y_axis(),
        rx.recharts.tooltip(),
        data=data,
        height=400
    )
```

### Styling Strategy

```python
# Theme configuration
THEME = {
    "font": {
        "family": "Inter, system-ui, sans-serif",
    },
    "colors": {
        "primary": "#0066FF",
        "secondary": "#7C3AED",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
    },
    "radius": {
        "sm": "0.25rem",
        "md": "0.375rem",
        "lg": "0.5rem",
    },
    "spacing": {
        "unit": "0.25rem",  # 4px
    }
}

# Apply theme
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        radius="medium",
    )
)
```

## Rationale

The hybrid approach provides:

1. **Pragmatic**: Use built-in components where sufficient
2. **Flexible**: Add specialized components as needed
3. **Consistent**: Unified theming across all components
4. **Performant**: Only include what we need
5. **Maintainable**: Leverage framework updates

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive

- Minimal custom component development
- Leverages Reflex's built-in capabilities
- Can add advanced components selectively
- Good documentation for all choices
- Future-proof architecture

### Negative

- Need to manage multiple component sources
- Potential styling inconsistencies
- Some wrapper code required
- Testing across component libraries

### Neutral

- Team needs to learn component APIs
- Documentation for component usage patterns
- Style guide development needed

## Migration Path

1. Start with Reflex built-in components
2. Identify gaps in functionality
3. Implement Radix primitives for complex interactions
4. Add Recharts for data visualization
5. Create wrapper components for consistency
6. Document component usage patterns

## Component Development Guidelines

### Creating Custom Components

```python
def custom_input_component(label: str, value: str, error: str = "", **props):
    """Template for custom input components"""
    return rx.vstack(
        rx.text(label),
        rx.input(
            value=value,
            **props
        ),
        rx.cond(
            error != "",
            rx.text(error, color="red", size="sm")
        )
    )
```

### Accessibility Requirements

- All custom components must support keyboard navigation
- ARIA labels for all interactive elements
- Focus management for modals and popovers
- Color contrast ratios meeting WCAG 2.2 AA
- Screen reader support

## Testing Strategy

- Component unit tests with Reflex testing utilities
- Visual regression testing for styling
- Accessibility testing with axe-core
- Cross-browser compatibility testing
- Performance profiling for complex components

## Related ADRs

- **ADR-012**: Reflex UI Framework Decision - Foundation framework for component choices
- **ADR-013**: State Management Architecture - Integration of components with state
- **ADR-014**: Real-time Updates Strategy - Real-time component patterns
- **ADR-016**: Routing and Navigation Design - Navigation component integration
- **ADR-020**: Reflex Local Development - Development-specific component usage

## References

- [Reflex Component Library](https://reflex.dev/docs/library/)
- [Radix UI Primitives](https://www.radix-ui.com/primitives)
- [Recharts Documentation](https://recharts.org/)
- [Reflex Custom Components Guide](https://reflex.dev/docs/wrapping-react/overview/)
- [WCAG 2.2 Guidelines](https://www.w3.org/WAI/WCAG22/quickref/)
