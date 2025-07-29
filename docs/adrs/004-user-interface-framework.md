# ADR-004: User Interface Framework

## Title

Selection of UI Framework for Interactive Dashboard

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Need a simple, interactive UI for job viewing/management (Streamlit excels in data apps with minimal code).

## Related Requirements

- Dashboard with tables, filters, edits.
- Integration with Python backend.

## Alternatives

- Dash: More complex for simple UIs.
- Flask + JS: High maintenance.

## Decision

Streamlit v1.47.1 for its ease in building data dashboards with widgets like data_editor/multiselect.

## Related Decisions

- ADR-005 (Enhances with views/features).

## Design

- **Setup**: st.set_page_config(layout="wide"); st.title("AI Job Tracker").
- **Integration**: Query DB to DF, display with widgets.
- **Implementation Notes**: Use session_state for persistence.
- **Testing**: Manual UI tests.

## Consequences

- Rapid development (Python-only).
- Reactive (state management).

- Less customizable (sufficient here).
