# ADR-005: UI Enhancements and Features

## Title

Dashboard Design with Views, Tabs, Search, Sorting, and Pagination

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Enhance UI for engagement/usability: Tabs for categories, toggle list/card views, filters/search/sorting/pagination, theming.

## Related Requirements

- Tabs (All/Favorites/Applied), views (list editable, card visual), per-tab search.
- Sorting/pagination in cards, global filters.

## Alternatives

- Single view: Less flexible.
- External components: Unneeded deps.

## Decision

Implement tabs with st.tabs, view toggle with st.radio, search with st.text_input (filter DF), sorting with st.selectbox (df.sort_values), pagination with buttons/session_state.page. Custom CSS for cards/theme.

## Related Decisions

- ADR-004 (Builds on Streamlit).
- ADR-003 (Queries from DB to DF).

## Design

- **Tabs/Views**: tab1, tab2 = st.tabs(...); if view_mode=='List': st.data_editor; else: columns for cards with HTML.
- **Search/Sort/Paginate**: search_term = st.text_input; df[contains(term)]; sorted_df = df.sort_values; paginated = iloc[start:end].
- **Theming**: st.markdown(CSS); Toggle with selectbox.
- **Integration**: display_jobs func per tab, with unique keys for widgets.
- **Implementation Notes**: Rerun on changes; Media queries for mobile.
- **Testing**: Emulate interactions, assert filters/sort/paginate work without overlaps.

## Consequences

- Highly usable (search/track in views/tabs).
- Responsive (CSS fixes mobile).

- Reruns on edits (Streamlit limitation, mitigated by keys).

**Changelog:**  

- 1.0 (July 29, 2025): Consolidated all UI features into one ADR for coherence.
