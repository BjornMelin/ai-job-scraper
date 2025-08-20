# ADR-010: Component-Based UI Architecture

## Title

Adoption of a Component-Based UI Architecture with Multi-Page Navigation

## Version/Date

1.0 / August 7, 2025

## Status

**SUPERSEDED** - *Archived August 19, 2025*

**Superseded By:**
- ADR-022: Reflex UI Framework (complete framework replacement)
- ADR-025: Component Library Selection (component architecture patterns)
- ADR-026: Routing Navigation Design (navigation patterns)
- ADR-040: Reflex Local Development (development architecture)
- ADR-023: State Management Architecture (state organization)

**Supersession Rationale:** This ADR defined a Streamlit-based component architecture that is incompatible with the new Reflex framework. All implementation details (st.navigation(), st.Page(), Streamlit patterns) cannot be used with Reflex. All valuable architectural concepts are fully covered in the superseding ADRs.

## Context

The initial proof-of-concept was a monolithic `app.py` file. This became unmaintainable as features were added. To support future growth and improve code quality, a structured, modular architecture for the user interface is required.

## Related Requirements

* `SYS-ARCH-01`: Component-Based Architecture

* `SYS-ARCH-03`: Multi-Page Navigation

* `NFR-MAINT-01`: Maintainability

## Decision

We will refactor the entire UI into a component-based architecture located under the `src/ui/` directory. This structure separates pages, reusable components, state, and styles into distinct modules. For navigation, we will use Streamlit's native `st.navigation()` feature, which is the modern, library-first approach for building multi-page applications.

### Directory Structure

```text
src/ui/
├── pages/
│   ├── jobs.py
│   ├── companies.py
│   ├── scraping.py
│   └── settings.py
├── components/
│   ├── cards/
│   │   └── job_card.py
│   ├── progress/
│   │   └── company_progress_card.py
│   └── sidebar.py
├── state/
│   └── session_state.py
├── styles/
│   └── theme.py
└── utils/
    ├── background_tasks.py
    ├── formatters.py
    └── ...
```

### Navigation

The main application entry point, `src/main.py`, will use `st.navigation` to define and route to the different page modules.

```python

# In src/main.py
def main():
    # ... page config and state init
    pages = [
        st.Page("src/ui/pages/jobs.py", title="Jobs", icon="📋", default=True),
        st.Page("src/ui/pages/companies.py", title="Companies", icon="🏢"),
        # ... other pages
    ]
    pg = st.navigation(pages)
    pg.run()
```

## Consequences

* **Positive:**
  * **Improved Maintainability:** Code is organized logically, making it easier to find, understand, and modify.
  * **Reusability:** Components like `job_card.py` can be reused across different parts of the application.
  * **Scalability:** Adding new pages or complex features is straightforward and does not bloat a single file.
  * **Better Performance:** `st.navigation` is optimized by Streamlit for efficient multi-page app performance.
  * **Team Collaboration:** Developers can work on different components or pages simultaneously with fewer merge conflicts.

* **Negative:**
  * Requires an initial, one-time effort to refactor the monolithic `app.py`. (This has already been completed).
  * Slightly more boilerplate for creating new pages compared to a single-file app.

* **Mitigations:** The long-term benefits of maintainability and scalability far outweigh the minimal initial overhead. The adopted structure is a standard, well-understood pattern in web development.
