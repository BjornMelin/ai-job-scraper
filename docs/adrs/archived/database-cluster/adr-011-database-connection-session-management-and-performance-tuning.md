# ADR-011: Database Connection, Session Management, and Performance Tuning

## Title

Database Connection, Session Management, and Performance Tuning for a Streamlit and SQLite Environment

## Version/Date

1.0 / August 7, 2025

## Status

Proposed

## Context

The application requires a stable, performant, and concurrent database connection strategy. The choice of Streamlit as the UI framework and SQLite as the default database presents a unique set of challenges. Streamlit's web server is multi-threaded, while SQLite, by default, is not designed for concurrent write access from multiple threads. This can lead to `database is locked` errors and performance bottlenecks, especially with a background scraping task running alongside user interactions in the UI. A deliberate and well-documented connection and session management strategy is therefore critical.

## Related Requirements

* `SYS-ARCH-04`: Background Task Execution
* `NFR-PERF-01`: UI Responsiveness
* `NFR-PERF-02`: Scalability
* `NFR-MAINT-01`: Maintainability

## Decision

We will implement a multi-faceted strategy specifically tailored to optimize the interaction between SQLAlchemy/SQLModel, SQLite, and Streamlit.

1. **Thread-Safe SQLite Configuration:** The SQLAlchemy engine for SQLite will be configured with `connect_args={"check_same_thread": False}` and `poolclass=StaticPool`. This combination allows the single SQLite connection to be safely accessed from Streamlit's multiple threads and the background scraping thread.

2. **Performance Optimization via PRAGMAs:** To enhance performance and concurrency, a set of SQLite PRAGMA statements will be executed on every new database connection via an SQLAlchemy event listener. The most critical of these is `PRAGMA journal_mode = WAL` (Write-Ahead Logging), which significantly improves concurrent read/write performance and reduces locking issues.

3. **Centralized Session Management:** All database interactions will use a centralized, thread-safe context manager (`db_session` in `src/database.py`). This ensures that every session is correctly opened, committed or rolled back, and closed, preventing resource leaks and ensuring transactional integrity.

4. **Optional Performance Monitoring:** To aid in debugging and optimization, SQLAlchemy event listeners for performance monitoring (`log_slow`, `start_timer`) will be included. This feature is disabled by default but can be activated via the `db_monitoring` setting for development.

## Design

The implementation is centralized in `src/database.py` and `src/database_listeners/`.

**1. Engine and Session Configuration (`src/database.py`):**

The engine is created with specific arguments for SQLite to ensure thread safety and a static connection pool.

```python
# In src/database.py

# SQLite-specific configuration for thread safety and performance
engine = create_engine(
    settings.db_url,
    echo=False,
    connect_args={
        "check_same_thread": False,  # Allow cross-thread access
    },
    poolclass=StaticPool,  # Single connection reused safely
)
```

**2. PRAGMA and Monitoring Listeners (`src/database.py`):**

Event listeners are attached to the engine to apply configurations on each connection.

```python
# In src/database.py

def _attach_sqlite_listeners(db_engine):
    """Attach SQLite event listeners for pragmas and optional performance monitoring."""
    # Always attach pragma handler for SQLite optimization
    event.listen(db_engine, "connect", apply_pragmas)

    # Only attach performance monitoring if enabled in settings
    if settings.db_monitoring:
        event.listen(db_engine, "before_cursor_execute", start_timer)
        event.listen(db_engine, "after_cursor_execute", log_slow)

_attach_sqlite_listeners(engine)
```

**3. Centralized Session Context Manager (`src/database.py`):**

A single, reusable context manager enforces correct session handling across the entire application.

```python
# In src/database.py

@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Database session context manager with automatic lifecycle management."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

## Consequences

* **Positive:**
  * **Stability:** The chosen configuration directly addresses and prevents common `database is locked` errors in a multi-threaded Streamlit environment.
  * **Performance:** The `WAL` journal mode and other PRAGMAs provide a significant performance boost for concurrent read/write operations.
  * **Maintainability:** Centralizing session logic in a single context manager and connection logic in one file makes the database layer easy to manage and debug.
  * **Observability:** The optional performance monitoring provides valuable insights into slow queries during development.
* **Negative:**
  * The configuration `check_same_thread=False` delegates thread safety responsibility to the application code.
  * This configuration is highly specific to SQLite and would not be portable to other database backends like PostgreSQL without changes.
* **Mitigations:**
  * The risk of `check_same_thread=False` is mitigated by enforcing the use of the `db_session` context manager, which ensures each unit of work gets its own session that is properly closed.
  * The database engine creation logic is already conditional, allowing for different configurations for PostgreSQL or other databases.
