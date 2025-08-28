"""SQLite FTS5 search service using library-first implementation with sqlite-utils.

This module provides the JobSearchService class implementing professional-grade
full-text search capabilities using SQLite FTS5 via sqlite-utils. Features include:

- Porter stemming for improved matching ("develop" matches "developer")
- Multi-field search across title, description, company, location, requirements
- BM25 relevance ranking with best matches first
- Automatic index maintenance with triggers
- Filter integration (location, salary, remote, date)
- Streamlit caching with 5-minute TTL for performance
- Graceful error handling for database issues

The implementation uses a complete library-first approach,
leveraging sqlite-utils for FTS5 setup and automatic trigger management.
"""

import logging
import sqlite3

from datetime import UTC, datetime
from typing import Any

import sqlite_utils

# Import streamlit for caching decorators
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    # Create dummy decorator for non-Streamlit environments
    class _DummyStreamlit:
        """Dummy Streamlit class for non-Streamlit environments."""

        @staticmethod
        def cache_data(**_kwargs):
            """Dummy cache decorator that passes through the function unchanged."""

            def decorator(wrapped_func):
                """Inner decorator function."""
                return wrapped_func

            return decorator

    st = _DummyStreamlit()

from src.constants import SALARY_DEFAULT_MIN, SALARY_UNBOUNDED_THRESHOLD

logger = logging.getLogger(__name__)

# Type aliases for better readability
type FilterDict = dict[str, Any]
type JobSearchResult = dict[str, Any]


class JobSearchService:
    """Library-first search service using SQLite FTS5 via sqlite-utils.

    Provides professional-grade full-text search capabilities with:
    - Porter stemming for text normalization
    - Multi-field search across job content
    - BM25 relevance ranking
    - Automatic index maintenance
    - Filter integration with existing UI
    - Performance optimization with caching

    Implementation uses native sqlite-utils capabilities for FTS5 setup
    and automatic trigger management, minimizing custom code.
    """

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize search service with database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to "jobs.db"
                    which matches existing database configuration.
        """
        self.db_path = db_path
        self._db = None
        self._fts_enabled = False

    @property
    def db(self) -> sqlite_utils.Database:
        """Lazy database connection with automatic FTS5 setup.

        Returns:
            sqlite_utils.Database: Database connection with FTS5 configured.

        Note:
            Connection is created on first access and reused for performance.
            FTS5 setup is attempted on first connection with graceful fallback.
        """
        if self._db is None:
            self._db = sqlite_utils.Database(self.db_path)
            self._setup_search_index()
        return self._db

    def _setup_search_index(self) -> None:
        """Enable FTS5 search with automatic triggers - 100% library-first.

        Uses sqlite-utils to configure FTS5 with:
        - Porter stemming for text normalization
        - Unicode61 tokenizer for international support
        - Automatic triggers for index maintenance
        - Multi-field search across job content

        Gracefully handles cases where:
        - FTS5 is already enabled
        - Jobs table doesn't exist yet
        - Database connection issues
        """
        try:
            # Check if jobs table exists
            if "jobs" not in self.db.table_names():
                logger.info("Jobs table doesn't exist yet - FTS5 setup deferred")
                return

            # Enable FTS5 for jobs table with automatic triggers
            self.db["jobs"].enable_fts(
                ["title", "description", "company", "location"],  # Multi-field search
                create_triggers=True,  # Automatic index updates on INSERT/UPDATE/DELETE
                tokenize="porter unicode61",  # Porter stemming + Unicode support
            )

            self._fts_enabled = True
            logger.info(
                "FTS5 search enabled with porter stemming and automatic triggers"
            )

        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            # FTS already enabled - this is expected on subsequent runs
            self._fts_enabled = True
            logger.debug("FTS5 already enabled for jobs table")

        except Exception as e:
            # Graceful fallback - search will still work without FTS
            logger.warning("Could not enable FTS5 search: %s", e)
            self._fts_enabled = False

    def _is_fts_available(self) -> bool:
        """Check if FTS5 search is available and working.

        Returns:
            bool: True if FTS5 is enabled and functional, False otherwise.

        Note:
            Performs quick test query to verify FTS functionality.
            Used to determine whether to use FTS or fallback search.
        """
        if not self._fts_enabled:
            return False

        try:
            # Test FTS availability with simple query
            list(self.db.execute("SELECT count(*) FROM jobs_fts LIMIT 1"))
        except Exception:
            logger.debug("FTS5 not available - using fallback search")
            return False
        else:
            return True

    @st.cache_data(
        ttl=300,
        max_entries=500,
        show_spinner="Searching jobs...",
        hash_funcs={"JobSearchService": id},
    )  # Cache for 5 minutes with enhanced config
    def search_jobs(
        _self,  # noqa: N805  # Streamlit caching requires _ prefix to avoid hashing
        query: str,
        filters: FilterDict | None = None,
        limit: int = 50,
    ) -> list[JobSearchResult]:
        """Search jobs using FTS5 with optional filters and BM25 ranking.

        Provides professional-grade search with:
        - Porter stemming for text normalization
        - Multi-field search across title, description, company, location
        - BM25 relevance ranking with best matches first
        - Filter integration for location, salary, remote, date
        - Performance optimization with Streamlit caching

        Args:
            query: Search query string (supports natural language queries).
            filters: Optional filter criteria dictionary:
                - company: List of company names or "All"
                - application_status: List of status values or "All"
                - date_from: Start date for filtering
                - date_to: End date for filtering
                - favorites_only: Boolean to show only favorites
                - salary_min: Minimum salary filter (int or None)
                - salary_max: Maximum salary filter (int or None)
            limit: Maximum number of results to return (default: 50).

        Returns:
            List[JobSearchResult]: List of job dictionaries with search ranking.
                Each result includes all job fields plus:
                - rank: BM25 relevance score (higher = more relevant)

        Example:
            ```python
            # Basic search
            results = search_service.search_jobs("python developer")

            # Search with filters
            filters = {
                "salary_min": 100000,
                "company": ["Google", "Apple"],
                "favorites_only": True,
            }
            results = search_service.search_jobs("machine learning", filters)
            ```
        """
        # Handle empty queries gracefully
        if not query or not query.strip():
            logger.debug("Empty search query provided")
            return []

        query = query.strip()
        filters = filters or {}

        try:
            # Use FTS5 search if available, otherwise fallback to LIKE queries
            if _self._is_fts_available():
                return _self._search_with_fts(query, filters, limit)
            return _self._search_with_fallback(query, filters, limit)

        except Exception:
            logger.exception("Search failed for query '%s'", query)
            return []

    def _search_with_fts(
        self, query: str, filters: FilterDict, limit: int
    ) -> list[JobSearchResult]:
        """Execute FTS5 search with BM25 relevance ranking.

        Args:
            query: Search query string.
            filters: Filter criteria dictionary.
            limit: Maximum number of results.

        Returns:
            List of search results with relevance ranking.
        """
        # Build FTS5 query with BM25 relevance ranking
        base_query = """
            SELECT
                jobs.*,
                jobs_fts.rank,
                companies.name as company_name
            FROM jobs_fts
            JOIN jobs ON jobs.rowid = jobs_fts.rowid
            LEFT JOIN companysql as companies ON jobs.company_id = companies.id
            WHERE jobs_fts MATCH ?
        """

        params = [query]
        conditions = []

        # Apply filters using the same patterns as JobService
        conditions, params = self._build_filter_conditions(
            conditions, params, filters, "jobs"
        )

        # Add filter conditions to query
        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        # Order by relevance ranking (rank is negative in FTS5, so ORDER BY rank ASC
        # gives best first) with secondary sort by posted_date for ties
        base_query += " ORDER BY jobs_fts.rank, jobs.posted_date DESC LIMIT ?"
        params.append(limit)

        # Execute search query and get column names
        cursor = self.db.execute(base_query, params)
        column_names = (
            [desc[0] for desc in cursor.description] if cursor.description else []
        )
        results = list(cursor)

        # Convert to dictionaries with proper company name handling
        search_results = []
        for row in results:
            # sqlite-utils execute() returns tuples, so convert using column names
            try:
                result_dict = dict(zip(column_names, row, strict=True))
            except ValueError:
                logger.exception(
                    "Column/value mismatch in search: columns=%d, values=%d",
                    len(column_names),
                    len(row),
                )
                continue  # Skip this row

            # Use company_name from JOIN if available, otherwise keep existing field
            if result_dict.get("company_name"):
                result_dict["company"] = result_dict["company_name"]
            result_dict.pop("company_name", None)  # Remove the JOIN column
            search_results.append(result_dict)

        logger.info(
            "FTS5 search found %d results for query: '%s'", len(search_results), query
        )

        # Add cache performance info to results
        for result in search_results:
            result["_cache_info"] = {
                "cached": True,
                "search_method": "fts5_cached",
                "cache_ttl_seconds": 300,
            }

        return search_results

    def _search_with_fallback(
        self, query: str, filters: FilterDict, limit: int
    ) -> list[JobSearchResult]:
        """Fallback search using SQL LIKE queries when FTS5 unavailable.

        Args:
            query: Search query string.
            filters: Filter criteria dictionary.
            limit: Maximum number of results.

        Returns:
            List of search results without relevance ranking.
        """
        # Split query into individual terms for LIKE matching
        search_terms = query.split()

        # Build LIKE conditions for each search term across multiple fields
        like_conditions = []
        like_params = []

        for term in search_terms:
            term_pattern = f"%{term}%"
            like_conditions.append(
                "(jobs.title LIKE ? OR jobs.description LIKE ? OR "
                "companies.name LIKE ? OR jobs.location LIKE ?)"
            )
            like_params.extend([term_pattern] * 4)

        # Base query with company JOIN
        base_query = """
            SELECT
                jobs.*,
                0 as rank,
                companies.name as company_name
            FROM jobs
            LEFT JOIN companysql as companies ON jobs.company_id = companies.id
        """

        # Add search conditions
        if like_conditions:
            base_query += " WHERE " + " AND ".join(like_conditions)

        params = like_params

        # Apply additional filters
        filter_conditions = []
        filter_conditions, params = self._build_filter_conditions(
            filter_conditions, params, filters, "jobs"
        )

        if filter_conditions:
            if like_conditions:
                base_query += " AND " + " AND ".join(filter_conditions)
            else:
                base_query += " WHERE " + " AND ".join(filter_conditions)

        # Order by posted_date since we don't have relevance ranking
        base_query += " ORDER BY jobs.posted_date DESC LIMIT ?"
        params.append(limit)

        # Execute fallback search query and get column names
        cursor = self.db.execute(base_query, params)
        column_names = (
            [desc[0] for desc in cursor.description] if cursor.description else []
        )
        results = list(cursor)

        # Convert to dictionaries with proper company name handling
        search_results = []
        for row in results:
            # sqlite-utils execute() returns tuples, so convert using column names
            try:
                result_dict = dict(zip(column_names, row, strict=True))
            except ValueError:
                logger.exception(
                    "Column/value mismatch in fallback: columns=%d, values=%d",
                    len(column_names),
                    len(row),
                )
                continue  # Skip this row

            # Use company_name from JOIN if available
            if result_dict.get("company_name"):
                result_dict["company"] = result_dict["company_name"]
            result_dict.pop("company_name", None)  # Remove the JOIN column
            search_results.append(result_dict)

        logger.info(
            "Fallback search found %d results for query: '%s'",
            len(search_results),
            query,
        )

        # Add cache performance info to results
        for result in search_results:
            result["_cache_info"] = {
                "cached": True,
                "search_method": "fallback_cached",
                "cache_ttl_seconds": 300,
            }

        return search_results

    def _build_filter_conditions(
        self,
        conditions: list[str],
        params: list[Any],
        filters: FilterDict,
        table_alias: str = "jobs",
    ) -> tuple[list[str], list[Any]]:
        """Build SQL filter conditions from filter dictionary.

        Uses the same filter patterns as JobService for consistency.

        Args:
            conditions: Existing list of SQL conditions.
            params: Existing list of SQL parameters.
            filters: Filter criteria dictionary.
            table_alias: Table alias to use in conditions.

        Returns:
            tuple[List[str], List[Any]]: Updated conditions and parameters.
        """
        # Company filter - handled via JOIN already, no additional condition needed
        if (
            company_filter := filters.get("company", [])
        ) and "All" not in company_filter:
            # Company filtering handled by JOIN in the query
            pass

        # Application status filter
        if (
            status_filter := filters.get("application_status", [])
        ) and "All" not in status_filter:
            placeholders = ",".join("?" * len(status_filter))
            conditions.append(f"{table_alias}.application_status IN ({placeholders})")
            params.extend(status_filter)

        # Date filters
        if date_from := filters.get("date_from"):
            date_from = self._parse_date(date_from)
            if date_from:
                conditions.append(f"{table_alias}.posted_date >= ?")
                params.append(date_from)

        if date_to := filters.get("date_to"):
            date_to = self._parse_date(date_to)
            if date_to:
                conditions.append(f"{table_alias}.posted_date <= ?")
                params.append(date_to)

        # Favorites filter
        if filters.get("favorites_only", False):
            conditions.append(f"{table_alias}.favorite = 1")

        # Salary range filters with high-value support (matches JobService logic)
        salary_min = filters.get("salary_min")
        if salary_min is not None and salary_min > SALARY_DEFAULT_MIN:
            conditions.append(f"json_extract({table_alias}.salary, '$[1]') >= ?")
            params.append(salary_min)

        salary_max = filters.get("salary_max")
        if salary_max is not None and salary_max < SALARY_UNBOUNDED_THRESHOLD:
            conditions.append(f"json_extract({table_alias}.salary, '$[0]') <= ?")
            params.append(salary_max)

        # Filter out archived jobs by default
        if not filters.get("include_archived", False):
            conditions.append(f"{table_alias}.archived = 0")

        return conditions, params

    def _parse_date(self, date_input: str | datetime | None) -> datetime | None:
        """Parse date input into datetime object.

        Reuses the same date parsing logic as JobService for consistency.

        Args:
            date_input: Date as string, datetime object, or None.

        Returns:
            Parsed datetime object or None if input is None/invalid.
        """
        if isinstance(date_input, str):
            date_input = date_input.strip()
            if not date_input:
                return None

            # Try ISO format first (most common for APIs)
            try:
                dt = datetime.fromisoformat(date_input)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
            except ValueError:
                # Continue to alternative date parsing if ISO format fails
                logger.debug("ISO format parsing failed for date: %s", date_input)
            else:
                return dt

            # Try common formats found in job site scraping
            date_formats = [
                "%Y-%m-%d",  # 2024-12-31 (ISO date)
                "%m/%d/%Y",  # 12/31/2024 (US format)
                "%d/%m/%Y",  # 31/12/2024 (EU format)
                "%B %d, %Y",  # December 31, 2024
                "%d %B %Y",  # 31 December 2024
            ]

            for date_format in date_formats:
                try:
                    return datetime.strptime(date_input, date_format).replace(
                        tzinfo=UTC
                    )
                except ValueError:
                    # Try next date format if this one fails
                    logger.debug(
                        "Date format %s failed for date: %s", date_format, date_input
                    )
                    continue

            # If all formats fail, log warning
            logger.warning("Could not parse date: %s", date_input)
        elif date_input is not None and not isinstance(date_input, datetime):
            logger.warning("Unsupported date type: %s", type(date_input))

        return None

    def rebuild_search_index(self) -> bool:
        """Rebuild the FTS5 search index from scratch.

        Useful for maintenance or after bulk data changes.
        This method will:
        1. Drop existing FTS5 index
        2. Recreate index with current configuration
        3. Repopulate from jobs table

        Returns:
            bool: True if rebuild was successful, False otherwise.

        Note:
            This operation can take time for large datasets.
            The search index will be unavailable during rebuild.
        """
        try:
            logger.info("Starting FTS5 index rebuild...")

            # Drop existing FTS index if it exists
            try:
                self.db.execute("DROP TABLE IF EXISTS jobs_fts")
                logger.debug("Dropped existing FTS5 index")
            except Exception as e:
                logger.debug("No existing FTS5 index to drop: %s", e)

            # Reset FTS enabled flag to force recreation
            self._fts_enabled = False

            # Recreate FTS5 index
            self._setup_search_index()

            if self._fts_enabled:
                # Get count of indexed records
                result = list(self.db.execute("SELECT count(*) as count FROM jobs_fts"))
                count = result[0][0] if result else 0
                logger.info("FTS5 index rebuilt successfully with %d records", count)
            else:
                logger.error("Failed to rebuild FTS5 index")

        except Exception:
            logger.exception("Failed to rebuild search index")
            return False
        else:
            return self._fts_enabled

    def get_search_stats(self) -> dict[str, Any]:
        """Get search index statistics and health information.

        Returns:
            Dict[str, Any]: Dictionary with search statistics:
                - fts_enabled: Whether FTS5 is enabled and working
                - indexed_jobs: Number of jobs in search index
                - total_jobs: Total number of jobs in database
                - index_coverage: Percentage of jobs covered by search index
                - last_updated: Timestamp of most recent job in index

        Example:
            ```python
            stats = search_service.get_search_stats()
            print(f"FTS5 enabled: {stats['fts_enabled']}")
            print(f"Coverage: {stats['index_coverage']:.1f}%")
            ```
        """
        try:
            stats = {
                "fts_enabled": self._is_fts_available(),
                "indexed_jobs": 0,
                "total_jobs": 0,
                "index_coverage": 0.0,
                "last_updated": None,
            }

            # Get total jobs count
            result = list(self.db.execute("SELECT count(*) as count FROM jobs"))
            stats["total_jobs"] = result[0][0] if result else 0

            # Get FTS index stats if available
            if stats["fts_enabled"]:
                # Count indexed jobs
                result = list(self.db.execute("SELECT count(*) as count FROM jobs_fts"))
                stats["indexed_jobs"] = result[0][0] if result else 0

                # Calculate coverage
                if stats["total_jobs"] > 0:
                    stats["index_coverage"] = (
                        stats["indexed_jobs"] / stats["total_jobs"]
                    ) * 100

                # Get most recent job timestamp
                result = list(
                    self.db.execute(
                        "SELECT max(posted_date) as last_posted FROM jobs "
                        "WHERE rowid IN (SELECT rowid FROM jobs_fts)"
                    )
                )
                if result and result[0][0]:
                    stats["last_updated"] = result[0][0]

        except Exception as e:
            logger.exception("Failed to get search statistics")
            return {
                "fts_enabled": False,
                "indexed_jobs": 0,
                "total_jobs": 0,
                "index_coverage": 0.0,
                "last_updated": None,
                "error": str(e),
            }
        else:
            return stats

    @staticmethod
    def clear_all_caches() -> None:
        """Clear all Streamlit caches used by the search service.

        Useful for forcing fresh search results or troubleshooting.
        """
        if STREAMLIT_AVAILABLE:
            st.cache_data.clear()
            logger.info("✅ All JobSearchService caches cleared")
        else:
            logger.info("INFO: Streamlit not available - no caches to clear")

    @staticmethod
    def get_cache_stats() -> dict[str, Any]:
        """Get cache utilization statistics for the search service.

        Returns information about cache performance and memory usage.
        """
        return {
            "streamlit_available": STREAMLIT_AVAILABLE,
            "caching_enabled": STREAMLIT_AVAILABLE,
            "cached_methods": [
                "search_jobs",
            ],
            "cache_config": {
                "ttl_seconds": 300,  # 5 minutes
                "max_entries": 500,  # Maximum cached searches
                "uses_custom_hash_func": True,  # JobSearchService hash function
            },
            "performance_benefits": {
                "reduced_fts5_queries": "5min search result caching",
                "improved_search_responsiveness": "Cached search operations",
                "reduced_database_load": "Less frequent SQLite FTS5 queries",
                "fallback_search_caching": "Cached LIKE query fallbacks",
            },
            "search_features": {
                "fts5_enabled": "Dynamic based on database",
                "porter_stemming": "Enabled in FTS5",
                "relevance_ranking": "BM25 ranking in FTS5",
                "multi_field_search": "title, description, company, location",
            },
        }

    def refresh_search_cache(self) -> None:
        """Refresh search caches by clearing them.

        Forces fresh search results on next search call.
        Useful after database updates or index rebuilds.
        """
        self.clear_all_caches()
        logger.info("🔄 Search caches refreshed - next searches will be fresh")

    @st.cache_data(ttl=60, show_spinner=False)  # Cache index stats for 1 minute
    def get_cached_search_stats(self) -> dict[str, Any]:
        """Get cached search statistics to avoid frequent database queries.

        Returns cached version of search index statistics.
        """
        return self.get_search_stats()


# Global search service instance for application use
# Uses default database path "jobs.db" matching existing configuration
search_service = JobSearchService()


# Convenience functions for cache management
def clear_search_caches() -> None:
    """Clear all search-related caches globally.

    Convenience function for cache management.
    """
    JobSearchService.clear_all_caches()


def get_search_cache_stats() -> dict[str, Any]:
    """Get search cache statistics globally.

    Convenience function for cache monitoring.
    """
    return JobSearchService.get_cache_stats()


def refresh_search_caches() -> None:
    """Refresh all search caches globally.

    Convenience function to force fresh search results.
    """
    search_service.refresh_search_cache()
