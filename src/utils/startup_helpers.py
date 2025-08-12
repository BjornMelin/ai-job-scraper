"""Startup helpers for application initialization.

This module provides utilities for application startup including:
- Simple cache warming using Streamlit's native caching
- Performance monitoring and initialization logging
- Streamlit session state initialization
"""

import logging
import threading
import time

from typing import Any

from src.constants import BACKGROUND_PREFETCH_INTERVAL

try:
    import streamlit as st
except ImportError:
    # Create dummy streamlit for non-Streamlit environments
    class _DummyStreamlit:
        @staticmethod
        def cache_data(**_kwargs):
            def decorator(func):
                return func

            return decorator

        class SessionState:
            cache_warmed = False

        session_state = SessionState()

    st = _DummyStreamlit()

from src.services.company_service import CompanyService
from src.services.job_service import JobService

logger = logging.getLogger(__name__)

# Startup configuration
CACHE_WARMUP_TIMEOUT = 30  # seconds


def warm_startup_cache(config: dict | None = None) -> dict[str, Any]:
    """Warm cache with commonly accessed data on application startup.

    This function pre-loads frequently accessed queries into the cache
    to achieve faster initial page loads. It runs essential queries first,
    then background queries if time permits.

    Args:
        config: Configuration dictionary with 'background' and 'timeout' keys

    Returns:
        Dictionary with warming statistics and performance info
    """
    if config is None:
        config = {"background": True, "timeout": CACHE_WARMUP_TIMEOUT}
    background = config.get("background", True)
    timeout = config.get("timeout", CACHE_WARMUP_TIMEOUT)
    if hasattr(st.session_state, "cache_warmed") and st.session_state.cache_warmed:
        logger.debug("Cache already warmed in this session, skipping")
        return {"status": "skipped", "reason": "already_warmed"}

    start_time = time.time()
    stats = {
        "status": "started",
        "background": background,
        "timeout": timeout,
        "queries_completed": 0,
        "items_cached": 0,
        "errors": 0,
        "duration_seconds": 0,
    }

    def _warm_cache_sync() -> dict[str, Any]:
        """Internal synchronous cache warming function."""
        try:
            logger.info("Starting cache warmup process...")

            # Priority 1: Essential statistics (fast queries)
            try:
                CompanyService.get_active_companies_count()
                stats["queries_completed"] += 1
                logger.debug("Warmed: active companies count")
            except Exception:
                logger.exception("Failed to warm active companies count")
                stats["errors"] += 1

            try:
                JobService.get_job_counts_by_status()
                stats["queries_completed"] += 1
                logger.debug("Warmed: job counts by status")
            except Exception:
                logger.exception("Failed to warm job counts by status")
                stats["errors"] += 1

            # Priority 2: Common job queries with pagination
            prefetch_stats = JobService.prefetch_common_queries(background=True)
            stats["queries_completed"] += prefetch_stats.get("queries_prefetched", 0)
            stats["items_cached"] += prefetch_stats.get("items_cached", 0)

            # Priority 3: Company statistics (if time permits)
            elapsed = time.time() - start_time
            if elapsed < timeout - 2:  # Leave 2 seconds buffer
                try:
                    CompanyService.get_companies_with_job_counts()
                    stats["queries_completed"] += 1
                    logger.debug("Warmed: companies with job counts")
                except Exception:
                    logger.exception("Failed to warm companies with job counts")
                    stats["errors"] += 1

            stats["status"] = "completed"
            stats["duration_seconds"] = round(time.time() - start_time, 2)

            # Mark cache as warmed in session state
            if hasattr(st, "session_state"):
                st.session_state.cache_warmed = True

            logger.info(
                "Cache warmup completed: %d queries, %d items cached, %d errors, %.2fs",
                stats["queries_completed"],
                stats["items_cached"],
                stats["errors"],
                stats["duration_seconds"],
            )

        except Exception:
            stats["status"] = "failed"
            stats["duration_seconds"] = round(time.time() - start_time, 2)
            logger.exception(
                "Cache warmup failed after %.2fs", stats["duration_seconds"]
            )
        else:
            return stats

        return stats

    if background and hasattr(st, "session_state"):
        # Run in background thread for Streamlit
        def _background_warmup():
            try:
                _warm_cache_sync()
            except Exception:
                logger.exception("Background cache warmup failed")

        thread = threading.Thread(target=_background_warmup, daemon=True)
        thread.start()

        stats["status"] = "background_started"
        logger.info("Started background cache warmup thread")
        return stats
    # Run synchronously
    return _warm_cache_sync()


def start_background_prefetching() -> bool:
    """Start background thread for periodic cache prefetching.

    This creates a daemon thread that periodically refreshes commonly
    accessed cache entries to maintain fast response times.

    Returns:
        True if background prefetching was started successfully
    """

    def _prefetch_loop():
        """Background loop for periodic cache prefetching."""
        while True:
            try:
                time.sleep(BACKGROUND_PREFETCH_INTERVAL)

                logger.debug("Running periodic cache prefetch...")

                # Prefetch job statistics and common queries
                prefetch_stats = JobService.prefetch_common_queries(background=True)

                # Refresh company statistics
                CompanyService.get_active_companies_count()
                CompanyService.get_companies_with_job_counts()

                logger.debug(
                    "Periodic prefetch completed: %d queries, %d items",
                    prefetch_stats.get("queries_prefetched", 0),
                    prefetch_stats.get("items_cached", 0),
                )

            except Exception:
                logger.exception("Error in background prefetch loop")
                # Continue the loop even on errors
                continue

    try:
        prefetch_thread = threading.Thread(target=_prefetch_loop, daemon=True)
        prefetch_thread.start()

        logger.info(
            "Started background prefetching thread (interval: %ds)",
            BACKGROUND_PREFETCH_INTERVAL,
        )
    except Exception:
        logger.exception("Failed to start background prefetching")
        return False
    else:
        return True


def initialize_performance_optimizations() -> dict[str, Any]:
    """Initialize all performance optimizations including caching and prefetching.

    This is the main entry point for startup performance optimizations.
    It coordinates cache warming and background prefetching.

    Returns:
        Dictionary with initialization results and statistics
    """
    logger.info("Initializing performance optimizations...")

    results = {
        "cache_warmup": {},
        "background_prefetch": False,
        "startup_time": time.time(),
    }

    try:
        # Initialize cache manager
        # Cache manager no longer needed - using native Streamlit caching
        results["cache_manager_initialized"] = True
        logger.debug("Cache manager initialized successfully")

        # Warm cache with common queries
        results["cache_warmup"] = warm_startup_cache(background=True)

        # Start background prefetching
        results["background_prefetch"] = start_background_prefetching()

        total_time = round(time.time() - results["startup_time"], 3)
        logger.info(
            "Performance optimizations initialized in %.3fs - "
            "cache warmup: %s, background prefetch: %s",
            total_time,
            results["cache_warmup"].get("status", "unknown"),
            "enabled" if results["background_prefetch"] else "disabled",
        )

    except Exception:
        logger.exception("Failed to initialize performance optimizations")

    return results


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cache_performance_stats() -> dict[str, Any]:
    """Get cache performance statistics for monitoring.

    Returns:
        Dictionary with current cache performance metrics
    """
    try:
        # No complex cache stats with Streamlit native caching
        stats = {"message": "Using native Streamlit caching"}

        # Add additional context
        stats["timestamp"] = time.time()
        stats["performance_optimizations_active"] = True

    except Exception:
        logger.exception("Failed to get cache performance stats")
        return {
            "error": "Failed to retrieve cache stats",
            "timestamp": time.time(),
            "performance_optimizations_active": False,
        }
    else:
        return stats
