"""Cache manager with multi-level caching strategy.

This module provides a comprehensive caching solution using:
- cachetools: Fast in-memory caching with TTL and LRU eviction
- diskcache: Persistent disk-based caching for larger datasets
- Multi-level strategy: Memory -> Disk -> Database fallback

Features:
- Library-first approach using proven caching libraries
- TTL-based expiration for cache freshness
- Size-limited memory cache to prevent memory bloat
- Persistent disk cache for expensive computations
- Type-safe cache operations with proper error handling
- Performance monitoring with hit/miss statistics
"""

import hashlib
import json
import logging

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

import cachetools
import diskcache

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Type variables for generic cache operations
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Cache configuration constants
MEMORY_CACHE_SIZE = 1000  # Max items in memory cache
MEMORY_CACHE_TTL = 300  # 5 minutes TTL for memory cache
DISK_CACHE_TTL = 3600  # 1 hour TTL for disk cache
DISK_CACHE_SIZE_LIMIT = 500 * 1024 * 1024  # 500MB disk cache limit

# Cache key prefixes for organization
JOB_CACHE_PREFIX = "job:"
COMPANY_CACHE_PREFIX = "company:"
STATS_CACHE_PREFIX = "stats:"


class CacheManager:
    """Multi-level cache manager with memory and disk caching.

    Implements a two-tier caching strategy:
    1. Fast in-memory cache using cachetools.TTLCache (hot data)
    2. Persistent disk cache using diskcache.Cache (warm data)

    Cache hierarchy:
    - Check memory cache first (fastest)
    - Fall back to disk cache (fast)
    - Fall back to database query (slow)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        memory_size: int = MEMORY_CACHE_SIZE,
        memory_ttl: int = MEMORY_CACHE_TTL,
        disk_ttl: int = DISK_CACHE_TTL,
        disk_size_limit: int = DISK_CACHE_SIZE_LIMIT,
    ):
        """Initialize cache manager with configurable settings.

        Args:
            cache_dir: Directory for disk cache storage
            memory_size: Maximum items in memory cache
            memory_ttl: TTL in seconds for memory cache
            disk_ttl: TTL in seconds for disk cache
            disk_size_limit: Maximum size of disk cache in bytes
        """
        # Initialize memory cache with TTL and LRU eviction
        self._memory_cache = cachetools.TTLCache(maxsize=memory_size, ttl=memory_ttl)

        # Initialize disk cache with size limits
        cache_directory = cache_dir or Path(settings.cache_directory or "./cache")
        cache_directory.mkdir(parents=True, exist_ok=True)

        self._disk_cache = diskcache.Cache(
            directory=str(cache_directory),
            size_limit=disk_size_limit,
            eviction_policy="least-recently-used",
        )

        self._memory_ttl = memory_ttl
        self._disk_ttl = disk_ttl

        # Statistics tracking
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_requests": 0,
        }

        logger.info(
            "Initialized CacheManager: memory_size=%d, memory_ttl=%ds, "
            "disk_ttl=%ds, disk_size_limit=%dMB",
            memory_size,
            memory_ttl,
            disk_ttl,
            disk_size_limit // 1024 // 1024,
        )

    def _generate_key(self, prefix: str, key: str) -> str:
        """Generate consistent cache key with prefix.

        Args:
            prefix: Cache key prefix for organization
            key: Original cache key

        Returns:
            Prefixed cache key
        """
        return f"{prefix}{key}"

    def _serialize_key(self, key: Any) -> str:
        """Serialize complex keys to consistent string representation.

        Args:
            key: Any hashable or serializable object

        Returns:
            Serialized string key suitable for caching
        """
        if isinstance(key, str):
            return key
        if isinstance(key, (int, float, bool)):
            return str(key)
        # Handle complex objects by serializing to JSON and hashing
        try:
            serialized = json.dumps(key, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()
        except (TypeError, ValueError):
            # Fall back to string representation
            return str(hash(key))

    def get(self, key: Any, prefix: str = "", default: T | None = None) -> T | None:
        """Retrieve value from cache hierarchy.

        Checks memory cache first, then disk cache, returns default if not found.

        Args:
            key: Cache key (will be serialized if complex)
            prefix: Cache key prefix for organization
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        self._stats["total_requests"] += 1
        cache_key = self._generate_key(prefix, self._serialize_key(key))

        try:
            # Check memory cache first (fastest)
            if cache_key in self._memory_cache:
                value = self._memory_cache[cache_key]
                self._stats["memory_hits"] += 1
                logger.debug("Memory cache hit for key: %s", cache_key)
                return value

            # Check disk cache (fast fallback)
            disk_value = self._disk_cache.get(cache_key)
            if disk_value is not None:
                # Promote to memory cache for faster future access
                self._memory_cache[cache_key] = disk_value
                self._stats["disk_hits"] += 1
                logger.debug("Disk cache hit for key: %s", cache_key)
                return disk_value

            # Cache miss
            self._stats["misses"] += 1
            logger.debug("Cache miss for key: %s", cache_key)
            return default

        except Exception:
            logger.exception("Error retrieving from cache: %s", cache_key)
            return default

    def set(
        self,
        key: Any,
        value: T,
        prefix: str = "",
        memory_ttl: int | None = None,
        disk_ttl: int | None = None,
    ) -> bool:
        """Store value in cache hierarchy.

        Stores in both memory and disk cache for redundancy and performance.

        Args:
            key: Cache key (will be serialized if complex)
            value: Value to cache
            prefix: Cache key prefix for organization
            memory_ttl: Override default memory TTL
            disk_ttl: Override default disk TTL

        Returns:
            True if stored successfully, False otherwise
        """
        cache_key = self._generate_key(prefix, self._serialize_key(key))

        try:
            # Store in memory cache (with TTL handled by TTLCache)
            self._memory_cache[cache_key] = value

            # Store in disk cache with explicit TTL
            expire_time = None
            if disk_ttl is not None:
                expire_time = datetime.now(timezone.utc) + timedelta(seconds=disk_ttl)
            elif self._disk_ttl > 0:
                expire_time = datetime.now(timezone.utc) + timedelta(
                    seconds=self._disk_ttl
                )

            success = self._disk_cache.set(
                cache_key,
                value,
                expire=expire_time.timestamp() if expire_time else None,
            )

            if success:
                logger.debug("Cached value for key: %s", cache_key)
            else:
                logger.warning("Failed to cache value for key: %s", cache_key)

            return success

        except Exception:
            logger.exception("Error storing in cache: %s", cache_key)
            return False

    def delete(self, key: Any, prefix: str = "") -> bool:
        """Delete value from cache hierarchy.

        Args:
            key: Cache key to delete
            prefix: Cache key prefix

        Returns:
            True if deleted successfully, False otherwise
        """
        cache_key = self._generate_key(prefix, self._serialize_key(key))

        try:
            # Delete from memory cache
            self._memory_cache.pop(cache_key, None)

            # Delete from disk cache
            success = self._disk_cache.delete(cache_key)

            logger.debug("Deleted cache key: %s", cache_key)
            return success

        except Exception:
            logger.exception("Error deleting from cache: %s", cache_key)
            return False

    def clear(self, prefix: str | None = None) -> bool:
        """Clear cache entries, optionally by prefix.

        Args:
            prefix: If provided, only clear keys with this prefix

        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if prefix is None:
                # Clear all caches
                self._memory_cache.clear()
                self._disk_cache.clear()
                logger.info("Cleared all cache entries")
            else:
                # Clear by prefix (more complex operation)
                keys_to_delete = []

                # Find memory cache keys with prefix
                for key in list(self._memory_cache.keys()):
                    if key.startswith(prefix):
                        keys_to_delete.append(key)

                # Delete from memory cache
                for key in keys_to_delete:
                    self._memory_cache.pop(key, None)

                # Find disk cache keys with prefix (expensive operation)
                # Note: diskcache doesn't have efficient prefix deletion
                # This is a limitation we accept for the library-first approach
                for key in list(self._disk_cache):
                    if key.startswith(prefix):
                        self._disk_cache.delete(key)

                logger.info(
                    "Cleared %d cache entries with prefix: %s",
                    len(keys_to_delete),
                    prefix,
                )

            return True

        except Exception:
            logger.exception("Error clearing cache with prefix: %s", prefix)
            return False

    def cached(
        self,
        key_func: Callable[..., str] | None = None,
        prefix: str = "",
        memory_ttl: int | None = None,
        disk_ttl: int | None = None,
    ) -> Callable[[F], F]:
        """Decorator for caching function results.

        Args:
            key_func: Function to generate cache key from args/kwargs
            prefix: Cache key prefix
            memory_ttl: Override memory TTL
            disk_ttl: Override disk TTL

        Returns:
            Decorated function with caching
        """

        def decorator(func: F) -> F:
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation from function name and args
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = "|".join(key_parts)

                # Try to get from cache
                cached_result = self.get(cache_key, prefix)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, prefix, memory_ttl, disk_ttl)

                return result

            return wrapper

        return decorator

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_hits = self._stats["memory_hits"] + self._stats["disk_hits"]
        total_requests = self._stats["total_requests"]
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "memory_hits": self._stats["memory_hits"],
            "disk_hits": self._stats["disk_hits"],
            "misses": self._stats["misses"],
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_size": len(self._disk_cache),
            "disk_cache_volume_mb": round(self._disk_cache.volume() / 1024 / 1024, 2),
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_requests": 0,
        }
        logger.info("Reset cache statistics")

    def warm_cache(
        self, warm_func: Callable[[], dict[str, Any]], prefix: str = ""
    ) -> int:
        """Warm cache with pre-computed values.

        Args:
            warm_func: Function that returns dict of key-value pairs to cache
            prefix: Cache key prefix for warmed entries

        Returns:
            Number of items warmed in cache
        """
        try:
            warm_data = warm_func()
            count = 0

            for key, value in warm_data.items():
                if self.set(key, value, prefix):
                    count += 1

            logger.info("Warmed cache with %d items (prefix: %s)", count, prefix)
            return count

        except Exception:
            logger.exception("Error warming cache")
            return 0

    def close(self) -> None:
        """Close cache connections and clean up resources."""
        try:
            self._disk_cache.close()
            self._memory_cache.clear()
            logger.info("Closed cache manager")
        except Exception:
            logger.exception("Error closing cache manager")


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance.

    Returns:
        Singleton CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_key_for_filters(filters: dict[str, Any]) -> str:
    """Generate consistent cache key for filter dictionaries.

    Args:
        filters: Dictionary of filter parameters

    Returns:
        Consistent string key for caching
    """
    # Sort and serialize filters for consistent cache keys
    sorted_filters = dict(sorted(filters.items()))

    # Handle special cases for filter values
    processed_filters = {}
    for key, value in sorted_filters.items():
        if isinstance(value, list):
            processed_filters[key] = sorted(str(v) for v in value)
        elif isinstance(value, (datetime,)):
            processed_filters[key] = value.isoformat()
        else:
            processed_filters[key] = str(value) if value is not None else "null"

    return json.dumps(processed_filters, sort_keys=True)
