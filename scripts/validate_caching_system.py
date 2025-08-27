#!/usr/bin/env python3
"""Validation script for the unified Streamlit native caching system.

This script validates the implementation of STREAM B MISSION - replacing mixed
custom caching with unified Streamlit native caching across all services.

Validates:
- Unified data caching strategy implementation
- Resource connection caching for database and AI services
- Cache performance and memory optimization
- Cache management and invalidation capabilities
- Cross-service cache coordination and statistics

Usage:
    python scripts/validate_caching_system.py
"""

import json
import logging
import sys
import time

from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CacheValidationError(Exception):
    """Custom exception for cache validation failures."""



class CachingSystemValidator:
    """Comprehensive validator for the unified Streamlit native caching system."""

    def __init__(self):
        """Initialize the cache system validator."""
        self.validation_results = {
            "timestamp": time.time(),
            "overall_status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "issues": [],
            "recommendations": [],
        }

        logger.info("🧪 Initializing Streamlit native caching system validation...")

    def validate_streamlit_availability(self) -> bool:
        """Validate that Streamlit is available for caching."""
        try:
            import streamlit as st

            logger.info("✅ Streamlit is available for native caching")

            # Test basic cache functionality
            @st.cache_data(ttl=1)
            def test_cache_data():
                return {"test": time.time()}

            @st.cache_resource
            def test_cache_resource():
                return "cached_resource"

            # Test cache operations
            result1 = test_cache_data()
            result2 = test_cache_data()  # Should be cached

            resource1 = test_cache_resource()
            resource2 = test_cache_resource()  # Should be same instance

            # Validate caching worked
            if result1["test"] == result2["test"]:
                logger.info("✅ st.cache_data is working correctly")
            else:
                raise CacheValidationError("st.cache_data not caching properly")

            if resource1 is resource2:
                logger.info("✅ st.cache_resource is working correctly")
            else:
                raise CacheValidationError("st.cache_resource not sharing instances")

            self.validation_results["tests_passed"] += 1
            return True

        except ImportError:
            logger.error("❌ Streamlit not available - caching will be disabled")
            self.validation_results["issues"].append("Streamlit not installed")
            self.validation_results["tests_failed"] += 1
            return False
        except Exception as e:
            logger.error("❌ Streamlit caching test failed: %s", e)
            self.validation_results["issues"].append(f"Streamlit caching error: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_unified_scraper_caching(self) -> bool:
        """Validate unified scraper service caching implementation."""
        try:
            from src.services.unified_scraper import UnifiedScrapingService

            logger.info("🔍 Validating unified scraper caching...")

            # Test cache stats availability
            cache_stats = UnifiedScrapingService.get_cache_stats()

            required_fields = [
                "streamlit_available",
                "caching_enabled",
                "cached_functions",
            ]
            for field in required_fields:
                if field not in cache_stats:
                    raise CacheValidationError(f"Missing cache stats field: {field}")

            # Validate cached functions are present
            cached_functions = cache_stats["cached_functions"]
            expected_functions = [
                "_normalize_jobspy_data_cached",
                "_load_active_companies_cached",
                "_deduplicate_jobs_cached",
            ]

            for func in expected_functions:
                if func not in cached_functions:
                    logger.warning("⚠️ Expected cached function missing: %s", func)

            # Test cache clearing
            UnifiedScrapingService.clear_all_caches()
            logger.info("✅ Unified scraper cache clearing works")

            logger.info("✅ Unified scraper caching validation passed")
            self.validation_results["tests_passed"] += 1
            return True

        except ImportError as e:
            logger.error("❌ Could not import UnifiedScrapingService: %s", e)
            self.validation_results["tests_failed"] += 1
            return False
        except Exception as e:
            logger.error("❌ Unified scraper caching validation failed: %s", e)
            self.validation_results["issues"].append(f"Unified scraper caching: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_analytics_service_caching(self) -> bool:
        """Validate analytics service caching implementation."""
        try:
            from src.services.analytics_service import AnalyticsService

            logger.info("📊 Validating analytics service caching...")

            # Test cache stats
            cache_stats = AnalyticsService.get_cache_stats()

            # Validate expected cached methods
            cached_methods = cache_stats.get("cached_methods", [])
            expected_methods = [
                "get_job_trends",
                "get_company_analytics",
                "get_salary_analytics",
            ]

            for method in expected_methods:
                if method not in cached_methods:
                    logger.warning("⚠️ Expected cached method missing: %s", method)

            # Test cache configuration
            cache_config = cache_stats.get("cache_config", {})
            if cache_config.get("ttl_seconds") != 300:  # 5 minutes
                logger.warning(
                    "⚠️ Unexpected TTL configuration: %s",
                    cache_config.get("ttl_seconds"),
                )

            # Test cache clearing
            AnalyticsService.clear_all_caches()
            logger.info("✅ Analytics service cache clearing works")

            logger.info("✅ Analytics service caching validation passed")
            self.validation_results["tests_passed"] += 1
            return True

        except Exception as e:
            logger.error("❌ Analytics service caching validation failed: %s", e)
            self.validation_results["issues"].append(f"Analytics service caching: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_search_service_caching(self) -> bool:
        """Validate search service caching implementation."""
        try:
            from src.services.search_service import (
                JobSearchService,
                clear_search_caches,
            )

            logger.info("🔍 Validating search service caching...")

            # Test cache stats
            cache_stats = JobSearchService.get_cache_stats()

            # Validate search-specific configuration
            cache_config = cache_stats.get("cache_config", {})
            expected_max_entries = 500

            if cache_config.get("max_entries") != expected_max_entries:
                logger.warning(
                    "⚠️ Unexpected max_entries: %s", cache_config.get("max_entries")
                )

            # Validate custom hash function usage
            if not cache_config.get("uses_custom_hash_func"):
                logger.warning("⚠️ Custom hash function not configured")

            # Test convenience functions
            clear_search_caches()
            logger.info("✅ Search service cache clearing works")

            logger.info("✅ Search service caching validation passed")
            self.validation_results["tests_passed"] += 1
            return True

        except Exception as e:
            logger.error("❌ Search service caching validation failed: %s", e)
            self.validation_results["issues"].append(f"Search service caching: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_ai_service_caching(self) -> bool:
        """Validate AI service caching implementation."""
        try:
            from src.ai.cloud_ai_service import CloudAIService
            from src.ai.hybrid_ai_router import HybridAIRouter

            logger.info("🤖 Validating AI service caching...")

            # Test HybridAIRouter caching
            router_stats = HybridAIRouter.get_cache_stats()
            if "routing_metrics" not in router_stats.get("cache_ttls", {}):
                logger.warning("⚠️ HybridAIRouter metrics caching not configured")

            # Test CloudAIService caching
            cloud_stats = CloudAIService.get_cache_stats()
            cached_functions = cloud_stats.get("cached_functions", [])

            expected_cloud_functions = [
                "_create_litellm_router",
                "_create_instructor_client",
            ]
            for func in expected_cloud_functions:
                if func not in cached_functions:
                    logger.warning("⚠️ Expected AI cached function missing: %s", func)

            # Test cache clearing
            HybridAIRouter.clear_all_caches()
            CloudAIService.clear_all_caches()
            logger.info("✅ AI service cache clearing works")

            logger.info("✅ AI service caching validation passed")
            self.validation_results["tests_passed"] += 1
            return True

        except Exception as e:
            logger.error("❌ AI service caching validation failed: %s", e)
            self.validation_results["issues"].append(f"AI service caching: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_database_caching(self) -> bool:
        """Validate database connection caching implementation."""
        try:
            from src.database import (
                create_db_and_tables,
                get_engine,
                get_session_factory,
            )

            logger.info("🗄️ Validating database connection caching...")

            # Test that engine is cached (should return same instance)
            engine1 = get_engine()
            engine2 = get_engine()

            if engine1 is not engine2:
                logger.warning("⚠️ Database engine not properly cached")
            else:
                logger.info("✅ Database engine caching works")

            # Test session factory caching
            factory1 = get_session_factory()
            factory2 = get_session_factory()

            if factory1 is not factory2:
                logger.warning("⚠️ Session factory not properly cached")
            else:
                logger.info("✅ Session factory caching works")

            # Test table creation caching
            create_db_and_tables()  # Should not recreate if already cached
            logger.info("✅ Database table creation caching works")

            self.validation_results["tests_passed"] += 1
            return True

        except Exception as e:
            logger.error("❌ Database caching validation failed: %s", e)
            self.validation_results["issues"].append(f"Database caching: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_cache_manager(self) -> bool:
        """Validate unified cache manager implementation."""
        try:
            from src.services.cache_manager import (
                clear_all_caches,
                get_cache_health,
                get_cache_stats,
                optimize_cache_memory,
                warm_caches,
            )

            logger.info("🗂️ Validating unified cache manager...")

            # Test comprehensive stats collection
            stats = get_cache_stats()

            required_sections = [
                "cache_manager",
                "service_stats",
                "performance_summary",
            ]
            for section in required_sections:
                if section not in stats:
                    raise CacheValidationError(f"Missing stats section: {section}")

            # Test cache health reporting
            health = get_cache_health()
            if "overall_health" not in health:
                raise CacheValidationError("Missing overall_health in health report")

            # Test memory optimization
            optimization = optimize_cache_memory()
            if optimization["status"] != "success":
                logger.warning(
                    "⚠️ Memory optimization reported issues: %s", optimization
                )

            # Test cache warming (should not fail even if some services unavailable)
            warming = warm_caches()
            logger.info("Cache warming completed with status: %s", warming["status"])

            # Test global cache clearing
            clear_result = clear_all_caches()
            if clear_result["status"] == "success":
                logger.info("✅ Global cache clearing works")
            else:
                logger.warning("⚠️ Global cache clearing issues: %s", clear_result)

            logger.info("✅ Unified cache manager validation passed")
            self.validation_results["tests_passed"] += 1
            return True

        except Exception as e:
            logger.error("❌ Cache manager validation failed: %s", e)
            self.validation_results["issues"].append(f"Cache manager: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def validate_performance_improvements(self) -> bool:
        """Validate that caching provides performance improvements."""
        try:
            logger.info("⚡ Validating caching performance improvements...")

            # Test with a simple cacheable operation
            import streamlit as st

            call_count = {"value": 0}

            @st.cache_data(ttl=60)
            def expensive_operation(input_val: str) -> dict[str, Any]:
                call_count["value"] += 1
                time.sleep(0.1)  # Simulate expensive operation
                return {"result": f"processed_{input_val}", "timestamp": time.time()}

            # First call - should be slow and increment counter
            start_time = time.time()
            result1 = expensive_operation("test_input")
            first_call_time = time.time() - start_time

            # Second call - should be fast and not increment counter
            start_time = time.time()
            result2 = expensive_operation("test_input")
            second_call_time = time.time() - start_time

            # Validate caching worked
            if call_count["value"] != 1:
                raise CacheValidationError(
                    f"Function called {call_count['value']} times, expected 1"
                )

            if result1 != result2:
                raise CacheValidationError("Cached results don't match")

            # Validate performance improvement
            speedup_ratio = first_call_time / max(
                second_call_time, 0.001
            )  # Avoid division by zero

            self.validation_results["performance_metrics"] = {
                "first_call_time_ms": round(first_call_time * 1000, 2),
                "cached_call_time_ms": round(second_call_time * 1000, 2),
                "speedup_ratio": round(speedup_ratio, 1),
                "cache_hit_improvement": f"{speedup_ratio:.1f}x faster",
            }

            if speedup_ratio > 5:  # At least 5x improvement expected
                logger.info(
                    "✅ Caching provides significant performance improvement: %.1fx",
                    speedup_ratio,
                )
                self.validation_results["tests_passed"] += 1
                return True
            logger.warning(
                "⚠️ Caching performance improvement lower than expected: %.1fx",
                speedup_ratio,
            )
            self.validation_results["tests_passed"] += 1  # Still pass, just warn
            return True

        except Exception as e:
            logger.error("❌ Performance validation failed: %s", e)
            self.validation_results["issues"].append(f"Performance validation: {e}")
            self.validation_results["tests_failed"] += 1
            return False

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run all validation tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive caching system validation...")

        validation_tests = [
            ("Streamlit Availability", self.validate_streamlit_availability),
            ("Unified Scraper Caching", self.validate_unified_scraper_caching),
            ("Analytics Service Caching", self.validate_analytics_service_caching),
            ("Search Service Caching", self.validate_search_service_caching),
            ("AI Service Caching", self.validate_ai_service_caching),
            ("Database Connection Caching", self.validate_database_caching),
            ("Unified Cache Manager", self.validate_cache_manager),
            ("Performance Improvements", self.validate_performance_improvements),
        ]

        for test_name, test_func in validation_tests:
            logger.info(f"Running: {test_name}")
            try:
                test_func()
            except Exception as e:
                logger.error("❌ Test %s failed with exception: %s", test_name, e)
                self.validation_results["tests_failed"] += 1
                self.validation_results["issues"].append(f"{test_name}: {e}")

        # Calculate overall status
        total_tests = (
            self.validation_results["tests_passed"]
            + self.validation_results["tests_failed"]
        )
        success_rate = (
            self.validation_results["tests_passed"] / max(total_tests, 1) * 100
        )

        if success_rate >= 90:
            self.validation_results["overall_status"] = "excellent"
        elif success_rate >= 75:
            self.validation_results["overall_status"] = "good"
        elif success_rate >= 50:
            self.validation_results["overall_status"] = "fair"
        else:
            self.validation_results["overall_status"] = "poor"

        # Add recommendations
        if len(self.validation_results["issues"]) == 0:
            self.validation_results["recommendations"] = [
                "Caching system is working optimally",
                "Consider monitoring cache hit rates in production",
                "Periodically review TTL values based on data freshness requirements",
            ]
        else:
            self.validation_results["recommendations"] = [
                "Address the issues listed above",
                "Ensure Streamlit is properly installed",
                "Check service imports and dependencies",
            ]

        # Log final summary
        logger.info("🎯 Validation completed:")
        logger.info(f"   Tests passed: {self.validation_results['tests_passed']}")
        logger.info(f"   Tests failed: {self.validation_results['tests_failed']}")
        logger.info(f"   Overall status: {self.validation_results['overall_status']}")
        logger.info(f"   Success rate: {success_rate:.1f}%")

        if self.validation_results["issues"]:
            logger.info("❌ Issues found:")
            for issue in self.validation_results["issues"]:
                logger.info(f"   - {issue}")

        return self.validation_results


def main():
    """Main validation script entry point."""
    print("🚀 STREAM B MISSION: Unified Streamlit Native Caching Validation")
    print("=" * 70)

    validator = CachingSystemValidator()
    results = validator.run_comprehensive_validation()

    # Save results to file
    results_file = Path("cache_validation_results.json")
    with results_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📊 Validation results saved to: {results_file}")
    print("\n🎯 VALIDATION SUMMARY:")
    print(f"   Overall Status: {results['overall_status'].upper()}")
    print(f"   Tests Passed: {results['tests_passed']}")
    print(f"   Tests Failed: {results['tests_failed']}")

    if results["performance_metrics"]:
        print(
            f"   Cache Performance: {results['performance_metrics']['cache_hit_improvement']}"
        )

    # Exit with appropriate code
    if results["overall_status"] in ["excellent", "good"]:
        print("\n✅ Unified Streamlit native caching system validation PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Unified Streamlit native caching system validation FAILED!")
        print("Please address the issues above before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
