"""Performance benchmarking configuration and fixtures.

This module provides specialized fixtures and configuration for performance
benchmarking tests, including timing measurements, memory usage tracking,
and performance regression detection.
"""

import gc
import os
import time
import tracemalloc

from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import pytest

from sqlmodel import Session

from tests.utils.db_utils import benchmark_data_creation

# Benchmark configuration

BENCHMARK_SCALES = {
    "micro": 10,  # 10 records
    "small": 100,  # 100 records
    "medium": 1000,  # 1K records
    "large": 10000,  # 10K records
    "xlarge": 50000,  # 50K records
}

PERFORMANCE_THRESHOLDS = {
    "database_creation_per_record": 0.01,  # 10ms per record max
    "search_response_time": 0.5,  # 500ms max
    "pagination_response_time": 0.1,  # 100ms max
    "memory_growth_mb": 100,  # 100MB max growth
}


@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "scales": BENCHMARK_SCALES,
        "thresholds": PERFORMANCE_THRESHOLDS,
        "warmup_iterations": 3,
        "measurement_iterations": 10,
        "memory_sampling_interval": 0.1,  # seconds
    }


@pytest.fixture(scope="session")
def process_monitor():
    """Monitor system process for resource usage."""
    return psutil.Process(os.getpid())


@pytest.fixture
def memory_tracker():
    """Track memory usage during test execution."""
    tracemalloc.start()

    # Force garbage collection for clean baseline
    gc.collect()

    snapshot_start = tracemalloc.take_snapshot()

    yield

    snapshot_end = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calculate memory differences
    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")

    return {
        "top_differences": top_stats[:10],
        "total_size_diff": sum(stat.size_diff for stat in top_stats),
        "total_count_diff": sum(stat.count_diff for stat in top_stats),
    }


@pytest.fixture
def performance_timer():
    """High-precision timing context manager."""

    @contextmanager
    def timer():
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        return end_time - start_time

    return timer


@pytest.fixture
def cpu_profiler():
    """CPU usage profiler for benchmark tests."""
    import cProfile
    import pstats

    from io import StringIO

    profiler = cProfile.Profile()

    profiler.enable()
    yield profiler
    profiler.disable()

    # Generate profiling report
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative").print_stats(20)

    return {
        "profiler": profiler,
        "stats": stats,
        "report": stream.getvalue(),
    }


@pytest.fixture(params=["micro", "small", "medium"])
def benchmark_scale(request, benchmark_config):
    """Parametrized fixture for different benchmark scales."""
    scale_name = request.param
    return {
        "name": scale_name,
        "size": benchmark_config["scales"][scale_name],
    }


@pytest.fixture(params=["large", "xlarge"])
def large_benchmark_scale(request, benchmark_config):
    """Parametrized fixture for large-scale benchmarks."""
    scale_name = request.param
    return {
        "name": scale_name,
        "size": benchmark_config["scales"][scale_name],
    }


@pytest.fixture
def benchmark_dataset(session: Session, benchmark_scale):
    """Create benchmark dataset with specified scale."""
    return benchmark_data_creation(session, scale=benchmark_scale["size"])


@pytest.fixture
def large_benchmark_dataset(session: Session, large_benchmark_scale):
    """Create large-scale benchmark dataset."""
    return benchmark_data_creation(session, scale=large_benchmark_scale["size"])


@pytest.fixture
def performance_metrics():
    """Collect and store performance metrics."""
    metrics = {
        "timings": {},
        "memory": {},
        "cpu": {},
        "database": {},
    }

    yield metrics

    # Optional: Store metrics to file for regression analysis
    if os.getenv("STORE_BENCHMARK_METRICS", "false").lower() == "true":
        _store_metrics(metrics)


def _store_metrics(metrics: dict[str, Any]) -> None:
    """Store benchmark metrics to file for analysis."""
    import json

    metrics_dir = Path("benchmark_results")
    metrics_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(UTC).isoformat().replace(":", "-")
    filename = metrics_dir / f"metrics_{timestamp}.json"

    with filename.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)


@pytest.fixture
def regression_detector(benchmark_config):
    """Detect performance regressions against baselines."""

    def check_regression(
        metric_name: str, current_value: float, baseline_value: float | None = None
    ) -> bool:
        """Check if current metric indicates regression.

        Args:
            metric_name: Name of the metric being checked
            current_value: Current measurement
            baseline_value: Baseline to compare against (optional)

        Returns:
            True if regression detected, False otherwise
        """
        if baseline_value is None:
            # Use threshold-based detection
            threshold = benchmark_config["thresholds"].get(metric_name)
            if threshold and current_value > threshold:
                return True

        else:
            # Use percentage-based regression detection (20% degradation)
            regression_threshold = baseline_value * 1.2
            if current_value > regression_threshold:
                return True

        return False

    return check_regression


@pytest.fixture
def warmup_session(session: Session):
    """Warm up database connections and caches before benchmarking."""
    # Perform some dummy queries to warm up connections
    session.execute("SELECT 1").scalar()
    session.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    # Force SQLite to load schema and optimize query plans
    session.execute("ANALYZE").scalar()

    return session


@contextmanager
def benchmark_isolation():
    """Ensure benchmark isolation by controlling system resources."""
    # Force garbage collection
    gc.collect()

    # Set high process priority (if possible)
    try:
        process = psutil.Process()
        original_priority = process.nice()
        process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -5)
    except (psutil.AccessDenied, OSError):
        original_priority = None

    try:
        yield
    finally:
        # Restore original priority
        if original_priority is not None:
            with suppress(psutil.AccessDenied, OSError):
                process.nice(original_priority)

        # Clean up after benchmark
        gc.collect()


@pytest.fixture
def system_resource_monitor(process_monitor):
    """Monitor system resources during benchmark execution."""

    def monitor_resources() -> dict[str, Any]:
        """Get current system resource usage."""
        return {
            "cpu_percent": process_monitor.cpu_percent(interval=0.1),
            "memory_rss": process_monitor.memory_info().rss,
            "memory_vms": process_monitor.memory_info().vms,
            "memory_percent": process_monitor.memory_percent(),
            "open_files": len(process_monitor.open_files()),
            "connections": len(process_monitor.connections()),
        }

    return monitor_resources


@pytest.fixture
def benchmark_reporter():
    """Generate benchmark reports and analysis."""

    def generate_report(results: dict[str, Any], scale: str) -> str:
        """Generate a formatted benchmark report.

        Args:
            results: Benchmark results dictionary
            scale: Scale of the benchmark (micro, small, medium, etc.)

        Returns:
            Formatted report string
        """
        report_lines = [
            f"Benchmark Report - Scale: {scale}",
            "=" * 50,
        ]

        if "timings" in results:
            report_lines.extend(
                [
                    "",
                    "Timing Results:",
                    "-" * 20,
                ]
            )
            for operation, timing in results["timings"].items():
                report_lines.append(f"{operation}: {timing:.3f}s")

        if "memory" in results:
            report_lines.extend(
                [
                    "",
                    "Memory Usage:",
                    "-" * 20,
                ]
            )
            for metric, value in results["memory"].items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"{metric}: {value:,.0f} bytes")

        if "database" in results:
            report_lines.extend(
                [
                    "",
                    "Database Metrics:",
                    "-" * 20,
                ]
            )
            for metric, value in results["database"].items():
                report_lines.append(f"{metric}: {value}")

        return "\n".join(report_lines)

    return generate_report


# Utility fixtures for specific benchmark types


@pytest.fixture
def database_benchmark_suite(session: Session):
    """Suite of database benchmark utilities."""

    def benchmark_inserts(record_count: int) -> dict[str, float]:
        """Benchmark database insert operations."""
        from tests.utils.db_utils import benchmark_data_creation

        start_time = time.perf_counter()
        result = benchmark_data_creation(session, scale=record_count)
        end_time = time.perf_counter()

        return {
            "total_time": end_time - start_time,
            "records_per_second": record_count / (end_time - start_time),
            "time_per_record": (end_time - start_time) / record_count,
            "records_created": len(result.get("jobs", [])),
        }

    def benchmark_queries(query_count: int = 100) -> dict[str, float]:
        """Benchmark database query operations."""
        from sqlmodel import select

        from src.models import JobSQL

        timings = []
        for _ in range(query_count):
            start = time.perf_counter()
            session.exec(select(JobSQL).limit(10)).fetchall()
            end = time.perf_counter()
            timings.append(end - start)

        return {
            "avg_query_time": sum(timings) / len(timings),
            "min_query_time": min(timings),
            "max_query_time": max(timings),
            "queries_per_second": query_count / sum(timings),
        }

    return {
        "benchmark_inserts": benchmark_inserts,
        "benchmark_queries": benchmark_queries,
    }


@pytest.fixture
def search_benchmark_suite():
    """Suite of search performance benchmark utilities."""

    def benchmark_fts_search(session: Session, query: str, iterations: int = 50):
        """Benchmark FTS search performance."""
        timings = []

        for _ in range(iterations):
            start = time.perf_counter()
            # Perform FTS search query here
            result = session.execute(
                "SELECT * FROM jobs_fts WHERE jobs_fts MATCH ? LIMIT 20", (query,)
            ).fetchall()
            end = time.perf_counter()

            timings.append(end - start)

        return {
            "avg_search_time": sum(timings) / len(timings),
            "searches_per_second": iterations / sum(timings),
            "results_found": len(result) if result else 0,
        }

    return {"benchmark_fts_search": benchmark_fts_search}
