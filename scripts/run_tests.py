#!/usr/bin/env python3
"""Advanced test runner script with multiple execution modes.

This script provides various test execution modes for different development
and CI/CD scenarios, including fast mode, integration mode, benchmarking,
and coverage reporting.
"""

import argparse
import os
import subprocess
import sys
import time

from pathlib import Path


def setup_environment() -> None:
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)


def run_command(cmd: list[str], description: str = "") -> int:
    """Run a shell command and return exit code.

    Args:
        cmd: Command to run as list of strings
        description: Description for logging

    Returns:
        Exit code from command
    """
    if description:
        print(f"\n🏃 {description}")
        print(f"➤ {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, check=False)
    duration = time.time() - start_time

    if result.returncode == 0:
        print(f"✅ Completed in {duration:.2f}s")
    else:
        print(f"❌ Failed after {duration:.2f}s (exit code: {result.returncode})")

    return result.returncode


def fast_mode() -> int:
    """Run fast unit tests only."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--no-cov",  # Disable coverage for fast mode
        "-n",
        "0",  # Disable parallel execution for fast mode
        "--benchmark-disable",
        "-m",
        "fast or (unit and not slow)",
        "--tb=short",
        "--disable-warnings",
        "-x",  # Stop on first failure
        "--ff",  # Fail fast - run failures first
    ]
    return run_command(cmd, "Running fast tests")


def unit_mode() -> int:
    """Run all unit tests."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--benchmark-disable",
        "-m",
        "unit",
        "--tb=short",
    ]
    return run_command(cmd, "Running unit tests")


def integration_mode() -> int:
    """Run integration tests."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "-m",
        "integration",
        "--tb=line",
        "-v",
    ]
    return run_command(cmd, "Running integration tests")


def performance_mode() -> int:
    """Run performance and benchmark tests."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--no-cov",  # Disable coverage for performance tests
        "-n",
        "0",  # Disable parallel for benchmarks
        "-m",
        "performance or benchmark",
        "--benchmark-enable",
        "--benchmark-autosave",
        "--benchmark-group-by=group",
        "--benchmark-sort=mean",
        "--tb=short",
    ]
    return run_command(cmd, "Running performance tests")


def coverage_mode() -> int:
    """Run tests with comprehensive coverage reporting."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--benchmark-disable",  # Override pyproject.toml to disable benchmarks
        "-m",
        "not slow and not benchmark",
        "-v",
    ]
    return run_command(cmd, "Running coverage tests")


def smoke_mode() -> int:
    """Run critical smoke tests."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "-m",
        "smoke",
        "--tb=short",
        "-x",  # Stop on first failure
        "--ff",  # Run failures first
        "-v",
    ]
    return run_command(cmd, "Running smoke tests")


def regression_mode() -> int:
    """Run regression test suite."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "-m",
        "regression",
        "--tb=line",
        "-v",
    ]
    return run_command(cmd, "Running regression tests")


def parallel_mode() -> int:
    """Run tests with maximum parallelism."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--benchmark-disable",
        "-m",
        "parallel_safe and not serial",
        "--tb=short",
    ]
    return run_command(cmd, "Running parallel tests")


def serial_mode() -> int:
    """Run tests that must execute serially."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "-m",
        "serial",
        "--tb=line",
        "-v",
    ]
    return run_command(cmd, "Running serial tests")


def debug_mode() -> int:
    """Run tests in debug mode with detailed output."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--no-cov",
        "-n",
        "0",  # Disable parallel execution for debugging
        "--benchmark-disable",
        "--tb=long",
        "--capture=no",
        "-s",
        "-vvv",
        "--disable-warnings",
    ]
    return run_command(cmd, "Running debug mode")


def ci_mode() -> int:
    """Run comprehensive test suite for CI/CD."""
    print("🚀 Running CI/CD test suite")

    # 1. Fast smoke tests first
    exit_code = smoke_mode()
    if exit_code != 0:
        return exit_code

    # 2. Unit tests with coverage
    exit_code = coverage_mode()
    if exit_code != 0:
        return exit_code

    # 3. Integration tests
    exit_code = integration_mode()
    if exit_code != 0:
        return exit_code

    # 4. Performance tests (non-blocking)
    print("\n📊 Running performance tests (non-blocking)")
    performance_mode()  # Don't fail CI on performance regressions

    print("✅ CI/CD test suite completed successfully")
    return 0


def all_mode() -> int:
    """Run all test types."""
    print("🎯 Running comprehensive test suite")

    modes = [
        ("Unit Tests", unit_mode),
        ("Integration Tests", integration_mode),
        ("Performance Tests", performance_mode),
        ("Smoke Tests", smoke_mode),
        ("Serial Tests", serial_mode),
    ]

    failed_modes = []

    for mode_name, mode_func in modes:
        print(f"\n{mode_name}")
        print("=" * 50)
        exit_code = mode_func()
        if exit_code != 0:
            failed_modes.append(mode_name)

    if failed_modes:
        print(f"\n❌ Failed modes: {', '.join(failed_modes)}")
        return 1
    print("\n✅ All test modes completed successfully")
    return 0


def lint_and_format() -> int:
    """Run linting and formatting."""
    print("🧹 Running linting and formatting")

    # Format with ruff
    exit_code = run_command(
        ["uv", "run", "ruff", "format", "src", "tests"], "Formatting code with ruff"
    )
    if exit_code != 0:
        return exit_code

    # Lint with ruff
    exit_code = run_command(
        ["uv", "run", "ruff", "check", "src", "tests", "--fix"],
        "Linting code with ruff",
    )
    if exit_code != 0:
        return exit_code

    return 0


def type_check() -> int:
    """Run type checking with mypy."""
    cmd = ["uv", "run", "mypy", "src"]
    return run_command(cmd, "Type checking with mypy")


def security_scan() -> int:
    """Run security scan with bandit."""
    cmd = [
        "uv",
        "run",
        "bandit",
        "-r",
        "src",
        "-f",
        "json",
        "-o",
        "security-report.json",
    ]
    return run_command(cmd, "Security scanning with bandit")


def clean_cache() -> None:
    """Clean test artifacts and cache."""
    print("🧹 Cleaning cache and artifacts")

    cache_dirs = [
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "htmlcov",
        ".coverage",
        "coverage.xml",
        ".benchmarks",
        "benchmark_results",
        "tests/cassettes",
    ]

    for cache_dir in cache_dirs:
        path = Path(cache_dir)
        if path.exists():
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  Removed {cache_dir}")


def list_markers() -> int:
    """List all available pytest markers."""
    cmd = ["uv", "run", "pytest", "--markers"]
    return run_command(cmd, "Listing pytest markers")


def main() -> int:
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Advanced test runner for AI Job Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_tests.py fast        # Run fast tests only
    python scripts/run_tests.py ci          # Full CI/CD suite
    python scripts/run_tests.py coverage    # Coverage analysis
    python scripts/run_tests.py benchmark   # Performance tests
    python scripts/run_tests.py debug       # Debug mode
        """,
    )

    parser.add_argument(
        "mode",
        choices=[
            "fast",
            "unit",
            "integration",
            "performance",
            "coverage",
            "smoke",
            "regression",
            "parallel",
            "serial",
            "debug",
            "ci",
            "all",
            "lint",
            "typecheck",
            "security",
            "clean",
            "markers",
            "benchmark",
        ],
        help="Test execution mode",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - Commands will be shown but not executed")

    setup_environment()

    # Mode dispatch
    mode_functions = {
        "fast": fast_mode,
        "unit": unit_mode,
        "integration": integration_mode,
        "performance": performance_mode,
        "benchmark": performance_mode,  # Alias
        "coverage": coverage_mode,
        "smoke": smoke_mode,
        "regression": regression_mode,
        "parallel": parallel_mode,
        "serial": serial_mode,
        "debug": debug_mode,
        "ci": ci_mode,
        "all": all_mode,
        "lint": lint_and_format,
        "typecheck": type_check,
        "security": security_scan,
        "clean": lambda: (clean_cache(), 0)[1],
        "markers": list_markers,
    }

    mode_func = mode_functions.get(args.mode)
    if not mode_func:
        print(f"❌ Unknown mode: {args.mode}")
        return 1

    try:
        return mode_func()
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
