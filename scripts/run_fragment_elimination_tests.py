#!/usr/bin/env python3
"""Fragment Elimination Test Runner.

This script runs the comprehensive fragment elimination validation test suite
with appropriate configuration and provides detailed reporting of results.

Usage:
    python scripts/run_fragment_elimination_tests.py [--mode=<mode>]

Modes:
    all (default): Run all fragment elimination tests
    smoke: Run only smoke tests for quick validation
    performance: Run only performance-critical tests
    detection: Run only fragment detection tests
    state: Run only session state tests
    ui: Run only UI functionality tests
    regression: Run only regression tests
"""

import argparse
import subprocess
import sys

from pathlib import Path


def run_tests(mode: str = "all", verbose: bool = True) -> int:
    """Run fragment elimination tests with specified mode.

    Args:
        mode: Test mode to run
        verbose: Whether to run with verbose output

    Returns:
        Exit code from pytest
    """
    # Base pytest command
    cmd = ["uv", "run", "pytest", "tests/fragment_elimination/"]

    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s", "--tb=short"])

    # Add mode-specific markers
    if mode == "smoke":
        cmd.extend(["-m", "smoke"])
    elif mode == "performance":
        cmd.extend(["-m", "performance_critical"])
    elif mode == "detection":
        cmd.extend(["-k", "fragment_detection"])
    elif mode == "state":
        cmd.extend(["-m", "session_state"])
    elif mode == "ui":
        cmd.extend(["-m", "ui_functionality"])
    elif mode == "regression":
        cmd.extend(["-m", "regression"])
    elif mode == "all":
        cmd.extend(["-m", "fragment_elimination"])
    else:
        print(f"Unknown mode: {mode}")
        return 1

    # Add coverage reporting for comprehensive runs
    if mode in ["all", "regression"]:
        cmd.extend(
            [
                "--cov=src/ui",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/fragment_elimination",
            ]
        )

    # Add performance options
    if mode == "performance":
        cmd.extend(["--benchmark-enable"])

    print(f"Running fragment elimination tests in {mode} mode...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    # Run the tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=False)

    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run fragment elimination validation tests"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "smoke",
            "performance",
            "detection",
            "state",
            "ui",
            "regression",
        ],
        default="all",
        help="Test mode to run (default: all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Run tests in quiet mode")

    args = parser.parse_args()

    # Run tests
    exit_code = run_tests(mode=args.mode, verbose=not args.quiet)

    # Report results
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ Fragment elimination tests PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Fragment elimination tests FAILED")
        print("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
