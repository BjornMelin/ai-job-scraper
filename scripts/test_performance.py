#!/usr/bin/env python3
"""Test performance comparison script.

This script demonstrates the performance improvements between different test
configurations.
"""

import subprocess
import time

from pathlib import Path


def run_command(cmd: list[str], timeout: int = 60) -> dict:
    """Run a command and measure execution time.

    Args:
        cmd: Command to run, must be a non-empty list of strings
        timeout: Maximum time to allow the command to run

    Returns:
        Dict with command execution details
    """
    # Validate input: cmd must be a non-empty list of strings
    if not cmd or not all(isinstance(arg, str) for arg in cmd):
        raise ValueError("Command must be a non-empty list of strings")

    # Sanitize command to prevent shell injection
    sanitized_cmd = [str(arg).replace("`", "").replace("$", "") for arg in cmd]

    start_time = time.time()
    try:
        # Safe execution with input validation and error checking
        result = subprocess.run(  # noqa: S603
            sanitized_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
            check=True,  # Raise CalledProcessError for non-zero exit codes
        )
        elapsed = time.time() - start_time
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": "Command timed out",
        }
    else:
        return {
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def main():
    """Run performance comparison tests."""
    print("🚀 AI Job Scraper Test Performance Analysis")
    print("=" * 50)

    # Test configurations to compare
    test_configs = [
        {
            "name": "Fast Unit Tests (No Coverage)",
            "cmd": [
                "uv",
                "run",
                "pytest",
                "tests/unit/core/test_config.py",
                "--no-cov",
                "-v",
            ],
            "timeout": 30,
        },
        {
            "name": "Unit Tests with Coverage",
            "cmd": [
                "uv",
                "run",
                "pytest",
                "tests/unit/core/test_config.py",
                "--cov=src",
                "-v",
            ],
            "timeout": 60,
        },
        {
            "name": "Fast Unit Tests (Optimized)",
            "cmd": [
                "uv",
                "run",
                "pytest",
                "tests/unit/core/test_config.py",
                "-m",
                "not slow",
                "--tb=short",
                "-q",
            ],
            "timeout": 30,
        },
        {
            "name": "Small Test Suite (10 tests)",
            "cmd": [
                "uv",
                "run",
                "pytest",
                "tests/unit/core/",
                "--no-cov",
                "-v",
                "--maxfail=10",
            ],
            "timeout": 45,
        },
    ]

    results = []

    for config in test_configs:
        print(f"\n📊 Running: {config['name']}")
        print(f"Command: {' '.join(config['cmd'])}")

        result = run_command(config["cmd"], config["timeout"])
        results.append(
            {
                "name": config["name"],
                "elapsed": result["elapsed"],
                "success": result["success"],
            },
        )

        if result["success"]:
            print(f"✅ Success in {result['elapsed']:.2f}s")
        else:
            print(f"❌ Failed after {result['elapsed']:.2f}s")
            if "timed out" in result["stderr"]:
                print("   Reason: Timeout")
            else:
                print(f"   Reason: {result['stderr'][:100]}...")

    # Summary
    print("\n📈 Performance Summary")
    print("=" * 50)

    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['name']}: {result['elapsed']:.2f}s")

    # Calculate improvements
    if len(results) >= 2:
        baseline = results[1]["elapsed"]  # Coverage test
        optimized = results[0]["elapsed"]  # Fast test
        if baseline > 0:
            improvement = (baseline - optimized) / baseline * 100
            print(f"\n🎯 Speed Improvement: {improvement:.1f}% faster without coverage")

    print("\n💡 Recommendations:")
    print("   • Use 'uv run pytest --no-cov' for development")
    print("   • Use 'uv run pytest -n auto' for parallel execution")
    print("   • Use coverage only in CI/CD pipelines")
    print("   • Mark slow tests with '@pytest.mark.slow'")


if __name__ == "__main__":
    main()
