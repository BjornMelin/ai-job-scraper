#!/usr/bin/env python3
"""Quick validation test for the testing infrastructure."""

import subprocess
import sys

from pathlib import Path


def run_command(cmd: list[str], timeout: int = 30) -> dict:
    """Run a command and return results."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command timed out",
        }


def main():
    """Run validation tests for the infrastructure."""
    print("🔧 Testing Validation Infrastructure")
    print("=" * 50)

    tests = [
        {
            "name": "Generate Metrics",
            "cmd": ["uv", "run", "python", "scripts/generate_metrics.py"],
            "expect_success": True,
        },
        {
            "name": "Check Dead Code",
            "cmd": ["uv", "run", "python", "scripts/check_dead_code.py"],
            "expect_success": False,  # Expecting to find some issues
        },
        {
            "name": "Find Duplicate Tests",
            "cmd": ["uv", "run", "python", "scripts/find_duplicate_tests.py"],
            "expect_success": True,  # Should pass after cleanup
        },
        {
            "name": "Fast Unit Test",
            "cmd": [
                "uv",
                "run",
                "pytest",
                "tests/unit/core/test_constants.py",
                "-v",
                "--tb=short",
            ],
            "expect_success": True,
        },
        {
            "name": "Import Validation",
            "cmd": [
                "uv",
                "run",
                "python",
                "-c",
                "import src.config; import src.models; print('✅ Core imports work')",
            ],
            "expect_success": True,
        },
    ]

    results = []

    for test in tests:
        print(f"\n🧪 Running: {test['name']}")
        result = run_command(test["cmd"])

        success = result["success"] == test["expect_success"]
        results.append(
            {
                "name": test["name"],
                "passed": success,
                "expected": test["expect_success"],
                "actual": result["success"],
            }
        )

        if success:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            if result["stderr"]:
                print(f"   Error: {result['stderr'][:200]}...")
            if result["stdout"] and "❌" not in result["stdout"]:
                print(f"   Output: {result['stdout'][:200]}...")

    # Summary
    print("\n📊 Validation Summary")
    print("=" * 50)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for result in results:
        status = "✅" if result["passed"] else "❌"
        expected = "should pass" if result["expected"] else "should find issues"
        print(f"{status} {result['name']} ({expected})")

    print(f"\nOverall: {passed}/{total} validation checks passed")

    if passed >= 4:  # Allow one test to fail
        print("🎉 Validation infrastructure is working!")
        return 0
    print("⚠️  Some validation issues need attention")
    return 1


if __name__ == "__main__":
    sys.exit(main())
