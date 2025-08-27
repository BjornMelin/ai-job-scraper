"""Week 1 Validation Test Runner.

Main entry point for running Week 1 stream validation tests.
Provides comprehensive validation with benchmark reporting.
"""

import sys

from pathlib import Path

import pytest

from tests.week1_validation.benchmark_runner import run_week1_benchmarks


def main():
    """Main entry point for Week 1 validation."""
    print("🚀 Week 1 Stream Validation Framework")
    print("=" * 50)

    # Get repo root
    repo_root = Path(__file__).parent.parent.parent
    week1_validation_dir = repo_root / "tests" / "week1_validation"

    print(f"📁 Validation directory: {week1_validation_dir}")
    print(f"🔍 Repository root: {repo_root}")

    # Run pytest with Week 1 validation tests
    print("\n🧪 Running Week 1 validation tests...")
    pytest_args = [
        str(week1_validation_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--disable-warnings",  # Clean output
    ]

    # Run tests
    test_exit_code = pytest.main(pytest_args)

    if test_exit_code == 0:
        print("✅ All Week 1 validation tests passed!")

        # Run comprehensive benchmarks
        print("\n📊 Running comprehensive benchmarks...")
        try:
            results = run_week1_benchmarks(iterations=3)  # Reduced iterations for CI

            # Print summary
            overall = results["summary"]["overall_assessment"]
            print("\n🎯 BENCHMARK RESULTS:")
            print(f"   Week 1 Targets Met: {overall['week1_targets_met']}")
            print(f"   Integration Successful: {overall['integration_successful']}")
            print(f"   Deployment Ready: {overall['deployment_ready']}")
            print(f"   Confidence Score: {overall['confidence_score']:.1f}%")
            print(f"   Status: {overall['overall_status']}")

            if overall["deployment_ready"]:
                print("\n🎉 Week 1 validation SUCCESSFUL!")
                print("   All streams meet their performance targets.")
                print("   Integration between streams works correctly.")
                print("   System is ready for deployment.")
                return 0
            print("\n⚠️ Week 1 validation completed with issues.")
            print("   Some targets may not be fully met.")
            print("   Check benchmark report for recommendations.")
            return 1

        except Exception as e:
            print(f"\n❌ Benchmark execution failed: {e}")
            return 1
    else:
        print("❌ Week 1 validation tests failed!")
        return test_exit_code


if __name__ == "__main__":
    sys.exit(main())
