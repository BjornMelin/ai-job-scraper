#!/usr/bin/env python3
"""Generate accurate metrics for the project."""

import json

from pathlib import Path


def count_lines(file_path):
    """Count non-empty lines in a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0
    except UnicodeDecodeError:
        # Skip binary files
        return 0


def count_directory_lines(directory, pattern="*.py"):
    """Count total lines in directory."""
    total = 0
    try:
        for file in Path(directory).rglob(pattern):
            if "__pycache__" not in str(file) and ".venv" not in str(file):
                total += count_lines(file)
    except FileNotFoundError:
        pass
    return total


def main():
    """Generate comprehensive project metrics."""
    project_root = Path(__file__).parent.parent

    # Calculate key service metrics
    search_service_current = count_lines(
        project_root / "src/services/search_service.py"
    )
    analytics_service_current = count_lines(
        project_root / "src/services/analytics_service.py"
    )
    background_helpers_current = count_lines(
        project_root / "src/ui/utils/background_helpers.py"
    )

    # Calculate directory totals
    src_lines = count_directory_lines(project_root / "src")
    tests_lines = count_directory_lines(project_root / "tests")
    total_lines = src_lines + tests_lines

    # Calculate reduction percentages based on optimization goals
    search_service_original = 653
    analytics_service_original = 915
    background_helpers_original = 432

    search_reduction = (
        max(
            0,
            round(
                (search_service_original - search_service_current)
                / search_service_original
                * 100
            ),
        )
        if search_service_original > 0
        else 0
    )
    analytics_reduction = (
        max(
            0,
            round(
                (analytics_service_original - analytics_service_current)
                / analytics_service_original
                * 100
            ),
        )
        if analytics_service_original > 0
        else 0
    )
    background_reduction = (
        max(
            0,
            round(
                (background_helpers_original - background_helpers_current)
                / background_helpers_original
                * 100
            ),
        )
        if background_helpers_original > 0
        else 0
    )

    metrics = {
        "timestamp": Path(__file__).stat().st_mtime,
        "search_service": {
            "current": search_service_current,
            "original": search_service_original,
            "reduction_pct": search_reduction,
        },
        "analytics_service": {
            "current": analytics_service_current,
            "original": analytics_service_original,
            "reduction_pct": analytics_reduction,
        },
        "background_helpers": {
            "current": background_helpers_current,
            "original": background_helpers_original,
            "reduction_pct": background_reduction,
        },
        "total_project": {
            "source": src_lines,
            "tests": tests_lines,
            "total": total_lines,
        },
        "test_coverage": {
            "unit_tests": count_directory_lines(project_root / "tests/unit"),
            "integration_tests": count_directory_lines(
                project_root / "tests/integration"
            ),
            "service_tests": count_directory_lines(project_root / "tests/services"),
            "ui_tests": count_directory_lines(project_root / "tests/ui"),
        },
    }

    # Write metrics to file
    metrics_file = project_root / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("📊 Project Metrics Generated")
    print("=" * 40)
    print(f"✅ Source code: {src_lines} lines")
    print(f"✅ Test code: {tests_lines} lines")
    print(f"✅ Total lines: {total_lines} lines")
    print()
    print("🎯 Optimization Results:")
    print(
        f"  • Search service: {search_reduction}% reduction ({search_service_current}/{search_service_original} lines)"
    )
    print(
        f"  • Analytics service: {analytics_reduction}% reduction ({analytics_service_current}/{analytics_service_original} lines)"
    )
    print(
        f"  • Background helpers: {background_reduction}% reduction ({background_helpers_current}/{background_helpers_original} lines)"
    )
    print()
    print(f"📁 Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
