#!/usr/bin/env python3
"""Check for dead code patterns."""

import os
import sys

from pathlib import Path


def main():
    """Check for various dead code patterns in the project."""
    project_root = Path(__file__).parent.parent
    issues = []

    print("🔍 Scanning for dead code patterns...")

    # Check for dead test files
    dead_patterns = [
        "*_old.py",
        "*_fixed.py",
        "*_backup.py",
        "*_test_old.py",
        "*_deprecated.py",
    ]
    for pattern in dead_patterns:
        for file in project_root.rglob(pattern):
            if ".venv" not in str(file) and "__pycache__" not in str(file):
                issues.append(f"Dead file: {file.relative_to(project_root)}")

    # Check for __pycache__ directories outside .venv
    for cache_dir in project_root.rglob("__pycache__"):
        if ".venv" not in str(cache_dir):
            issues.append(f"Cache directory: {cache_dir.relative_to(project_root)}")

    # Check for .pyc files outside .venv
    for pyc in project_root.rglob("*.pyc"):
        if ".venv" not in str(pyc):
            issues.append(f"Python cache: {pyc.relative_to(project_root)}")

    # Check for temporary files
    temp_patterns = ["*.tmp", "*.temp", "*~", "*.bak", ".DS_Store"]
    for pattern in temp_patterns:
        for file in project_root.rglob(pattern):
            if ".venv" not in str(file) and ".git" not in str(file):
                issues.append(f"Temporary file: {file.relative_to(project_root)}")

    # Check for empty directories (except known empty ones)
    known_empty = {".git", "__pycache__", ".venv", ".pytest_cache", ".coverage"}
    for dirpath, dirnames, filenames in os.walk(project_root):
        rel_path = Path(dirpath).relative_to(project_root)

        # Skip if it's a known empty or system directory
        if any(part in known_empty or part.startswith(".") for part in rel_path.parts):
            continue

        # Check if directory is empty (no files, no subdirs with files)
        if not filenames and not any(
            Path(dirpath, d).iterdir() for d in dirnames if not d.startswith(".")
        ):
            issues.append(f"Empty directory: {rel_path}")

    # Check for duplicate or conflicting test files
    test_files = {}
    for test_file in (project_root / "tests").rglob("test_*.py"):
        if "__pycache__" not in str(test_file):
            base_name = test_file.stem
            if base_name in test_files:
                issues.append(
                    f"Potential duplicate test: {test_file.relative_to(project_root)} vs {test_files[base_name].relative_to(project_root)}"
                )
            else:
                test_files[base_name] = test_file

    # Check for unused imports in __init__.py files
    for init_file in project_root.rglob("__init__.py"):
        if ".venv" not in str(init_file):
            try:
                with open(init_file, encoding="utf-8") as f:
                    content = f.read().strip()
                    if (
                        content
                        and not content.startswith('"""')
                        and not content.startswith("'''")
                    ):
                        # Only flag if it has imports but no docstring
                        if "import " in content and len(content.split("\n")) > 5:
                            issues.append(
                                f"Complex __init__.py (potential dead imports): {init_file.relative_to(project_root)}"
                            )
            except (UnicodeDecodeError, FileNotFoundError):
                pass

    # Report results
    if issues:
        print("\n❌ Dead code found:")
        print("=" * 50)
        for issue in sorted(issues):
            print(f"  • {issue}")
        print("=" * 50)
        print(f"Total issues: {len(issues)}")
        print("\n💡 Cleanup suggestions:")
        print("  • Remove dead test files")
        print("  • Run: find . -name '__pycache__' -type d -exec rm -rf {} +")
        print("  • Run: find . -name '*.pyc' -delete")
        print("  • Remove empty directories")
        print("  • Consolidate duplicate tests")
        sys.exit(1)
    else:
        print("✅ No dead code found!")
        sys.exit(0)


if __name__ == "__main__":
    main()
