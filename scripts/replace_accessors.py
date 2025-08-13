#!/usr/bin/env python3
"""Replace simple accessor functions with inline expressions."""

import re

from pathlib import Path


def replace_accessors(file_path: Path) -> bool:
    """Replace accessor functions with inline expressions."""
    content = file_path.read_text()
    original = content

    # Replace get_salary_min() calls
    # Pattern: get_salary_min(expr) -> (expr[0] if expr else None)
    content = re.sub(
        r"get_salary_min\(([^)]+)\)",
        lambda m: f"({m.group(1)}[0] if {m.group(1)} else None)",
        content,
    )

    # Replace get_salary_max() calls
    # Pattern: get_salary_max(expr) -> (expr[1] if expr else None)
    content = re.sub(
        r"get_salary_max\(([^)]+)\)",
        lambda m: f"({m.group(1)}[1] if {m.group(1)} else None)",
        content,
    )

    # Replace get_job_company_name() calls
    # Pattern: get_job_company_name(expr) -> (expr.name if expr else "Unknown")
    content = re.sub(
        r"get_job_company_name\(([^)]+)\)",
        lambda m: f'({m.group(1)}.name if {m.group(1)} else "Unknown")',
        content,
    )

    if content != original:
        file_path.write_text(content)
        return True
    return False


def main():
    """Main replacement script."""
    # Files to update
    files_to_update = [
        "tests/unit/models/test_salary_parser_comprehensive.py",
        "tests/ui/utils/test_ui_helpers_computed.py",
        "tests/ui/utils/test_ui_helpers_job_company.py",
    ]

    project_root = Path("/home/bjorn/repos/ai-job-scraper")

    for file_path in files_to_update:
        full_path = project_root / file_path
        if full_path.exists():
            if replace_accessors(full_path):
                print(f"✓ Updated {file_path}")
            else:
                print(f"  No changes needed in {file_path}")
        else:
            print(f"✗ File not found: {file_path}")


if __name__ == "__main__":
    main()
