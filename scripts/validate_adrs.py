#!/usr/bin/env python3
"""Validate ADR filename and title consistency."""

import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "docs/adrs"


def main():
    """Validate all ADR files have matching IDs in filename and title."""
    if not os.path.exists(ROOT):
        print(f"ADR directory not found: {ROOT}")
        sys.exit(1)

    ok = True
    issues = []

    for f in sorted(os.listdir(ROOT)):
        if not f.lower().endswith(".md"):
            continue

        # Skip non-ADR files
        if f in [
            "ADR_TEMPLATE.md",
            "ARCHITECTURE_OVERVIEW.md",
            "IMPLEMENTATION_GUIDE.md",
            "README.md",
        ]:
            continue

        # Check filename format
        m = re.match(r"ADR-(\d{3,4})-.*\.md$", f, re.IGNORECASE)
        if not m:
            issues.append(f"Bad filename format: {f}")
            ok = False
            continue

        file_id = m.group(1).zfill(4)  # Normalize to 4 digits

        # Check title in file
        filepath = os.path.join(ROOT, f)
        try:
            with open(filepath, encoding="utf-8") as fh:
                first_line = fh.readline().strip()

            # Look for ADR ID in title
            m2 = re.match(r"#\s*ADR[-\s]?0*(\d+)\s*:?", first_line, re.IGNORECASE)
            if not m2:
                issues.append(f"No ADR ID in title of {f}: '{first_line}'")
                ok = False
                continue

            title_id = f"{int(m2.group(1)):04d}"  # Normalize to 4 digits

            if file_id != title_id:
                issues.append(
                    f"ID mismatch in {f}: "
                    f"filename has ADR-{file_id} but title has ADR-{title_id}"
                )
                ok = False

        except Exception as e:
            issues.append(f"Error reading {f}: {e}")
            ok = False

    if not ok:
        print("❌ ADR VALIDATION FAILED")
        print("=" * 60)
        for issue in issues:
            print(f"  • {issue}")
        print("=" * 60)
        print(f"Total issues: {len(issues)}")
        sys.exit(1)
    else:
        print("✅ All ADRs have consistent IDs")
        sys.exit(0)


if __name__ == "__main__":
    main()
