#!/usr/bin/env python3
"""Find duplicate test files based on normalized content."""

import hashlib
import os
import re
import sys

from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "tests"


def normalize_py(text):
    """Normalize Python code by removing comments and blank lines."""
    lines = []
    for line in text.splitlines():
        # Skip comments
        if re.match(r"^\s*#", line):
            continue
        # Skip blank lines
        if not line.strip():
            continue
        lines.append(line.strip())
    return "\n".join(lines)


def main():
    """Find duplicate test files."""
    hash_map = defaultdict(list)

    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(".py"):
                continue
            if f == "__init__.py":
                continue

            p = os.path.join(dirpath, f)
            try:
                with open(p, encoding="utf-8") as fh:
                    content = fh.read()
                    norm = normalize_py(content)
                    if len(norm) < 100:  # Skip very small files
                        continue

                h = hashlib.sha256(norm.encode()).hexdigest()
                hash_map[h].append(p)
            except Exception as e:
                print(f"Error processing {p}: {e}", file=sys.stderr)
                continue

    # Find duplicates
    dups = [v for v in hash_map.values() if len(v) > 1]

    if dups:
        print("DUPLICATE TEST FILES FOUND:")
        print("=" * 60)
        total_lines = 0

        for group in dups:
            print("\nDuplicate group:")
            for p in group:
                lines = sum(1 for _ in open(p))
                print(f"  {p} ({lines} lines)")
                if group.index(p) > 0:  # Count all but first as duplicates
                    total_lines += lines

        print("\n" + "=" * 60)
        print(f"Total duplicate lines that could be removed: {total_lines}")
        sys.exit(1)
    else:
        print("✅ No duplicate test files found")
        sys.exit(0)


if __name__ == "__main__":
    main()
