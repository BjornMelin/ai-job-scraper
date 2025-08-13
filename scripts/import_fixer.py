#!/usr/bin/env python3
"""Automated import refactoring script for the AI Job Scraper project."""

import re

from pathlib import Path


def fix_typing_imports(file_path: str) -> None:
    """Move typing-only imports into TYPE_CHECKING block."""
    with Path(file_path).open() as f:
        content = f.read()

    # Define patterns for import types
    patterns = {
        "any_imports": re.compile(
            r"^from typing import (\w+(?:, \w+)*)$", re.MULTILINE
        ),
        "local_imports": re.compile(
            r"^from (src\.\w+|[\w\.]+) import (\w+(?:, \w+)*)$", re.MULTILINE
        ),
        "stdlib_imports": re.compile(
            r"^from (collections\.abc|typing) import (\w+(?:, \w+)*)$", re.MULTILINE
        ),
        "thirdparty_imports": re.compile(
            r"^from (sqlmodel|sqlalchemy|pydantic) import (\w+(?:, \w+)*)$",
            re.MULTILINE,
        ),
    }

    # Collect different types of imports
    imports = {}
    for key, pattern in patterns.items():
        matches = pattern.findall(content)
        if matches:
            # Split comma-separated imports
            imports[key] = []
            for match in matches:
                if isinstance(match, tuple):
                    mod, imps = match
                    imports[key].extend([(mod, imp.strip()) for imp in imps.split(",")])
                else:
                    imports[key].append(match)

    # Remove all these imports from the main content
    for pattern in patterns.values():
        content = pattern.sub("", content)

    # Determine if TYPE_CHECKING block exists
    type_checking_exists = (
        re.search(r"^if TYPE_CHECKING:", content, re.MULTILINE) is not None
    )

    # Prepare TYPE_CHECKING content
    type_checking_content = "\n\nif TYPE_CHECKING:\n"

    # Add imports in a logical order
    import_order = [
        "any_imports",
        "stdlib_imports",
        "thirdparty_imports",
        "local_imports",
    ]
    for key in import_order:
        if imports.get(key):
            if key == "any_imports":
                # Simplified Any import
                type_checking_content += "    from typing import Any\n"
            else:
                # Other imports with module handling
                mod_imports = {}
                for mod, imp in imports[key]:
                    if mod not in mod_imports:
                        mod_imports[mod] = []
                    mod_imports[mod].append(imp)

                for mod, imps in mod_imports.items():
                    type_checking_content += (
                        f"    from {mod} import {', '.join(imps)}\n"
                    )

    # Insert TYPE_CHECKING block
    if not type_checking_exists:
        # Find first import block, add TYPE_CHECKING right after
        import_block = re.search(r"(from \w+ import [^\n]+\n)+", content)
        if import_block:
            end_pos = import_block.end()
            content = content[:end_pos] + type_checking_content + content[end_pos:]
    else:
        # If TYPE_CHECKING already exists, append imports
        type_checking_pattern = re.compile(r"(if TYPE_CHECKING:)\n", re.MULTILINE)
        content = type_checking_pattern.sub(
            r"\1\n" + type_checking_content[len("\n\nif TYPE_CHECKING:\n") :], content
        )

    with Path(file_path).open("w") as f:
        f.write(content)


def process_files(directory: str, extensions: list[str] | None = None) -> None:
    """Process all files in the directory with specified extensions."""
    if extensions is None:
        extensions = [".py"]

    for root_path in Path(directory).rglob("*"):
        if root_path.is_file() and any(root_path.suffix == ext for ext in extensions):
            file_path = str(root_path)
            try:
                fix_typing_imports(file_path)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def main():
    """Main execution function."""
    project_root = str(Path(__file__).resolve().parent.parent)
    process_files(project_root)


if __name__ == "__main__":
    main()
