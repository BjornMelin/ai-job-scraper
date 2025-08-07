"""CLI entry point for the Streamlit app.

This module provides a CLI command to run the Streamlit dashboard.
"""

import subprocess
import sys

from pathlib import Path


def main() -> None:
    """Run the Streamlit dashboard."""
    # Get the directory containing main.py (same as src/)
    src_dir = Path(__file__).resolve().parent
    main_path = src_dir / "main.py"

    # Validate the main.py file exists and is in the expected location
    if not main_path.exists():
        print(f"Error: main.py not found at {main_path}")
        sys.exit(1)

    if not main_path.is_file():
        print(f"Error: {main_path} is not a file")
        sys.exit(1)

    # Ensure the main.py file is within our expected directory structure
    try:
        main_path.resolve().relative_to(src_dir.resolve())
    except ValueError:
        print(f"Error: main.py is not within expected directory {src_dir}")
        sys.exit(1)

    # Use absolute path to avoid any path resolution issues
    main_path_str = str(main_path.resolve())

    # Run streamlit with the validated main.py file
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", main_path_str],
            check=True,
            shell=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
