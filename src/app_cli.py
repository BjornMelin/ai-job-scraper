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

    # Run streamlit with the main.py file
    # nosec B603: Using subprocess with controlled input (hardcoded command)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(main_path)],
            check=True,
            shell=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
