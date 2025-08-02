"""CLI entry point for the Streamlit app.

This module provides a CLI command to run the Streamlit dashboard.
"""

import os
import subprocess
import sys


def main() -> None:
    """Run the Streamlit dashboard."""
    # Get the directory containing app.py (parent of src/)
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(app_dir, "app.py")

    # Run streamlit with the app.py file
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
