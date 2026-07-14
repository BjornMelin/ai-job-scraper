"""Regression tests for the application-owned SQLModel registry."""

import subprocess
import sys


def test_database_models_can_be_reloaded() -> None:
    """Keep Streamlit source reloads from redefining shared SQLModel tables."""
    script = """
from importlib import reload
from src import database_models

reload(database_models)
assert set(database_models.AppSQLModel.metadata.tables) == {
    "companysql", "cost_entries", "jobsql", "savedsearchsql"
}
"""

    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
