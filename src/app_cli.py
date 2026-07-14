"""CLI entry point for the Streamlit application."""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from src.db.migrations import run_migrations

app = typer.Typer(add_completion=False)


@app.command()
def dashboard(
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8501,
    address: Annotated[str, typer.Option()] = "127.0.0.1",
) -> None:
    """Run the Streamlit dashboard."""
    run_migrations()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).with_name("main.py")),
            f"--server.port={port}",
            f"--server.address={address}",
        ],
        check=False,
    )
    if result.returncode:
        raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
