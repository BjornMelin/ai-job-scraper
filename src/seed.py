"""Seed script for populating the database with initial companies.

This module provides a Typer CLI to insert predefined AI companies into the
database if they do not already exist, based on their URL.
"""

import sqlmodel
import typer

from .config import Settings
from .models import CompanySQL

settings = Settings()
engine = sqlmodel.create_engine(settings.db_url)

app = typer.Typer()


@app.command()
def seed() -> None:
    """Seed the database with initial active AI companies.

    This function defines a hardcoded list of core AI companies, checks for their
    existence in the database by name (to avoid duplicates), adds any missing ones,
    commits the changes, and prints the count of added companies. It is designed
    to be idempotent, allowing safe repeated executions without creating duplicates.

    Returns:
        None: This function does not return a value but prints the result to stdout.
    """
    # Define the list of core AI companies with their names, career page URLs,
    # and active status
    companies = [
        CompanySQL(
            name="Anthropic", url="https://www.anthropic.com/careers", active=True
        ),
        CompanySQL(name="OpenAI", url="https://openai.com/careers", active=True),
        CompanySQL(
            name="Google DeepMind",
            url="https://deepmind.google/about/careers/",
            active=True,
        ),
        CompanySQL(name="xAI", url="https://x.ai/careers/", active=True),
        CompanySQL(name="Meta", url="https://www.metacareers.com/jobs", active=True),
        CompanySQL(
            name="Microsoft",
            url="https://jobs.careers.microsoft.com/global/en/search",
            active=True,
        ),
        CompanySQL(
            name="NVIDIA",
            url="https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
            active=True,
        ),
    ]

    # Open a database session for transactions
    with sqlmodel.Session(engine) as session:
        # Initialize counter for newly added companies
        added = 0
        # Iterate over each company in the list
        for comp in companies:
            # Query the database to check if a company with this name already exists
            existing = session.exec(
                sqlmodel.select(CompanySQL).where(CompanySQL.name == comp.name)
            ).first()
            # If no existing entry, add the new company and increment the counter
            if not existing:
                session.add(comp)
                added += 1
        # Commit all changes to the database
        session.commit()
        # Print the number of companies successfully seeded
        print(f"Seeded {added} companies.")


if __name__ == "__main__":
    app()
