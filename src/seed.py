"""Seed script for populating the database with initial companies.

This module provides a Typer CLI to insert predefined AI companies into the
database if they do not already exist, based on their URL.
"""

import sqlmodel
import typer

from src.config import Settings
from src.models import CompanySQL

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
    # Define the list of AI/ML companies with their names, career page URLs,
    # and active status - dataset for real job searching
    companies = [
        # Top AI Research Labs & Foundations
        CompanySQL(
            name="Anthropic",
            url="https://www.anthropic.com/careers",
            active=True,
        ),
        CompanySQL(name="OpenAI", url="https://openai.com/careers", active=True),
        CompanySQL(
            name="Google DeepMind",
            url="https://deepmind.google/about/careers/",
            active=True,
        ),
        CompanySQL(name="xAI", url="https://x.ai/careers/", active=True),
        # Big Tech AI Divisions
        CompanySQL(name="Meta", url="https://www.metacareers.com/jobs", active=True),
        CompanySQL(
            name="Microsoft",
            url="https://jobs.careers.microsoft.com/global/en/search",
            active=True,
        ),
        CompanySQL(name="Google", url="https://careers.google.com/jobs/", active=True),
        CompanySQL(
            name="Apple",
            url="https://jobs.apple.com/en-us/search",
            active=True,
        ),
        CompanySQL(name="Amazon", url="https://www.amazon.jobs/en/search", active=True),
        CompanySQL(
            name="Amazon Web Services (AWS)",
            url="https://aws.amazon.com/careers/",
            active=True,
        ),
        CompanySQL(name="IBM", url="https://www.ibm.com/careers/search", active=True),
        # Hardware & Compute
        CompanySQL(
            name="NVIDIA",
            url="https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
            active=True,
        ),
        CompanySQL(name="AMD", url="https://jobs.amd.com/jobs/", active=True),
        CompanySQL(
            name="Intel",
            url="https://jobs.intel.com/en/search-jobs",
            active=True,
        ),
        # AI Startups & Scale-ups
        CompanySQL(name="Scale AI", url="https://scale.com/careers", active=True),
        CompanySQL(
            name="Databricks",
            url="https://www.databricks.com/company/careers",
            active=True,
        ),
        CompanySQL(
            name="Snowflake",
            url="https://careers.snowflake.com/us/en",
            active=True,
        ),
        CompanySQL(
            name="Palantir",
            url="https://www.palantir.com/careers/",
            active=True,
        ),
        CompanySQL(
            name="Weights & Biases",
            url="https://www.wandb.com/careers",
            active=True,
        ),
        CompanySQL(
            name="Hugging Face",
            url="https://apply.workable.com/huggingface/",
            active=True,
        ),
        CompanySQL(name="Cohere", url="https://cohere.com/careers", active=True),
        CompanySQL(name="Adept", url="https://www.adept.ai/careers", active=True),
        CompanySQL(
            name="Inflection AI",
            url="https://inflection.ai/careers",
            active=True,
        ),
        CompanySQL(
            name="Character.AI",
            url="https://character.ai/careers",
            active=True,
        ),
        CompanySQL(
            name="Perplexity",
            url="https://www.perplexity.ai/careers",
            active=True,
        ),
        # Cloud Providers & MLOps
        CompanySQL(
            name="Anyscale",
            url="https://www.anyscale.com/careers",
            active=True,
        ),
        CompanySQL(name="Modal", url="https://modal.com/careers", active=True),
        CompanySQL(name="Replicate", url="https://replicate.com/careers", active=True),
        CompanySQL(
            name="Together AI",
            url="https://www.together.ai/careers",
            active=True,
        ),
        CompanySQL(name="RunPod", url="https://www.runpod.io/careers", active=True),
        # Autonomous Vehicles
        CompanySQL(
            name="Tesla",
            url="https://www.tesla.com/careers/search/",
            active=True,
        ),
        CompanySQL(name="Waymo", url="https://waymo.com/careers/", active=True),
        CompanySQL(name="Argo AI", url="https://www.argo.ai/careers/", active=True),
        # Robotics
        CompanySQL(
            name="Boston Dynamics",
            url="https://www.bostondynamics.com/careers",
            active=True,
        ),
        CompanySQL(
            name="Agility Robotics",
            url="https://agilityrobotics.com/careers/",
            active=True,
        ),
        CompanySQL(name="Figure", url="https://www.figure.ai/careers", active=True),
        CompanySQL(
            name="1X Technologies",
            url="https://www.1x.tech/careers",
            active=True,
        ),
        # Computer Vision & Edge AI
        CompanySQL(name="Sentry", url="https://sentry.io/careers/", active=True),
        # Financial AI
        CompanySQL(
            name="Renaissance Technologies",
            url="https://www.rentec.com/Careers.action",
            active=True,
        ),
        # Enterprise AI
        CompanySQL(name="C3.ai", url="https://c3.ai/careers/", active=True),
        CompanySQL(
            name="DataRobot",
            url="https://www.datarobot.com/careers/",
            active=True,
        ),
        CompanySQL(name="H2O.ai", url="https://h2o.ai/careers/", active=True),
        CompanySQL(name="Alteryx", url="https://www.alteryx.com/careers", active=True),
        # AI Tools & Infrastructure
        CompanySQL(
            name="Pinecone",
            url="https://www.pinecone.io/careers/",
            active=True,
        ),
        CompanySQL(
            name="LangChain",
            url="https://www.langchain.com/careers",
            active=True,
        ),
        CompanySQL(name="Weaviate", url="https://weaviate.io/careers", active=True),
        CompanySQL(name="Qdrant", url="https://qdrant.tech/careers/", active=True),
        # Healthcare AI
        # Gaming & Entertainment AI
        CompanySQL(
            name="Midjourney",
            url="https://www.midjourney.com/careers",
            active=True,
        ),
        CompanySQL(
            name="Stability AI",
            url="https://stability.ai/careers",
            active=True,
        ),
        CompanySQL(name="Runway", url="https://runwayml.com/careers/", active=True),
        # Consulting & Services
        # Research Institutions (with industry partnerships)
        CompanySQL(
            name="Allen Institute for AI",
            url="https://allenai.org/careers",
            active=True,
        ),
        CompanySQL(
            name="FAIR (Meta AI)",
            url="https://ai.facebook.com/join-us/",
            active=True,
        ),
        # Additional High-Growth AI Companies
        CompanySQL(name="Notion", url="https://www.notion.so/careers", active=True),
        CompanySQL(name="Figma", url="https://www.figma.com/careers/", active=True),
        CompanySQL(name="Linear", url="https://linear.app/careers", active=True),
        CompanySQL(name="Retool", url="https://retool.com/careers/", active=True),
        CompanySQL(name="Vercel", url="https://vercel.com/careers", active=True),
        CompanySQL(name="Supabase", url="https://supabase.com/careers", active=True),
    ]

    # Open a database session for transactions
    with sqlmodel.Session(engine) as session:
        # Initialize counter for newly added companies
        added = 0
        # Iterate over each company in the list
        for comp in companies:
            # Query the database to check if a company with this name already exists
            existing = session.exec(
                sqlmodel.select(CompanySQL).where(CompanySQL.name == comp.name),
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
