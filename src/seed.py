"""Seed useful starter searches without inventing company records."""

import typer
from sqlmodel import select

from src.database import db_session
from src.database_models import SavedSearchSQL
from src.schemas import SavedSearchCreate

app = typer.Typer()

STARTER_SEARCHES = (
    ("AI Engineering", "AI engineer"),
    ("Machine Learning", "machine learning engineer"),
    ("Data Science", "data scientist"),
    ("MLOps", "MLOps engineer"),
    ("AI Product", "AI product manager"),
)


@app.command()
def seed() -> None:
    """Create idempotent starter saved searches."""
    added = 0
    with db_session() as session:
        existing_names = set(session.exec(select(SavedSearchSQL.name)).all())
        for name, query in STARTER_SEARCHES:
            if name in existing_names:
                continue
            data = SavedSearchCreate(name=name, query=query)
            session.add(SavedSearchSQL.model_validate(data.model_dump(mode="json")))
            added += 1
    typer.echo(f"Seeded {added} saved searches.")


if __name__ == "__main__":
    app()
