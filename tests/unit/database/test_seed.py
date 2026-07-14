"""Starter saved-search seeding."""

from sqlmodel import Session, select
from src.database_models import SavedSearchSQL
from src.seed import STARTER_SEARCHES, app, seed
from typer.testing import CliRunner


def test_seed_creates_starter_searches(session: Session) -> None:
    seed()

    searches = session.exec(select(SavedSearchSQL)).all()
    assert {(search.name, search.query) for search in searches} == set(STARTER_SEARCHES)
    assert all(search.enabled for search in searches)


def test_seed_is_idempotent(session: Session) -> None:
    seed()
    seed()

    assert len(session.exec(select(SavedSearchSQL)).all()) == len(STARTER_SEARCHES)


def test_seed_cli(session: Session) -> None:
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert "Seeded" in result.output
