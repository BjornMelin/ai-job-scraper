"""Saved-search persistence and run-health ownership."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import or_, update
from sqlalchemy.engine import CursorResult
from sqlmodel import Session, col, select

from src.database import db_session
from src.database_models import SavedSearchSQL
from src.models.job_models import SavedSearchRunStatus
from src.schemas import (
    SavedSearch,
    SavedSearchCreate,
    SavedSearchRunHealth,
    SavedSearchUpdate,
)

RUN_LEASE = timedelta(minutes=30)


class SavedSearchService:
    """CRUD and latest-run health for user-owned scrape definitions."""

    @staticmethod
    def _to_dto(search: SavedSearchSQL) -> SavedSearch:
        return SavedSearch.model_validate(search)

    def list(self, *, enabled_only: bool = False) -> list[SavedSearch]:
        with db_session() as session:
            statement = select(SavedSearchSQL)
            if enabled_only:
                statement = statement.where(col(SavedSearchSQL.enabled).is_(True))
            searches = session.exec(statement.order_by(SavedSearchSQL.name)).all()
            return [self._to_dto(search) for search in searches]

    def get(self, search_id: int) -> SavedSearch | None:
        with db_session() as session:
            search = session.get(SavedSearchSQL, search_id)
            return self._to_dto(search) if search else None

    def create(self, data: SavedSearchCreate) -> SavedSearch:
        with db_session() as session:
            search = SavedSearchSQL.model_validate(data.model_dump())
            session.add(search)
            session.flush()
            return self._to_dto(search)

    def update(self, search_id: int, data: SavedSearchUpdate) -> SavedSearch | None:
        with db_session() as session:
            search = session.get(SavedSearchSQL, search_id)
            if search is None:
                return None
            for field, value in data.model_dump(exclude_unset=True).items():
                setattr(search, field, value)
            session.flush()
            return self._to_dto(search)

    def delete(self, search_id: int) -> bool:
        """Delete only the search definition; persisted jobs remain untouched."""
        with db_session() as session:
            search = session.get(SavedSearchSQL, search_id)
            if search is None:
                return False
            session.delete(search)
            return True

    def claim_run(
        self,
        search_id: int,
        started_at: datetime,
    ) -> SavedSearch | None:
        """Atomically claim one run or reclaim a stale run lease."""
        stale_before = started_at - RUN_LEASE
        with db_session() as session:
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(SavedSearchSQL)
                    .where(
                        col(SavedSearchSQL.id) == search_id,
                        or_(
                            col(SavedSearchSQL.last_run_status)
                            != SavedSearchRunStatus.RUNNING,
                            col(SavedSearchSQL.last_run_at).is_(None),
                            col(SavedSearchSQL.last_run_at) < stale_before,
                        ),
                    )
                    .values(
                        last_run_at=started_at,
                        last_run_status=SavedSearchRunStatus.RUNNING,
                        jobs_seen=0,
                        jobs_new=0,
                        duration_ms=None,
                        last_error=None,
                    )
                ),
            )
            if result.rowcount != 1:
                return None
            search = session.get(SavedSearchSQL, search_id)
            return self._to_dto(search) if search else None

    def record_run(
        self,
        search_id: int,
        health: SavedSearchRunHealth,
        *,
        expected_started_at: datetime | None = None,
        session: Session | None = None,
    ) -> SavedSearch | None:
        """Record terminal health, optionally inside a caller-owned transaction."""
        if session is not None:
            return self._record_run(
                session,
                search_id,
                health,
                expected_started_at=expected_started_at,
            )
        with db_session() as session:
            return self._record_run(
                session,
                search_id,
                health,
                expected_started_at=expected_started_at,
            )

    def _record_run(
        self,
        session: Session,
        search_id: int,
        health: SavedSearchRunHealth,
        *,
        expected_started_at: datetime | None,
    ) -> SavedSearch | None:
        statement = update(SavedSearchSQL).where(col(SavedSearchSQL.id) == search_id)
        if expected_started_at is not None:
            statement = statement.where(
                col(SavedSearchSQL.last_run_status) == SavedSearchRunStatus.RUNNING,
                col(SavedSearchSQL.last_run_at) == expected_started_at,
            )
        result = cast(
            CursorResult[Any],
            session.execute(
                statement.values(
                    **health.model_dump(exclude={"last_run_at"}),
                    last_run_at=datetime.now(UTC),
                )
            ),
        )
        if result.rowcount != 1:
            return None
        search = session.get(SavedSearchSQL, search_id)
        return self._to_dto(search) if search else None


saved_search_service = SavedSearchService()
