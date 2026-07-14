"""Import legacy costs and repair canonical salary storage.

Revision ID: c91e7a4d2b6f
Revises: 8f4b2c91a3d7
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision: str = "c91e7a4d2b6f"
down_revision: str | Sequence[str] | None = "8f4b2c91a3d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IMPORT_TABLE = "_alembic_c91e7a4d2b6f_legacy_cost_imports"
_REQUIRED_COST_COLUMNS = {
    "id",
    "timestamp",
    "service",
    "operation",
    "cost_usd",
    "extra_data",
}


def _legacy_cost_path(bind: sa.Connection) -> Path | None:
    """Locate the sibling database used by the retired CostMonitor default."""
    if bind.dialect.name != "sqlite":
        return None
    main_database = next(
        (
            row.file
            for row in bind.exec_driver_sql("PRAGMA database_list").mappings()
            if row.name == "main"
        ),
        "",
    )
    if not main_database:
        return None
    target_path = Path(main_database).resolve()
    source_path = target_path.with_name("costs.db")
    if source_path == target_path:
        return None
    return source_path


def _normalize_extra_data(raw_value: object) -> dict[str, object]:
    """Keep JSON objects and retain every other nonblank legacy value."""
    if isinstance(raw_value, dict):
        return raw_value
    raw_text = "" if raw_value is None else str(raw_value)
    if not raw_text.strip():
        return {}
    try:
        parsed = json.loads(raw_text)
    except (TypeError, ValueError):
        parsed = None
    return parsed if isinstance(parsed, dict) else {"legacy_raw": raw_text}


def _read_legacy_costs(source_path: Path) -> list[dict[str, object]]:
    """Read and validate the complete legacy source before target mutation."""
    try:
        source = sqlite3.connect(f"{source_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise RuntimeError(
            f"Cannot open legacy cost database: {source_path}"
        ) from error

    source.row_factory = sqlite3.Row
    try:
        table_exists = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cost_entries'"
        ).fetchone()
        if table_exists is None:
            raise RuntimeError(
                f"Legacy cost database has no cost_entries table: {source_path}"
            )
        columns = {
            row["name"] for row in source.execute("PRAGMA table_info(cost_entries)")
        }
        missing_columns = sorted(_REQUIRED_COST_COLUMNS - columns)
        if missing_columns:
            raise RuntimeError(
                "Legacy cost database is missing required columns: "
                + ", ".join(missing_columns)
            )
        raw_rows = source.execute(
            "SELECT id, timestamp, service, operation, cost_usd, extra_data "
            "FROM cost_entries ORDER BY id"
        ).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(
            f"Cannot read legacy cost database: {source_path}"
        ) from error
    finally:
        source.close()

    rows: list[dict[str, object]] = []
    invalid: list[str] = []
    for raw in raw_rows:
        source_id = raw["id"]
        timestamp = raw["timestamp"]
        service = raw["service"]
        operation = raw["operation"]
        cost = raw["cost_usd"]
        reasons: list[str] = []
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            reasons.append("invalid id")
        try:
            if isinstance(timestamp, datetime):
                parsed_timestamp = timestamp
            else:
                parsed_timestamp = datetime.fromisoformat(str(timestamp))
        except (TypeError, ValueError):
            parsed_timestamp = None
            reasons.append("invalid timestamp")
        if not isinstance(service, str) or not service.strip():
            reasons.append("blank service")
        if not isinstance(operation, str) or not operation.strip():
            reasons.append("blank operation")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, int | float)
            or not math.isfinite(float(cost))
            or float(cost) < 0
        ):
            reasons.append("invalid cost_usd")
        if reasons:
            invalid.append(f"id={source_id!r} ({', '.join(reasons)})")
            continue
        rows.append(
            {
                "source_id": source_id,
                "timestamp": parsed_timestamp,
                "service": service,
                "operation": operation,
                "cost_usd": float(cost),
                "extra_data": _normalize_extra_data(raw["extra_data"]),
            }
        )
    if invalid:
        raise RuntimeError(
            "Cannot import invalid legacy cost entries; repair the source rows and "
            "retry (" + "; ".join(invalid) + ")"
        )
    return rows


def _ensure_import_table(bind: sa.Connection) -> None:
    if _IMPORT_TABLE in sa.inspect(bind).get_table_names():
        return
    op.create_table(
        _IMPORT_TABLE,
        sa.Column("source_path", sa.String(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("source_path", "source_id"),
        sa.UniqueConstraint("target_id"),
    )


def _insert_cost(
    bind: sa.Connection,
    row: dict[str, object],
    *,
    target_id: int | None,
) -> int:
    values = {
        "timestamp": row["timestamp"],
        "service": row["service"],
        "operation": row["operation"],
        "cost_usd": row["cost_usd"],
        "extra_data": row["extra_data"],
    }
    if target_id is None:
        statement = sa.text(
            "INSERT INTO cost_entries "
            "(timestamp, service, operation, cost_usd, extra_data) VALUES "
            "(:timestamp, :service, :operation, :cost_usd, :extra_data)"
        )
    else:
        values["target_id"] = target_id
        statement = sa.text(
            "INSERT INTO cost_entries "
            "(id, timestamp, service, operation, cost_usd, extra_data) VALUES "
            "(:target_id, :timestamp, :service, :operation, :cost_usd, :extra_data)"
        )
    statement = statement.bindparams(
        sa.bindparam("timestamp", type_=sa.DateTime()),
        sa.bindparam("extra_data", type_=sa.JSON()),
    )
    result = bind.execute(statement, values)
    return int(target_id if target_id is not None else result.lastrowid)


def _import_legacy_costs(bind: sa.Connection) -> None:
    source_path = _legacy_cost_path(bind)
    if source_path is None:
        return
    source_key = str(source_path)
    import_table_exists = _IMPORT_TABLE in sa.inspect(bind).get_table_names()
    mappings: dict[int, int] = {}
    if import_table_exists:
        mappings = {
            int(row.source_id): int(row.target_id)
            for row in bind.execute(
                sa.text(
                    f"SELECT source_id, target_id FROM {_IMPORT_TABLE} "
                    "WHERE source_path = :source_path"
                ),
                {"source_path": source_key},
            )
        }

    if mappings:
        missing_targets = {
            source_id: target_id
            for source_id, target_id in mappings.items()
            if bind.execute(
                sa.text("SELECT 1 FROM cost_entries WHERE id = :target_id"),
                {"target_id": target_id},
            ).first()
            is None
        }
        if not missing_targets:
            return
        if not source_path.is_file():
            raise RuntimeError(
                "Imported cost rows are missing and their legacy source is unavailable: "
                f"{source_path}"
            )
        source_rows = {
            int(row["source_id"]): row for row in _read_legacy_costs(source_path)
        }
        missing_sources = sorted(set(missing_targets) - set(source_rows))
        if missing_sources:
            raise RuntimeError(
                "Imported cost rows are missing from both databases: "
                + ", ".join(map(str, missing_sources))
            )
        for source_id, target_id in missing_targets.items():
            target_is_free = (
                bind.execute(
                    sa.text("SELECT 1 FROM cost_entries WHERE id = :target_id"),
                    {"target_id": target_id},
                ).first()
                is None
            )
            restored_id = _insert_cost(
                bind,
                source_rows[source_id],
                target_id=target_id if target_is_free else None,
            )
            if restored_id != target_id:
                bind.execute(
                    sa.text(
                        f"UPDATE {_IMPORT_TABLE} SET target_id = :target_id "
                        "WHERE source_path = :source_path AND source_id = :source_id"
                    ),
                    {
                        "source_path": source_key,
                        "source_id": source_id,
                        "target_id": restored_id,
                    },
                )
        return

    if not source_path.is_file():
        return
    source_rows = _read_legacy_costs(source_path)
    if not source_rows:
        return
    _ensure_import_table(bind)
    for row in source_rows:
        source_id = int(row["source_id"])
        target_is_free = (
            bind.execute(
                sa.text("SELECT 1 FROM cost_entries WHERE id = :target_id"),
                {"target_id": source_id},
            ).first()
            is None
        )
        target_id = _insert_cost(
            bind,
            row,
            target_id=source_id if target_is_free else None,
        )
        bind.execute(
            sa.text(
                f"INSERT INTO {_IMPORT_TABLE} (source_path, source_id, target_id) "
                "VALUES (:source_path, :source_id, :target_id)"
            ),
            {
                "source_path": source_key,
                "source_id": source_id,
                "target_id": target_id,
            },
        )


def _validate_target_costs(bind: sa.Connection) -> None:
    """Reject corrupt already-adopted rows before any follow-up mutation."""
    invalid_rows: list[str] = []
    rows = bind.execute(
        sa.text(
            "SELECT id, timestamp, service, operation, cost_usd "
            "FROM cost_entries ORDER BY id"
        )
    ).mappings()
    for row in rows:
        reasons: list[str] = []
        try:
            timestamp = row["timestamp"]
            if not isinstance(timestamp, datetime):
                datetime.fromisoformat(str(timestamp))
        except (TypeError, ValueError):
            reasons.append("invalid timestamp")
        service = row["service"]
        if not isinstance(service, str) or not service.strip():
            reasons.append("blank service")
        operation = row["operation"]
        if not isinstance(operation, str) or not operation.strip():
            reasons.append("blank operation")
        cost = row["cost_usd"]
        if (
            isinstance(cost, bool)
            or not isinstance(cost, int | float)
            or not math.isfinite(float(cost))
            or float(cost) < 0
        ):
            reasons.append("invalid cost_usd")
        if reasons:
            invalid_rows.append(f"id={row['id']!r} ({', '.join(reasons)})")
    if invalid_rows:
        raise RuntimeError(
            "Cannot repair invalid canonical cost entries; repair the target rows "
            "and retry (" + "; ".join(invalid_rows) + ")"
        )


def _normalize_target_cost_extra_data(bind: sa.Connection) -> None:
    """Make every already-adopted metadata value safe for JSON ORM reads."""
    values = [
        {
            "entry_id": row["id"],
            "extra_data": _normalize_extra_data(row["extra_data"]),
        }
        for row in bind.execute(
            sa.text("SELECT id, extra_data FROM cost_entries ORDER BY id")
        ).mappings()
    ]
    if not values:
        return
    statement = sa.text(
        "UPDATE cost_entries SET extra_data = :extra_data WHERE id = :entry_id"
    ).bindparams(sa.bindparam("extra_data", type_=sa.JSON()))
    bind.execute(statement, values)


def _is_canonical_salary(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            item is None or (isinstance(item, int) and not isinstance(item, bool))
            for item in value
        )
    )


def _normalize_salary_amount(value: object) -> int | None:
    """Return a lossless canonical amount or reject it."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError from None
    if not amount.is_finite() or amount != amount.to_integral_value():
        raise ValueError
    return int(amount)


def _normalize_salary_pair(value: object) -> list[int | None]:
    """Normalize one finite numeric pair without discarding information."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError
    return [_normalize_salary_amount(item) for item in value]


def _normalize_salaries(bind: sa.Connection) -> None:
    rows = bind.execute(sa.text("SELECT id, salary FROM jobsql ORDER BY id")).mappings()
    invalid_rows: list[str] = []
    updates: list[dict[str, object]] = []
    for row in rows:
        raw_value = row["salary"]
        if isinstance(raw_value, list):
            parsed = raw_value
        else:
            try:
                parsed = json.loads(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                parsed = None
        try:
            normalized = _normalize_salary_pair(parsed)
        except ValueError:
            invalid_rows.append(f"id={row['id']} salary={raw_value!r}")
            continue
        if not _is_canonical_salary(parsed):
            updates.append({"job_id": int(row["id"]), "salary": normalized})
    if invalid_rows:
        raise RuntimeError(
            "Cannot migrate noncanonical salaries without data loss; repair these "
            "rows and retry (" + "; ".join(invalid_rows) + ")"
        )
    if not updates:
        return
    statement = sa.text(
        "UPDATE jobsql SET salary = :salary WHERE id = :job_id"
    ).bindparams(sa.bindparam("salary", type_=sa.JSON()))
    bind.execute(statement, updates)


def upgrade() -> None:
    """Preserve retired cost history and enforce the canonical salary shape."""
    bind = op.get_bind()
    _normalize_salaries(bind)
    _validate_target_costs(bind)
    _normalize_target_cost_extra_data(bind)
    _import_legacy_costs(bind)
    salary_column = next(
        column
        for column in sa.inspect(bind).get_columns("jobsql")
        if column["name"] == "salary"
    )
    if salary_column["nullable"]:
        with op.batch_alter_table("jobsql") as batch_op:
            batch_op.alter_column(
                "salary",
                existing_type=sa.JSON(),
                nullable=False,
            )


def downgrade() -> None:
    """Keep imported history, provenance, and the repaired forward-only schema."""
