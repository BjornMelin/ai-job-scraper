"""Make saved searches the scrape source of truth.

Revision ID: 8f4b2c91a3d7
Revises: d555e0170c65
"""

import json
import math
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlmodel.sql.sqltypes import AutoString

revision: str = "8f4b2c91a3d7"
down_revision: str | Sequence[str] | None = "d555e0170c65"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MIGRATION_STATE_TABLE = "_alembic_8f4b2c91a3d7_state"


def _create_saved_search_table() -> None:
    op.create_table(
        "savedsearchsql",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", AutoString(), nullable=False),
        sa.Column("query", AutoString(), nullable=False),
        sa.Column("location", AutoString(), nullable=False),
        sa.Column("sites", sa.JSON(), nullable=False),
        sa.Column("remote_only", sa.Boolean(), nullable=False),
        sa.Column(
            "job_type",
            sa.Enum(
                "FULLTIME",
                "PARTTIME",
                "CONTRACT",
                "INTERNSHIP",
                "TEMPORARY",
                name="jobtype",
                native_enum=False,
            ),
            nullable=True,
        ),
        sa.Column("results_limit", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column(
            "last_run_status",
            sa.Enum(
                "NEVER",
                "RUNNING",
                "SUCCEEDED",
                "PARTIAL",
                "FAILED",
                "CANCELLED",
                name="savedsearchrunstatus",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("jobs_seen", sa.Integer(), nullable=False),
        sa.Column("jobs_new", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("last_error", AutoString(), nullable=True),
        sa.CheckConstraint(
            "results_limit BETWEEN 1 AND 1000",
            name="results_limit",
        ),
        sa.CheckConstraint("length(trim(name)) > 0", name="name_not_blank"),
        sa.CheckConstraint("length(trim(query)) > 0", name="query_not_blank"),
        sa.CheckConstraint(
            "length(trim(location)) > 0",
            name="location_not_blank",
        ),
        sa.CheckConstraint("jobs_seen >= 0", name="jobs_seen_nonnegative"),
        sa.CheckConstraint("jobs_new >= 0", name="jobs_new_nonnegative"),
        sa.CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="duration_ms_nonnegative",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("savedsearchsql") as batch_op:
        batch_op.create_index("ix_savedsearchsql_enabled", ["enabled"])
        batch_op.create_index(
            "ix_savedsearchsql_last_run_status",
            ["last_run_status"],
        )
        batch_op.create_index("ix_savedsearchsql_name", ["name"], unique=True)


def _unique_recovery_value(base: str, used_values: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used_values:
        candidate = f"{base} ({suffix})"
        suffix += 1
    used_values.add(candidate)
    return candidate


def _repair_legacy_rows(bind: sa.Connection) -> None:
    """Preserve malformed legacy rows with explicit, deterministic placeholders."""
    used_company_names = set(
        bind.execute(
            sa.text(
                "SELECT name FROM companysql "
                "WHERE name IS NOT NULL AND trim(name) != ''"
            )
        ).scalars()
    )
    blank_company_ids = list(
        bind.execute(
            sa.text("SELECT id FROM companysql WHERE name IS NULL OR trim(name) = ''")
        ).scalars()
    )
    for company_id in blank_company_ids:
        recovered_name = _unique_recovery_value(
            f"Recovered company {company_id}", used_company_names
        )
        bind.execute(
            sa.text("UPDATE companysql SET name = :name WHERE id = :company_id"),
            {"name": recovered_name, "company_id": company_id},
        )

    missing_company_count = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM jobsql "
            "WHERE company_id IS NULL OR NOT EXISTS ("
            "SELECT 1 FROM companysql WHERE companysql.id = jobsql.company_id"
            ")"
        )
    ).scalar_one()
    if missing_company_count:
        recovered_name = _unique_recovery_value(
            "Unknown company (recovered)", used_company_names
        )
        bind.execute(
            sa.text(
                "INSERT INTO companysql "
                "(name, url, active, last_scraped, scrape_count, success_rate) "
                "VALUES (:name, '', 1, NULL, 0, 1)"
            ),
            {"name": recovered_name},
        )
        recovered_company_id = bind.execute(
            sa.text("SELECT id FROM companysql WHERE name = :name"),
            {"name": recovered_name},
        ).scalar_one()
        bind.execute(
            sa.text(
                "UPDATE jobsql SET company_id = :company_id "
                "WHERE company_id IS NULL OR NOT EXISTS ("
                "SELECT 1 FROM companysql "
                "WHERE companysql.id = jobsql.company_id"
                ")"
            ),
            {"company_id": recovered_company_id},
        )

    blank_title_ids = list(
        bind.execute(
            sa.text("SELECT id FROM jobsql WHERE title IS NULL OR trim(title) = ''")
        ).scalars()
    )
    for job_id in blank_title_ids:
        bind.execute(
            sa.text("UPDATE jobsql SET title = :title WHERE id = :job_id"),
            {"title": f"Untitled role {job_id}", "job_id": job_id},
        )

    used_links = set(
        bind.execute(
            sa.text(
                "SELECT link FROM jobsql WHERE link IS NOT NULL AND trim(link) != ''"
            )
        ).scalars()
    )
    blank_link_ids = list(
        bind.execute(
            sa.text("SELECT id FROM jobsql WHERE link IS NULL OR trim(link) = ''")
        ).scalars()
    )
    for job_id in blank_link_ids:
        recovered_link = _unique_recovery_value(
            f"legacy://recovered-job/{job_id}", used_links
        )
        bind.execute(
            sa.text("UPDATE jobsql SET link = :link WHERE id = :job_id"),
            {"link": recovered_link, "job_id": job_id},
        )

    bind.execute(
        sa.text(
            "UPDATE jobsql SET salary = '[null, null]' "
            "WHERE salary IS NULL OR trim(CAST(salary AS TEXT)) IN ('', 'null')"
        )
    )


def _validate_legacy_cost_rows(bind: sa.Connection) -> None:
    """Reject legacy cost data that cannot satisfy the canonical schema."""
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
            "Cannot migrate invalid legacy cost entries; repair the source rows "
            "and retry (" + "; ".join(invalid_rows) + ")"
        )


def _normalize_legacy_cost_extra_data(bind: sa.Connection) -> None:
    """Convert legacy text metadata to JSON objects without discarding raw values."""
    rows = bind.execute(
        sa.text("SELECT id, extra_data FROM cost_entries ORDER BY id")
    ).mappings()
    for row in rows:
        raw_value = row["extra_data"]
        raw_text = "" if raw_value is None else str(raw_value)
        normalized: dict[str, object]
        if isinstance(raw_value, dict):
            normalized = raw_value
        elif not raw_text.strip():
            normalized = {}
        else:
            try:
                parsed = json.loads(raw_text)
            except (TypeError, ValueError):
                parsed = None
            normalized = (
                parsed if isinstance(parsed, dict) else {"legacy_raw": raw_text}
            )
        bind.execute(
            sa.text(
                "UPDATE cost_entries SET extra_data = :extra_data WHERE id = :entry_id"
            ),
            {
                "entry_id": row["id"],
                "extra_data": json.dumps(normalized, sort_keys=True),
            },
        )


def _record_cost_table_origin(bind: sa.Connection, *, preexisting: bool) -> None:
    """Persist cost-table provenance so downgrade never deletes adopted data."""
    op.create_table(
        _MIGRATION_STATE_TABLE,
        sa.Column("cost_entries_preexisting", sa.Boolean(), nullable=False),
    )
    bind.execute(
        sa.text(
            f"INSERT INTO {_MIGRATION_STATE_TABLE} "
            "(cost_entries_preexisting) VALUES (:preexisting)"
        ),
        {"preexisting": preexisting},
    )


def upgrade() -> None:
    """Replace company scrape state with saved-search run state."""
    bind = op.get_bind()
    legacy_stages = {"New", "Interested", "Applied", "Rejected"}
    persisted_stages = {
        row[0]
        for row in bind.execute(
            sa.text("SELECT DISTINCT application_status FROM jobsql")
        )
    }
    unknown_stages = persisted_stages - legacy_stages
    if unknown_stages:
        raise RuntimeError(
            "Cannot migrate unknown application statuses: "
            + ", ".join(sorted(unknown_stages))
        )

    inspector = sa.inspect(bind)
    cost_entries_preexisting = "cost_entries" in inspector.get_table_names()
    if cost_entries_preexisting:
        _validate_legacy_cost_rows(bind)
        _normalize_legacy_cost_extra_data(bind)
    _record_cost_table_origin(bind, preexisting=cost_entries_preexisting)

    if not cost_entries_preexisting:
        op.create_table(
            "cost_entries",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("service", AutoString(), nullable=False),
            sa.Column("operation", AutoString(), nullable=False),
            sa.Column("cost_usd", sa.Float(), nullable=False),
            sa.Column("extra_data", sa.JSON(), nullable=False),
            sa.CheckConstraint("cost_usd >= 0", name="cost_usd_nonnegative"),
            sa.CheckConstraint(
                "length(trim(service)) > 0",
                name="service_not_blank",
            ),
            sa.CheckConstraint(
                "length(trim(operation)) > 0",
                name="operation_not_blank",
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("cost_entries") as batch_op:
            batch_op.create_index("ix_cost_entries_service", ["service"])
            batch_op.create_index("ix_cost_entries_timestamp", ["timestamp"])
    else:
        cost_columns = {
            column["name"]: column for column in inspector.get_columns("cost_entries")
        }
        with op.batch_alter_table("cost_entries") as batch_op:
            batch_op.alter_column(
                "id",
                existing_type=sa.Integer(),
                nullable=False,
            )
            if not isinstance(cost_columns["extra_data"]["type"], sa.JSON):
                batch_op.alter_column(
                    "extra_data",
                    existing_type=sa.String(),
                    type_=sa.JSON(),
                    nullable=False,
                )
            batch_op.create_check_constraint("cost_usd_nonnegative", "cost_usd >= 0")
            batch_op.create_check_constraint(
                "service_not_blank", "length(trim(service)) > 0"
            )
            batch_op.create_check_constraint(
                "operation_not_blank", "length(trim(operation)) > 0"
            )

    _create_saved_search_table()
    _repair_legacy_rows(bind)

    op.execute(
        "INSERT INTO savedsearchsql "
        "(name, query, location, sites, remote_only, job_type, results_limit, "
        "enabled, last_run_at, last_run_status, jobs_seen, jobs_new, duration_ms, "
        "last_error) "
        "SELECT 'Migrated company: ' || name, name, 'United States', "
        "'[\"linkedin\"]', 0, NULL, 50, active, NULL, 'NEVER', 0, 0, NULL, NULL "
        "FROM companysql"
    )

    op.execute(
        "UPDATE jobsql SET application_status = 'Inbox' WHERE application_status = 'New'"
    )
    op.execute(
        "UPDATE jobsql SET application_status = 'Saved' WHERE application_status = 'Interested'"
    )
    op.execute(
        "UPDATE jobsql SET application_status = 'Closed' WHERE application_status = 'Rejected'"
    )

    with op.batch_alter_table("jobsql") as batch_op:
        batch_op.create_check_constraint("link_not_blank", "length(trim(link)) > 0")
        batch_op.create_check_constraint("title_not_blank", "length(trim(title)) > 0")
        batch_op.create_check_constraint(
            "application_stage",
            "application_status IN "
            "('Inbox', 'Saved', 'Applied', 'Interviews', 'Closed')",
        )
        batch_op.alter_column(
            "company_id",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.alter_column(
            "salary",
            existing_type=sa.JSON(),
            nullable=False,
        )
        batch_op.create_index("ix_jobsql_company_id", ["company_id"])
        batch_op.create_index("ix_jobsql_favorite", ["favorite"])
        batch_op.create_index("ix_jobsql_location", ["location"])
        batch_op.create_index("ix_jobsql_posted_date", ["posted_date"])
        batch_op.create_index("ix_jobsql_title", ["title"])

    with op.batch_alter_table("companysql") as batch_op:
        batch_op.drop_index("ix_companysql_active")
        batch_op.drop_index("ix_companysql_last_scraped")
        batch_op.drop_column("active")
        batch_op.drop_column("last_scraped")
        batch_op.drop_column("scrape_count")
        batch_op.drop_column("success_rate")
        batch_op.alter_column("url", existing_type=sa.String(), nullable=True)
        batch_op.create_check_constraint("name_not_blank", "length(trim(name)) > 0")


def downgrade() -> None:
    """Restore the legacy company scrape-state columns."""
    bind = op.get_bind()
    op.execute("UPDATE companysql SET url = '' WHERE url IS NULL")
    with op.batch_alter_table("companysql") as batch_op:
        batch_op.drop_constraint("name_not_blank", type_="check")
        batch_op.alter_column("url", existing_type=sa.String(), nullable=False)
        batch_op.add_column(
            sa.Column("success_rate", sa.Float(), nullable=False, server_default="1")
        )
        batch_op.add_column(
            sa.Column("scrape_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("last_scraped", sa.DateTime(), nullable=True))
        batch_op.add_column(
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true())
        )
        batch_op.create_index("ix_companysql_last_scraped", ["last_scraped"])
        batch_op.create_index("ix_companysql_active", ["active"])
    op.execute(
        "UPDATE companysql SET active = COALESCE(("
        "SELECT enabled FROM savedsearchsql "
        "WHERE savedsearchsql.name = 'Migrated company: ' || companysql.name"
        "), active)"
    )
    op.drop_table("savedsearchsql")
    with op.batch_alter_table("jobsql") as batch_op:
        batch_op.drop_index("ix_jobsql_title")
        batch_op.drop_index("ix_jobsql_posted_date")
        batch_op.drop_index("ix_jobsql_location")
        batch_op.drop_index("ix_jobsql_favorite")
        batch_op.drop_index("ix_jobsql_company_id")
        batch_op.alter_column(
            "company_id",
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch_op.alter_column(
            "salary",
            existing_type=sa.JSON(),
            nullable=True,
        )
        batch_op.drop_constraint("title_not_blank", type_="check")
        batch_op.drop_constraint("link_not_blank", type_="check")
        batch_op.drop_constraint("application_stage", type_="check")
    op.execute(
        "UPDATE jobsql SET application_status = 'New' WHERE application_status = 'Inbox'"
    )
    op.execute(
        "UPDATE jobsql SET application_status = 'Interested' WHERE application_status IN ('Saved', 'Interviews')"
    )
    op.execute(
        "UPDATE jobsql SET application_status = 'Rejected' WHERE application_status = 'Closed'"
    )
    inspector = sa.inspect(bind)
    state_table_exists = _MIGRATION_STATE_TABLE in inspector.get_table_names()
    cost_entries_preexisting = True
    if state_table_exists:
        cost_entries_preexisting = bool(
            bind.execute(
                sa.text(
                    f"SELECT cost_entries_preexisting FROM {_MIGRATION_STATE_TABLE}"
                )
            ).scalar_one()
        )

    if cost_entries_preexisting:
        with op.batch_alter_table("cost_entries") as batch_op:
            batch_op.drop_constraint("operation_not_blank", type_="check")
            batch_op.drop_constraint("service_not_blank", type_="check")
            batch_op.drop_constraint("cost_usd_nonnegative", type_="check")
            batch_op.alter_column(
                "extra_data",
                existing_type=sa.JSON(),
                type_=sa.String(),
                nullable=False,
            )
            batch_op.alter_column(
                "id",
                existing_type=sa.Integer(),
                nullable=True,
            )
    else:
        op.drop_table("cost_entries")

    if state_table_exists:
        op.drop_table(_MIGRATION_STATE_TABLE)
