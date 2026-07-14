"""Migration startup behavior."""

import json
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.util.exc import CommandError
from sqlalchemy import create_engine, inspect, text
from sqlmodel import Session, select
from src.database_models import CostEntry, JobSQL
from src.db.migrations import run_migrations

_MIGRATION_STATE_TABLE = "_alembic_8f4b2c91a3d7_state"
_LEGACY_COST_IMPORT_TABLE = "_alembic_c91e7a4d2b6f_legacy_cost_imports"


def _create_legacy_cost_table(connection) -> None:
    connection.execute(
        text(
            "CREATE TABLE cost_entries ("
            "id INTEGER PRIMARY KEY, timestamp DATETIME NOT NULL, "
            "service VARCHAR NOT NULL, operation VARCHAR NOT NULL, "
            "cost_usd FLOAT NOT NULL, extra_data VARCHAR NOT NULL)"
        )
    )
    connection.execute(
        text("CREATE INDEX ix_cost_entries_timestamp ON cost_entries (timestamp)")
    )
    connection.execute(
        text("CREATE INDEX ix_cost_entries_service ON cost_entries (service)")
    )


def setup_function() -> None:
    run_migrations.cache_clear()


def teardown_function() -> None:
    run_migrations.cache_clear()


def test_migrations_run_once_per_process() -> None:
    with patch("src.db.migrations.command.upgrade") as upgrade:
        run_migrations()
        run_migrations()

    upgrade.assert_called_once()
    config, revision = upgrade.call_args.args
    assert config.config_file_name.endswith("alembic.ini")
    assert revision == "head"


def test_migration_failure_stops_startup() -> None:
    with (
        patch(
            "src.db.migrations.command.upgrade",
            side_effect=CommandError("invalid schema"),
        ),
        pytest.raises(CommandError, match="invalid schema"),
    ):
        run_migrations()


def test_sqlite_ddl_interruption_rolls_back_and_clean_retry_succeeds(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'interrupted.db'}")
    config = Config("alembic.ini")
    interrupted = False

    def interrupt_after_state_marker(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        nonlocal interrupted
        normalized = " ".join(statement.lower().split())
        if not interrupted and "create table savedsearchsql" in normalized:
            interrupted = True
            raise RuntimeError("injected migration interruption")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        sa.event.listen(
            engine,
            "before_cursor_execute",
            interrupt_after_state_marker,
        )
        with pytest.raises(RuntimeError, match="injected migration interruption"):
            command.upgrade(config, "head")
        connection.rollback()
        sa.event.remove(
            engine,
            "before_cursor_execute",
            interrupt_after_state_marker,
        )

        tables_after_failure = inspect(connection).get_table_names()
        revision_after_failure = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        command.upgrade(config, "head")
        command.check(config)
        final_revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()

    assert interrupted is True
    assert revision_after_failure == "d555e0170c65"
    assert _MIGRATION_STATE_TABLE not in tables_after_failure
    assert "cost_entries" not in tables_after_failure
    assert "savedsearchsql" not in tables_after_failure
    assert final_revision == "c91e7a4d2b6f"


def test_dashboard_migrates_before_starting_streamlit() -> None:
    from src.app_cli import dashboard

    with (
        patch("src.app_cli.run_migrations") as migrate,
        patch("src.app_cli.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        dashboard(port=8511, address="127.0.0.1")

    migrate.assert_called_once_with()
    command_args = run.call_args.args[0]
    assert command_args[-2:] == [
        "--server.port=8511",
        "--server.address=127.0.0.1",
    ]


def test_head_migration_rewrites_the_canonical_workflow(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        connection.execute(
            text(
                "INSERT INTO companysql "
                "(id, name, url, active, last_scraped, scrape_count, success_rate) "
                "VALUES (1, 'Acme', 'https://acme.test', 1, NULL, 0, 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) "
                "VALUES (1, :title, '', :link, 'Remote', NULL, '[null, null]', "
                "0, '', :hash, :stage, NULL, 0, NULL)"
            ),
            [
                {
                    "title": stage,
                    "link": f"https://acme.test/{stage}",
                    "hash": stage,
                    "stage": stage,
                }
                for stage in ("New", "Interested", "Applied", "Rejected")
            ],
        )
        connection.commit()

        command.upgrade(config, "head")

        stages = list(
            connection.execute(
                text("SELECT application_status FROM jobsql ORDER BY id")
            ).scalars()
        )
        constraints = inspect(connection).get_check_constraints("jobsql")
        migrated_source = connection.execute(
            text(
                "SELECT query, enabled, last_run_status FROM savedsearchsql "
                "WHERE name = 'Migrated company: Acme'"
            )
        ).one()

    assert stages == ["Inbox", "Saved", "Applied", "Closed"]
    assert any(
        "application_status IN ('Inbox', 'Saved', 'Applied', 'Interviews', 'Closed')"
        in constraint["sqltext"]
        for constraint in constraints
    )
    assert migrated_source == ("Acme", 1, "NEVER")


def test_head_migration_preserves_and_repairs_legacy_rows(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-recovery.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        connection.execute(
            text(
                "INSERT INTO companysql "
                "(id, name, url, active, last_scraped, scrape_count, success_rate) "
                "VALUES "
                "(1, 'Acme', 'https://acme.test', 0, '2026-01-01', 4, 0.75), "
                "(2, '', '', 1, NULL, 0, 1), "
                "(3, 'Orphaned facet', '', 1, NULL, 0, 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(id, company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) "
                "VALUES "
                "(10, 2, '', 'Preserve me', '', 'Remote', NULL, '[null, null]', "
                "1, 'Follow up', 'recovered', 'Interested', NULL, 0, NULL), "
                "(11, NULL, 'Unattributed role', '', 'https://example.test/11', "
                "'Remote', NULL, '[null, null]', 0, '', 'unattributed', 'New', "
                "NULL, 0, NULL), "
                "(12, 1, 'ML Engineer', '', 'https://acme.test/12', 'Denver', "
                "NULL, '[null, null]', 0, '', 'acme', 'Applied', NULL, 0, NULL)"
            )
        )
        connection.commit()

        command.upgrade(config, "head")

        jobs = connection.execute(
            text(
                "SELECT jobsql.id, jobsql.title, jobsql.link, companysql.name, "
                "jobsql.favorite, jobsql.notes, jobsql.application_status "
                "FROM jobsql JOIN companysql ON companysql.id = jobsql.company_id "
                "ORDER BY jobsql.id"
            )
        ).all()
        company_names = list(
            connection.execute(
                text("SELECT name FROM companysql ORDER BY id")
            ).scalars()
        )
        migrated_searches = connection.execute(
            text(
                "SELECT name, query, sites, enabled, last_run_status "
                "FROM savedsearchsql ORDER BY name"
            )
        ).all()

    assert len(jobs) == 3
    assert jobs[0] == (
        10,
        "Untitled role 10",
        "legacy://recovered-job/10",
        "Recovered company 2",
        1,
        "Follow up",
        "Saved",
    )
    assert jobs[1][3] == "Unknown company (recovered)"
    assert jobs[1][6] == "Inbox"
    assert company_names == [
        "Acme",
        "Recovered company 2",
        "Orphaned facet",
        "Unknown company (recovered)",
    ]
    assert [search[:2] for search in migrated_searches] == [
        ("Migrated company: Acme", "Acme"),
        ("Migrated company: Orphaned facet", "Orphaned facet"),
        ("Migrated company: Recovered company 2", "Recovered company 2"),
        (
            "Migrated company: Unknown company (recovered)",
            "Unknown company (recovered)",
        ),
    ]
    assert all(json.loads(search.sites) == ["linkedin"] for search in migrated_searches)
    assert [search.enabled for search in migrated_searches] == [0, 1, 1, 1]
    assert all(search.last_run_status == "NEVER" for search in migrated_searches)


def test_salary_migration_is_lossless_and_retryable(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'salary-recovery.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "8f4b2c91a3d7")
        connection.execute(
            text("INSERT INTO companysql (id, name, url) VALUES (1, 'Acme', NULL)")
        )
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (1, '2026-01-01', 'ai', 'legacy', 1.0, 'not-json')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(id, company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) VALUES "
                "(1, 1, 'Canonical', '', 'https://example.test/1', '', NULL, "
                "'[90000, 120000]', 0, '', 'one', 'Inbox', NULL, 0, NULL), "
                "(2, 1, 'Numeric strings', '', 'https://example.test/2', '', NULL, "
                "'[\"90000\", \"120000\"]', 0, '', 'two', 'Inbox', NULL, 0, NULL), "
                "(3, 1, 'Integral floats', '', 'https://example.test/3', '', NULL, "
                "'[90000.0, 120000.0]', 0, '', 'three', 'Inbox', NULL, 0, NULL), "
                "(4, 1, 'Fractional', '', 'https://example.test/4', '', NULL, "
                "'[90000.5, 120000]', 0, '', 'four', 'Inbox', NULL, 0, NULL), "
                "(5, 1, 'Malformed', '', 'https://example.test/5', '', NULL, "
                "'not-json', 0, '', 'five', 'Inbox', NULL, 0, NULL), "
                "(6, 1, 'Open range', '', 'https://example.test/6', '', NULL, "
                "'[null, 120000]', 0, '', 'six', 'Inbox', NULL, 0, NULL)"
            )
        )
        connection.commit()
        salaries_before = list(
            connection.execute(text("SELECT salary FROM jobsql ORDER BY id")).scalars()
        )

        with pytest.raises(RuntimeError) as exc_info:
            command.upgrade(config, "head")
        connection.rollback()

        assert "id=4" in str(exc_info.value)
        assert "90000.5" in str(exc_info.value)
        assert "id=5" in str(exc_info.value)
        assert "not-json" in str(exc_info.value)
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "8f4b2c91a3d7"
        )
        assert (
            list(
                connection.execute(
                    text("SELECT salary FROM jobsql ORDER BY id")
                ).scalars()
            )
            == salaries_before
        )
        assert (
            connection.execute(
                text("SELECT extra_data FROM cost_entries WHERE id = 1")
            ).scalar_one()
            == "not-json"
        )

        connection.execute(
            text(
                "UPDATE jobsql SET salary = CASE id "
                "WHEN 4 THEN '[90000, 120000]' "
                "WHEN 5 THEN '[null, null]' ELSE salary END WHERE id IN (4, 5)"
            )
        )
        connection.commit()
        command.upgrade(config, "head")

        salary_column = next(
            column
            for column in inspect(connection).get_columns("jobsql")
            if column["name"] == "salary"
        )
        raw_salaries = list(
            connection.execute(text("SELECT salary FROM jobsql ORDER BY id")).scalars()
        )
        normalized_cost_metadata = connection.execute(
            text("SELECT extra_data FROM cost_entries WHERE id = 1")
        ).scalar_one()

    with Session(engine) as session:
        jobs = list(session.exec(select(JobSQL).order_by(JobSQL.id)))
        salaries = [job.salary for job in jobs]

    assert salary_column["nullable"] is False
    assert raw_salaries == [
        "[90000, 120000]",
        "[90000, 120000]",
        "[90000, 120000]",
        "[90000, 120000]",
        "[null, null]",
        "[null, 120000]",
    ]
    assert salaries == [
        [90000, 120000],
        [90000, 120000],
        [90000, 120000],
        [90000, 120000],
        [None, None],
        [None, 120000],
    ]
    assert json.loads(normalized_cost_metadata) == {"legacy_raw": "not-json"}


def test_head_migration_rejects_unknown_stage_before_mutation(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'unknown-stage.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        connection.execute(
            text(
                "INSERT INTO companysql "
                "(id, name, url, active, last_scraped, scrape_count, success_rate) "
                "VALUES (1, 'Acme', 'https://acme.test', 1, NULL, 0, 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) "
                "VALUES (1, 'ML Engineer', '', 'https://acme.test/ml', 'Remote', "
                "NULL, '[null, null]', 0, '', 'unknown', 'Unexpected', NULL, 0, NULL)"
            )
        )
        connection.commit()

        with pytest.raises(
            RuntimeError,
            match="Cannot migrate unknown application statuses: Unexpected",
        ):
            command.upgrade(config, "head")
        connection.rollback()

        inspector = inspect(connection)
        tables = inspector.get_table_names()
        company_columns = {
            column["name"] for column in inspector.get_columns("companysql")
        }
        stages = list(
            connection.execute(text("SELECT application_status FROM jobsql")).scalars()
        )
        current_revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()

    assert "savedsearchsql" not in tables
    assert "active" in company_columns
    assert stages == ["Unexpected"]
    assert current_revision == "d555e0170c65"


def test_head_migration_creates_cost_table_from_deployed_schema(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'historical.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")

        command.upgrade(config, "head")

        inspector = inspect(connection)
        cost_columns = {
            column["name"]: column for column in inspector.get_columns("cost_entries")
        }
        cost_constraints = {
            constraint["name"]
            for constraint in inspector.get_check_constraints("cost_entries")
        }
        cost_indexes = {
            index["name"] for index in inspector.get_indexes("cost_entries")
        }

    assert isinstance(cost_columns["extra_data"]["type"], sa.JSON)
    assert cost_constraints == {
        "cost_usd_nonnegative",
        "operation_not_blank",
        "service_not_blank",
    }
    assert cost_indexes == {"ix_cost_entries_service", "ix_cost_entries_timestamp"}


def test_followup_imports_sibling_legacy_costs_without_mutating_source(
    tmp_path,
) -> None:
    target_engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    source_engine = create_engine(f"sqlite:///{tmp_path / 'costs.db'}")
    config = Config("alembic.ini")
    with source_engine.begin() as source:
        _create_legacy_cost_table(source)
        source.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) VALUES "
                "(7, '2026-01-01 01:02:03', 'ai', 'valid', 1.25, "
                '\'{"model": "x"}\'), '
                "(8, '2026-01-02 01:02:03', 'proxy', 'blank', 2.5, ''), "
                "(9, '2026-01-03 01:02:03', 'scraping', 'malformed', 3.75, "
                '\'{"company": "ACME "quoted""}\'), '
                "(10, '2026-01-04 01:02:03', 'ai', 'non-object', 4.0, '[1, 2]')"
            )
        )
        source_before = source.execute(
            text("SELECT id, extra_data FROM cost_entries ORDER BY id")
        ).all()
    source_engine.dispose()

    with target_engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        command.upgrade(config, "head")
        command.check(config)
        mappings = connection.execute(
            text(
                f"SELECT source_id, target_id FROM {_LEGACY_COST_IMPORT_TABLE} "
                "ORDER BY source_id"
            )
        ).all()

    with Session(target_engine) as session:
        imported = list(session.exec(select(CostEntry).order_by(CostEntry.id)))
    source_engine = create_engine(f"sqlite:///{tmp_path / 'costs.db'}")
    with source_engine.connect() as source:
        source_after = source.execute(
            text("SELECT id, extra_data FROM cost_entries ORDER BY id")
        ).all()

    assert mappings == [(source_id, source_id) for source_id in range(7, 11)]
    assert [entry.id for entry in imported] == [7, 8, 9, 10]
    assert {entry.id: entry.extra_data for entry in imported} == {
        7: {"model": "x"},
        8: {},
        9: {"legacy_raw": '{"company": "ACME "quoted""}'},
        10: {"legacy_raw": "[1, 2]"},
    }
    assert source_after == source_before


def test_followup_repairs_stamped_schema_and_imports_collisions_idempotently(
    tmp_path,
) -> None:
    target_engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    source_engine = create_engine(f"sqlite:///{tmp_path / 'costs.db'}")
    config = Config("alembic.ini")
    with source_engine.begin() as source:
        _create_legacy_cost_table(source)
        source.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (1, '2026-01-01 01:02:03', 'ai', 'legacy', 1.25, '{}')"
            )
        )
    source_engine.dispose()

    with target_engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "8f4b2c91a3d7")
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (1, '2026-02-01 01:02:03', 'ai', 'canonical', 9.5, "
                "'{\"target\": true}')"
            )
        )
        connection.execute(
            text("INSERT INTO companysql (id, name, url) VALUES (1, 'Acme', NULL)")
        )
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(id, company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) VALUES "
                "(1, 1, 'Numeric salary', '', 'https://example.test/1', '', NULL, "
                "'[\"90000\", \"120000\"]', 0, '', 'numeric', 'Inbox', NULL, 0, NULL)"
            )
        )
        connection.commit()
        operations = Operations(MigrationContext.configure(connection))
        with operations.batch_alter_table("jobsql") as batch_op:
            batch_op.alter_column(
                "salary",
                existing_type=sa.JSON(),
                nullable=True,
            )
        connection.commit()

        command.upgrade(config, "head")
        command.check(config)
        mapping = connection.execute(
            text(f"SELECT source_id, target_id FROM {_LEGACY_COST_IMPORT_TABLE}")
        ).one()
        first_count = connection.execute(
            text("SELECT COUNT(*) FROM cost_entries")
        ).scalar_one()
        salary_raw = connection.execute(
            text("SELECT salary FROM jobsql WHERE id = 1")
        ).scalar_one()
        salary_nullable = next(
            column["nullable"]
            for column in inspect(connection).get_columns("jobsql")
            if column["name"] == "salary"
        )

        command.downgrade(config, "8f4b2c91a3d7")
        command.upgrade(config, "head")
        second_count = connection.execute(
            text("SELECT COUNT(*) FROM cost_entries")
        ).scalar_one()

        command.downgrade(config, "8f4b2c91a3d7")
        connection.execute(
            text("DELETE FROM cost_entries WHERE id = :target_id"),
            {"target_id": mapping.target_id},
        )
        connection.commit()
        command.upgrade(config, "head")
        restored = connection.execute(
            text(
                "SELECT id, service, operation, cost_usd, extra_data "
                "FROM cost_entries WHERE id = :target_id"
            ),
            {"target_id": mapping.target_id},
        ).one()
        final_count = connection.execute(
            text("SELECT COUNT(*) FROM cost_entries")
        ).scalar_one()

    assert mapping == (1, 2)
    assert first_count == second_count == final_count == 2
    assert salary_raw == "[90000, 120000]"
    assert salary_nullable is False
    assert restored[:4] == (2, "ai", "legacy", 1.25)
    assert json.loads(restored.extra_data) == {}


def test_followup_normalizes_stamped_target_cost_metadata_for_orm_reads(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "8f4b2c91a3d7")
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) VALUES "
                "(1, '2026-01-01', 'ai', 'object', 1.0, '{\"model\": \"x\"}'), "
                "(2, '2026-01-01', 'ai', 'blank', 1.0, ''), "
                "(3, '2026-01-01', 'ai', 'malformed', 1.0, 'not-json'), "
                "(4, '2026-01-01', 'ai', 'non-object', 1.0, '[1, 2]'), "
                "(5, '2026-01-01', 'ai', 'json-null', 1.0, 'null')"
            )
        )
        connection.commit()

        command.upgrade(config, "head")

    with Session(engine) as session:
        metadata = {
            entry.id: entry.extra_data
            for entry in session.exec(select(CostEntry).order_by(CostEntry.id))
        }

    assert metadata == {
        1: {"model": "x"},
        2: {},
        3: {"legacy_raw": "not-json"},
        4: {"legacy_raw": "[1, 2]"},
        5: {"legacy_raw": "null"},
    }


def test_followup_rejects_invalid_external_costs_without_partial_import_and_retries(
    tmp_path,
) -> None:
    target_engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    source_engine = create_engine(f"sqlite:///{tmp_path / 'costs.db'}")
    config = Config("alembic.ini")
    with source_engine.begin() as source:
        _create_legacy_cost_table(source)
        source.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (1, 'not-a-date', '', 'legacy', 'not-a-number', '{}')"
            )
        )
    source_engine.dispose()

    with target_engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "8f4b2c91a3d7")
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (99, '2026-02-01', 'ai', 'canonical', 2.5, 'not-json')"
            )
        )
        connection.commit()
        with pytest.raises(
            RuntimeError,
            match="invalid timestamp, blank service, invalid cost_usd",
        ):
            command.upgrade(config, "head")
        connection.rollback()
        tables_after_failure = inspect(connection).get_table_names()
        revision_after_failure = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        cost_count_after_failure = connection.execute(
            text("SELECT COUNT(*) FROM cost_entries")
        ).scalar_one()
        metadata_after_failure = connection.execute(
            text("SELECT extra_data FROM cost_entries WHERE id = 99")
        ).scalar_one()

    source_engine = create_engine(f"sqlite:///{tmp_path / 'costs.db'}")
    with source_engine.begin() as source:
        source.execute(
            text(
                "UPDATE cost_entries SET timestamp = '2026-01-01', "
                "service = 'ai', cost_usd = 1.25"
            )
        )
    source_engine.dispose()
    with target_engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        imported_count = connection.execute(
            text("SELECT COUNT(*) FROM cost_entries")
        ).scalar_one()
        normalized_metadata = connection.execute(
            text("SELECT extra_data FROM cost_entries WHERE id = 99")
        ).scalar_one()

    assert revision_after_failure == "8f4b2c91a3d7"
    assert _LEGACY_COST_IMPORT_TABLE not in tables_after_failure
    assert cost_count_after_failure == 1
    assert metadata_after_failure == "not-json"
    assert imported_count == 2
    assert json.loads(normalized_metadata) == {"legacy_raw": "not-json"}


def test_followup_rejects_invalid_timestamp_in_stamped_target_and_retries(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "8f4b2c91a3d7")
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (41, 'not-a-date', 'ai', 'legacy', 1.25, 'not-json')"
            )
        )
        connection.commit()

        with pytest.raises(
            RuntimeError,
            match=r"canonical cost entries.*id=41 \(invalid timestamp\)",
        ):
            command.upgrade(config, "head")
        connection.rollback()
        revision_after_failure = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        tables_after_failure = inspect(connection).get_table_names()
        metadata_after_failure = connection.execute(
            text("SELECT extra_data FROM cost_entries WHERE id = 41")
        ).scalar_one()

        connection.execute(
            text("UPDATE cost_entries SET timestamp = '2026-01-01' WHERE id = 41")
        )
        connection.commit()
        command.upgrade(config, "head")

    assert revision_after_failure == "8f4b2c91a3d7"
    assert _LEGACY_COST_IMPORT_TABLE not in tables_after_failure
    assert metadata_after_failure == "not-json"


def test_head_migration_rejects_invalid_legacy_costs_before_ddl_and_can_retry(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'invalid-costs.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        _create_legacy_cost_table(connection)
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (:id, '2026-01-01', :service, 'legacy', :cost, '{}')"
            ),
            [
                {"id": 1, "service": "", "cost": -1.25},
                {"id": 2, "service": "legacy", "cost": "not-a-number"},
            ],
        )
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (3, 'not-a-date', 'legacy', 'legacy', 1.25, '{}')"
            )
        )
        connection.commit()

        with pytest.raises(
            RuntimeError,
            match=(
                r"id=1 \(blank service, invalid cost_usd\); "
                r"id=2 \(invalid cost_usd\); id=3 \(invalid timestamp\)"
            ),
        ):
            command.upgrade(config, "head")
        connection.rollback()

        tables_after_failure = inspect(connection).get_table_names()
        revision_after_failure = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        connection.execute(
            text(
                "UPDATE cost_entries SET timestamp = '2026-01-01', "
                "service = 'legacy', cost_usd = 1.25"
            )
        )
        connection.commit()
        command.upgrade(config, "head")
        tables_after_retry = inspect(connection).get_table_names()

    assert revision_after_failure == "d555e0170c65"
    assert _MIGRATION_STATE_TABLE not in tables_after_failure
    assert not any(name.startswith("_alembic_tmp") for name in tables_after_failure)
    assert "savedsearchsql" not in tables_after_failure
    assert _MIGRATION_STATE_TABLE in tables_after_retry
    assert not any(name.startswith("_alembic_tmp") for name in tables_after_retry)


def test_head_migration_restores_an_adopted_cost_table_on_downgrade(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'adopted-costs.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        _create_legacy_cost_table(connection)
        connection.execute(
            text(
                "INSERT INTO cost_entries "
                "(id, timestamp, service, operation, cost_usd, extra_data) "
                "VALUES (:id, '2026-01-01', 'ai', 'legacy', 1.25, :extra_data)"
            ),
            [
                {"id": 7, "extra_data": '{"model": "x"}'},
                {"id": 8, "extra_data": ""},
                {"id": 9, "extra_data": '{"company": "ACME "quoted""}'},
                {"id": 10, "extra_data": "[1, 2]"},
            ],
        )
        connection.commit()

        command.upgrade(config, "head")
        assert _MIGRATION_STATE_TABLE in inspect(connection).get_table_names()
        command.check(config)
        with Session(engine) as session:
            upgraded_extra_data = {
                entry.id: entry.extra_data
                for entry in session.exec(select(CostEntry).order_by(CostEntry.id))
            }
        command.downgrade(config, "d555e0170c65")

        inspector = inspect(connection)
        tables_after_downgrade = inspector.get_table_names()
        cost_column = next(
            column
            for column in inspector.get_columns("cost_entries")
            if column["name"] == "extra_data"
        )
        cost_constraints = inspector.get_check_constraints("cost_entries")
        cost_rows = connection.execute(
            text(
                "SELECT id, service, operation, cost_usd, extra_data "
                "FROM cost_entries ORDER BY id"
            )
        ).all()

    expected_extra_data = {
        7: {"model": "x"},
        8: {},
        9: {"legacy_raw": '{"company": "ACME "quoted""}'},
        10: {"legacy_raw": "[1, 2]"},
    }
    assert upgraded_extra_data == expected_extra_data
    assert _MIGRATION_STATE_TABLE not in tables_after_downgrade
    assert isinstance(cost_column["type"], sa.String)
    assert cost_constraints == []
    assert [row[:4] for row in cost_rows] == [
        (entry_id, "ai", "legacy", 1.25) for entry_id in range(7, 11)
    ]
    assert {row.id: json.loads(row.extra_data) for row in cost_rows} == (
        expected_extra_data
    )


def test_head_migration_downgrades_to_the_declared_previous_schema(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'downgrade.db'}")
    config = Config("alembic.ini")

    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "d555e0170c65")
        connection.execute(
            text(
                "INSERT INTO companysql "
                "(id, name, url, active, last_scraped, scrape_count, success_rate) "
                "VALUES (1, 'Acme', 'https://acme.test', 0, NULL, 0, 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) "
                "VALUES (1, :title, '', :link, 'Remote', NULL, '[null, null]', "
                "0, '', :hash, :stage, NULL, 0, NULL)"
            ),
            [
                {
                    "title": stage,
                    "link": f"https://acme.test/{stage}",
                    "hash": stage,
                    "stage": stage,
                }
                for stage in ("New", "Interested", "Applied", "Rejected")
            ],
        )
        connection.commit()
        command.upgrade(config, "head")
        connection.execute(
            text(
                "INSERT INTO jobsql "
                "(company_id, title, description, link, location, posted_date, "
                "salary, favorite, notes, content_hash, application_status, "
                "application_date, archived, last_seen) "
                "VALUES (1, 'Interview', '', 'https://acme.test/interview', "
                "'Remote', NULL, '[null, null]', 0, '', 'interview', "
                "'Interviews', NULL, 0, NULL)"
            )
        )
        connection.commit()

        command.downgrade(config, "d555e0170c65")

        inspector = inspect(connection)
        tables = inspector.get_table_names()
        company_columns = {
            column["name"] for column in inspector.get_columns("companysql")
        }
        stages = list(
            connection.execute(
                text("SELECT application_status FROM jobsql ORDER BY id")
            ).scalars()
        )
        company_active = connection.execute(
            text("SELECT active FROM companysql WHERE id = 1")
        ).scalar_one()
        salary_column = next(
            column
            for column in inspector.get_columns("jobsql")
            if column["name"] == "salary"
        )

    assert "savedsearchsql" not in tables
    assert {"active", "last_scraped", "scrape_count", "success_rate"} <= (
        company_columns
    )
    assert "cost_entries" not in tables
    assert _MIGRATION_STATE_TABLE not in tables
    assert stages == ["New", "Interested", "Applied", "Rejected", "Interested"]
    assert company_active == 0
    assert salary_column["nullable"] is True
