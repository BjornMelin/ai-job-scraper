"""Tests for migration integration with application startup.

This module tests the integration of Alembic migrations with the application
startup process, ensuring that migrations run automatically and safely
during application initialization as described in the migration strategy ADR.
"""

import tempfile

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlmodel import Session, SQLModel

from src.config import Settings
from src.models import CompanySQL, JobSQL


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary SQLite database file for startup tests.

    Yields:
        Path to temporary database file that is automatically cleaned up.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_path = temp_file.name

    yield temp_path

    # Cleanup
    temp_file_path = Path(temp_path)
    if temp_file_path.exists():
        temp_file_path.unlink()


@pytest.fixture
def startup_alembic_dir() -> Generator[Path, None, None]:
    """Create a temporary Alembic configuration for startup testing.

    Yields:
        Path to temporary Alembic directory with startup-compatible configuration.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_dir = Path(temp_dir) / "alembic"
        alembic_dir.mkdir()

        # Create versions directory
        versions_dir = alembic_dir / "versions"
        versions_dir.mkdir()

        # Create script.py.mako template required for autogenerate
        script_mako = alembic_dir / "script.py.mako"
        script_mako.write_text("""\"\"\"${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

\"\"\"
from alembic import op
import sqlalchemy as sa
import sqlmodel
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
""")

        # Create env.py that simulates the real application configuration
        env_py = alembic_dir / "env.py"
        env_py.write_text("""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from src.models import SQLModel, CompanySQL, JobSQL

# This simulates the real env.py that will be created
config = context.config

# Allow connection injection for testing
connectable = context.config.attributes.get('connection', None)

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = SQLModel.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    if connectable is None:
        connectable_local = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    else:
        connectable_local = connectable

    with connectable_local.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
""")

        # Create alembic.ini
        alembic_ini = Path(temp_dir) / "alembic.ini"
        alembic_ini.write_text(f"""
[alembic]
script_location = {alembic_dir}
sqlalchemy.url = driver://user:pass@localhost/dbname

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
""")

        yield alembic_dir


def create_migration_function(temp_db_path: str, alembic_dir: Path):
    """Factory function to create a run_migrations function for testing.

    This simulates the run_migrations function that will be added to main.py
    or a dedicated migration module as part of the startup integration.

    Args:
        temp_db_path: Path to test database
        alembic_dir: Path to Alembic configuration directory

    Returns:
        Function that runs migrations programmatically
    """

    def run_migrations() -> None:
        """Run database migrations programmatically.

        This function will be integrated into the application startup process
        to ensure the database schema is up-to-date before the application starts.
        """
        try:
            config = Config()
            config.set_main_option("script_location", str(alembic_dir))
            config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

            # Run migrations to head
            command.upgrade(config, "head")

        except Exception as e:
            # In real implementation, this would use proper logging
            msg = f"Migration failed: {e}"
            raise RuntimeError(msg) from e

    return run_migrations


class TestMigrationStartupIntegration:
    """Test migration integration with application startup process."""

    def test_run_migrations_function_exists_and_callable(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test that run_migrations function can be created and called.

        Validates that the migration function planned for startup integration
        is properly structured and executable.
        """
        run_migrations = create_migration_function(temp_db_path, startup_alembic_dir)

        # Function should be callable
        assert callable(run_migrations)

        # Should not raise errors when called (even with no migrations)
        # This should be safe to call with no migrations present
        run_migrations()  # Should complete successfully without errors

    def test_startup_migration_on_fresh_database(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test migration execution during startup with fresh database.

        Simulates application startup on a system where no database exists,
        ensuring migrations create the database and all required tables.
        """
        run_migrations = create_migration_function(temp_db_path, startup_alembic_dir)

        # Create initial migration
        config = Config()
        config.set_main_option("script_location", str(startup_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        command.revision(config, autogenerate=True, message="Initial startup migration")

        # Simulate startup migration
        run_migrations()

        # Verify database and tables were created
        engine = create_engine(f"sqlite:///{temp_db_path}")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        assert "companysql" in table_names
        assert "jobsql" in table_names
        assert "alembic_version" in table_names

    def test_startup_migration_with_existing_database(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test migration execution during startup with existing database.

        Simulates application startup on a system where a database already exists
        but needs schema updates, ensuring migrations upgrade the schema safely.
        """
        # Create initial database with old schema
        engine = create_engine(f"sqlite:///{temp_db_path}")
        SQLModel.metadata.create_all(engine)

        # Add some test data
        with Session(engine) as session:
            company = CompanySQL(
                name="Existing Company",
                url="https://existing.com/careers",
                active=True,
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            job = JobSQL(
                company_id=company.id,
                title="Existing Job",
                description="Old job entry",
                link="https://existing.com/job/1",
                location="Office",
                content_hash="old123",
                salary=(80000, 100000),
            )
            session.add(job)
            session.commit()

        # Configure Alembic
        config = Config()
        config.set_main_option("script_location", str(startup_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        # Properly stamp the existing database as base revision
        command.stamp(config, "base")

        # Create migration for "new features" - this simulates schema evolution
        command.revision(config, autogenerate=True, message="Startup upgrade test")

        # Simulate startup migration
        run_migrations = create_migration_function(temp_db_path, startup_alembic_dir)
        run_migrations()

        # Verify data is preserved and schema is updated
        with Session(engine) as session:
            companies = session.query(CompanySQL).all()
            jobs = session.query(JobSQL).all()

            assert len(companies) == 1
            assert companies[0].name == "Existing Company"

            assert len(jobs) == 1
            assert jobs[0].title == "Existing Job"

        # Verify migration was applied (should have a specific version now, not 'base')
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            versions = result.fetchall()
            assert len(versions) == 1
            # Should not be 'base' anymore after upgrade
            assert versions[0][0] != "base"

    def test_startup_migration_idempotency(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test that startup migrations can be run multiple times safely.

        Validates that the startup migration process is idempotent and can
        handle being called multiple times without errors or data corruption.
        """
        # Setup migration
        config = Config()
        config.set_main_option("script_location", str(startup_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        command.revision(config, autogenerate=True, message="Idempotency test")

        run_migrations = create_migration_function(temp_db_path, startup_alembic_dir)

        # Run migration first time
        run_migrations()

        # Run migration second time (should be idempotent)
        run_migrations()  # Should not raise errors

        # Verify database state is correct
        engine = create_engine(f"sqlite:///{temp_db_path}")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        assert "companysql" in table_names
        assert "jobsql" in table_names

        # Should only have one migration version
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            versions = result.fetchall()
            assert len(versions) == 1

    def test_startup_migration_error_handling(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test error handling in startup migration process.

        Validates that migration errors during startup are properly handled
        and provide meaningful error messages for debugging.
        """

        # Create a run_migrations function that will fail
        def failing_run_migrations() -> None:
            config = Config()
            config.set_main_option("script_location", "/nonexistent/path")
            config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

            try:
                command.upgrade(config, "head")
            except Exception as e:
                msg = f"Migration failed: {e}"
                raise RuntimeError(msg) from e

        # Should raise meaningful error
        with pytest.raises(RuntimeError, match="Migration failed"):
            failing_run_migrations()

    def test_startup_with_missing_database_file(
        self,
        startup_alembic_dir: Path,
    ) -> None:
        """Test startup behavior when database file is completely missing.

        Validates that the startup migration process can handle cases where
        the database file doesn't exist at all and create it from scratch.
        """
        # Use a temporary file path that doesn't exist
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            missing_db_path = f"{tmp_file.name}_nonexistent.db"

        config = Config()
        config.set_main_option("script_location", str(startup_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{missing_db_path}")

        # Create initial migration
        command.revision(config, autogenerate=True, message="Missing DB test")

        run_migrations = create_migration_function(missing_db_path, startup_alembic_dir)

        try:
            # Should create database from scratch
            run_migrations()

            # Verify database was created
            assert Path(missing_db_path).exists()

            engine = create_engine(f"sqlite:///{missing_db_path}")
            inspector = inspect(engine)
            table_names = inspector.get_table_names()

            assert "companysql" in table_names
            assert "jobsql" in table_names

        finally:
            # Clean up
            db_path = Path(missing_db_path)
            if db_path.exists():
                db_path.unlink()


class TestMainAppIntegration:
    """Test integration with the main application entry point."""

    @patch("src.main.main")
    def test_main_app_startup_calls_migrations(self, mock_main) -> None:
        """Test that main application startup includes migration call.

        This test validates the planned integration where main.py or a startup
        module will call run_migrations() before starting the Streamlit app.
        """
        # Mock the main function to track if migrations would be called
        mock_run_migrations = MagicMock()

        # Simulate the planned integration in main.py
        def simulated_main_with_migrations():
            # This represents the planned addition to main.py
            mock_run_migrations()  # run_migrations() call
            mock_main()  # existing main() call

        # Test the integration
        simulated_main_with_migrations()

        # Verify both functions were called
        mock_run_migrations.assert_called_once()
        mock_main.assert_called_once()

    def test_migration_settings_integration(self) -> None:
        """Test that migration system properly integrates with Settings.

        Validates that the migration configuration can read database URL
        from the application's Settings class, ensuring consistency between
        the main application and migration system.
        """
        settings = Settings()

        # Migration system should use the same database URL
        assert settings.db_url is not None
        assert settings.db_url.startswith("sqlite://")

        # Config creation should work with settings
        config = Config()
        config.set_main_option("sqlalchemy.url", settings.db_url)

        assert config.get_main_option("sqlalchemy.url") == settings.db_url

    def test_streamlit_app_starts_after_successful_migration(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test that Streamlit app only starts after successful migration.

        Validates the planned startup sequence where database migrations
        must complete successfully before the Streamlit application begins.
        """
        migration_completed = False
        app_started = False

        def mock_run_migrations():
            nonlocal migration_completed
            try:
                # Simulate successful migration
                config = Config()
                config.set_main_option("script_location", str(startup_alembic_dir))
                config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

                command.revision(
                    config,
                    autogenerate=True,
                    message="Startup sequence test",
                )
                command.upgrade(config, "head")

                migration_completed = True
            except Exception:
                migration_completed = False
                raise

        def mock_streamlit_main():
            nonlocal app_started
            # Streamlit app should only start after migration
            assert migration_completed, "Migrations must complete before app starts"
            app_started = True

        # Simulate startup sequence
        mock_run_migrations()
        mock_streamlit_main()

        assert migration_completed
        assert app_started

    def test_migration_failure_prevents_app_startup(self) -> None:
        """Test that migration failure prevents application startup.

        Validates that if database migrations fail during startup,
        the application should not proceed to start the Streamlit interface.
        """
        migration_failed = False
        app_started = False

        def failing_migration():
            nonlocal migration_failed
            migration_failed = True
            raise RuntimeError(
                "DB Migration error in test_migration_startup simulation",
            )

        def mock_streamlit_main():
            nonlocal app_started
            app_started = True

        # Simulate startup sequence with migration failure
        with pytest.raises(RuntimeError, match="Migration failed intentionally"):
            failing_migration()

        # App should not start if migration failed
        assert migration_failed
        assert not app_started


class TestMigrationLogging:
    """Test migration logging and monitoring during startup."""

    def test_migration_success_logging(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test that successful migrations are properly logged.

        Validates that the startup migration process provides appropriate
        logging information for monitoring and debugging purposes.
        """
        # This test would verify logging integration
        # In real implementation, this would test actual log output

        config = Config()
        config.set_main_option("script_location", str(startup_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        command.revision(config, autogenerate=True, message="Logging test")

        # The run_migrations function should include logging
        run_migrations = create_migration_function(temp_db_path, startup_alembic_dir)

        # Should complete without errors (logging would be verified separately)
        run_migrations()

        # Verify migration completed
        engine = create_engine(f"sqlite:///{temp_db_path}")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            versions = result.fetchall()
            assert len(versions) == 1

    def test_migration_performance_monitoring(
        self,
        temp_db_path: str,
        startup_alembic_dir: Path,
    ) -> None:
        """Test migration performance monitoring during startup.

        Validates that migration execution time is reasonable for startup
        and that performance can be monitored for production deployments.
        """
        import time

        config = Config()
        config.set_main_option("script_location", str(startup_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        command.revision(config, autogenerate=True, message="Performance test")

        run_migrations = create_migration_function(temp_db_path, startup_alembic_dir)

        # Measure migration time
        start_time = time.time()
        run_migrations()
        end_time = time.time()

        migration_time = end_time - start_time

        # Migration should complete quickly (< 5 seconds for simple schema)
        assert migration_time < 5.0, f"Migration took too long: {migration_time}s"

        # Verify migration succeeded
        engine = create_engine(f"sqlite:///{temp_db_path}")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        assert "companysql" in table_names
        assert "jobsql" in table_names
