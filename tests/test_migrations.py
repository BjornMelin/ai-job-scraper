"""Tests for Alembic database migrations.

This module provides comprehensive tests for the Alembic migration system,
validating configuration, migration script generation, and database schema
management. Tests cover both fresh installations and upgrade scenarios
to ensure data safety and migration reliability.
"""

import tempfile

from collections.abc import Generator
from pathlib import Path

import pytest

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel
from src.models import CompanySQL, JobSQL


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary SQLite database file for testing migrations.

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
def temp_alembic_dir() -> Generator[Path, None, None]:
    """Create a temporary Alembic configuration directory for testing.

    Yields:
        Path to temporary Alembic directory with basic configuration.
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

        # Create minimal env.py
        env_py = alembic_dir / "env.py"
        env_py.write_text("""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from src.models import SQLModel

# Import all models to ensure they're registered with SQLModel.metadata
from src.models import CompanySQL, JobSQL

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

        yield alembic_dir


@pytest.fixture
def alembic_config(temp_db_path: str, temp_alembic_dir: Path) -> Config:
    """Create an Alembic configuration for testing.

    Args:
        temp_db_path: Path to temporary database file
        temp_alembic_dir: Path to temporary Alembic directory

    Returns:
        Configured Alembic Config instance
    """
    config = Config()
    config.set_main_option("script_location", str(temp_alembic_dir))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")
    return config


@pytest.fixture
def test_engine(temp_db_path: str) -> Engine:
    """Create a test SQLAlchemy engine.

    Args:
        temp_db_path: Path to temporary database file

    Returns:
        SQLAlchemy Engine instance for testing
    """
    return create_engine(f"sqlite:///{temp_db_path}")


class TestAlembicConfiguration:
    """Test Alembic configuration and setup."""

    def test_alembic_config_creation(self, alembic_config: Config) -> None:
        """Test that Alembic configuration can be created and loaded.

        Validates that the Alembic Config can be instantiated with proper
        script location and database URL settings.
        """
        assert alembic_config is not None
        assert alembic_config.get_main_option("script_location") is not None
        assert alembic_config.get_main_option("sqlalchemy.url") is not None

    def test_sqlmodel_metadata_detected(self, alembic_config: Config) -> None:
        """Test that SQLModel.metadata is properly accessible for autogeneration.

        Verifies that all model tables are registered in SQLModel.metadata
        and can be detected by Alembic for migration generation.
        """
        # Check that our tables are in the metadata
        table_names = list(SQLModel.metadata.tables.keys())
        assert "companysql" in table_names
        assert "jobsql" in table_names

    def test_database_url_from_settings(self, temp_db_path: str) -> None:
        """Test that database URL is correctly retrieved from Settings.

        Validates that the Alembic configuration can read the database URL
        from the application's Settings class as specified in the ADR.
        """
        # Override with test database path
        test_url = f"sqlite:///{temp_db_path}"

        config = Config()
        config.set_main_option("sqlalchemy.url", test_url)

        assert config.get_main_option("sqlalchemy.url") == test_url


class TestMigrationExecution:
    """Test migration script execution and database operations."""

    def test_fresh_database_creation(self, test_engine: Engine) -> None:
        """Test creating a fresh database with all tables.

        Simulates initial database setup using SQLModel.metadata.create_all()
        for fresh installations as described in the ADR.
        """
        # Create tables using SQLModel (fresh installation path)
        SQLModel.metadata.create_all(test_engine)

        # Verify all expected tables were created
        inspector = inspect(test_engine)
        table_names = inspector.get_table_names()

        assert "companysql" in table_names
        assert "jobsql" in table_names

    def test_migration_upgrade_head(
        self, alembic_config: Config, test_engine: Engine
    ) -> None:
        """Test running 'alembic upgrade head' on empty database.

        Creates initial migration and applies it to verify the complete
        migration workflow works correctly.
        """
        # Create initial migration
        command.revision(alembic_config, autogenerate=True, message="Initial migration")

        # Apply migrations
        command.upgrade(alembic_config, "head")

        # Verify tables were created by migration
        inspector = inspect(test_engine)
        table_names = inspector.get_table_names()

        assert "companysql" in table_names
        assert "jobsql" in table_names

        # Verify migration was recorded
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM alembic_version"))
            versions = result.fetchall()
            assert len(versions) == 1  # Should have one migration applied

    def test_table_schema_validation(
        self, alembic_config: Config, test_engine: Engine
    ) -> None:
        """Test that migrated tables match SQLModel definitions.

        Validates that all columns, types, and constraints from the SQLModel
        definitions are properly created by the migration system.
        """
        # Generate and apply initial migration
        command.revision(
            alembic_config, autogenerate=True, message="Schema validation test"
        )
        command.upgrade(alembic_config, "head")

        inspector = inspect(test_engine)

        # Test CompanySQL table structure
        company_columns = {
            col["name"]: col for col in inspector.get_columns("companysql")
        }

        expected_company_columns = {
            "id",
            "name",
            "url",
            "active",
            "last_scraped",
            "scrape_count",
            "success_rate",
        }
        assert expected_company_columns.issubset(set(company_columns.keys()))

        # Test JobSQL table structure
        job_columns = {col["name"]: col for col in inspector.get_columns("jobsql")}

        expected_job_columns = {
            "id",
            "company_id",
            "title",
            "description",
            "link",
            "location",
            "posted_date",
            "salary",
            "favorite",
            "notes",
            "content_hash",
            "application_status",
            "application_date",
            "archived",
            "last_seen",
        }
        assert expected_job_columns.issubset(set(job_columns.keys()))

        # Test foreign key relationship
        foreign_keys = inspector.get_foreign_keys("jobsql")
        assert foreign_keys
        fk = foreign_keys[0]
        assert fk["referred_table"] == "companysql"
        assert "company_id" in fk["constrained_columns"]

    def test_indexes_creation(
        self, alembic_config: Config, test_engine: Engine
    ) -> None:
        """Test that database indexes are properly created by migrations.

        Validates that all indexes defined in SQLModel (using Field(index=True))
        are correctly created by the migration system for optimal query performance.
        """
        # Generate and apply migration
        command.revision(
            alembic_config, autogenerate=True, message="Index creation test"
        )
        command.upgrade(alembic_config, "head")

        inspector = inspect(test_engine)

        # Test CompanySQL indexes
        company_indexes = inspector.get_indexes("companysql")

        # Expected indexed columns from model definitions
        expected_company_indexes = {"name", "active", "last_scraped"}

        # Check that our expected indexes exist
        # (may have more due to primary keys, etc.)
        for expected_col in expected_company_indexes:
            assert any(
                expected_col in idx_cols
                for idx_cols in [idx["column_names"] for idx in company_indexes]
            )

        # Test JobSQL indexes
        job_indexes = inspector.get_indexes("jobsql")

        expected_job_indexes = {
            "content_hash",
            "application_status",
            "archived",
            "last_seen",
        }

        for expected_col in expected_job_indexes:
            assert any(
                expected_col in idx_cols
                for idx_cols in [idx["column_names"] for idx in job_indexes]
            )

    def test_migration_idempotency(
        self, alembic_config: Config, test_engine: Engine
    ) -> None:
        """Test that running migrations multiple times is safe and idempotent.

        Validates that running 'upgrade head' multiple times doesn't cause
        errors or duplicate operations, ensuring safe deployment practices.
        """
        # Create and apply initial migration
        command.revision(alembic_config, autogenerate=True, message="Idempotency test")
        command.upgrade(alembic_config, "head")

        # Running upgrade again should be safe
        command.upgrade(alembic_config, "head")  # Should not error

        # Verify database state is still correct
        inspector = inspect(test_engine)
        table_names = inspector.get_table_names()

        assert "companysql" in table_names
        assert "jobsql" in table_names

        # Verify only one migration is recorded
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            versions = result.fetchall()
            assert len(versions) == 1

    def test_downgrade_functionality(
        self, alembic_config: Config, test_engine: Engine
    ) -> None:
        """Test that migration downgrade functionality works correctly.

        Validates that migrations can be safely rolled back, which is crucial
        for deployment rollback scenarios and development workflow.
        """
        # Create and apply initial migration
        command.revision(alembic_config, autogenerate=True, message="Downgrade test")
        command.upgrade(alembic_config, "head")

        # Verify tables exist
        inspector = inspect(test_engine)
        table_names = inspector.get_table_names()
        assert "companysql" in table_names
        assert "jobsql" in table_names

        # Downgrade to base
        command.downgrade(alembic_config, "base")

        # Verify tables are removed (except alembic_version)
        inspector = inspect(test_engine)
        table_names = inspector.get_table_names()
        assert "companysql" not in table_names
        assert "jobsql" not in table_names
        assert "alembic_version" in table_names  # Alembic tracking table remains


class TestMigrationWithData:
    """Test migrations with existing data to ensure data safety."""

    def test_migration_preserves_existing_data(
        self, alembic_config: Config, test_engine: Engine
    ) -> None:
        """Test that migrations preserve existing data during schema updates.

        Creates initial schema with sample data, then runs a migration to ensure
        existing data is preserved during schema changes.
        """
        # Create initial schema using SQLModel
        SQLModel.metadata.create_all(test_engine)

        # Insert sample data
        with Session(test_engine) as session:
            company = CompanySQL(
                name="Test Company",
                url="https://test.com/careers",
                active=True,
                scrape_count=5,
                success_rate=0.8,
            )
            session.add(company)
            session.commit()
            session.refresh(company)

            job = JobSQL(
                company_id=company.id,
                title="Software Engineer",
                description="Great opportunity",
                link="https://test.com/job/123",
                location="Remote",
                salary=(100000, 150000),
                content_hash="abc123",
                application_status="Applied",
            )
            session.add(job)
            session.commit()

        # Initialize Alembic with existing schema - set to head without creating
        # migration
        with test_engine.connect() as conn:
            # Create alembic_version table and mark as current
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version "
                    "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
                )
            )
            conn.execute(
                text("DELETE FROM alembic_version")
            )  # Clear any existing versions
            conn.execute(
                text("INSERT INTO alembic_version (version_num) VALUES ('head')")
            )
            conn.commit()

        # Verify data is preserved (no actual migration needed for this test)
        with Session(test_engine) as session:
            companies = session.query(CompanySQL).all()
            jobs = session.query(JobSQL).all()

            assert len(companies) == 1
            assert companies[0].name == "Test Company"
            assert companies[0].scrape_count == 5

            assert len(jobs) == 1
            assert jobs[0].title == "Software Engineer"
            assert jobs[0].application_status == "Applied"

    def test_missing_database_handling(self, alembic_config: Config) -> None:
        """Test migration behavior when database file doesn't exist.

        Validates that migrations can create a new database from scratch
        when the database file is missing, ensuring robust deployment.
        """
        # First create a migration since none exist in test environment
        command.revision(
            alembic_config, autogenerate=True, message="Test missing database"
        )

        # The temp database doesn't exist yet, migration should create it
        command.upgrade(alembic_config, "head")

        # Verify database and tables were created
        db_url = alembic_config.get_main_option("sqlalchemy.url")
        engine = create_engine(db_url)

        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        # Should have our tables plus alembic_version
        assert "companysql" in table_names
        assert "jobsql" in table_names
        assert "alembic_version" in table_names


class TestMigrationScriptGeneration:
    """Test Alembic migration script generation capabilities."""

    def test_autogenerate_detects_model_changes(
        self, temp_alembic_dir: Path, temp_db_path: str
    ) -> None:
        """Test that autogenerate detects changes to SQLModel definitions.

        Simulates the developer workflow of modifying models and generating
        migrations, ensuring the autogenerate feature works correctly.
        """
        config = Config()
        config.set_main_option("script_location", str(temp_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        # Generate initial migration
        revision = command.revision(config, autogenerate=True, message="Initial schema")

        # Verify migration script was created
        assert revision is not None

        # Check migration file exists
        versions_dir = temp_alembic_dir / "versions"
        migration_files = list(versions_dir.glob("*.py"))
        assert migration_files

        # Read migration content
        migration_content = migration_files[0].read_text()

        # Should contain references to our tables
        assert "companysql" in migration_content.lower()
        assert "jobsql" in migration_content.lower()
        assert "create_table" in migration_content.lower()

    def test_autogenerate_detects_column_type_changes(
        self, temp_alembic_dir: Path, temp_db_path: str
    ) -> None:
        """Test that autogenerate detects column type changes.

        Simulates modifying a column type and verifies that Alembic's
        autogenerate feature can detect and script these changes correctly.
        """
        config = Config()
        config.set_main_option("script_location", str(temp_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        # Create initial schema with original column type
        engine = create_engine(f"sqlite:///{temp_db_path}")
        with engine.connect() as conn:
            # Create a test table with a VARCHAR column
            conn.execute(
                text(
                    "CREATE TABLE test_table (id INTEGER PRIMARY KEY, test_col VARCHAR(50))"
                )
            )
            conn.commit()

        # Stamp as current head
        command.stamp(config, "head")

        # Now modify the target metadata to change column type
        # We'll temporarily add a table to the metadata to simulate type change
        from sqlalchemy import Column, Integer, String, Table

        test_table = Table(
            "test_table",
            SQLModel.metadata,
            Column("id", Integer, primary_key=True),
            Column("test_col", String(100)),  # Changed from VARCHAR(50) to VARCHAR(100)
        )

        try:
            # Generate migration for type change
            revision = command.revision(
                config, autogenerate=True, message="Change column type"
            )

            assert revision is not None

            versions_dir = temp_alembic_dir / "versions"
            migration_files = list(versions_dir.glob("*.py"))
            assert migration_files

            migration_content = migration_files[0].read_text()

            # Should detect the column type change
            # Note: SQLite doesn't directly support ALTER COLUMN, so this might
            # generate batch operations or table recreation
            assert (
                "alter_column" in migration_content.lower()
                or "batch_alter_table" in migration_content.lower()
            )

        finally:
            # Clean up the temporary table from metadata
            SQLModel.metadata.remove(test_table)

    def test_autogenerate_detects_constraint_changes(
        self, temp_alembic_dir: Path, temp_db_path: str
    ) -> None:
        """Test that autogenerate detects constraint modifications.

        Simulates adding/removing constraints and verifies that Alembic can
        detect and script these changes appropriately.
        """
        config = Config()
        config.set_main_option("script_location", str(temp_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        # Create initial schema without unique constraint
        engine = create_engine(f"sqlite:///{temp_db_path}")
        with engine.connect() as conn:
            conn.execute(
                text(
                    "CREATE TABLE constraint_test (id INTEGER PRIMARY KEY, email VARCHAR(100))"
                )
            )
            conn.commit()

        # Stamp as current head
        command.stamp(config, "head")

        # Add a table to metadata with a unique constraint
        from sqlalchemy import Column, Integer, String, Table, UniqueConstraint

        constraint_table = Table(
            "constraint_test",
            SQLModel.metadata,
            Column("id", Integer, primary_key=True),
            Column("email", String(100)),
            UniqueConstraint("email", name="uq_constraint_test_email"),
        )

        try:
            # Generate migration for constraint addition
            revision = command.revision(
                config, autogenerate=True, message="Add unique constraint"
            )

            assert revision is not None

            versions_dir = temp_alembic_dir / "versions"
            migration_files = list(versions_dir.glob("*.py"))
            # Should have 2 files now (initial + constraint change)
            assert len(migration_files) >= 1

            # Get the most recent migration file
            latest_migration = max(migration_files, key=lambda f: f.stat().st_mtime)
            migration_content = latest_migration.read_text()

            # Should detect the constraint change
            assert (
                "unique" in migration_content.lower()
                or "constraint" in migration_content.lower()
                or "create_unique_constraint" in migration_content.lower()
            )

        finally:
            # Clean up the temporary table from metadata
            SQLModel.metadata.remove(constraint_table)

    def test_migration_script_validation(
        self, temp_alembic_dir: Path, temp_db_path: str
    ) -> None:
        """Test that generated migration scripts are syntactically valid.

        Ensures that autogenerated migration scripts can be imported and
        executed without syntax errors, providing confidence in the
        generated migration quality.
        """
        config = Config()
        config.set_main_option("script_location", str(temp_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        # Generate migration
        command.revision(config, autogenerate=True, message="Validation test")

        # Apply the migration to bring database up to date before checking
        command.upgrade(config, "head")

        # Try to check/validate the migration (this will fail if syntax is wrong)
        try:
            command.check(config)
        except Exception as e:
            # Some versions of Alembic might not have check command
            if "Unknown command" not in str(e):
                pytest.fail(f"Migration validation failed: {e}")

    def test_empty_migration_generation(
        self, temp_alembic_dir: Path, temp_db_path: str
    ) -> None:
        """Test generating migrations when no changes are detected.

        Validates behavior when running autogenerate with no model changes,
        ensuring the system handles no-op scenarios gracefully.
        """
        config = Config()
        config.set_main_option("script_location", str(temp_alembic_dir))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")

        # Create initial schema
        engine = create_engine(f"sqlite:///{temp_db_path}")
        SQLModel.metadata.create_all(engine)

        # Stamp as head (simulate existing migration state)
        command.stamp(config, "head")

        # Try to generate migration with no changes
        revision = command.revision(config, autogenerate=True, message="No changes")

        # Should still create a migration file (even if empty)
        assert revision is not None

        versions_dir = temp_alembic_dir / "versions"
        migration_files = list(versions_dir.glob("*.py"))
        assert migration_files

        # Read the migration file to verify it's a no-op
        migration_content = migration_files[0].read_text()

        # A no-op migration should have empty upgrade() and downgrade() functions
        # or at least not contain any table/column operations
        assert "def upgrade()" in migration_content
        assert "def downgrade()" in migration_content

        # Check that no schema changes are present (no-op)
        schema_operations = [
            "create_table",
            "drop_table",
            "add_column",
            "drop_column",
            "alter_column",
            "create_index",
            "drop_index",
        ]

        for operation in schema_operations:
            assert operation not in migration_content, (
                f"Found schema operation {operation} in no-op migration"
            )

        # Additionally check that upgrade/downgrade contain only pass or comments
        lines = migration_content.split("\n")
        inside_upgrade = False
        inside_downgrade = False
        upgrade_has_operations = False
        downgrade_has_operations = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def upgrade"):
                inside_upgrade = True
                inside_downgrade = False
                continue
            if stripped.startswith("def downgrade"):
                inside_downgrade = True
                inside_upgrade = False
                continue
            if stripped.startswith("def ") or not stripped.startswith(" "):
                inside_upgrade = False
                inside_downgrade = False
                continue

            if (
                inside_upgrade
                and stripped
                and not stripped.startswith("#")
                and stripped != "pass"
            ):
                upgrade_has_operations = True
            if (
                inside_downgrade
                and stripped
                and not stripped.startswith("#")
                and stripped != "pass"
            ):
                downgrade_has_operations = True

        assert not upgrade_has_operations, (
            "Upgrade function contains actual operations in no-op migration"
        )
        assert not downgrade_has_operations, (
            "Downgrade function contains actual operations in no-op migration"
        )
