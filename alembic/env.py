"""Alembic environment configuration for SQLModel integration.

This module configures Alembic to work with SQLModel and the application's
database schema. It sets up dynamic database URL configuration, naming
conventions, and batch mode for SQLite compatibility.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# Import Settings and models for autogeneration
from src.config import Settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Set database URL from Settings
settings = Settings()
config.set_main_option("sqlalchemy.url", settings.db_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models to ensure they are registered with SQLModel.metadata
# ruff: noqa: E402

# Set target_metadata to SQLModel.metadata for autogenerate support
target_metadata = SQLModel.metadata

# Naming convention for cleaner constraint names
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Common configuration shared by online and offline modes
COMMON_CFG = {
    "target_metadata": target_metadata,
    "compare_type": True,
    "render_as_batch": True,  # Enable batch mode for SQLite
    "naming_convention": NAMING_CONVENTION,
}

# other values from the config, defined by the needs of env.py,
# can be acquired using config.get_main_option("option_name")


def _configure_offline() -> None:
    """Configure context for offline migrations."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **COMMON_CFG,
    )


def _configure_online(connection) -> None:
    """Configure context for online migrations."""
    context.configure(connection=connection, **COMMON_CFG)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    _configure_offline()
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Check if a connection was passed programmatically via config attributes
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        # Create engine from config if no connection was provided
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    if hasattr(connectable, "connect"):
        # We have an engine, create a connection
        with connectable.connect() as connection:
            _configure_online(connection)
            with context.begin_transaction():
                context.run_migrations()
    else:
        # We already have a connection
        _configure_online(connectable)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
