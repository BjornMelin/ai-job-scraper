# ADR-012: Database Schema Migration Strategy

## Title

Adoption of Alembic for Database Schema Migrations

## Version/Date

1.0 / August 7, 2025

## Status

Proposed

## Context

The current method for managing the database schema relies on `SQLModel.metadata.create_all()`. This approach is sufficient for initial application setup but is fundamentally flawed for long-term maintenance and upgrades. Any change to the `SQLModel` definitions in `src/models.py` would require developers to manually delete the `jobs.db` file to apply the changes. For end-users, this would result in the **complete loss of all their data**, including favorited jobs, notes, and application history, which is unacceptable for a production application.

## Related Requirements

* `DB-SCHEMA-01`, `DB-SCHEMA-02`, `DB-SCHEMA-03`: These define the schema which will inevitably need to evolve.
* `NFR-MAINT-01`: A robust migration strategy is essential for long-term maintainability and safe deployments.

## Decision

We will adopt **Alembic** as the official tool for managing all database schema migrations. Alembic is the de-facto standard for migration management in the SQLAlchemy ecosystem and integrates seamlessly with SQLModel.

1. **Tool Adoption:** Alembic will be added as a development dependency.
2. **Migration Workflow:**
    * For any schema change, a developer will first modify the model classes in `src/models.py`.
    * They will then generate a new migration script using `alembic revision --autogenerate -m "description_of_change"`.
    * The developer will review the auto-generated script for correctness and commit it to the repository.
3. **Deployment Process:** As part of the application startup sequence or deployment script, the command `alembic upgrade head` will be executed. This command will apply any pending migration scripts to the database, bringing its schema up to date with the latest model definitions non-destructively.
4. **Initial Setup vs. Upgrades:**
    * For a **fresh installation**, the database will still be created using `SQLModel.metadata.create_all()`.
    * For any **existing installation**, schema updates will *only* be handled by running Alembic migrations.

## Design

The implementation requires setting up an `alembic/` directory and configuring it to work with our SQLModel definitions.

**1. Alembic Configuration (`alembic.ini`):**

This file configures the location of migration scripts and the database connection URL.

```ini
# alembic.ini
[alembic]
script_location = alembic
sqlalchemy.url = sqlite:///jobs.db # This will be read from our app's settings

...
```

**2. Environment Configuration (`alembic/env.py`):**

This is the key file that connects Alembic to our application's models.

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy.pool import NullPool
from alembic import context

# Import your application's settings and models
from src.config import Settings
from src.models import SQLModel # This imports all your models

# Get database URL from our application's settings
app_settings = Settings()

config = context.config
config.set_main_option("sqlalchemy.url", app_settings.db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLModel metadata as the target for autogeneration
target_metadata = SQLModel.metadata

def run_migrations_offline() -> None:
    # ... (standard offline configuration)
    ...

def run_migrations_online() -> None:
    # ... (standard online configuration)
    ...

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**3. Example CLI Commands:**

```bash
# Generate a new migration script after changing models.py
alembic revision --autogenerate -m "Add job_type column to JobSQL"

# Apply the latest migrations to the database
alembic upgrade head
```

## Consequences

* **Positive:**
  * **Non-Destructive Updates:** Users can update the application without losing their existing data.
  * **Version Control for Schema:** The database schema's history is now stored as code in version control, providing a clear audit trail.
  * **Reliable Deployments:** The deployment process becomes more robust and repeatable, as schema changes are automated and predictable.
  * **Decoupling:** Separates the concern of schema management from the application's runtime logic.
* **Negative:**
  * **Added Complexity:** Introduces a new tool (Alembic) and an additional step in the development workflow.
  * **Autogenerate Limitations:** Alembic's autogenerate feature is powerful but not perfect; it may not detect all types of changes (e.g., changes to server defaults or check constraints) and generated scripts must always be reviewed by the developer.
* **Mitigations:**
  * Alembic is a mature and well-documented tool, and its workflow is a standard practice for professional Python development.
  * Team guidelines will enforce the mandatory review of all auto-generated migration scripts before they are committed.
