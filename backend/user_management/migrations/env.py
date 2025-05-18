### userManagementMicroservice/migrations/env.py ###

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

import sys
from pathlib import Path

# Add parent directory to path so we can import our app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import your models and database configuration
from user_management.app.db.session import Base
from user_management.app.core.settings import get_settings
# Import models to populate Base.metadata
from user_management.app.db.models import User, UserToken, Shift, Activity, WorkHours, Task
version_table = "alembic_version_user_management"

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Get database URI from settings
settings = get_settings()

print(settings.DATABASE_URI)
config.set_main_option('sqlalchemy.url', settings.DATABASE_URI.replace('%', '%%'))

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata to Base.metadata for autogeneration support
target_metadata = Base.metadata

# Add under the imports
SCHEMA_NAME = "user_mgmt"  # Unique schema for this service  
TARGET_TABLES = {'users', 'shifts'}  # Your actual table names

def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table":
        return name in TARGET_TABLES
    if hasattr(object, "schema"):
        return object.schema == SCHEMA_NAME
    return True

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=version_table,
        include_object=include_object,
        include_schemas=True,
        version_table_schema="user_mgmt"
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            version_table=version_table,
            include_object=include_object,
            include_schemas=True,
            version_table_schema="user_mgmt" 
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
