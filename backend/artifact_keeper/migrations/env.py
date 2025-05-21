from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

import sys
from pathlib import Path

# Add parent directory to path so we can import our app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import your models and database configuration
from artifact_keeper.app.db.session import Base
from artifact_keeper.app.core.settings import get_settings

version_table = "alembic_version_artifact_keeper"

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
SCHEMA_NAME = "artifact_keeper"  # Unique schema for this service  

# This is the key function that determines what's included in migrations
def include_object(object, name, type_, reflected, compare_to):
    # Skip other schemas' tables during migration
    if type_ == "table" and hasattr(object, "schema") and object.schema != SCHEMA_NAME:
        return False
    
    # Skip other schemas' indexes
    if type_ == "index" and hasattr(object, "table") and hasattr(object.table, "schema") and object.table.schema != SCHEMA_NAME:
        return False
        
    # For other object types, check schema when it's available
    if hasattr(object, "schema") and object.schema != SCHEMA_NAME:
        return False
        
    # Include everything else in our schema
    return True

# This is called before generating migrations to filter the metadata
def process_revision_directives(context, revision, directives):
    if not directives:
        return
        
    # Process each migration script
    for directive in directives:
        # Focus only on the upgrade operations
        if hasattr(directive, 'upgrade_ops'):
            # Remove operations affecting other schemas
            directive.upgrade_ops.ops = [
                op for op in directive.upgrade_ops.ops 
                if not (hasattr(op, 'schema') and op.schema != SCHEMA_NAME)
            ]
            
            # Handle operations that don't have a schema attribute directly
            # but might contain nested operations affecting other schemas
            for op in directive.upgrade_ops.ops:
                if hasattr(op, 'ops'):
                    op.ops = [
                        nested_op for nested_op in op.ops
                        if not (hasattr(nested_op, 'schema') and nested_op.schema != SCHEMA_NAME)
                    ]

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
        version_table_schema=SCHEMA_NAME,
        process_revision_directives=process_revision_directives
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
            version_table_schema=SCHEMA_NAME,
            process_revision_directives=process_revision_directives
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()