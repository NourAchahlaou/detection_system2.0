from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

import sys
from pathlib import Path

# Add parent directory to path so we can import our app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import your models and database configuration
from annotation.app.db.session import Base
from annotation.app.core.settings import get_settings

from annotation.app.db.models.annotation import Annotation

version_table = "alembic_version_annotation"

config = context.config

settings = get_settings()

print(settings.DATABASE_URI)
config.set_main_option('sqlalchemy.url', settings.DATABASE_URI.replace('%', '%%'))


if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

SCHEMA_NAME = "annotation" 
OWNED_TABLES = {
    'annotation', 
}
REFERENCED_TABLES = {
    'piece', 
    'piece_image'
}

def include_object(object, name, type_, reflected, compare_to):
    """
    Enhanced filtering logic for shared models across schemas
    """
    if type_ == "table":
        # Check if this service owns this table (regardless of schema)
        table_name = name
        if hasattr(object, 'name'):
            table_name = object.name
            
        # Only include tables this service owns
        if table_name not in OWNED_TABLES:
            print(f"SKIPPING table '{table_name}' - not owned by {SCHEMA_NAME}")
            return False
        if table_name in REFERENCED_TABLES:
            print(f"SKIPPING referenced table: {table_name}")
            return False
                    
        print(f"INCLUDING table '{table_name}' - owned by {SCHEMA_NAME}")
        return True
    
    if type_ == "column":
        # Include columns only if their table is owned by this service
        table_name = object.table.name if hasattr(object, 'table') else None
        if table_name and table_name not in OWNED_TABLES:
            return False
        return True
    
    if type_ == "index":
        # Include indexes only if their table is owned by this service
        table_name = None
        if hasattr(object, 'table') and hasattr(object.table, 'name'):
            table_name = object.table.name
        elif hasattr(object, 'table_name'):
            table_name = object.table_name
            
        if table_name and table_name not in OWNED_TABLES:
            return False
        return True
    
    if type_ == "unique_constraint" or type_ == "foreign_key_constraint":
        # Include constraints only if their table is owned by this service
        table_name = None
        if hasattr(object, 'table') and hasattr(object.table, 'name'):
            table_name = object.table.name
        elif hasattr(object, 'table_name'):
            table_name = object.table_name
            
        if table_name and table_name not in OWNED_TABLES:
            return False
        return True
    
    # Include other object types by default
    return True

def process_revision_directives(context, revision, directives):
    """
    Enhanced directive processing for shared models
    """
    if not directives:
        return
        
    for directive in directives:
        if hasattr(directive, 'upgrade_ops') and directive.upgrade_ops:
            # Filter operations to only include owned tables
            filtered_ops = []
            
            for op in directive.upgrade_ops.ops:
                should_include = True
                
                # Check various operation types for table ownership
                if hasattr(op, 'table_name'):
                    should_include = op.table_name in OWNED_TABLES
                elif hasattr(op, 'source_table'):
                    should_include = op.source_table in OWNED_TABLES
                elif hasattr(op, 'target_table'):
                    should_include = op.target_table in OWNED_TABLES
                elif hasattr(op, 'table') and hasattr(op.table, 'name'):
                    should_include = op.table.name in OWNED_TABLES
                
                if should_include:
                    # Also filter nested operations
                    if hasattr(op, 'ops'):
                        op.ops = [
                            nested_op for nested_op in op.ops
                            if not hasattr(nested_op, 'table_name') or 
                               nested_op.table_name in OWNED_TABLES
                        ]
                    filtered_ops.append(op)
                else:
                    print(f"FILTERED OUT operation for non-owned table: {getattr(op, 'table_name', 'unknown')}")
            
            directive.upgrade_ops.ops = filtered_ops

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
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
    """Run migrations in 'online' mode."""
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