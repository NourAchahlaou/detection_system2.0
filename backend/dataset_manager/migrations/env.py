from logging.config import fileConfig

from sqlalchemy import engine_from_config, text
from sqlalchemy import pool

from alembic import context
import sys
from pathlib import Path
from dataset_manager.app.db.session import Base
from dataset_manager.app.core.settings import get_settings
# Import ALL models for relationship resolution, but only migrate owned ones
from dataset_manager.app.db.models.piece import  Piece  # Read-onlyp    
from dataset_manager.app.db.models.piece_image import PieceImage  # Read-only
# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
sys.path.append(str(Path(__file__).parent.parent.parent))
version_table = "alembic_version_dataset_manager"
config = context.config
setting = get_settings()

config.set_main_option('sqlalchemy.url', setting.DATABASE_URI.replace('%', '%%'))
# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata
SCHEMA_NAME = "dataset_mng"
# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.
OWNED_TABLES = {}
REFERENCED_TABLES = {
    'piece',        # Owned by artifact_keeper
    'piece_image'   # Owned by artifact_keeper  
    }
def include_object(object, name, type_, reflected, compare_to):
    """
    Enhanced filtering logic for shared models across schemas
    """
    if type_ == "table":
        table_name = name
        if hasattr(object, 'name'):
            table_name = object.name
            
        # Only include tables this service owns
        if table_name not in OWNED_TABLES:
            print(f"SKIPPING table '{table_name}' - not owned by {SCHEMA_NAME}")
            return False
            
        # Explicitly skip referenced tables
        if table_name in REFERENCED_TABLES:
            print(f"SKIPPING referenced table: {table_name}")
            return False
                    
        print(f"INCLUDING table '{table_name}' - owned by {SCHEMA_NAME}")
        return True
    
    if type_ == "column":
        table_name = object.table.name if hasattr(object, 'table') else None
        if table_name and table_name not in OWNED_TABLES:
            return False
        return True
    
    if type_ == "index":
        table_name = None
        if hasattr(object, 'table') and hasattr(object.table, 'name'):
            table_name = object.table.name
        elif hasattr(object, 'table_name'):
            table_name = object.table_name
            
        if table_name and table_name not in OWNED_TABLES:
            return False
        return True
    
    if type_ == "unique_constraint" or type_ == "foreign_key_constraint":
        table_name = None
        if hasattr(object, 'table') and hasattr(object.table, 'name'):
            table_name = object.table.name
        elif hasattr(object, 'table_name'):
            table_name = object.table_name
            
        if table_name and table_name not in OWNED_TABLES:
            return False
        return True
    
    return True

def process_revision_directives(context, revision, directives):
    """
    Enhanced directive processing for shared models
    """
    if not directives:
        return
        
    for directive in directives:
        if hasattr(directive, 'upgrade_ops') and directive.upgrade_ops:
            filtered_ops = []
            
            for op in directive.upgrade_ops.ops:
                should_include = True
                
                if hasattr(op, 'table_name'):
                    should_include = op.table_name in OWNED_TABLES
                elif hasattr(op, 'source_table'):
                    should_include = op.source_table in OWNED_TABLES
                elif hasattr(op, 'target_table'):
                    should_include = op.target_table in OWNED_TABLES
                elif hasattr(op, 'table') and hasattr(op.table, 'name'):
                    should_include = op.table.name in OWNED_TABLES
                
                if should_include:
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