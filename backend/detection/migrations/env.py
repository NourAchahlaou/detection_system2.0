import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any app imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from logging.config import fileConfig
from sqlalchemy import engine_from_config, text
from sqlalchemy import pool
from alembic import context

# Import your models and database configuration
from detection.app.db.session import Base
from detection.app.core.settings import get_settings

# Import ALL models for relationship resolution, but only migrate owned ones
from detection.app.db.models.detectionSession  import DetectionSession 
from detection.app.db.models.piece import Piece  # Read-only
from detection.app.db.models.detectionLot import DetectionLot  # Read-only
from detection.app.db.models.camera import Camera  # Read-only
from detection.app.db.models.camera_settings import CameraSettings  # Read-only
version_table = "alembic_version_detection "

config = context.config
settings = get_settings()

print(f"detection service DATABASE_URI: {settings.DATABASE_URI}")
config.set_main_option('sqlalchemy.url', settings.DATABASE_URI.replace('%', '%%'))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

SCHEMA_NAME = "detection"

# Tables this service owns and should manage migrations for
OWNED_TABLES = {
    'detection_session', 
    'detection_lot' # This service owns detection s
}

# Tables from other services that we reference but don't own
REFERENCED_TABLES = {
    'piece',
    'camera',
    'camera_settings'         # Owned by artifact_keeper
  # Owned by artifact_keeper
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

def create_cross_schema_foreign_keys(connection):
    """
    Create foreign key constraints that reference other schemas
    This runs AFTER the main migration to ensure referenced tables exist
    """
    print("Creating cross-schema foreign key constraints...")
    
    try:
        # Check if FK constraint already exists for detection_session.piece_id
        result = connection.execute(text("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE constraint_name = 'fk_detection_session_piece_id'
            AND table_schema = 'detection'
        """))
        
        if result.scalar() == 0:
            # Create FK constraint for detection_session.piece_id -> artifact_keeper.piece.id
            connection.execute(text("""
                ALTER TABLE detection.detection_session
                ADD CONSTRAINT fk_detection_session_piece_id
                FOREIGN KEY (piece_id)
                REFERENCES artifact_keeper.piece(id)
                ON DELETE CASCADE
            """))
            print("✓ Created FK constraint: detection_session.piece_id -> artifact_keeper.piece.id")
        else:
            print("✓ FK constraint already exists: detection_session.piece_id -> artifact_keeper.piece.id")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not create FK constraint for detection_session.piece_id: {e}")
        print("This is expected if the referenced table doesn't exist yet.")
        print("Run artifact_keeper migrations first, then re-run detection migrations.")

    try:
        # Check if FK constraint already exists for detection_lot.expected_piece_id
        result = connection.execute(text("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE constraint_name = 'fk_detection_lot_expected_piece_id'
            AND table_schema = 'detection'
        """))
        
        if result.scalar() == 0:
            # Create FK constraint for detection_lot.expected_piece_id -> artifact_keeper.piece.id
            connection.execute(text("""
                ALTER TABLE detection.detection_lot
                ADD CONSTRAINT fk_detection_lot_expected_piece_id
                FOREIGN KEY (expected_piece_id)
                REFERENCES artifact_keeper.piece(id)
                ON DELETE CASCADE
            """))
            print("✓ Created FK constraint: detection_lot.expected_piece_id -> artifact_keeper.piece.id")
        else:
            print("✓ FK constraint already exists: detection_lot.expected_piece_id -> artifact_keeper.piece.id")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not create FK constraint for detection_lot.expected_piece_id: {e}")
        print("This is expected if the referenced table doesn't exist yet.")
        print("Run artifact_keeper migrations first, then re-run detection migrations.")

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
            
        create_cross_schema_foreign_keys(connection)

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()