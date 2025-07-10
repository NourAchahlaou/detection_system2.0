from training.app.core.settings import get_settings
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Generator
from sqlalchemy import create_engine
import datetime
from sqlalchemy import DateTime 
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase
import contextlib
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

engine = create_engine(
    settings.DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=20,
    max_overflow=0,
    # Add these parameters for better connection handling
    pool_timeout=30,
    echo=False,  # Set to True for debugging SQL queries
)

SessionLocal = sessionmaker(
    bind=engine, 
    autocommit=False, 
    autoflush=False,
    # Add expire_on_commit=False to prevent issues with detached instances
    expire_on_commit=False
)

class Base(DeclarativeBase):
    """Base class for all models"""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_`%(constraint_name)s`",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {
        datetime.datetime: DateTime(timezone=True),
    }

def get_session() -> Generator:
    """
    Dependency function that provides a database session.
    Fixed to handle concurrent operations properly.
    """
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        # Rollback on any exception
        try:
            session.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback session: {str(rollback_error)}")
        raise
    finally:
        # Ensure session is properly closed
        try:
            if session.in_transaction():
                # If there's an active transaction, rollback before closing
                session.rollback()
            session.close()
        except Exception as close_error:
            logger.error(f"Error closing session: {str(close_error)}")

@contextlib.contextmanager
def get_db_session():
    """
    Context manager for database sessions.
    Use this for operations that need explicit session management.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

def create_new_session():
    """
    Create a new database session.
    Use this when you need a fresh session in background threads.
    """
    return SessionLocal()

def safe_commit(session):
    """
    Safely commit a session with error handling.
    """
    try:
        session.commit()
        return True
    except Exception as e:
        logger.error(f"Commit failed: {str(e)}")
        try:
            session.rollback()
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {str(rollback_error)}")
        return False

def safe_close(session):
    """
    Safely close a session with proper state checking.
    """
    try:
        if session.in_transaction():
            session.rollback()
        session.close()
    except Exception as e:
        logger.error(f"Error closing session: {str(e)}")
        # Force close if possible
        try:
            session.close()
        except:
            pass