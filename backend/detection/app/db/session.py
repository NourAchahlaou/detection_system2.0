from detection.app.core.settings import get_settings
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Generator
from sqlalchemy import create_engine
import datetime
from sqlalchemy import DateTime 
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

settings = get_settings()

engine = create_engine(settings.DATABASE_URI,
                       pool_pre_ping=True,
                       pool_recycle=3600,
                       pool_size=20,
                       max_overflow=0)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

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
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
