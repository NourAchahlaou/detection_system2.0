# artifact_keeper/app/api/endpoints/health.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from artifact_keeper.app.db.session import get_session
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

health_router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={404: {"description": "Not found"}},
)

@health_router.get("/")
@health_router.get("")  # Handle both /health/ and /health
async def health_check(db: Session = Depends(get_session)):
    """
    Comprehensive health check endpoint that verifies:
    1. Service is running
    2. Database connection works
    3. Schema exists
    4. Required tables exist
    """
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        
        # Test that our schema exists
        schema_result = db.execute(text("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = 'artifact_keeper'
        """)).fetchone()
        
        if not schema_result:
            logger.error("artifact_keeper schema not found")
            raise HTTPException(status_code=503, detail="artifact_keeper schema not found")
        
        # Test that our tables exist
        tables_result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'artifact_keeper'
        """)).fetchall()
        
        available_tables = [row[0] for row in tables_result]
        expected_tables = ['piece', 'piece_image', 'camera', 'cameraSettings']
        missing_tables = [table for table in expected_tables if table not in available_tables]
        
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            return {
                "status": "degraded",
                "service": "artifact_keeper",
                "database": "connected",
                "schema": "artifact_keeper",
                "available_tables": available_tables,
                "missing_tables": missing_tables,
                "warning": "Some expected tables are missing"
            }
        
        logger.info("Health check passed - all systems operational")
        return {
            "status": "healthy",
            "service": "artifact_keeper",
            "database": "connected",
            "schema": "artifact_keeper",
            "tables": available_tables,
            "timestamp": "2025-06-04T12:37:30Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )

@health_router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_session)):
    """
    Detailed health check with additional diagnostics
    """
    try:
        # Basic connectivity
        db.execute(text("SELECT 1"))
        
        # Get database version
        db_version = db.execute(text("SELECT version()")).fetchone()[0]
        
        # Get all schemas
        schemas = db.execute(text("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        """)).fetchall()
        
        # Get connection info
        connection_info = db.execute(text("""
            SELECT current_database(), current_user, inet_server_addr(), inet_server_port()
        """)).fetchone()
        
        return {
            "status": "healthy",
            "service": "artifact_keeper",
            "database": {
                "status": "connected",
                "version": db_version,
                "database": connection_info[0],
                "user": connection_info[1],
                "host": connection_info[2],
                "port": connection_info[3]
            },
            "schemas": [schema[0] for schema in schemas],
            "timestamp": "2025-06-04T12:37:30Z"
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Detailed health check failed: {str(e)}"
        )