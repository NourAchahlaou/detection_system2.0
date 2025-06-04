
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from annotation.app.db.session import get_session
from sqlalchemy import text

health_router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={404: {"description": "Not found"}},
)

@health_router.get("/health")
async def health_check(db: Session = Depends(get_session)):
    """
    Health check endpoint that verifies:
    1. Service is running
    2. Database connection works
    3. Schema exists
    4. Referenced tables exist
    """
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        
        # Test that our schema exists
        result = db.execute(text("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = 'annotation'
        """)).fetchone()
        
        if not result:
            return {"status": "unhealthy", "reason": "annotation schema not found"}
        
        # Test that our tables exist
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'annotation' 
            AND table_name = 'annotation'
        """)).fetchone()
        
        if not result:
            return {"status": "unhealthy", "reason": "annotation table not found"}
        
        # Test that referenced tables exist in other schemas
        referenced_tables = db.execute(text("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE (table_schema = 'artifact_keeper' AND table_name IN ('piece', 'piece_image'))
        """)).fetchall()
        
        if len(referenced_tables) < 2:
            return {"status": "unhealthy", "reason": "Referenced tables missing from artifact_keeper schema"}
        
        # Test cross-schema foreign key constraint exists
        fk_exists = db.execute(text("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_schema = 'annotation' 
            AND table_name = 'annotation' 
            AND constraint_name = 'fk_annotation_piece_image_id'
        """)).fetchone()
        
        return {
            "status": "healthy",
            "service": "annotation",
            "database": "connected",
            "schema": "annotation",
            "tables": ["annotation"],
            "referenced_tables": [f"{row[0]}.{row[1]}" for row in referenced_tables],
            "cross_schema_fk": "exists" if fk_exists else "missing"
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
