
import logging

from detection.app.service.DetectionSessionManager import DetectionSessionManager
from detection.app.service.DetectionService import load_model_once
from fastapi import APIRouter, HTTPException
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

detection_router = APIRouter(
    prefix="/detection",
    tags=["Detection Processing"],
    responses={404: {"description": "Not found"}},
)

# Global detection system
detection_system = None

# Thread pool for CPU-bound tasks
# Active detection sessions
active_detections = {}
detection_locks = {}


session_manager = DetectionSessionManager()

    


@detection_router.post("/model/reload")
async def load_model_endpoint():
    """Endpoint to load the model once when the inspection page is accessed."""
    try:
        await load_model_once()
        return {"message": "Model loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the model: {e}")

@detection_router.get("/health")
async def detection_health_check():
    """Health check for detection service."""
    try:
        model_loaded = detection_system is not None
        active_session_count = len([s for s in session_manager.sessions.keys() 
                                  if session_manager.is_session_active(s)])
        
        return {
            "status": "healthy" if model_loaded else "degraded",
            "service": "detection_processing",
            "model_loaded": model_loaded,
            "active_sessions": active_session_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "detection_processing",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
