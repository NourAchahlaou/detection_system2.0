from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Annotated
import logging
import requests

from artifact_keeper.app.db.models.camera_settings import CameraSettings
from artifact_keeper.app.db.session import get_session
from artifact_keeper.app.services.camera import CameraService
from artifact_keeper.app.request.camera import (

    CameraIDRequest,
    UpdateCameraSettings,
)
from artifact_keeper.app.response.camera import (

    CameraResponse,
    CameraStatusResponse,
    CameraStopResponse,
    CircuitBreakerStatusResponse,
)


camera_router = APIRouter(
    prefix="/camera",
    tags=["Camera"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)
camera_service = CameraService()

# Database dependency
db_dependency = Annotated[Session, Depends(get_session)]

@camera_router.get("/detect", response_model=List[Dict[str, Any]])
async def detect_cameras(db: db_dependency):
    """
    Detect all available cameras (both regular OpenCV and Basler) and save to database.
    """
    logger.info("Detecting cameras")
    return camera_service.detect_and_save_cameras(db)

@camera_router.get("/get_allcameras", response_model=List[CameraResponse])
async def get_all_cameras(db: db_dependency):
    """
    Get all registered cameras from the database.
    """
    cameras = camera_service.get_all_cameras(db)
    return cameras

@camera_router.get("/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera_by_id(camera_id: int, db: db_dependency):
    """
    Get a camera by its ID.
    """
    return camera_service.get_camera_by_id(db, camera_id)

@camera_router.post("/start", response_model=Dict[str, str])
async def start_camera(request: CameraIDRequest, db: db_dependency):
    """
    Start a camera by its ID.
    """
    return camera_service.start_camera(db, request.camera_id)

@camera_router.post("/stop", response_model=CameraStopResponse)
async def stop_camera(db: db_dependency):
    """
    Stop the currently running camera.
    """
    return camera_service.stop_camera(db)

@camera_router.get("/check_camera", response_model=CameraStatusResponse)
async def check_camera():
    """
    Check if the camera is running.
    """
    return camera_service.check_camera()


@camera_router.get("/circuit-breaker-status", response_model=Dict[str, CircuitBreakerStatusResponse])
async def get_circuit_breaker_status():
    """
    Get the status of all circuit breakers.
    """
    return camera_service.get_circuit_breaker_status()

@camera_router.post("/reset-circuit-breaker/{breaker_name}", response_model=Dict[str, str])
async def reset_circuit_breaker(breaker_name: str):
    """
    Reset a specific circuit breaker.
    """
    return camera_service.reset_circuit_breaker(breaker_name)

# Video feed endpoint - this will proxy the video feed from the hardware service
@camera_router.get("/video_feed")
async def video_feed():
    """
    Stream video from the hardware service.
    """
    from fastapi.responses import StreamingResponse
    
    try:
        hardware_response = requests.get(
            "http://host.docker.internal:8003/camera/video_feed",
            stream=True
        )
        return StreamingResponse(
            hardware_response.iter_content(chunk_size=1024),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Hardware service unavailable")
    except Exception as e:
        logger.error(f"Error streaming video feed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stream video feed: {str(e)}")

@camera_router.put("/settings/{camera_id}", response_model=CameraResponse)
async def update_camera_settings(
    camera_id: int, 
    settings: UpdateCameraSettings,
    db: db_dependency
):
    """
    Update camera settings.
    """
    camera = camera_service.get_camera_by_id(db, camera_id)
    camera_settings = db.query(CameraSettings).filter(CameraSettings.id == camera.settings_id).first()
    
    if not camera_settings:
        raise HTTPException(status_code=404, detail="Camera settings not found")
    
    # Update the settings in the database
    for key, value in settings.dict(exclude_unset=True).items():
        setattr(camera_settings, key, value)
    
    db.commit()
    db.refresh(camera_settings)
    db.refresh(camera)
    
    return camera
