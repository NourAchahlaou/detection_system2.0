import os
from typing import Annotated, Generator, List, Optional
import cv2
from fastapi import APIRouter, Depends, HTTPException, Response, Path, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from piece_registry.app.db.session import get_session
from piece_registry.app.db.schemas.camera_settings import UpdateCameraSettings
from piece_registry.app.db.models.camera import Camera 
from piece_registry.app.service.frameSource import FrameSource
from piece_registry.app.service.camera_manager import CameraManager
from piece_registry.app.service.camera_capture import ImageCapture
import re
from piece_registry.app.response.camera import (
    CameraStartResponse, CameraStopResponse, CameraStatusResponse,
      CameraIndexResponse, CameraWithSettingsResponse)
from piece_registry.app.response.piece_image import (
    CleanupResponse, SaveImagesResponse
)
from piece_registry.app.db.schemas.camera import (
    CameraID,
    CameraListItem
)

router = APIRouter()
db_dependency = Annotated[Session, Depends(get_session)]

frame_source = FrameSource()


@router.get("/get-index", response_model=CameraIndexResponse)
def get_camera_index(camera_id: int, db: db_dependency):
    camera_index = frame_source.get_camera_by_index(camera_id, db)
    if camera_index is None:
        raise HTTPException(status_code=404, detail=f"Camera with id {camera_id} not found")
    return {"camera_index": camera_index}


@router.get("/cameras/", response_model=List[tuple])
def read_cameras(db: db_dependency):
    cameras = frame_source.get_camera_model_and_ids(db)
    if not cameras:
        raise HTTPException(status_code=404, detail="No cameras found")
    return cameras


@router.post("/start-camera", response_model=CameraStartResponse)
def start_camera(camera: CameraID, db: db_dependency):
    frame_source.start(camera.camera_id, db)
    return {"message": "Camera started"}


@router.get("/video_feed")
def video_feed():
    return StreamingResponse(frame_source.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/camera_info/{camera_id}", response_model=CameraWithSettingsResponse)
def get_camera_info(
    camera_id: int = Path(..., title="The ID of the camera to get"),
    db: db_dependency = Depends()
):
    return CameraManager.get_camera(camera_id, db)


@router.put("/{camera_id}", response_model=Camera)
async def change_camera_settings(
    camera_id: int = Path(..., title="The ID of the camera to update settings for"),
    camera_settings_update: UpdateCameraSettings = None,
    db: db_dependency = Depends()
):
    return CameraManager.change_camera_settings(camera_id, camera_settings_update, db)


@router.get("/capture_images/{piece_label}")
async def capture_images(
    piece_label: str = Path(..., title="The label of the piece to capture images for")
):
    if not frame_source.camera_is_running:
        raise HTTPException(status_code=400, detail="Camera is not running. Please start the camera first.")
    
    # Extract the part before the dot in the format "A123.4567"
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    extracted_label = match.group(1)
    url = os.path.join('Pieces','Pieces','images',"valid",extracted_label, piece_label)
   
    # Define save_folder using the extracted_label
    save_folder = os.path.join('dataset','Pieces','Pieces','images','valid',extracted_label,piece_label)
    os.makedirs(save_folder, exist_ok=True)
    
    try:
        frame = ImageCapture().capture_images(frame_source, save_folder, url, piece_label)
    except SystemError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from the camera.")
    
    _, buffer = cv2.imencode('.jpg', frame)
    print("Frame captured and encoded")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.post("/cleanup-temp-photos", response_model=CleanupResponse)
async def cleanup_temp_photos_endpoint():
    ImageCapture().cleanup_temp_photos(frame_source)
    return {"message": "Temporary photos cleaned up successfully"}


@router.post("/save-images", response_model=SaveImagesResponse)
def save_images(
    piece_label: str = Query(..., title="The label of the piece to save images for"),
    db: db_dependency = Depends()
):
    try:
        ImageCapture().save_images_to_database(frame_source, db, piece_label)
    except SystemError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Images saved to database"}


@router.post("/stop", response_model=CameraStopResponse)
def stop_camera():
    frame_source.stop()
    return {"message": "Camera stopped"}


@router.get("/check_camera", response_model=CameraStatusResponse)
async def check_camera():
    try:
        camera_status = frame_source._check_camera()
        return {"camera_opened": camera_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_allcameras/", response_model=List[CameraListItem])
def read_all_cameras(db: db_dependency):
    cameras = db.query(Camera.camera_index, Camera.model, Camera.id).all()
    return [{"camera_index": cam.camera_index, "model": cam.model, "camera_id": cam.id} for cam in cameras]


@router.get("/cameraByModelIndex/", response_model=int)
def read_camera_id(
    model: str = Query(..., title="Camera model name"),
    camera_index: int = Query(..., title="Camera index"),
    db: db_dependency = Depends()
):
    camera_id = db.query(Camera.id).filter(
        Camera.model == model,
        Camera.camera_index == camera_index
    ).scalar()
    
    if camera_id is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return camera_id