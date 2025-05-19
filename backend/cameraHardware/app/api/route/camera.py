import os
from typing import Annotated, List
import cv2
from fastapi import APIRouter, Depends, HTTPException, Response, Path
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from db.session import get_session
from service.frameSource import FrameSource
from service.camera_capture import ImageCapture
import re
from db.schemas.camera import BaslerCameraRequest, OpenCVCameraRequest
from response.camera import (
     CameraStopResponse, CameraStatusResponse,
      )
from response.piece_image import (
    CleanupResponse
)


camera_router = APIRouter(
    prefix="/camera",
    tags=["Camera"],
    responses={404: {"description": "Not found"}},
)


frame_source = FrameSource()



# OpenCV camera routes
@camera_router.post("/opencv/start")
def start_opencv_camera(request: OpenCVCameraRequest):
    """Start an OpenCV camera using the provided index"""
    try:
        frame_source.start_opencv_camera(request.camera_index)
        return {"status": "success", "message": f"OpenCV camera with index {request.camera_index} started successfully"}
    except SystemError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start OpenCV camera: {str(e)}")

# Basler camera routes
@camera_router.post("/basler/start")
def start_basler_camera(request: BaslerCameraRequest):
    """Start a Basler camera using the provided serial number"""
    try:
        frame_source.start_basler_camera(request.serial_number)
        return {"status": "success", "message": f"Basler camera with serial number {request.serial_number} started successfully"}
    except SystemError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Basler camera: {str(e)}")




@camera_router.get("/video_feed")
def video_feed():
    return StreamingResponse(frame_source.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")



@camera_router.get("/capture_images/{piece_label}")
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


@camera_router.post("/cleanup-temp-photos", response_model=CleanupResponse)
async def cleanup_temp_photos_endpoint():
    ImageCapture().cleanup_temp_photos(frame_source)
    return {"message": "Temporary photos cleaned up successfully"}




@camera_router.post("/stop", response_model=CameraStopResponse)
def stop_camera():
    frame_source.stop()
    return {"message": "Camera stopped"}


@camera_router.get("/check_camera", response_model=CameraStatusResponse)
async def check_camera():
    try:
        camera_status = frame_source._check_camera()
        return {"camera_opened": camera_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

