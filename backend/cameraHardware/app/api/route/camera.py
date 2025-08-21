import os
from typing import List, Optional, Dict, Any
import cv2
from fastapi import APIRouter, HTTPException, Response, Path, Depends
from fastapi.responses import StreamingResponse
from app.service.frameSource import FrameSource
from app.service.camera_capture import ImageCapture
import re
from app.schemas.camera import BaslerCameraRequest, OpenCVCameraRequest
from app.response.camera import (
     CameraStopResponse, CameraStatusResponse, CameraResponse
      )
from app.response.piece_image import (
    CleanupResponse
)
from app.service.camera_manager import CameraManager
from app.service.circuitBreaker import CircuitBreaker
from app.response.circuitBreaker import CircuitBreakerStatusResponse

# Create circuit breakers for different camera operations
opencv_camera_cb = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    name="opencv_camera"
)

basler_camera_cb = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    name="basler_camera"
)

detect_camera_cb = CircuitBreaker(
    failure_threshold=5,  # More tolerant for detection
    recovery_timeout=30,
    name="detect_cameras"
)

camera_router = APIRouter(
    prefix="/camera",
    tags=["Camera"],
    responses={404: {"description": "Not found"}},
)

frame_source = FrameSource()

# Fallback functions
def detect_cameras_fallback() -> List[CameraResponse]:
    """Fallback for camera detection - return cached results if available"""
    # Return empty list or cached results if available
    return []

def start_camera_fallback(camera_type: str, identifier: str) -> Dict[str, str]:
    """Fallback for camera start operations"""
    return {
        "status": "warning", 
        "message": f"Unable to start {camera_type} camera. Hardware service is temporarily unavailable."
    }

@camera_router.get("/detect", response_model=List[CameraResponse])
async def detect_cameras():
    """
    Detect all available cameras (both regular OpenCV and Basler).
    """
    try:
        # Use circuit breaker to protect this operation
        return detect_camera_cb.execute(
            CameraManager.detect_cameras,
            fallback=detect_cameras_fallback
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Camera service unavailable: {str(e)}")

# OpenCV camera routes
@camera_router.post("/opencv/start")
def start_opencv_camera(request: OpenCVCameraRequest):
    """Start an OpenCV camera using the provided index"""
    try:
        def start_opencv():
            frame_source.start_opencv_camera(request.camera_index)
            return {
                "status": "success",
                "message": f"OpenCV camera with index {request.camera_index} started successfully"
            }
        
        # Use circuit breaker to protect camera start
        return opencv_camera_cb.execute(
            start_opencv,
            fallback=lambda: start_camera_fallback("OpenCV", str(request.camera_index))
        )
    except Exception as e:
        raise HTTPException(
            status_code=503 if "Circuit" in str(e) else 500,
            detail=f"Failed to start OpenCV camera: {str(e)}"
        )

# Basler camera routes
@camera_router.post("/basler/start")
def start_basler_camera(request: BaslerCameraRequest):
    """Start a Basler camera using the provided serial number"""
    try:
        def start_basler():
            frame_source.start_basler_camera(request.serial_number)
            return {
                "status": "success",
                "message": f"Basler camera with serial number {request.serial_number} started successfully"
            }
        
        # Use circuit breaker to protect camera start
        return basler_camera_cb.execute(
            start_basler,
            fallback=lambda: start_camera_fallback("Basler", request.serial_number)
        )
    except Exception as e:
        raise HTTPException(
            status_code=503 if "Circuit" in str(e) else 500,
            detail=f"Failed to start Basler camera: {str(e)}"
        )

@camera_router.get("/video_feed")
def video_feed():
    """Stream video from the current camera in MJPEG format"""
    if not frame_source.camera_is_running:
        raise HTTPException(
            status_code=503,
            detail="Camera is not running or hardware is unavailable"
        )
    return StreamingResponse(
        frame_source.generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@camera_router.get("/raw_jpeg_stream")
def raw_jpeg_stream():
    """Stream raw JPEG frames for video streaming service consumption"""
    if not frame_source.camera_is_running:
        raise HTTPException(
            status_code=503,
            detail="Camera is not running or hardware is unavailable"
        )
    
    print(f"Starting raw JPEG stream for {frame_source.type} camera")
    
    return StreamingResponse(
        frame_source.generate_raw_jpeg_stream(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"
        }
    )

@camera_router.get("/capture_images/{piece_label}")
async def capture_images(
    piece_label: str = Path(..., title="The label of the piece to capture images for")
):
    """Capture images with resilience using circuit breaker - NO LOCAL STORAGE"""
    if not frame_source.camera_is_running:
        raise HTTPException(
            status_code=503,
            detail="Camera is not running. Please start the camera first."
        )
    
    # Extract the part before the dot in the format "A123.4567"
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    
    # Use a specific circuit breaker for image capture
    image_capture_cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30, name="image_capture")
    
    def capture_image():
        """Capture an image using the ImageCapture service"""
        return ImageCapture().capture_image_only(frame_source, piece_label)
    
    def capture_fallback():
        # Return a default "service unavailable" image or None
        return None
    
    try:
        frame = image_capture_cb.execute(capture_image, fallback=capture_fallback)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Camera service temporarily unavailable: {str(e)}"
        )
    
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="Failed to capture frame from the camera. Service may be temporarily unavailable."
        )
    
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@camera_router.post("/cleanup-temp-photos", response_model=CleanupResponse)
async def cleanup_temp_photos_endpoint():
    """Clean up temporary photos with circuit breaker protection"""
    cleanup_cb = CircuitBreaker(failure_threshold=2, recovery_timeout=15, name="cleanup")
    
    def cleanup():
        ImageCapture().cleanup_temp_photos(frame_source)
        return {"message": "Temporary photos cleaned up successfully"}
    
    def cleanup_fallback():
        return {"message": "Unable to clean up temporary photos. Will try later."}
    
    return cleanup_cb.execute(cleanup, fallback=cleanup_fallback)

@camera_router.post("/stop", response_model=CameraStopResponse)
def stop_camera():
    """Stop camera with circuit breaker protection"""
    stop_cb = CircuitBreaker(failure_threshold=2, recovery_timeout=15, name="camera_stop")
    
    def stop():
        frame_source.stop()
        return {"message": "Camera stopped"}
    
    def stop_fallback():
        return {"message": "Camera stop operation registered but may not have completed"}
    
    return stop_cb.execute(stop, fallback=stop_fallback)

@camera_router.get("/check_camera", response_model=CameraStatusResponse)
async def check_camera():
    """Check camera status with circuit breaker protection"""
    check_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=20, name="camera_check")
    
    def check():
        camera_status = frame_source._check_camera()
        return {"camera_opened": camera_status}
    
    def check_fallback():
        return {"camera_opened": False, "circuit_breaker_active": True}
    
    try:
        return check_cb.execute(check, fallback=check_fallback)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@camera_router.get("/status")
async def get_camera_status():
    """Get detailed camera status"""
    return {
        "camera_is_running": frame_source.camera_is_running,
        "camera_type": frame_source.type,
        "camera_id": frame_source.cam_id,
        "is_open": frame_source._check_camera() if frame_source.camera_is_running else False
    }
# Add this to your camera router
@camera_router.get("/single_frame")
def get_single_frame():
    """Get a single frame as JPEG for video streaming service"""
    if not frame_source.camera_is_running:
        raise HTTPException(
            status_code=503,
            detail="Camera is not running or hardware is unavailable"
        )
    
    try:
        # Get a single frame directly from the camera
        frame = frame_source.frame()
        
        if frame is None:
            raise HTTPException(status_code=503, detail="Failed to capture frame")
        
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise HTTPException(status_code=503, detail="Failed to encode frame as JPEG")
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except Exception as e:
       
        raise HTTPException(status_code=503, detail=f"Failed to get frame: {str(e)}")

@camera_router.get("/frame_info")
def get_frame_info():
    """Get information about the camera and frame status"""
    return {
        "camera_is_running": frame_source.camera_is_running,
        "camera_type": frame_source.type,
        "camera_id": frame_source.cam_id,
        "is_open": frame_source._check_camera() if frame_source.camera_is_running else False
    }
# Circuit breaker management endpoints
@camera_router.get("/circuit-breaker-status", response_model=Dict[str, CircuitBreakerStatusResponse])
async def get_circuit_breaker_status():
    """Get the status of all circuit breakers"""
    return {
        "opencv_camera": {
            "state": opencv_camera_cb.current_state,
            "failure_count": opencv_camera_cb.failure_count,
            "last_failure_time": opencv_camera_cb.last_failure_time
        },
        "basler_camera": {
            "state": basler_camera_cb.current_state,
            "failure_count": basler_camera_cb.failure_count,
            "last_failure_time": basler_camera_cb.last_failure_time
        },
        "detect_cameras": {
            "state": detect_camera_cb.current_state,
            "failure_count": detect_camera_cb.failure_count,
            "last_failure_time": detect_camera_cb.last_failure_time
        }
    }

@camera_router.post("/reset-circuit-breaker/{breaker_name}")
async def reset_circuit_breaker(breaker_name: str):
    """Manually reset a specific circuit breaker"""
    breakers = {
        "opencv_camera": opencv_camera_cb,
        "basler_camera": basler_camera_cb,
        "detect_cameras": detect_camera_cb
    }
    
    if breaker_name not in breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{breaker_name}' not found")
    
    breakers[breaker_name].reset()
    return {"message": f"Circuit breaker '{breaker_name}' has been reset"}