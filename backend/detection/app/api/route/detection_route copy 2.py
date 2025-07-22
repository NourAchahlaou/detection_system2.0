import cv2
import aiohttp
import asyncio
import io
import logging
import threading
import numpy as np
from typing import Annotated, AsyncGenerator, Optional
from fastapi import APIRouter, Depends, HTTPException

from fastapi import File, UploadFile, Form

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import time
from datetime import datetime
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
import queue

from detection.app.service.detection_service import DetectionSystem
from detection.app.db.session import get_session
from detection.app.service.hardwareServiceClient import CameraClient

# Import the camera service from artifact_keeper to get camera details
from detection.app.service.camera import CameraService
import base64


class FrameProcessingResponse(BaseModel):
    processed_frame: str  # Base64 encoded frame
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("training_logs.log", mode='a')])

logger = logging.getLogger(__name__)
detection_router = APIRouter(
    prefix="/detection",
    tags=["Detection"],
    responses={404: {"description": "Not found"}},
)

db_dependency = Annotated[Session, Depends(get_session)]

# Configuration for direct hardware connection
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"

# Initialize hardware client and camera service
hardware_client = CameraClient(base_url=HARDWARE_SERVICE_URL)
camera_service = CameraService()

# Global variables for detection system and stop event
detection_system = None
stop_event = threading.Event()

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=2)

# Frame processing queue to prevent backlog
frame_queue = queue.Queue(maxsize=3)  # Limit queue size to prevent memory buildup

async def load_model_once():
    """Load the model once when the application starts to avoid reloading it for each frame."""
    global detection_system
    if detection_system is None:
        detection_system = DetectionSystem()
        detection_system.get_my_model()
        logger.info("Detection model loaded successfully")

@detection_router.get("/load_model")
async def load_model_endpoint():
    """Endpoint to load the model once when the inspection page is accessed."""
    try:
        await load_model_once()
        return {"message": "Model loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the model: {e}")

def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
    """Optimized frame resizing with better interpolation."""
    if frame.shape[:2] != target_size[::-1]:  # Check if resize is needed
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return frame

def process_frame_sync(frame: np.ndarray, target_label: str):
    """Synchronous frame processing to run in thread pool."""
    try:
        # Ensure frame is contiguous for better performance
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
            
        detection_results = detection_system.detect_and_contour(frame, target_label)
        if isinstance(detection_results, tuple):
            processed_frame = detection_results[0]
            detected_target = detection_results[1] if len(detection_results) > 1 else False
            non_target_count = detection_results[2] if len(detection_results) > 2 else 0
        else:
            processed_frame = detection_results
            detected_target = False
            non_target_count = 0
        
        return processed_frame, detected_target, non_target_count
    except cv2.error as e:
        logger.error(f"OpenCV error: {e}")
        return frame, False, 0
    except Exception as e:
        logger.error(f"Unexpected error in frame processing: {e}")
        return frame, False, 0

async def process_frame_async(frame: np.ndarray, target_label: str):
    """Asynchronously process a single frame using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_frame_sync, frame, target_label)

async def wait_for_camera_ready(camera_index: int, max_wait_time: int = 15) -> bool:
    """Optimized camera readiness check."""
    logger.info(f"Waiting for camera index {camera_index} to be ready...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            check_response = await check_camera_status_safe()
            
            if check_response and check_response.get("camera_opened", False):
                logger.info(f"Camera index {camera_index} reports as ready")
                
                # Single verification attempt
                test_frame = await capture_frame_from_hardware_simple()
                if test_frame is not None:
                    logger.info("Camera verified with successful frame capture")
                    return True
                    
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.warning(f"Error during camera readiness check: {e}")
            await asyncio.sleep(1.0)
    
    logger.error(f"Camera index {camera_index} not ready after {max_wait_time} seconds")
    return False

async def check_camera_status_safe() -> Optional[dict]:
    """Safely check camera status with proper error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=3.0)  # Reduced timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HARDWARE_SERVICE_URL}/camera/check_camera") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Camera check returned status {response.status}")
                    return None
    except asyncio.TimeoutError:
        logger.warning("Camera check timed out")
        return None
    except Exception as e:
        logger.warning(f"Error checking camera status: {e}")
        return None

async def capture_frame_from_hardware_simple() -> Optional[np.ndarray]:
    """Optimized simple frame capture."""
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                if response.status == 200:
                    buffer = bytearray()
                    chunk_size = 16384  # Increased chunk size
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        buffer.extend(chunk)
                        
                        # Look for complete JPEG frame
                        jpeg_start = buffer.find(b'\xff\xd8')
                        if jpeg_start != -1:
                            jpeg_end = buffer.find(b'\xff\xd9', jpeg_start)
                            if jpeg_end != -1:
                                # Extract complete JPEG
                                jpeg_data = bytes(buffer[jpeg_start:jpeg_end+2])
                                
                                # Decode the frame
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    return frame
                        
                        # Break after reasonable buffer size
                        if len(buffer) > 200000:  # 200KB
                            break
                
                return None
                
    except Exception as e:
        logger.debug(f"Simple frame capture failed: {e}")
        return None

async def capture_frame_from_hardware_optimized() -> Optional[np.ndarray]:
    """Optimized frame capture with better error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=8.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                if response.status == 200:
                    buffer = bytearray()
                    chunk_size = 32768  # Larger chunks for better performance
                    max_buffer_size = 1024 * 1024  # 1MB max buffer
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        buffer.extend(chunk)
                        
                        # Look for JPEG boundaries
                        jpeg_start = buffer.find(b'\xff\xd8')
                        if jpeg_start != -1:
                            jpeg_end = buffer.find(b'\xff\xd9', jpeg_start + 2)
                            if jpeg_end != -1:
                                # Extract and decode JPEG
                                jpeg_data = bytes(buffer[jpeg_start:jpeg_end + 2])
                                
                                try:
                                    nparr = np.frombuffer(jpeg_data, np.uint8)
                                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if frame is not None and frame.size > 0:
                                        return frame
                                except Exception:
                                    # Continue if decode fails
                                    pass
                                    
                                # Remove processed data
                                buffer = buffer[jpeg_end + 2:]
                        
                        # Prevent buffer overflow
                        if len(buffer) > max_buffer_size:
                            buffer = buffer[-max_buffer_size//2:]  # Keep recent half
                            
                        # Stop if we have reasonable data
                        if len(buffer) > 500000:  # 500KB
                            break
                            
    except Exception as e:
        logger.error(f"Optimized frame capture failed: {e}")
    
    return None

async def generate_frames(camera_id: int, target_label: str, db: Session) -> AsyncGenerator[bytes, None]:
    """Generate video frames with optimized processing."""
    
    # Get camera details
    try:
        camera = camera_service.get_camera_by_id(db, camera_id)
        camera_index = camera.camera_index
        logger.info(f"Using camera ID {camera_id} with camera index {camera_index}")
    except Exception as e:
        logger.error(f"Failed to get camera details for ID {camera_id}: {e}")
        return
    
    # Start camera
    camera_started = await start_camera_via_hardware_service(camera_index)
    if not camera_started:
        logger.error("Failed to start camera via hardware service")
        return
    
    # Wait for camera readiness
    camera_ready = await wait_for_camera_ready(camera_index)
    if not camera_ready:
        logger.error("Camera did not become ready in time")
        await stop_camera_via_hardware_service()
        return
    
    frame_counter = 0
    consecutive_failures = 0
    max_consecutive_failures = 3  # Reduced for faster failure detection
    last_successful_frame = None
    
    # Pre-encode settings for better performance
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Slightly lower quality for speed
    target_fps = 25  # Reduced FPS for better processing
    frame_time = 1.0 / target_fps
    
    # Skip detection on every N frames for better performance
    detection_skip = 2  # Process every 2nd frame
    
    try:
        while not stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                # Capture frame
                frame = await capture_frame_from_hardware_optimized()

                if frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Frame capture failed {consecutive_failures} times")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame failures, stopping camera")
                        break
                        
                    # Use cached frame if available
                    if last_successful_frame is not None:
                        frame = last_successful_frame
                    else:
                        await asyncio.sleep(0.05)
                        continue
                else:
                    consecutive_failures = 0
                    # Cache successful frame (resize to save memory)
                    last_successful_frame = cv2.resize(frame, (320, 240))  # Smaller cache
                
                # Resize frame once
                frame = resize_frame_optimized(frame, (640, 480))
                
                # Process detection (skip frames for better performance)
                if frame_counter % detection_skip == 0:
                    try:
                        processed_frame, detected_target, non_target_count = await process_frame_async(frame, target_label)
                        
                        if detected_target:
                            logger.info(f"Target object '{target_label}' detected!")
                        if non_target_count > 0:
                            logger.debug(f"Detected {non_target_count} non-target objects")
                            
                        frame = processed_frame
                        
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                
                # Encode and yield frame
                if frame is not None and len(frame.shape) == 3:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, encode_params)
                        if buffer is not None:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except Exception as e:
                        logger.error(f"Error encoding frame: {e}")

                frame_counter += 1
                
                # Maintain target FPS
                elapsed = time.time() - frame_start_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                
            except Exception as e:
                logger.error(f"Error in frame generation loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
    finally:
        await stop_camera_via_hardware_service()
        stop_event.clear()
        logger.info("Frame generation stopped and camera resources released")

async def start_camera_via_hardware_service(camera_index: int) -> bool:
    """Start camera using the hardware service."""
    try:
        # Cleanup first
        try:
            await hardware_client.cleanup_temp_photos()
        except:
            pass
        
        timeout = aiohttp.ClientTimeout(total=8.0)  # Reduced timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {"camera_index": camera_index}
            
            async with session.post(f"{HARDWARE_SERVICE_URL}/camera/opencv/start", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Camera start response: {result}")
                    
                    if result.get("status") == "success" or "started successfully" in result.get("message", "").lower():
                        logger.info(f"Camera index {camera_index} started successfully")
                        return True
                    else:
                        logger.error(f"Camera start failed: {result}")
                        return False
                else:
                    logger.error(f"Camera start request failed with status {response.status}")
                    return False
            
    except Exception as e:
        logger.error(f"Failed to start camera via hardware service: {e}")
        return False

async def stop_camera_via_hardware_service():
    """Stop camera using the hardware service."""
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)  # Reduced timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{HARDWARE_SERVICE_URL}/camera/stop") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Camera stop response: {result}")
                else:
                    logger.warning(f"Camera stop request returned status {response.status}")
        
        # Cleanup
        try:
            await hardware_client.cleanup_temp_photos()
        except:
            pass
        
        return True
    except Exception as e:
        logger.error(f"Failed to stop camera via hardware service: {e}")
        return False

@detection_router.get("/video_feed")
async def video_feed(camera_id: int, target_label: str, db: Session = Depends(get_session)):
    """Endpoint to start the video feed and detect objects using hardware service."""
    await load_model_once()
    stop_event.clear()
    
    return StreamingResponse(
        generate_frames(camera_id, target_label, db), 
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

def stop_video():
    """Stop video streaming."""
    stop_event.set()
    logger.info("Stop video signal sent.")

@detection_router.post("/stop_camera_feed")
async def stop_camera_feed():
    """Stop the camera feed."""
    try:
        stop_video()
        await stop_camera_via_hardware_service()
        return {"message": "Camera feed stopped successfully."}
    except Exception as e:
        logger.error(f"Error stopping camera feed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Directory to save captured images
SAVE_DIR = "captured_images"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

async def capture_single_frame_via_hardware(camera_id: int, target_label: str, user_id: str, of: str, db: Session):
    """Optimized single frame capture."""
    await load_model_once()
    
    try:
        camera = camera_service.get_camera_by_id(db, camera_id)
        camera_index = camera.camera_index
        logger.info(f"Using camera ID {camera_id} with camera index {camera_index}")
    except Exception as e:
        logger.error(f"Failed to get camera details for ID {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get camera details: {e}")
    
    # Start camera
    camera_started = await start_camera_via_hardware_service(camera_index)
    if not camera_started:
        raise HTTPException(status_code=500, detail="Failed to start camera via hardware service")
    
    # Wait for camera readiness
    camera_ready = await wait_for_camera_ready(camera_index)
    if not camera_ready:
        await stop_camera_via_hardware_service()
        raise HTTPException(status_code=500, detail="Camera did not become ready in time")
    
    try:
        # Capture with fewer retries for faster response
        frame = await capture_frame_from_hardware_optimized()
        if frame is None:
            # Single retry
            await asyncio.sleep(0.5)
            frame = await capture_frame_from_hardware_optimized()
            
        if frame is None:
            raise HTTPException(status_code=500, detail="No frame captured from the hardware service.")
        
        # Resize and process
        frame = resize_frame_optimized(frame, (640, 480))
        processed_frame, detected_target, non_target_count = await process_frame_async(frame, target_label)
        
        if detected_target:
            # Save detected image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"captured_{target_label}_{timestamp}_{user_id}.jpg"
            image_path = os.path.join(SAVE_DIR, image_name)
            
            success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode frame.")
            
            with open(image_path, 'wb') as f:
                f.write(buffer.tobytes())
            
            logger.info(f"Captured image saved at {image_path}")
            return {
                "message": "Target object detected and image captured.", 
                "file_path": image_path,
                "non_target_count": non_target_count
            }
        else:
            return {
                "message": "No target object detected in the frame.",
                "non_target_count": non_target_count
            }
    
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while capturing the frame: {e}")
    
    finally:
        await stop_camera_via_hardware_service()

@detection_router.get("/capture_image")
async def capture_image(camera_id: int, of: str, target_label: str, user_id: str, db: Session = Depends(get_session)):
    """Capture an image using the hardware service."""
    return await capture_single_frame_via_hardware(camera_id, target_label, user_id, of, db)

@detection_router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        hardware_status = await check_camera_status_safe()
        hardware_accessible = hardware_status is not None
        model_loaded = detection_system is not None
        
        return {
            "status": "healthy" if hardware_accessible and model_loaded else "degraded",
            "hardware_service": "accessible" if hardware_accessible else "unavailable",
            "detection_model": "loaded" if model_loaded else "not loaded",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@detection_router.post("/process_frame", response_model=FrameProcessingResponse)
async def process_frame_endpoint(
    frame: UploadFile = File(..., description="Frame image file"),
    target_label: str = Form(..., description="Target label to detect")
):
    """
    Process a single frame for object detection.
    This endpoint is designed to be called by the external camera service.
    
    Args:
        frame: Image file (JPEG format)
        target_label: Target object label to detect
    
    Returns:
        FrameProcessingResponse with processed frame and detection results
    """
    import time
    
    start_time = time.time()
    
    try:
        # Ensure model is loaded
        await load_model_once()
        
        # Read frame data
        frame_data = await frame.read()
        if not frame_data:
            raise HTTPException(status_code=400, detail="Empty frame data")
        
        # Decode frame
        nparr = np.frombuffer(frame_data, np.uint8)
        cv_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Resize frame for consistent processing
        cv_frame = resize_frame_optimized(cv_frame, (640, 480))
        
        # Process frame with detection
        processed_frame, detected_target, non_target_count = await process_frame_async(
            cv_frame, target_label
        )
        
        # Encode processed frame to base64
        success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed frame")
        
        processed_frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return FrameProcessingResponse(
            processed_frame=processed_frame_b64,
            detected_target=detected_target,
            non_target_count=non_target_count,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {e}")

@detection_router.get("/model/status")
async def get_model_status():
    """Get the current status of the detection model."""
    global detection_system
    
    return {
        "model_loaded": detection_system is not None,
        "device": str(detection_system.device) if detection_system else "unknown",
        "model_type": "YOLO" if detection_system else "unknown",
        "timestamp": datetime.now().isoformat()
    }

@detection_router.post("/model/reload")
async def reload_model():
    """Force reload the detection model."""
    global detection_system
    
    try:
        detection_system = None  # Clear existing model
        await load_model_once()  # Reload model
        
        return {
            "message": "Model reloaded successfully",
            "device": str(detection_system.device),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")    
    