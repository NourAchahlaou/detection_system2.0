import cv2
import aiohttp
import asyncio
import io
import logging
import threading
import numpy as np
from typing import Annotated, AsyncGenerator, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import time
from datetime import datetime
import uuid
import os

from detection.app.service.detection_service import DetectionSystem
from detection.app.db.session import get_session
from detection.app.service.hardwareServiceClient import CameraClient


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

# Initialize hardware client
hardware_client = CameraClient(base_url=HARDWARE_SERVICE_URL)

# Global variables for detection system and stop event
detection_system = None
stop_event = threading.Event()

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

def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to the nearest size divisible by 32."""
    height, width = frame.shape[:2]
    new_height = (height // 32) * 32
    new_width = (width // 32) * 32
    return cv2.resize(frame, (new_width, new_height))

async def process_frame_async(frame: np.ndarray, target_label: str):
    """Asynchronously process a single frame to perform detection and contouring."""
    try:
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

async def wait_for_camera_ready(camera_id: int, max_wait_time: int = 15) -> bool:
    """Wait for the camera to be ready and streaming frames."""
    logger.info(f"Waiting for camera {camera_id} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            # Check if camera is running with proper error handling
            check_response = await check_camera_status_safe()
            if check_response and check_response.get("camera_opened", False):
                logger.info(f"Camera {camera_id} is ready and running")
                # Additional verification - try to capture a test frame
                test_frame = await capture_frame_from_hardware_simple()
                if test_frame is not None:
                    logger.info("Camera verified with successful frame capture")
                    return True
                else:
                    logger.warning("Camera reports as open but no frame captured")
            
            logger.debug(f"Camera {camera_id} not ready yet, waiting...")
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Error checking camera readiness: {e}")
            await asyncio.sleep(1.0)
    
    logger.error(f"Camera {camera_id} not ready after {max_wait_time} seconds")
    return False

async def check_camera_status_safe() -> Optional[dict]:
    """Safely check camera status with proper error handling."""
    try:
        # Use aiohttp for async request
        timeout = aiohttp.ClientTimeout(total=5.0)
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
    """Simple frame capture for testing camera availability."""
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                if response.status == 200:
                    # Read first chunk of data
                    content = await response.read()
                    if content:
                        # Find JPEG start marker
                        jpeg_start = content.find(b'\xff\xd8')
                        if jpeg_start != -1:
                            # Find JPEG end marker
                            jpeg_end = content.find(b'\xff\xd9', jpeg_start)
                            if jpeg_end != -1:
                                jpeg_data = content[jpeg_start:jpeg_end+2]
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                return frame
                
                return None
    except Exception as e:
        logger.debug(f"Simple frame capture failed: {e}")
        return None

async def capture_frame_from_hardware_with_retry(max_retries: int = 3, retry_delay: float = 1.0) -> Optional[np.ndarray]:
    """Capture frame from hardware service with improved error handling."""
    
    for attempt in range(max_retries):
        try:
            # Use async approach with aiohttp for better performance
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                    if response.status == 200:
                        # Read the response content
                        content = await response.read()
                        
                        if not content:
                            logger.warning(f"Attempt {attempt + 1}: No content received")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            return None
                        
                        # Handle multipart stream format
                        if b'Content-Type: image/jpeg' in content:
                            # Extract JPEG data from multipart response
                            jpeg_start = content.find(b'\xff\xd8')  # JPEG start marker
                            jpeg_end = content.find(b'\xff\xd9')    # JPEG end marker
                            
                            if jpeg_start != -1 and jpeg_end != -1:
                                jpeg_data = content[jpeg_start:jpeg_end+2]
                            else:
                                # If no markers found, look for boundary
                                boundary_start = content.find(b'\r\n\r\n')
                                if boundary_start != -1:
                                    jpeg_data = content[boundary_start+4:]
                                else:
                                    jpeg_data = content
                        else:
                            jpeg_data = content
                        
                        # Decode JPEG
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            logger.debug(f"Successfully captured frame on attempt {attempt + 1}")
                            return frame
                        else:
                            logger.warning(f"Attempt {attempt + 1}: Failed to decode frame data")
                            
                    elif response.status == 503:
                        logger.warning(f"Attempt {attempt + 1}: Hardware service unavailable (503)")
                    else:
                        logger.error(f"Attempt {attempt + 1}: Hardware service returned status {response.status}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt + 1}: Timeout when capturing frame")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Unexpected error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
    
    logger.error(f"Failed to capture frame after {max_retries} attempts")
    return None

async def generate_frames(camera_id: int, target_label: str, db: Session) -> AsyncGenerator[bytes, None]:
    """Generate video frames with improved error handling and retry logic."""
    
    # Start camera via hardware service
    camera_started = await start_camera_via_hardware_service(camera_id)
    if not camera_started:
        logger.error("Failed to start camera via hardware service")
        return
    
    # Wait for camera to be ready
    camera_ready = await wait_for_camera_ready(camera_id)
    if not camera_ready:
        logger.error("Camera did not become ready in time")
        await stop_camera_via_hardware_service()
        return
    
    frame_counter = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    last_successful_frame = None
    
    # Add initialization delay
    await asyncio.sleep(2.0)

    try:
        while not stop_event.is_set():
            try:
                # Capture frame with retry logic
                frame = await capture_frame_from_hardware_with_retry(max_retries=2, retry_delay=0.3)

                if frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Frame capture failed {consecutive_failures} times")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame failures, stopping camera")
                        break
                        
                    # Use last successful frame if available
                    if last_successful_frame is not None:
                        frame = last_successful_frame.copy()
                        logger.debug("Using last successful frame as fallback")
                    else:
                        await asyncio.sleep(0.1)
                        continue
                else:
                    # Reset failure counter and store successful frame
                    consecutive_failures = 0
                    last_successful_frame = frame.copy()
                
                # Resize frame to standard size
                frame = cv2.resize(frame, (640, 480))
                
                # Process frame for detection
                if frame_counter % 1 == 0:  # Process every frame
                    try:
                        processed_frame, detected_target, non_target_count = await process_frame_async(frame, target_label)
                        
                        if non_target_count > 0:
                            logger.info(f"Detected {non_target_count} non-target objects")
                        
                        if detected_target:
                            logger.info(f"Target object '{target_label}' detected!")
                            
                        frame = processed_frame
                        
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        # Use original frame if processing fails
                
                # Encode frame for streaming
                if frame is not None and len(frame.shape) == 3:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if buffer is not None:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except Exception as e:
                        logger.error(f"Error encoding frame: {e}")

                frame_counter += 1
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in frame generation loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
    finally:
        await stop_camera_via_hardware_service()
        stop_event.clear()
        logger.info("Frame generation stopped and camera resources released")

async def start_camera_via_hardware_service(camera_id: int) -> bool:
    """Start camera using the hardware service with verification."""
    try:
        # First cleanup any existing camera state
        try:
            await hardware_client.cleanup_temp_photos()
        except:
            pass  # Ignore cleanup errors
        
        # Use aiohttp for async request
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Prepare the request payload
            payload = {"camera_index": camera_id}
            
            async with session.post(f"{HARDWARE_SERVICE_URL}/camera/opencv/start", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Camera start response: {result}")
                    
                    # Check if response indicates success
                    if result.get("status") == "success" or "started successfully" in result.get("message", "").lower():
                        logger.info(f"Camera {camera_id} started successfully")
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
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{HARDWARE_SERVICE_URL}/camera/stop") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Camera stop response: {result}")
                else:
                    logger.warning(f"Camera stop request returned status {response.status}")
        
        # Also cleanup temp photos
        try:
            await hardware_client.cleanup_temp_photos()
        except:
            pass  # Ignore cleanup errors
        
        return True
    except Exception as e:
        logger.error(f"Failed to stop camera via hardware service: {e}")
        return False

@detection_router.get("/video_feed")
async def video_feed(camera_id: int, target_label: str, db: Session = Depends(get_session)):
    """Endpoint to start the video feed and detect objects using hardware service."""
    # Ensure the model is loaded once before generating frames
    await load_model_once()
    
    # Reset stop event
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
    """Capture a single frame using the hardware service with improved error handling."""
    await load_model_once()
    
    # Start camera via hardware service
    camera_started = await start_camera_via_hardware_service(camera_id)
    if not camera_started:
        raise HTTPException(status_code=500, detail="Failed to start camera via hardware service")
    
    # Wait for camera to be ready
    camera_ready = await wait_for_camera_ready(camera_id)
    if not camera_ready:
        await stop_camera_via_hardware_service()
        raise HTTPException(status_code=500, detail="Camera did not become ready in time")
    
    try:
        # Capture frame from hardware service with retry logic
        frame = await capture_frame_from_hardware_with_retry(max_retries=5, retry_delay=1.0)
        if frame is None:
            raise HTTPException(status_code=500, detail="No frame captured from the hardware service after retries.")
        
        # Resize frame to standard size
        frame = cv2.resize(frame, (640, 480))
        
        # Detect object in the frame
        processed_frame, detected_target, non_target_count = await process_frame_async(frame, target_label)
        
        if detected_target:
            # Save the frame with the detected object
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"captured_{target_label}_{timestamp}_{user_id}.jpg"
            image_path = os.path.join(SAVE_DIR, image_name)
            
            # Encode and save the image as a JPEG file
            success, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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
        # Stop camera via hardware service
        await stop_camera_via_hardware_service()

@detection_router.get("/capture_image")
async def capture_image(camera_id: int, of: str, target_label: str, user_id: str, db: Session = Depends(get_session)):
    """Capture an image using the hardware service."""
    return await capture_single_frame_via_hardware(camera_id, target_label, user_id, of, db)

@detection_router.get("/health")
async def health_check():
    """Health check endpoint to verify detection service status."""
    try:
        # Check if hardware service is accessible
        hardware_status = await check_camera_status_safe()
        hardware_accessible = hardware_status is not None
        
        # Check if detection model is loaded
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