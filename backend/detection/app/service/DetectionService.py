from detection.app.service.DetectionSessionManager import DetectionSessionManager
import cv2
import aiohttp
import asyncio

import logging

import numpy as np
from typing import Annotated, Optional, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import time
from datetime import datetime

import os

from concurrent.futures import ThreadPoolExecutor
import queue

from detection.app.service.detection_service import DetectionSystem
from detection.app.db.session import get_session
from detection.app.service.hardwareServiceClient import CameraClient
from detection.app.service.camera import CameraService

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

detection_router = APIRouter(
    prefix="/detection",
    tags=["Detection Processing"],
    responses={404: {"description": "Not found"}},
)

db_dependency = Annotated[Session, Depends(get_session)]

# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"
SAVE_DIR = "captured_images"

# Initialize services
hardware_client = CameraClient(base_url=HARDWARE_SERVICE_URL)
camera_service = CameraService()

# Global detection system
detection_system = None

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=3)

# Active detection sessions
active_detections = {}
detection_locks = {}

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class DetectionResult(BaseModel):
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    confidence_scores: Dict[str, float]
    bounding_boxes: list
    timestamp: float


class DetectionSession(BaseModel):
    session_id: str
    camera_id: int
    target_label: str
    detection_frequency: float = 5.0  # Hz
    confidence_threshold: float = 0.5
    save_detections: bool = False
    user_id: Optional[str] = None

session_manager = DetectionSessionManager()

async def load_model_once():
    """Load the detection model once when needed."""
    global detection_system
    if detection_system is None:
        detection_system = DetectionSystem()
        detection_system.get_my_model()
        logger.info("Detection model loaded successfully")

def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
    """Optimized frame resizing."""
    if frame.shape[:2] != target_size[::-1]:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return frame

def process_frame_sync(frame: np.ndarray, target_label: str, confidence_threshold: float = 0.5):
    """Synchronous frame processing for thread pool."""
    try:
        # Ensure frame is contiguous
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        # Temporarily adjust confidence threshold if needed
        original_threshold = detection_system.confidence_threshold
        detection_system.confidence_threshold = confidence_threshold
        
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
        finally:
            # Restore original threshold
            detection_system.confidence_threshold = original_threshold
        
        return processed_frame, detected_target, non_target_count
    
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
        return frame, False, 0

async def process_frame_async(frame: np.ndarray, target_label: str, confidence_threshold: float = 0.5):
    """Asynchronously process frame using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_frame_sync, frame, target_label, confidence_threshold)

async def capture_frame_from_hardware() -> Optional[np.ndarray]:
    """Capture a single frame from hardware service."""
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                if response.status == 200:
                    buffer = bytearray()
                    chunk_size = 16384
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        buffer.extend(chunk)
                        
                        # Look for complete JPEG frame
                        jpeg_start = buffer.find(b'\xff\xd8')
                        if jpeg_start != -1:
                            jpeg_end = buffer.find(b'\xff\xd9', jpeg_start)
                            if jpeg_end != -1:
                                jpeg_data = bytes(buffer[jpeg_start:jpeg_end+2])
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    return frame
                        
                        if len(buffer) > 200000:  # 200KB limit
                            break
                
                return None
    except Exception as e:
        logger.debug(f"Frame capture failed: {e}")
        return None

async def detection_processing_loop(session: DetectionSession, db: Session):
    """Main detection processing loop for a session."""
    session_id = session.session_id
    stop_event = session_manager.stop_events[session_id]
    results_queue = session_manager.results_queues[session_id]
    
    logger.info(f"Starting detection processing for session {session_id}")
    
    frame_interval = 1.0 / session.detection_frequency
    detection_counter = 0
    
    try:
        while not stop_event.is_set():
            start_time = time.time()
            
            try:
                # Capture frame
                frame = await capture_frame_from_hardware()
                
                if frame is None:
                    logger.debug("No frame captured, skipping detection cycle")
                    await asyncio.sleep(frame_interval)
                    continue
                
                # Resize frame
                frame = resize_frame_optimized(frame, (640, 480))
                
                # Process detection
                processed_frame, detected_target, non_target_count = await process_frame_async(
                    frame, session.target_label, session.confidence_threshold
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Create detection result
                result = DetectionResult(
                    detected_target=detected_target,
                    non_target_count=non_target_count,
                    processing_time_ms=processing_time,
                    confidence_scores={},  # Can be extended to include actual scores
                    bounding_boxes=[],     # Can be extended to include bounding box data
                    timestamp=time.time()
                )
                
                # Save detection if requested and target detected
                if session.save_detections and detected_target:
                    await save_detection_image(processed_frame, session, detection_counter)
                
                # Store result in queue (non-blocking)
                try:
                    results_queue.put_nowait({
                        'result': result,
                        'processed_frame': processed_frame,
                        'detection_id': f"{session_id}_{detection_counter}"
                    })
                except queue.Full:
                    # Remove oldest result and add new one
                    try:
                        results_queue.get_nowait()
                        results_queue.put_nowait({
                            'result': result,
                            'processed_frame': processed_frame,
                            'detection_id': f"{session_id}_{detection_counter}"
                        })
                    except queue.Empty:
                        pass
                
                detection_counter += 1
                
                if detected_target:
                    logger.info(f"Target '{session.target_label}' detected in session {session_id}")
                
                # Maintain detection frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in detection loop for session {session_id}: {e}")
                await asyncio.sleep(frame_interval)
    
    except Exception as e:
        logger.error(f"Fatal error in detection processing loop for session {session_id}: {e}")
    finally:
        logger.info(f"Detection processing loop ended for session {session_id}")

async def save_detection_image(frame: np.ndarray, session: DetectionSession, counter: int):
    """Save detected image to disk."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"detection_{session.target_label}_{timestamp}_{session.session_id}_{counter}.jpg"
        image_path = os.path.join(SAVE_DIR, image_name)
        
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            with open(image_path, 'wb') as f:
                f.write(buffer.tobytes())
            logger.info(f"Detection image saved: {image_path}")
    except Exception as e:
        logger.error(f"Error saving detection image: {e}")
