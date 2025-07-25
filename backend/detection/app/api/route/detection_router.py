# detection_service_refactored.py
# Remove all video streaming logic, focus only on detection

import cv2
import asyncio
import logging
import numpy as np
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
import time
from datetime import datetime
import base64
from concurrent.futures import ThreadPoolExecutor

from detection.app.service.detection_service import DetectionSystem
from detection.app.db.session import get_session

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

detection_router = APIRouter(
    prefix="/detection",
    tags=["Detection"],
    responses={404: {"description": "Not found"}},
)

class FrameProcessingResponse(BaseModel):
    processed_frame: str  # Base64 encoded frame
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    detection_confidence: Optional[float] = None
    bounding_boxes: Optional[list] = None

class DetectionHealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_type: str
    timestamp: str

# Global variables - only for detection
detection_system = None
executor = ThreadPoolExecutor(max_workers=2)

async def load_model_once():
    """Load the model once when the application starts."""
    global detection_system
    if detection_system is None:
        detection_system = DetectionSystem()
        detection_system.get_my_model()
        logger.info("Detection model loaded successfully")

@detection_router.get("/load_model")
async def load_model_endpoint():
    """Endpoint to load the model."""
    try:
        await load_model_once()
        return {"message": "Model loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

def process_frame_sync(frame: np.ndarray, target_label: str):
    """Synchronous frame processing to run in thread pool."""
    try:
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
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
        return frame, False, 0

async def process_frame_async(frame: np.ndarray, target_label: str):
    """Asynchronously process a single frame using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_frame_sync, frame, target_label)

@detection_router.post("/process_frame", response_model=FrameProcessingResponse)
async def process_frame_endpoint(
    frame: UploadFile = File(..., description="Frame image file"),
    target_label: str = Form(..., description="Target label to detect")
):
    """
    Process a single frame for object detection.
    This is the main endpoint called by the video streaming service.
    """
    start_time = time.time()
    
    try:
        # Ensure model is loaded
        await load_model_once()
        
        # Read and validate frame data
        frame_data = await frame.read()
        if not frame_data:
            raise HTTPException(status_code=400, detail="Empty frame data")
        
        # Decode frame
        nparr = np.frombuffer(frame_data, np.uint8)
        cv_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
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

@detection_router.get("/health", response_model=DetectionHealthResponse)
async def health_check():
    """Health check endpoint for detection service."""
    try:
        model_loaded = detection_system is not None
        device = str(detection_system.device) if detection_system else "unknown"
        
        return DetectionHealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            device=device,
            model_type="YOLO" if detection_system else "unknown",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return DetectionHealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            model_type="unknown",
            timestamp=datetime.now().isoformat()
        )

@detection_router.get("/model/status")
async def get_model_status():
    """Get detailed model status."""
    global detection_system
    
    return {
        "model_loaded": detection_system is not None,
        "device": str(detection_system.device) if detection_system else "unknown",
        "model_type": "YOLO" if detection_system else "unknown",
        "memory_usage": "N/A",  # Could add GPU memory usage here
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

# Remove all video streaming endpoints and hardware client logic
# This service now only handles detection!""