# basic_detection_service.py - On-demand detection for low-performance mode
import cv2
import logging
import numpy as np
import aiohttp
from typing import Optional, Dict, Any
import time
import base64
from dataclasses import dataclass
import asyncio

# Import your detection system (adapt to your actual import)
from detection.app.service.detection_service import DetectionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"

@dataclass
class DetectionRequest:
    """Simple detection request structure"""
    camera_id: int
    target_label: str
    timestamp: float
    quality: int = 85

@dataclass
class DetectionResponse:
    """Simple detection response structure"""
    camera_id: int
    target_label: str
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    confidence: Optional[float]
    frame_with_overlay: str  # Base64 encoded image
    timestamp: float

class BasicDetectionProcessor:
    """Simple detection processor for on-demand detection"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        self.stats = {
            'detections_performed': 0,
            'targets_detected': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0
        }
    
    async def initialize(self):
        """Initialize detection system"""
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing basic detection system...")
                
                # Initialize detection system in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, DetectionSystem
                )
                
                self.is_initialized = True
                logger.info(f"✅ Basic detection system initialized on device: {self.detection_system.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize basic detection system: {e}")
            raise
    
    async def get_frame_from_camera(self, camera_id: int) -> Optional[np.ndarray]:
        """Capture a single frame from the camera"""
        try:
            logger.info(f"📸 Capturing frame from camera {camera_id}")
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                    if response.status != 200:
                        logger.error(f"Failed to get frame: HTTP {response.status}")
                        return None
                    
                    # Read first frame from stream
                    buffer = bytearray()
                    frame_found = False
                    
                    async for chunk in response.content.iter_chunked(8192):
                        buffer.extend(chunk)
                        
                        # Look for complete JPEG frame
                        jpeg_start = buffer.find(b'\xff\xd8')
                        if jpeg_start != -1:
                            jpeg_end = buffer.find(b'\xff\xd9', jpeg_start + 2)
                            if jpeg_end != -1:
                                jpeg_data = bytes(buffer[jpeg_start:jpeg_end + 2])
                                
                                # Decode frame
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    logger.info(f"✅ Captured frame from camera {camera_id}: {frame.shape}")
                                    return frame
                                
                                frame_found = True
                                break
                        
                        # Prevent excessive buffering
                        if len(buffer) > 50000:
                            buffer = buffer[-25000:]
                        
                        # Stop after reasonable attempt
                        if len(buffer) > 100000:
                            break
            
            if not frame_found:
                logger.warning(f"⚠️ No complete frame found from camera {camera_id}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error capturing frame from camera {camera_id}: {e}")
            return None
    
    async def detect_on_frame(self, request: DetectionRequest) -> DetectionResponse:
        """Perform detection on a single frame"""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting detection for camera {request.camera_id}, target: '{request.target_label}'")
            
            # Get frame from camera
            frame = await self.get_frame_from_camera(request.camera_id)
            if frame is None:
                raise Exception(f"Could not capture frame from camera {request.camera_id}")
            
            # Resize frame for processing efficiency
            frame = cv2.resize(frame, (640, 480))
            
            # Ensure frame is contiguous for better performance
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Perform detection using your detection system
            logger.info(f"🎯 Running detection on frame from camera {request.camera_id}")
            
            # Run detection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            detection_results = await loop.run_in_executor(
                None, 
                self.detection_system.detect_and_contour, 
                frame, 
                request.target_label
            )
            
            # Handle different return types from detection
            processed_frame = None
            detected_target = False
            non_target_count = 0
            confidence = None
            
            if isinstance(detection_results, tuple):
                processed_frame = detection_results[0]  # Frame with overlays
                detected_target = detection_results[1] if len(detection_results) > 1 else False
                non_target_count = detection_results[2] if len(detection_results) > 2 else 0
                confidence = detection_results[3] if len(detection_results) > 3 else None
            else:
                # If single return value, it's the processed frame
                processed_frame = detection_results
                detected_target = False
            
            # Encode processed frame as base64 for response
            frame_b64 = ""
            if processed_frame is not None:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        logger.info(f"✅ Encoded processed frame: {len(frame_b64)} chars")
                    else:
                        logger.error("❌ Failed to encode processed frame")
                except Exception as e:
                    logger.error(f"❌ Error encoding frame: {e}")
            else:
                # Fallback: encode original frame
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        logger.warning("⚠️ Using original frame (no overlay available)")
                except Exception as e:
                    logger.error(f"❌ Error encoding original frame: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.stats['detections_performed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['detections_performed']
            
            if detected_target:
                self.stats['targets_detected'] += 1
                logger.info(f"🎯 TARGET DETECTED: '{request.target_label}' found in camera {request.camera_id}!")
            else:
                logger.info(f"🔍 Detection complete: No '{request.target_label}' found in camera {request.camera_id}")
            
            response = DetectionResponse(
                camera_id=request.camera_id,
                target_label=request.target_label,
                detected_target=detected_target,
                non_target_count=non_target_count,
                processing_time_ms=round(processing_time, 2),
                confidence=confidence,
                frame_with_overlay=frame_b64,
                timestamp=time.time()
            )
            
            logger.info(f"✅ Detection completed for camera {request.camera_id} in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in detection for camera {request.camera_id}: {e}")
            
            # Return error response
            return DetectionResponse(
                camera_id=request.camera_id,
                target_label=request.target_label,
                detected_target=False,
                non_target_count=0,
                processing_time_ms=round(processing_time, 2),
                confidence=None,
                frame_with_overlay="",
                timestamp=time.time()
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'is_initialized': self.is_initialized,
            'device': str(self.detection_system.device) if self.detection_system else "unknown",
            **self.stats
        }

# Global basic detection processor
basic_detection_processor = BasicDetectionProcessor()
