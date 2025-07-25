import cv2
import aiohttp
import asyncio
import logging
import numpy as np
from typing import AsyncGenerator, Optional, Dict
import time
import base64
from io import BytesIO
from collections import deque

# Your existing imports
from video_streaming.app.service.videoStremManager import VideoStreamManager
from video_streaming.app.service.hardwareServiceClient import CameraClient
from video_streaming.app.service.camera import CameraService

logger = logging.getLogger(__name__)

# Detection service configuration
DETECTION_SERVICE_URL = "http://detection:8000"  # Adjust to your detection service URL

class DetectionIntegration:
    """Handles communication with the detection service"""
    
    def __init__(self):
        self.detection_enabled = False
        self.target_label = None
        self.session = None
        
    async def init_session(self):
        """Initialize HTTP session for detection requests"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=5.0)  # 5 second timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def process_frame_with_detection(self, frame: np.ndarray, target_label: str) -> tuple:
        """
        Send frame to detection service for processing
        Returns: (processed_frame, detected_target, non_target_count, processing_time)
        """
        try:
            await self.init_session()
            
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return frame, False, 0, 0.0
            
            # Prepare multipart form data
            frame_bytes = buffer.tobytes()
            
            data = aiohttp.FormData()
            data.add_field('target_label', target_label)
            data.add_field('frame', frame_bytes, filename='frame.jpg', content_type='image/jpeg')
            
            # Send request to detection service
            async with self.session.post(f"{DETECTION_SERVICE_URL}/detection/process_frame", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Decode processed frame from base64
                    processed_frame_data = base64.b64decode(result['processed_frame'])
                    nparr = np.frombuffer(processed_frame_data, np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if processed_frame is None:
                        return frame, False, 0, 0.0
                    
                    return (
                        processed_frame,
                        result['detected_target'],
                        result['non_target_count'],
                        result['processing_time_ms']
                    )
                else:
                    logger.warning(f"Detection service returned status {response.status}")
                    return frame, False, 0, 0.0
                    
        except asyncio.TimeoutError:
            logger.warning("Detection service timeout")
            return frame, False, 0, 0.0
        except Exception as e:
            logger.error(f"Error calling detection service: {e}")
            return frame, False, 0, 0.0

# Enhanced OptimizedStreamState with detection integration
class OptimizedStreamStateWithDetection:
    """Enhanced stream state with optional detection integration"""
    def __init__(self, camera_id: int):
        # Your existing initialization code
        self.camera_id = camera_id
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.frame_count = 0
        self.last_frame_time = 0
        self.consecutive_failures = 0
        self.session = None
        
        # Frame buffering
        self.frame_buffer = deque(maxlen=5)
        self.buffer_lock = asyncio.Lock()
        self.latest_frame = None
        self.frame_ready_event = asyncio.Event()
        
        # Consumer management
        self.connection_task = None
        self.consumers = set()
        
        # Detection integration
        self.detection_integration = DetectionIntegration()
        self.detection_enabled = False
        self.target_label = None
        
    async def add_consumer(self, consumer_id: str):
        """Add a consumer to this stream"""
        self.consumers.add(consumer_id)
        logger.info(f"Added consumer {consumer_id} to camera {self.camera_id}. Total consumers: {len(self.consumers)}")
        
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"Removed consumer {consumer_id} from camera {self.camera_id}. Total consumers: {len(self.consumers)}")
        
        # If no consumers left, stop the stream
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for camera {self.camera_id}, stopping stream")
            await self.cleanup()
    
    async def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest frame for streaming"""
        if self.latest_frame is not None:
            return self.latest_frame
        
        # Wait for first frame if none available
        try:
            await asyncio.wait_for(self.frame_ready_event.wait(), timeout=1.0)
            return self.latest_frame
        except asyncio.TimeoutError:
            return None
        
    async def enable_detection(self, target_label: str):
        """Enable detection for this stream"""
        self.detection_enabled = True
        self.target_label = target_label
        logger.info(f"Detection enabled for camera {self.camera_id} with target: {target_label}")
    
    async def disable_detection(self):
        """Disable detection for this stream"""
        self.detection_enabled = False
        self.target_label = None
        await self.detection_integration.close_session()
        logger.info(f"Detection disabled for camera {self.camera_id}")
        
    async def add_frame_with_detection(self, frame: np.ndarray):
        """Add a frame to the buffer, optionally processing with detection"""
        processed_frame = frame
        
        # Process with detection if enabled
        if self.detection_enabled and self.target_label:
            try:
                processed_frame, detected_target, non_target_count, processing_time = \
                    await self.detection_integration.process_frame_with_detection(frame, self.target_label)
                
                if detected_target:
                    logger.info(f"Target '{self.target_label}' detected in camera {self.camera_id}")
                
            except Exception as e:
                logger.error(f"Detection processing failed: {e}")
                processed_frame = frame  # Fall back to original frame
        
        # Your existing frame buffering logic
        async with self.buffer_lock:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
            if buffer is not None:
                frame_bytes = buffer.tobytes()
                self.frame_buffer.append({
                    'data': frame_bytes,
                    'timestamp': time.time()
                })
                self.latest_frame = frame_bytes
                self.frame_ready_event.set()
                self.frame_count += 1
                self.last_frame_time = time.time()
    
    async def add_frame(self, frame: np.ndarray):
        """Regular add_frame method (for compatibility with existing code)"""
        await self.add_frame_with_detection(frame)
    
    async def cleanup(self):
        """Clean up stream resources including detection"""
        await self.disable_detection()
        
        # Your existing cleanup code
        self.is_active = False
        self.stop_event.set()
        
        if self.connection_task and not self.connection_task.done():
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
        
        if self.session and not self.session.closed:
            await self.session.close()
            
        self.consumers.clear()
        self.frame_buffer.clear()