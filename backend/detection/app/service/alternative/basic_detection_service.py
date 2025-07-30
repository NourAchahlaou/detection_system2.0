# basic_detection_service.py - HTTP client approach for cross-container communication
import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any
import time
import base64
import aiohttp
import asyncio
from dataclasses import dataclass

# Import your detection system (adapt to your actual import)
from detection.app.service.detection_service import DetectionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for video streaming service
VIDEO_STREAMING_SERVICE_URL = "http://video_streaming:8000"  # Docker service name

@dataclass
class DetectionRequest:
    """Detection request structure"""
    camera_id: int
    target_label: str
    timestamp: float
    quality: int = 85

@dataclass
class DetectionResponse:
    """Detection response structure"""
    camera_id: int
    target_label: str
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    confidence: Optional[float]
    frame_with_overlay: str  # Base64 encoded image
    timestamp: float
    stream_frozen: bool

class VideoStreamingClient:
    """HTTP client to communicate with video streaming service"""
    
    def __init__(self, base_url: str = VIDEO_STREAMING_SERVICE_URL):
        self.base_url = base_url
        self.session = None
    
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def get_current_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get current frame from video streaming service"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/current-frame"
            
            async with session.get(url) as response:
                if response.status == 200:
                    frame_data = await response.read()
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return frame
                else:
                    logger.error(f"Failed to get frame: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current frame: {e}")
            return None
    
    async def freeze_stream(self, camera_id: int) -> bool:
        """Freeze video stream"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/freeze"
            
            async with session.post(url) as response:
                if response.status == 200:
                    logger.info(f"🧊 Stream frozen for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to freeze stream: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error freezing stream: {e}")
            return False
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze video stream"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/unfreeze"
            
            async with session.post(url) as response:
                if response.status == 200:
                    logger.info(f"🔥 Stream unfrozen for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to unfreeze stream: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error unfreezing stream: {e}")
            return False
    
    async def update_frozen_frame(self, camera_id: int, frame_bytes: bytes) -> bool:
        """Update frozen frame with detection results"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/update-frozen-frame"
            
            data = aiohttp.FormData()
            data.add_field('frame', frame_bytes, content_type='image/jpeg')
            
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    logger.info(f"✅ Updated frozen frame for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to update frozen frame: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating frozen frame: {e}")
            return False
    
    async def get_stream_status(self, camera_id: int) -> Dict[str, Any]:
        """Get stream status"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/status"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        'camera_id': camera_id,
                        'stream_active': False,
                        'is_frozen': False,
                        'error': f'HTTP {response.status}'
                    }
                    
        except Exception as e:
            logger.error(f"Error getting stream status: {e}")
            return {
                'camera_id': camera_id,
                'stream_active': False,
                'is_frozen': False,
                'error': str(e)
            }
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

class BasicDetectionProcessor:
    """Detection processor that communicates with video streaming service via HTTP"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        self.stats = {
            'detections_performed': 0,
            'targets_detected': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0
        }
        self.video_client = VideoStreamingClient()
    
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
    
    async def detect_on_current_frame(self, request: DetectionRequest) -> DetectionResponse:
        """
        Perform detection on current frame from video stream via HTTP
        """
        start_time = time.time()
        stream_frozen = False
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting detection for camera {request.camera_id}, target: '{request.target_label}'")
            
            # Freeze the stream first
            freeze_success = await self.video_client.freeze_stream(request.camera_id)
            if freeze_success:
                stream_frozen = True
                logger.info(f"🧊 Stream frozen for detection on camera {request.camera_id}")
            
            # Get current frame from video streaming service
            frame = await self.video_client.get_current_frame(request.camera_id)
            if frame is None:
                raise Exception(f"Could not get current frame from camera {request.camera_id}")
            
            # Ensure frame is the right size and format
            if frame.shape[:2] != (480, 640):
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
            
            # Update the frozen frame with detection results
            if processed_frame is not None and stream_frozen:
                try:
                    # Encode the processed frame
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        # Update the frozen frame in the video streaming service
                        await self.video_client.update_frozen_frame(request.camera_id, buffer.tobytes())
                        logger.info(f"✅ Updated frozen frame with detection results")
                except Exception as e:
                    logger.error(f"❌ Error updating frozen frame: {e}")
            
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
                timestamp=time.time(),
                stream_frozen=stream_frozen
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
                timestamp=time.time(),
                stream_frozen=stream_frozen
            )
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze the stream to resume live video"""
        return await self.video_client.unfreeze_stream(camera_id)
    
    def get_stream_status(self, camera_id: int) -> Dict[str, Any]:
        """Get current stream status via HTTP client"""
        # Note: This needs to be async, but keeping sync for compatibility
        # You might want to refactor this to be async
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a task
            task = asyncio.create_task(self.video_client.get_stream_status(camera_id))
            return {'note': 'Status check initiated', 'camera_id': camera_id}
        else:
            return loop.run_until_complete(self.video_client.get_stream_status(camera_id))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'is_initialized': self.is_initialized,
            'device': str(self.detection_system.device) if self.detection_system else "unknown",
            **self.stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.video_client.close()

# Global basic detection processor
basic_detection_processor = BasicDetectionProcessor()