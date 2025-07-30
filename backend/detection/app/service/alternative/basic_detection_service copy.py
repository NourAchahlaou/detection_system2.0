# redis_basic_detection_service.py - Redis-coordinated basic detection with stream management
import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any
import time
import base64
import asyncio
import redis.asyncio as async_redis
import redis as sync_redis
import pickle
from dataclasses import dataclass

# Import your detection system (adapt to your actual import)
from detection.app.service.detection_service import DetectionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = "redis"
REDIS_PORT = 6379

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

@dataclass
class StreamCommand:
    """Stream control command structure"""
    camera_id: int
    command: str  # 'freeze', 'unfreeze', 'get_frame', 'status'
    timestamp: float
    session_id: str

@dataclass
class StreamResponse:
    """Stream control response structure"""
    camera_id: int
    command: str
    success: bool
    data: Optional[Any]
    message: str
    timestamp: float

class RedisBasicDetectionProcessor:
    """Detection processor that coordinates with video streaming via Redis"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        self.stats = {
            'detections_performed': 0,
            'targets_detected': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'redis_commands_sent': 0,
            'stream_operations': 0
        }
        
        # Redis clients
        self.async_redis_client = None
        self.sync_redis_client = None
        
        # Stream coordination
        self.active_sessions = {}  # session_id -> camera_id mapping
    
    async def initialize(self):
        """Initialize detection system and Redis connections"""
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing Redis-based basic detection system...")
                
                # Initialize Redis connections
                await self._initialize_redis()
                
                # Initialize detection system in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, DetectionSystem
                )
                
                self.is_initialized = True
                logger.info(f"✅ Redis-based basic detection system initialized on device: {self.detection_system.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis-based basic detection system: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connections"""
        try:
            # Async Redis client for stream operations
            self.async_redis_client = async_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
                socket_connect_timeout=10,
                socket_timeout=30,
                max_connections=20
            )
            
            # Sync Redis client for quick operations
            self.sync_redis_client = sync_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=10
            )
            
            # Test both connections
            await self.async_redis_client.ping()
            self.sync_redis_client.ping()
            
            logger.info("✅ Redis connections established successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis connections: {e}")
            raise
    
    async def detect_on_current_frame(self, request: DetectionRequest) -> DetectionResponse:
        """
        Perform detection on current frame from video stream via Redis coordination
        """
        start_time = time.time()
        stream_frozen = False
        session_id = f"detection_{request.camera_id}_{int(time.time() * 1000)}"
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting Redis-coordinated detection for camera {request.camera_id}, target: '{request.target_label}'")
            
            # Step 1: Send freeze command to video streaming service
            logger.info(f"🧊 Sending freeze command for camera {request.camera_id}")
            freeze_response = await self._send_stream_command(
                camera_id=request.camera_id,
                command='freeze',
                session_id=session_id
            )
            
            if not freeze_response.success:
                raise Exception(f"Failed to freeze stream: {freeze_response.message}")
            
            stream_frozen = True
            logger.info(f"✅ Stream frozen for camera {request.camera_id}")
            
            # Step 2: Get current frame from the frozen stream
            logger.info(f"📸 Requesting current frame from camera {request.camera_id}")
            frame_response = await self._send_stream_command(
                camera_id=request.camera_id,
                command='get_frame',
                session_id=session_id
            )
            
            if not frame_response.success or not frame_response.data:
                raise Exception(f"Failed to get frame: {frame_response.message}")
            
            # Decode frame from Redis response
            frame_data = frame_response.data
            if isinstance(frame_data, str):
                # Base64 encoded
                frame_bytes = base64.b64decode(frame_data)
            else:
                # Raw bytes
                frame_bytes = frame_data
            
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise Exception("Failed to decode frame from Redis response")
            
            logger.info(f"✅ Received frame from camera {request.camera_id}: {frame.shape}")
            
            # Step 3: Perform detection
            logger.info(f"🎯 Running detection on frame from camera {request.camera_id}")
            
            # Ensure frame is the right size and format
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            
            # Ensure frame is contiguous for better performance
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
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
            
            # Step 4: Update the frozen frame with detection results via Redis
            if processed_frame is not None:
                logger.info(f"🎨 Updating frozen frame with detection overlay for camera {request.camera_id}")
                
                # Encode processed frame
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                
                if success:
                    # Send update command with processed frame
                    processed_frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    update_response = await self._send_stream_command(
                        camera_id=request.camera_id,
                        command='update_frozen_frame',
                        session_id=session_id,
                        data=processed_frame_b64
                    )
                    
                    if update_response.success:
                        logger.info(f"✅ Updated frozen frame with detection results for camera {request.camera_id}")
                    else:
                        logger.warning(f"⚠️ Failed to update frozen frame: {update_response.message}")
            
            # Step 5: Encode processed frame for response
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
            
            logger.info(f"✅ Redis-coordinated detection completed for camera {request.camera_id} in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in Redis-coordinated detection for camera {request.camera_id}: {e}")
            
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
        try:
            logger.info(f"🔥 Sending unfreeze command for camera {camera_id}")
            
            session_id = f"unfreeze_{camera_id}_{int(time.time() * 1000)}"
            response = await self._send_stream_command(
                camera_id=camera_id,
                command='unfreeze',
                session_id=session_id
            )
            
            if response.success:
                logger.info(f"✅ Stream unfrozen for camera {camera_id}")
                return True
            else:
                logger.error(f"❌ Failed to unfreeze stream: {response.message}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error unfreezing stream for camera {camera_id}: {e}")
            return False
    
    def get_stream_status(self, camera_id: int) -> Dict[str, Any]:
        """Get current stream status via Redis"""
        try:
            session_id = f"status_{camera_id}_{int(time.time() * 1000)}"
            
            # Use sync Redis for this quick operation
            command = StreamCommand(
                camera_id=camera_id,
                command='status',
                timestamp=time.time(),
                session_id=session_id
            )
            
            # Send command
            command_key = f"stream_command:{camera_id}"
            response_key = f"stream_response:{session_id}"
            
            # Store command
            self.sync_redis_client.setex(
                command_key, 
                10,  # 10 second TTL
                pickle.dumps(command)
            )
            
            # Wait for response with timeout
            timeout = 5.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                response_data = self.sync_redis_client.get(response_key)
                if response_data:
                    response = pickle.loads(response_data)
                    
                    if response.success and response.data:
                        return response.data
                    else:
                        return {
                            'camera_id': camera_id,
                            'stream_active': False,
                            'is_frozen': False,
                            'error': response.message
                        }
                
                time.sleep(0.1)
            
            # Timeout
            logger.warning(f"⚠️ Timeout waiting for stream status from camera {camera_id}")
            return {
                'camera_id': camera_id,
                'stream_active': False,
                'is_frozen': False,
                'error': 'Timeout waiting for response'
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting stream status for camera {camera_id}: {e}")
            return {
                'camera_id': camera_id,
                'stream_active': False,
                'is_frozen': False,
                'error': str(e)
            }
    
    async def _send_stream_command(self, camera_id: int, command: str, session_id: str, data: Any = None) -> StreamResponse:
        """Send command to video streaming service via Redis"""
        try:
            self.stats['redis_commands_sent'] += 1
            self.stats['stream_operations'] += 1
            
            # Create command
            stream_command = StreamCommand(
                camera_id=camera_id,
                command=command,
                timestamp=time.time(),
                session_id=session_id
            )
            
            # Add data if provided
            if data is not None:
                # Store data separately for large payloads
                data_key = f"stream_data:{session_id}"
                await self.async_redis_client.setex(data_key, 30, data)  # 30 second TTL
            
            # Send command
            command_key = f"stream_command:{camera_id}"
            response_key = f"stream_response:{session_id}"
            
            # Store command with TTL
            await self.async_redis_client.setex(
                command_key, 
                30,  # 30 second TTL
                pickle.dumps(stream_command)
            )
            
            logger.info(f"📤 Sent stream command '{command}' for camera {camera_id} (session: {session_id})")
            
            # Wait for response
            timeout = 10.0  # 10 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                response_data = await self.async_redis_client.get(response_key)
                if response_data:
                    response = pickle.loads(response_data)
                    
                    # Clean up response key
                    await self.async_redis_client.delete(response_key)
                    if data is not None:
                        await self.async_redis_client.delete(f"stream_data:{session_id}")
                    
                    logger.info(f"📥 Received response for command '{command}' from camera {camera_id}: {response.success}")
                    return response
                
                await asyncio.sleep(0.1)
            
            # Timeout
            logger.error(f"⏰ Timeout waiting for response to command '{command}' from camera {camera_id}")
            return StreamResponse(
                camera_id=camera_id,
                command=command,
                success=False,
                data=None,
                message=f"Timeout waiting for response after {timeout}s",
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Error sending stream command '{command}' to camera {camera_id}: {e}")
            return StreamResponse(
                camera_id=camera_id,
                command=command,
                success=False,
                data=None,
                message=f"Error: {str(e)}",
                timestamp=time.time()
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        redis_connected = False
        try:
            if self.sync_redis_client:
                self.sync_redis_client.ping()
                redis_connected = True
        except:
            redis_connected = False
        
        return {
            'is_initialized': self.is_initialized,
            'device': str(self.detection_system.device) if self.detection_system else "unknown",
            'redis_connected': redis_connected,
            **self.stats
        }
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        try:
            if self.async_redis_client:
                await self.async_redis_client.aclose()
                self.async_redis_client = None
            
            if self.sync_redis_client:
                self.sync_redis_client.close()
                self.sync_redis_client = None
            
            logger.info("✅ Redis connections cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# Global Redis-based detection processor
redis_basic_detection_processor = RedisBasicDetectionProcessor()