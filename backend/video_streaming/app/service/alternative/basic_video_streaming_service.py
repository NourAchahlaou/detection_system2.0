# hybrid_video_streaming_service.py - Adaptive approach for both camera types
import cv2
import asyncio
import logging
import numpy as np
import aiohttp
from typing import Optional, Dict, Any
import time
import uuid
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"

class CameraType(Enum):
    REGULAR = "regular"
    BASLER = "basler"
    UNKNOWN = "unknown"

@dataclass
class HybridStreamConfig:
    """Configuration for hybrid streaming that adapts to camera type"""
    camera_id: int
    stream_quality: int = 85
    target_fps: int = 25  # Regular cameras can handle higher FPS
    polling_fps: int = 15  # Basler cameras work better with polling
    camera_type: CameraType = CameraType.UNKNOWN

class HybridStreamManager:
    """Stream manager that adapts to different camera types"""
    
    def __init__(self):
        self.active_streams: Dict[str, 'HybridStreamState'] = {}
        self.stats = {
            'total_frames_streamed': 0,
            'active_streams_count': 0,
            'regular_streams': 0,
            'basler_streams': 0
        }
    
    async def create_stream(self, config: HybridStreamConfig) -> str:
        """Create a new hybrid stream with camera type detection"""
        stream_key = f"hybrid_{config.camera_id}_{uuid.uuid4().hex[:8]}"
        
        if stream_key not in self.active_streams:
            # Detect camera type if not specified
            if config.camera_type == CameraType.UNKNOWN:
                config.camera_type = await self._detect_camera_type()
            
            stream_state = HybridStreamState(config)
            await stream_state.initialize()
            self.active_streams[stream_key] = stream_state
            self.stats['active_streams_count'] = len(self.active_streams)
            
            # Update camera type stats
            if config.camera_type == CameraType.BASLER:
                self.stats['basler_streams'] += 1
            else:
                self.stats['regular_streams'] += 1
            
            logger.info(f"✅ Created {config.camera_type.value} stream: {stream_key}")
        
        return stream_key
    
    async def _detect_camera_type(self) -> CameraType:
        """Detect camera type by querying hardware service"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{HARDWARE_SERVICE_URL}/camera/status") as response:
                    if response.status == 200:
                        status = await response.json()
                        camera_type_str = status.get('camera_type', 'unknown')
                        
                        if camera_type_str == 'basler':
                            return CameraType.BASLER
                        elif camera_type_str == 'regular':
                            return CameraType.REGULAR
                        
        except Exception as e:
            logger.warning(f"Could not detect camera type: {e}")
        
        return CameraType.REGULAR  # Default to regular if detection fails
    
    async def remove_stream(self, stream_key: str):
        """Remove and cleanup stream"""
        if stream_key in self.active_streams:
            stream_state = self.active_streams[stream_key]
            camera_type = stream_state.config.camera_type
            
            await stream_state.cleanup()
            del self.active_streams[stream_key]
            self.stats['active_streams_count'] = len(self.active_streams)
            
            # Update camera type stats
            if camera_type == CameraType.BASLER:
                self.stats['basler_streams'] = max(0, self.stats['basler_streams'] - 1)
            else:
                self.stats['regular_streams'] = max(0, self.stats['regular_streams'] - 1)
            
            logger.info(f"Removed {camera_type.value} stream: {stream_key}")
    
    def get_stream(self, stream_key: str) -> Optional['HybridStreamState']:
        """Get stream by key"""
        return self.active_streams.get(stream_key)
    
    def get_stream_by_camera_id(self, camera_id: int) -> Optional['HybridStreamState']:
        """Get stream by camera ID"""
        for stream_state in self.active_streams.values():
            if stream_state.config.camera_id == camera_id:
                return stream_state
        return None

class HybridStreamState:
    """Stream state that adapts its behavior based on camera type"""
    
    def __init__(self, config: HybridStreamConfig):
        self.config = config
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.consumers = set()
        
        # Frame management with freeze capability
        self.latest_frame = None
        self.frozen_frame = None
        self.is_frozen = False
        self.frame_lock = asyncio.Lock()
        self.last_frame_time = 0
        self.frame_count = 0
        
        # HTTP session for camera communication
        self.session = None
        self.frame_producer_task = None
    
    async def initialize(self):
        """Initialize stream with appropriate method for camera type"""
        try:
            logger.info(f"🚀 Initializing {self.config.camera_type.value} stream for camera {self.config.camera_id}")
            
            # Create HTTP session with appropriate timeout
            if self.config.camera_type == CameraType.BASLER:
                timeout = aiohttp.ClientTimeout(total=10, connect=5)  # Shorter timeout for polling
            else:
                timeout = aiohttp.ClientTimeout(total=None)  # No timeout for streaming
            
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.is_active = True
            
            # Start appropriate frame producer
            if self.config.camera_type == CameraType.BASLER:
                self.frame_producer_task = asyncio.create_task(self._basler_frame_producer())
            else:
                self.frame_producer_task = asyncio.create_task(self._regular_frame_producer())
            
            logger.info(f"✅ {self.config.camera_type.value} stream initialized for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.config.camera_type.value} stream: {e}")
            await self.cleanup()
            raise
    
    async def _basler_frame_producer(self):
        """Polling-based frame producer optimized for Basler cameras"""
        logger.info(f"🎥 Starting Basler polling producer for camera {self.config.camera_id}")
        
        consecutive_failures = 0
        max_failures = 10
        poll_interval = 1.0 / self.config.polling_fps
        last_poll_time = 0
        
        try:
            while self.is_active and not self.stop_event.is_set():
                current_time = time.time()
                
                # Rate limiting
                time_since_last_poll = current_time - last_poll_time
                if time_since_last_poll < poll_interval:
                    sleep_time = poll_interval - time_since_last_poll
                    await asyncio.sleep(sleep_time)
                
                # Skip if no consumers and not frozen
                if not self.consumers and not self.is_frozen:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Poll for a single frame
                    frame_bytes = await self._poll_single_frame()
                    
                    if frame_bytes is not None:
                        consecutive_failures = 0
                        
                        # Store frame (only if not frozen)
                        if not self.is_frozen:
                            await self._store_frame(frame_bytes)
                        
                        last_poll_time = time.time()
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > max_failures:
                            logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), stopping Basler polling")
                            break
                        
                        await asyncio.sleep(min(2.0, 0.1 * consecutive_failures))
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in Basler frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"🛑 Basler frame producer stopped for camera {self.config.camera_id}")
    
    async def _regular_frame_producer(self):
        """Streaming-based frame producer optimized for regular cameras"""
        logger.info(f"🎥 Starting regular streaming producer for camera {self.config.camera_id}")
        
        consecutive_failures = 0
        
        try:
            while self.is_active and not self.stop_event.is_set():
                try:
                    # Connect to hardware service streaming endpoint
                    async with self.session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                        if response.status != 200:
                            logger.error(f"Stream failed with status {response.status}")
                            consecutive_failures += 1
                            if consecutive_failures > 5:
                                break
                            await asyncio.sleep(1.0)
                            continue
                        
                        consecutive_failures = 0
                        buffer = bytearray()
                        
                        async for chunk in response.content.iter_chunked(8192):
                            if self.stop_event.is_set() or not self.consumers:
                                break
                                
                            buffer.extend(chunk)
                            
                            # Process complete JPEG frames
                            while True:
                                jpeg_start = buffer.find(b'\xff\xd8')
                                if jpeg_start == -1:
                                    break
                                
                                jpeg_end = buffer.find(b'\xff\xd9', jpeg_start + 2)
                                if jpeg_end == -1:
                                    break
                                
                                jpeg_data = bytes(buffer[jpeg_start:jpeg_end + 2])
                                buffer = buffer[jpeg_end + 2:]
                                
                                try:
                                    # Store frame (only if not frozen)
                                    if not self.is_frozen:
                                        await self._store_frame(jpeg_data)
                                except Exception as e:
                                    logger.debug(f"Frame processing error: {e}")
                                    
                            # Prevent buffer from growing too large
                            if len(buffer) > 100000:
                                buffer = buffer[-50000:]
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in regular frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.error("Too many consecutive failures, stopping regular producer")
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"🛑 Regular frame producer stopped for camera {self.config.camera_id}")
    
    async def _poll_single_frame(self) -> Optional[bytes]:
        """Poll for a single frame (used by Basler cameras)"""
        try:
            url = f"{HARDWARE_SERVICE_URL}/camera/single_frame"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    frame_bytes = await response.read()
                    
                    # Validate frame
                    if len(frame_bytes) > 1000:  # Reasonable minimum size for JPEG
                        return frame_bytes
                    else:
                        logger.debug(f"Frame too small: {len(frame_bytes)} bytes")
                        return None
                else:
                    logger.debug(f"Frame poll failed with status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.debug("Frame poll timeout")
            return None
        except Exception as e:
            logger.debug(f"Frame poll error: {e}")
            return None
    
    async def _store_frame(self, frame_bytes: bytes):
        """Store frame for streaming (respects freeze state)"""
        try:
            # If frozen, don't update the latest frame
            if self.is_frozen:
                return
            
            # Optional: resize frame for efficiency based on camera type
            if self.config.stream_quality < 85 or self.config.camera_type == CameraType.BASLER:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Resize if needed (Basler cameras might need different sizing)
                    height, width = frame.shape[:2]
                    if self.config.camera_type == CameraType.BASLER and width > 800:
                        frame = cv2.resize(frame, (1920, 1152))
                    elif self.config.camera_type == CameraType.REGULAR and width > 1024:
                        frame = cv2.resize(frame, (1920, 1152))
                    
                    # Re-encode with specified quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.stream_quality]
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    
                    if success:
                        frame_bytes = buffer.tobytes()
            
            async with self.frame_lock:
                self.latest_frame = frame_bytes
                self.frame_count += 1
                self.last_frame_time = time.time()
                
                # Log progress with camera type info
                if self.frame_count % 100 == 0:
                    logger.info(f"📦 Stored {self.frame_count} frames for {self.config.camera_type.value} camera {self.config.camera_id}")
                        
        except Exception as e:
            logger.debug(f"Error storing frame: {e}")
    
    async def freeze_stream(self):
        """Freeze the stream to current frame"""
        async with self.frame_lock:
            if self.latest_frame is not None:
                self.frozen_frame = self.latest_frame
                self.is_frozen = True
                logger.info(f"🧊 {self.config.camera_type.value} stream frozen for camera {self.config.camera_id}")
                return True
            else:
                logger.warning(f"⚠️ No frame available to freeze for {self.config.camera_type.value} camera {self.config.camera_id}")
                return False
    
    async def unfreeze_stream(self):
        """Unfreeze the stream to resume live feed"""
        async with self.frame_lock:
            self.is_frozen = False
            self.frozen_frame = None
            logger.info(f"🔥 {self.config.camera_type.value} stream unfrozen for camera {self.config.camera_id}")
    
    async def update_frozen_frame(self, new_frame_bytes: bytes):
        """Update the frozen frame with detection results"""
        async with self.frame_lock:
            if self.is_frozen:
                self.frozen_frame = new_frame_bytes
                logger.info(f"🎯 Updated frozen frame with detection results for {self.config.camera_type.value} camera {self.config.camera_id}")
    
    async def get_current_frame_for_detection(self) -> Optional[np.ndarray]:
        """Get current frame as numpy array for detection"""
        async with self.frame_lock:
            frame_to_use = self.latest_frame
            
        if frame_to_use is not None:
            try:
                nparr = np.frombuffer(frame_to_use, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
            except Exception as e:
                logger.error(f"Error decoding frame for detection: {e}")
        return None
    
    async def add_consumer(self, consumer_id: str):
        """Add a consumer to this stream"""
        self.consumers.add(consumer_id)
        logger.info(f"👤 Added consumer {consumer_id} to {self.config.camera_type.value} camera {self.config.camera_id}. Total: {len(self.consumers)}")
    
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"👤 Removed consumer {consumer_id} from {self.config.camera_type.value} camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # Stop stream if no consumers
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for {self.config.camera_type.value} camera {self.config.camera_id}, stopping stream")
            await self.cleanup()
    
    async def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest frame for streaming (frozen or live)"""
        async with self.frame_lock:
            if self.is_frozen and self.frozen_frame is not None:
                return self.frozen_frame
            return self.latest_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        return {
            'camera_id': self.config.camera_id,
            'camera_type': self.config.camera_type.value,
            'is_active': self.is_active,
            'is_frozen': self.is_frozen,
            'consumers_count': len(self.consumers),
            'frames_processed': self.frame_count,
            'last_frame_time': self.last_frame_time,
            'stream_quality': self.config.stream_quality,
            'target_fps': self.config.target_fps if self.config.camera_type == CameraType.REGULAR else self.config.polling_fps
        }
    
    async def cleanup(self):
        """Cleanup stream resources"""
        logger.info(f"🧹 Cleaning up {self.config.camera_type.value} stream for camera {self.config.camera_id}")
        
        self.is_active = False
        self.stop_event.set()
        
        # Cancel frame producer task
        if self.frame_producer_task and not self.frame_producer_task.done():
            self.frame_producer_task.cancel()
            try:
                await asyncio.wait_for(self.frame_producer_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning(f"Frame producer task cleanup timeout for {self.config.camera_type.value} camera {self.config.camera_id}")
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Clear consumers and frames
        self.consumers.clear()
        async with self.frame_lock:
            self.latest_frame = None
            self.frozen_frame = None
            self.is_frozen = False
        
        logger.info(f"✅ {self.config.camera_type.value} stream cleanup complete for camera {self.config.camera_id}")

# Global hybrid stream manager
hybrid_stream_manager = HybridStreamManager()