# basic_video_streaming_service.py - Polling approach for reliable frame capture
import cv2
import asyncio
import logging
import numpy as np
import aiohttp
from typing import Optional, Dict, Any
import time
import uuid
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"

@dataclass
class BasicStreamConfig:
    """Simple configuration for basic streaming"""
    camera_id: int
    stream_quality: int = 85
    target_fps: int = 15  # Reduced for polling approach

class BasicStreamManager:
    """Lightweight stream manager for basic mode with freeze capability"""
    
    def __init__(self):
        self.active_streams: Dict[str, 'BasicStreamState'] = {}
        self.stats = {
            'total_frames_streamed': 0,
            'active_streams_count': 0
        }
    
    async def create_stream(self, config: BasicStreamConfig) -> str:
        """Create a new basic stream"""
        stream_key = f"basic_{config.camera_id}_{uuid.uuid4().hex[:8]}"
        
        if stream_key not in self.active_streams:
            stream_state = BasicStreamState(config)
            await stream_state.initialize()
            self.active_streams[stream_key] = stream_state
            self.stats['active_streams_count'] = len(self.active_streams)
            
            logger.info(f"✅ Created basic stream: {stream_key}")
        
        return stream_key
    
    async def remove_stream(self, stream_key: str):
        """Remove and cleanup stream"""
        if stream_key in self.active_streams:
            await self.active_streams[stream_key].cleanup()
            del self.active_streams[stream_key]
            self.stats['active_streams_count'] = len(self.active_streams)
            logger.info(f"Removed basic stream: {stream_key}")
    
    def get_stream(self, stream_key: str) -> Optional['BasicStreamState']:
        """Get stream by key"""
        return self.active_streams.get(stream_key)
    
    def get_stream_by_camera_id(self, camera_id: int) -> Optional['BasicStreamState']:
        """Get stream by camera ID"""
        for stream_state in self.active_streams.values():
            if stream_state.config.camera_id == camera_id:
                return stream_state
        return None

class BasicStreamState:
    """Stream state with freeze capability using polling approach"""
    
    def __init__(self, config: BasicStreamConfig):
        self.config = config
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.consumers = set()
        
        # Frame management with freeze capability
        self.latest_frame = None
        self.frozen_frame = None  # Frame to display when frozen
        self.is_frozen = False
        self.frame_lock = asyncio.Lock()
        self.last_frame_time = 0
        self.frame_count = 0
        
        # HTTP session for camera communication
        self.session = None
        self.frame_producer_task = None
    
    async def initialize(self):
        """Initialize basic stream"""
        try:
            logger.info(f"🚀 Initializing basic stream for camera {self.config.camera_id}")
            
            # Create HTTP session with reasonable timeouts
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test connection to hardware service
            try:
                async with self.session.get(f"{HARDWARE_SERVICE_URL}/camera/frame_info") as response:
                    if response.status == 200:
                        info = await response.json()
                        logger.info(f"📡 Hardware service info: {info}")
                    else:
                        logger.warning(f"Hardware service responded with status {response.status}")
            except Exception as e:
                logger.warning(f"Could not get hardware service info: {e}")
            
            self.is_active = True
            
            # Start frame producer
            self.frame_producer_task = asyncio.create_task(self._frame_producer_polling())
            
            logger.info(f"✅ Basic stream initialized for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize basic stream: {e}")
            await self.cleanup()
            raise
    
    async def _frame_producer_polling(self):
        """Frame producer using polling approach - much more reliable"""
        logger.info(f"🎥 Starting polling-based frame producer for camera {self.config.camera_id}")
        
        consecutive_failures = 0
        max_failures = 10
        poll_interval = 1.0 / self.config.target_fps  # Convert FPS to seconds
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
                            logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), stopping polling")
                            break
                        
                        # Exponential backoff for failures
                        await asyncio.sleep(min(2.0, 0.1 * consecutive_failures))
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in polling frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"🛑 Polling frame producer stopped for camera {self.config.camera_id}")
    
    async def _poll_single_frame(self) -> Optional[bytes]:
        """Poll for a single frame from the hardware service"""
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
            
            # Optional: resize frame for efficiency
            if self.config.stream_quality < 85:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Resize if needed
                    height, width = frame.shape[:2]
                    if width > 800:
                        frame = cv2.resize(frame, (800, 600))
                    
                    # Re-encode with specified quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.stream_quality]
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    
                    if success:
                        frame_bytes = buffer.tobytes()
            
            async with self.frame_lock:
                self.latest_frame = frame_bytes
                self.frame_count += 1
                self.last_frame_time = time.time()
                
                # Log progress
                if self.frame_count % 50 == 0:
                    logger.info(f"📦 Stored {self.frame_count} frames for camera {self.config.camera_id}")
                        
        except Exception as e:
            logger.debug(f"Error storing frame: {e}")
    
    async def freeze_stream(self):
        """Freeze the stream to current frame"""
        async with self.frame_lock:
            if self.latest_frame is not None:
                self.frozen_frame = self.latest_frame
                self.is_frozen = True
                logger.info(f"🧊 Stream frozen for camera {self.config.camera_id}")
                return True
            else:
                logger.warning(f"⚠️ No frame available to freeze for camera {self.config.camera_id}")
                return False
    
    async def unfreeze_stream(self):
        """Unfreeze the stream to resume live feed"""
        async with self.frame_lock:
            self.is_frozen = False
            self.frozen_frame = None
            logger.info(f"🔥 Stream unfrozen for camera {self.config.camera_id}")
    
    async def update_frozen_frame(self, new_frame_bytes: bytes):
        """Update the frozen frame with detection results"""
        async with self.frame_lock:
            if self.is_frozen:
                self.frozen_frame = new_frame_bytes
                logger.info(f"🎯 Updated frozen frame with detection results for camera {self.config.camera_id}")
    
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
        logger.info(f"👤 Added consumer {consumer_id} to basic camera {self.config.camera_id}. Total: {len(self.consumers)}")
    
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"👤 Removed consumer {consumer_id} from basic camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # Stop stream if no consumers
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for basic camera {self.config.camera_id}, stopping stream")
            await self.cleanup()
    
    async def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest frame for streaming (frozen or live)"""
        async with self.frame_lock:
            if self.is_frozen and self.frozen_frame is not None:
                return self.frozen_frame
            return self.latest_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic stream statistics"""
        return {
            'camera_id': self.config.camera_id,
            'is_active': self.is_active,
            'is_frozen': self.is_frozen,
            'consumers_count': len(self.consumers),
            'frames_processed': self.frame_count,
            'last_frame_time': self.last_frame_time,
            'stream_quality': self.config.stream_quality,
            'target_fps': self.config.target_fps
        }
    
    async def cleanup(self):
        """Cleanup stream resources"""
        logger.info(f"🧹 Cleaning up basic stream for camera {self.config.camera_id}")
        
        self.is_active = False
        self.stop_event.set()
        
        # Cancel frame producer task
        if self.frame_producer_task and not self.frame_producer_task.done():
            self.frame_producer_task.cancel()
            try:
                await asyncio.wait_for(self.frame_producer_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning(f"Frame producer task cleanup timeout for camera {self.config.camera_id}")
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Clear consumers and frames
        self.consumers.clear()
        async with self.frame_lock:
            self.latest_frame = None
            self.frozen_frame = None
            self.is_frozen = False
        
        logger.info(f"✅ Basic stream cleanup complete for camera {self.config.camera_id}")

# Global basic stream manager
basic_stream_manager = BasicStreamManager()