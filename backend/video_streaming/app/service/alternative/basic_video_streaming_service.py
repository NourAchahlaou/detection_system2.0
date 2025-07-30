# basic_video_streaming_service.py - Streaming with freeze capability for basic mode
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
    target_fps: int = 25

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
    """Stream state with freeze capability for detection"""
    
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
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.is_active = True
            
            # Start frame producer
            self.frame_producer_task = asyncio.create_task(self._frame_producer())
            
            logger.info(f"✅ Basic stream initialized for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize basic stream: {e}")
            await self.cleanup()
            raise
    
    async def _frame_producer(self):
        """Frame producer that can be frozen for detection"""
        logger.info(f"🎥 Starting basic frame producer for camera {self.config.camera_id}")
        
        consecutive_failures = 0
        
        try:
            while self.is_active and not self.stop_event.is_set():
                try:
                    # Connect to hardware service
                    async with self.session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                        if response.status != 200:
                            logger.error(f"Frame stream failed with status {response.status}")
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
                    logger.error(f"Error in basic frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.error("Too many consecutive failures, stopping producer")
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"🛑 Basic frame producer stopped for camera {self.config.camera_id}")
    
    async def _store_frame(self, jpeg_data: bytes):
        """Store frame for streaming (respects freeze state)"""
        try:
            # If frozen, don't update the latest frame
            if self.is_frozen:
                return
                
            # Decode and resize frame for efficiency
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None and frame.size > 0:
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Re-encode with specified quality
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.stream_quality]
                success, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                if success:
                    async with self.frame_lock:
                        self.latest_frame = buffer.tobytes()
                        self.frame_count += 1
                        self.last_frame_time = time.time()
                        
        except Exception as e:
            logger.error(f"Error storing frame: {e}")
    
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
            'stream_quality': self.config.stream_quality
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
                await asyncio.wait_for(self.frame_producer_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
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