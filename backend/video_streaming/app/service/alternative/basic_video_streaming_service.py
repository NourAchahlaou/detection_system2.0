# basic_video_streaming_service.py - Simple streaming for low-performance mode
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
    """Lightweight stream manager for basic mode"""
    
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

class BasicStreamState:
    """Simple stream state for basic mode"""
    
    def __init__(self, config: BasicStreamConfig):
        self.config = config
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.consumers = set()
        
        # Frame management
        self.latest_frame = None
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
        """Simple frame producer for basic streaming"""
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
                                    # Store frame for streaming
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
        """Store frame for streaming"""
        try:
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
        """Get the latest frame for streaming"""
        async with self.frame_lock:
            return self.latest_frame
    
    async def get_current_frame_for_detection(self) -> Optional[np.ndarray]:
        """Get current frame as numpy array for detection"""
        async with self.frame_lock:
            if self.latest_frame is not None:
                try:
                    nparr = np.frombuffer(self.latest_frame, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return frame
                except Exception as e:
                    logger.error(f"Error decoding frame for detection: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic stream statistics"""
        return {
            'camera_id': self.config.camera_id,
            'is_active': self.is_active,
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
        
        logger.info(f"✅ Basic stream cleanup complete for camera {self.config.camera_id}")

# Global basic stream manager
basic_stream_manager = BasicStreamManager()

