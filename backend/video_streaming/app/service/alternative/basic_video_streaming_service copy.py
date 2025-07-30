# redis_basic_video_streaming_service.py - Redis-coordinated streaming with freeze capability
import cv2
import asyncio
import logging
import numpy as np
import aiohttp
import redis.asyncio as async_redis
import redis as sync_redis
import pickle
import base64
from typing import Optional, Dict, Any
import time
import uuid
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"
REDIS_HOST = "redis"
REDIS_PORT = 6379

@dataclass
class BasicStreamConfig:
    """Simple configuration for basic streaming"""
    camera_id: int
    stream_quality: int = 85
    target_fps: int = 25

@dataclass
class StreamCommand:
    """Stream control command structure"""
    camera_id: int
    command: str  # 'freeze', 'unfreeze', 'get_frame', 'status', 'update_frozen_frame'
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

class RedisStreamCoordinator:
    """Handles Redis communication for stream commands"""
    
    def __init__(self):
        self.async_redis_client = None
        self.sync_redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.command_listener_task = None
        self.is_running = False
    
    async def initialize(self):
        """Initialize Redis connections"""
        try:
            # Async Redis client
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
            
            # Sync Redis client
            self.sync_redis_client = sync_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=10
            )
            
            # Test connections
            await self.async_redis_client.ping()
            self.sync_redis_client.ping()
            
            logger.info("✅ Redis stream coordinator connections established")
            
            # Start command listener
            self.is_running = True
            self.command_listener_task = asyncio.create_task(self._command_listener())
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis stream coordinator: {e}")
            raise
    
    async def _command_listener(self):
        """Listen for stream commands from detection service"""
        logger.info("🔊 Starting Redis stream command listener")
        
        try:
            while self.is_running:
                try:
                    # Check for commands for all active cameras
                    for stream_key, stream_state in redis_basic_stream_manager.active_streams.items():
                        camera_id = stream_state.config.camera_id
                        await self._check_camera_commands(camera_id, stream_state)
                    
                    await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Error in command listener: {e}")
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in command listener: {e}")
        finally:
            logger.info("🛑 Redis stream command listener stopped")
    
    async def _check_camera_commands(self, camera_id: int, stream_state: 'RedisBasicStreamState'):
        """Check for commands for a specific camera"""
        try:
            command_key = f"stream_command:{camera_id}"
            command_data = await self.async_redis_client.get(command_key)
            
            if command_data:
                try:
                    command = pickle.loads(command_data)
                    logger.info(f"📥 Received command '{command.command}' for camera {camera_id}")
                    
                    # Process command
                    response = await self._process_command(command, stream_state)
                    
                    # Send response
                    response_key = f"stream_response:{command.session_id}"
                    await self.async_redis_client.setex(
                        response_key, 
                        30,  # 30 second TTL
                        pickle.dumps(response)
                    )
                    
                    # Clean up command
                    await self.async_redis_client.delete(command_key)
                    
                    logger.info(f"📤 Sent response for command '{command.command}' from camera {camera_id}: {response.success}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing command for camera {camera_id}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error checking commands for camera {camera_id}: {e}")
    
    async def _process_command(self, command: StreamCommand, stream_state: 'RedisBasicStreamState') -> StreamResponse:
        """Process a stream command"""
        try:
            if command.command == 'freeze':
                success = await stream_state.freeze_stream()
                return StreamResponse(
                    camera_id=command.camera_id,
                    command=command.command,
                    success=success,
                    data=None,
                    message="Stream frozen successfully" if success else "Failed to freeze stream",
                    timestamp=time.time()
                )
            
            elif command.command == 'unfreeze':
                await stream_state.unfreeze_stream()
                return StreamResponse(
                    camera_id=command.camera_id,
                    command=command.command,
                    success=True,
                    data=None,
                    message="Stream unfrozen successfully",
                    timestamp=time.time()
                )
            
            elif command.command == 'get_frame':
                frame_data = await stream_state.get_current_frame_for_detection()
                if frame_data is not None:
                    # Encode frame as base64
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                    success, buffer = cv2.imencode('.jpg', frame_data, encode_params)
                    
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        return StreamResponse(
                            camera_id=command.camera_id,
                            command=command.command,
                            success=True,
                            data=frame_b64,
                            message="Frame retrieved successfully",
                            timestamp=time.time()
                        )
                
                return StreamResponse(
                    camera_id=command.camera_id,
                    command=command.command,
                    success=False,
                    data=None,
                    message="No frame available",
                    timestamp=time.time()
                )
            
            elif command.command == 'update_frozen_frame':
                # Get the frame data from Redis
                data_key = f"stream_data:{command.session_id}"
                frame_b64 = await self.async_redis_client.get(data_key)
                
                if frame_b64:
                    if isinstance(frame_b64, bytes):
                        frame_b64 = frame_b64.decode('utf-8')
                    
                    # Decode base64 to bytes
                    frame_bytes = base64.b64decode(frame_b64)
                    
                    # Update frozen frame
                    await stream_state.update_frozen_frame(frame_bytes)
                    
                    return StreamResponse(
                        camera_id=command.camera_id,
                        command=command.command,
                        success=True,
                        data=None,
                        message="Frozen frame updated successfully",
                        timestamp=time.time()
                    )
                else:
                    return StreamResponse(
                        camera_id=command.camera_id,
                        command=command.command,
                        success=False,
                        data=None,
                        message="No frame data found",
                        timestamp=time.time()
                    )
            
            elif command.command == 'status':
                stats = stream_state.get_stats()
                return StreamResponse(
                    camera_id=command.camera_id,
                    command=command.command,
                    success=True,
                    data=stats,
                    message="Status retrieved successfully",
                    timestamp=time.time()
                )
            
            else:
                return StreamResponse(
                    camera_id=command.camera_id,
                    command=command.command,
                    success=False,
                    data=None,
                    message=f"Unknown command: {command.command}",
                    timestamp=time.time()
                )
                
        except Exception as e:
            logger.error(f"❌ Error processing command '{command.command}': {e}")
            return StreamResponse(
                camera_id=command.camera_id,
                command=command.command,
                success=False,
                data=None,
                message=f"Error: {str(e)}",
                timestamp=time.time()
            )
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        try:
            self.is_running = False
            
            if self.command_listener_task and not self.command_listener_task.done():
                self.command_listener_task.cancel()
                try:
                    await self.command_listener_task
                except asyncio.CancelledError:
                    pass
            
            if self.async_redis_client:
                await self.async_redis_client.aclose()
                self.async_redis_client = None
            
            if self.sync_redis_client:
                self.sync_redis_client.close()
                self.sync_redis_client = None
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            logger.info("✅ Redis stream coordinator cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Error during Redis coordinator cleanup: {e}")

class RedisBasicStreamManager:
    """Lightweight stream manager for basic mode with Redis coordination"""
    
    def __init__(self):
        self.active_streams: Dict[str, 'RedisBasicStreamState'] = {}
        self.stats = {
            'total_frames_streamed': 0,
            'active_streams_count': 0,
            'redis_commands_processed': 0
        }
        self.redis_coordinator = RedisStreamCoordinator()
    
    async def initialize(self):
        """Initialize the Redis coordinator"""
        await self.redis_coordinator.initialize()
        logger.info("✅ Redis basic stream manager initialized")
    
    async def create_stream(self, config: BasicStreamConfig) -> str:
        """Create a new basic stream"""
        stream_key = f"redis_basic_{config.camera_id}_{uuid.uuid4().hex[:8]}"
        
        if stream_key not in self.active_streams:
            stream_state = RedisBasicStreamState(config)
            await stream_state.initialize()
            self.active_streams[stream_key] = stream_state
            self.stats['active_streams_count'] = len(self.active_streams)
            
            logger.info(f"✅ Created Redis basic stream: {stream_key}")
        
        return stream_key
    
    async def remove_stream(self, stream_key: str):
        """Remove and cleanup stream"""
        if stream_key in self.active_streams:
            await self.active_streams[stream_key].cleanup()
            del self.active_streams[stream_key]
            self.stats['active_streams_count'] = len(self.active_streams)
            logger.info(f"Removed Redis basic stream: {stream_key}")
    
    def get_stream(self, stream_key: str) -> Optional['RedisBasicStreamState']:
        """Get stream by key"""
        return self.active_streams.get(stream_key)
    
    def get_stream_by_camera_id(self, camera_id: int) -> Optional['RedisBasicStreamState']:
        """Get stream by camera ID"""
        for stream_state in self.active_streams.values():
            if stream_state.config.camera_id == camera_id:
                return stream_state
        return None
    
    async def cleanup(self):
        """Cleanup all streams and Redis coordinator"""
        # Cleanup all streams
        for stream_key in list(self.active_streams.keys()):
            await self.remove_stream(stream_key)
        
        # Cleanup Redis coordinator
        await self.redis_coordinator.cleanup()
        
        logger.info("✅ Redis basic stream manager cleanup complete")

class RedisBasicStreamState:
    """Stream state with freeze capability and Redis coordination"""
    
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
            logger.info(f"🚀 Initializing Redis basic stream for camera {self.config.camera_id}")
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.is_active = True
            
            # Start frame producer
            self.frame_producer_task = asyncio.create_task(self._frame_producer())
            
            logger.info(f"✅ Redis basic stream initialized for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis basic stream: {e}")
            await self.cleanup()
            raise
    
    async def _frame_producer(self):
        """Frame producer that can be frozen for detection"""
        logger.info(f"🎥 Starting Redis basic frame producer for camera {self.config.camera_id}")
        
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
                    logger.error(f"Error in Redis basic frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.error("Too many consecutive failures, stopping producer")
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"🛑 Redis basic frame producer stopped for camera {self.config.camera_id}")
    
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
                logger.info(f"🧊 Redis stream frozen for camera {self.config.camera_id}")
                return True
            else:
                logger.warning(f"⚠️ No frame available to freeze for camera {self.config.camera_id}")
                return False
    
    async def unfreeze_stream(self):
        """Unfreeze the stream to resume live feed"""
        async with self.frame_lock:
            self.is_frozen = False
            self.frozen_frame = None
            logger.info(f"🔥 Redis stream unfrozen for camera {self.config.camera_id}")
    
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
        logger.info(f"👤 Added consumer {consumer_id} to Redis basic camera {self.config.camera_id}. Total: {len(self.consumers)}")
    
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"👤 Removed consumer {consumer_id} from Redis basic camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # Stop stream if no consumers
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for Redis basic camera {self.config.camera_id}, stopping stream")
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
        logger.info(f"🧹 Cleaning up Redis basic stream for camera {self.config.camera_id}")
        
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
        
        logger.info(f"✅ Redis basic stream cleanup complete for camera {self.config.camera_id}")

# Global Redis basic stream manager
redis_basic_stream_manager = RedisBasicStreamManager()