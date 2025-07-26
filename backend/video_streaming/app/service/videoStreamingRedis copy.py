# redis_video_streaming_service.py (FIXED - Detection overlay display with proper debugging)
import cv2
import asyncio
import logging
import redis.asyncio as async_redis
import redis
import pickle
import numpy as np
import aiohttp
from typing import AsyncGenerator, Optional, Dict, Any
import time
import uuid
from dataclasses import dataclass, field
from collections import deque
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"
REDIS_HOST = "redis"
REDIS_PORT = 6379

@dataclass
class StreamConfig:
    """Configuration for video stream with detection"""
    camera_id: int
    target_label: str
    detection_enabled: bool = True
    detection_frequency: float = 5.0  # Hz (frames per second to process for detection)
    stream_quality: int = 85  # JPEG quality
    target_fps: int = 25
    frame_skip_detection: int = 5  # Process every Nth frame for detection
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass 
class DetectionResult:
    """Detection result from Redis"""
    camera_id: int
    session_id: str
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    timestamp: float
    confidence: Optional[float] = None
    bounding_boxes: Optional[list] = None

class AdaptiveFrameProcessor:
    """Handles intelligent frame processing and detection scheduling"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.frame_counter = 0
        
        # Calculate detection interval based on target FPS and detection frequency
        self.detection_interval = max(1, int(config.target_fps / config.detection_frequency))
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_backlog = 0
        self.avg_processing_time = 0
        
        # Frame quality adaptation
        self.current_quality = config.stream_quality
        self.quality_adaptation_enabled = True
        
    def should_process_for_detection(self) -> bool:
        """Determine if current frame should be processed for detection"""
        self.frame_counter += 1
        
        # Skip detection if backlog is too high
        if self.detection_backlog > 3:
            return False
        
        # Process every Nth frame based on detection frequency
        return self.frame_counter % self.detection_interval == 0
    
    def update_detection_performance(self, processing_time_ms: float):
        """Update performance metrics and adapt if needed"""
        self.avg_processing_time = (self.avg_processing_time * 0.9) + (processing_time_ms * 0.1)
        
        # Adapt detection frequency if processing is too slow
        if self.avg_processing_time > 200:  # 200ms threshold
            self.detection_interval = min(self.detection_interval + 1, 10)
        elif self.avg_processing_time < 50:  # 50ms threshold
            self.detection_interval = max(self.detection_interval - 1, 1)
    
    def get_current_quality(self) -> int:
        """Get current JPEG quality based on performance"""
        if not self.quality_adaptation_enabled:
            return self.current_quality
            
        # Reduce quality if detection is slow
        if self.avg_processing_time > 150:
            return max(60, self.current_quality - 10)
        elif self.avg_processing_time < 75:
            return min(95, self.current_quality + 5)
        
        return self.current_quality

class OptimizedStreamManager:
    """Manages optimized video streams with Redis-based detection"""
    
    def __init__(self):
        # Redis connection with connection pooling (sync for stats)
        self.redis_pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            max_connections=20,
            retry_on_timeout=True
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        # Active streams
        self.active_streams: Dict[str, 'OptimizedStreamState'] = {}
        self.stream_lock = asyncio.Lock()
        
        # Performance monitoring
        self.stats = {
            'total_frames_streamed': 0,
            'total_frames_detected': 0,
            'active_streams_count': 0,
            'redis_operations': 0
        }
    
    async def create_stream(self, config: StreamConfig) -> str:
        """Create a new optimized stream"""
        stream_key = f"{config.camera_id}_{config.session_id}"
        
        async with self.stream_lock:
            if stream_key not in self.active_streams:
                stream_state = OptimizedStreamState(config, self.redis_client)
                await stream_state.initialize()
                self.active_streams[stream_key] = stream_state
                self.stats['active_streams_count'] = len(self.active_streams)
                
                logger.info(f"Created optimized stream: {stream_key}")
            
            return stream_key
    
    async def remove_stream(self, stream_key: str):
        """Remove and cleanup stream"""
        async with self.stream_lock:
            if stream_key in self.active_streams:
                await self.active_streams[stream_key].cleanup()
                del self.active_streams[stream_key]
                self.stats['active_streams_count'] = len(self.active_streams)
                
                logger.info(f"Removed stream: {stream_key}")
    
    def get_stream(self, stream_key: str) -> Optional['OptimizedStreamState']:
        """Get stream by key"""
        return self.active_streams.get(stream_key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        try:
            # Get Redis connection pool info safely
            redis_pool_info = self._get_redis_pool_info()
            
            return {
                **self.stats,
                **redis_pool_info,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                **self.stats,
                'redis_pool_available': 0,
                'redis_pool_created': 0,
                'redis_pool_max': 20,
                'timestamp': datetime.now().isoformat(),
                'stats_error': str(e)
            }
    
    def _get_redis_pool_info(self) -> Dict[str, Any]:
        """Safely get Redis connection pool information"""
        try:
            # Get connection pool stats safely
            pool = self.redis_pool
            max_connections = pool.connection_kwargs.get('max_connections', 20)
            
            # Get current pool state
            created_connections = getattr(pool, '_created_connections', 0)
            available_connections = getattr(pool, '_available_connections', [])
            
            # Handle different types of available_connections
            if hasattr(available_connections, '__len__'):
                available_count = len(available_connections)
            else:
                available_count = 0
            
            # Handle different types of created_connections
            if isinstance(created_connections, int):
                created_count = created_connections
            elif hasattr(created_connections, '__len__'):
                created_count = len(created_connections)
            else:
                created_count = 0
            
            return {
                'redis_pool_max': max_connections,
                'redis_pool_created': created_count,
                'redis_pool_available': available_count,
                'redis_pool_in_use': max(0, created_count - available_count)
            }
            
        except Exception as e:
            logger.warning(f"Could not get Redis pool info: {e}")
            return {
                'redis_pool_max': 20,
                'redis_pool_created': 0,
                'redis_pool_available': 0,
                'redis_pool_in_use': 0
            }

class OptimizedStreamState:
    """FIXED stream state with proper detection overlay display and debug logging"""
    
    def __init__(self, config: StreamConfig, redis_client: redis.Redis):
        self.config = config
        self.redis_client = redis_client  # Sync Redis client for sending detection requests
        
        # Create ASYNC Redis client for pubsub
        self.async_redis_client = None
        
        # Stream management
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.consumers = set()
        
        # Frame processing
        self.frame_processor = AdaptiveFrameProcessor(config)
        self.latest_frame_original = None  # Store original frame
        self.latest_frame_with_overlay = None  # Store processed frame with overlays
        self.frame_lock = asyncio.Lock()
        
        # Detection state
        self.last_detection_result = None
        self.detection_subscriber_task = None
        self.has_recent_detection = False  # Track if we have recent detection overlay
        self.overlay_last_updated = 0  # Track when overlay was last updated
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.last_frame_time = 0
        self.overlay_served_count = 0  # Track how many overlay frames served
        self.original_served_count = 0  # Track how many original frames served
        
        # HTTP session for camera communication
        self.session = None
        self.frame_producer_task = None
        
        # DEBUG: Connection tracking
        self.pubsub_connected = False
        self.detection_messages_received = 0
        self.processed_frames_retrieved = 0
        
    async def initialize(self):
        """Initialize stream state"""
        try:
            # Create ASYNC Redis client for pubsub
            self.async_redis_client = async_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            # Test async Redis connection
            await self.async_redis_client.ping()
            logger.info(f"✓ Async Redis connection established for camera {self.config.camera_id}")
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Start detection result subscriber FIRST
            if self.config.detection_enabled:
                self.detection_subscriber_task = asyncio.create_task(
                    self._detection_result_subscriber()
                )
                # Give subscriber time to establish connection
                await asyncio.sleep(1.0)
                logger.info(f"✓ Detection subscriber started for camera {self.config.camera_id}")
            
            # Start frame producer
            self.frame_producer_task = asyncio.create_task(
                self._frame_producer()
            )
            
            self.is_active = True
            logger.info(f"✓ Initialized optimized stream for camera {self.config.camera_id} (detection: {self.config.detection_enabled})")
            
        except Exception as e:
            logger.error(f"Failed to initialize stream: {e}")
            await self.cleanup()
            raise
    
    async def _detection_result_subscriber(self):
        """FIXED: Subscribe to detection results from Redis using async client"""
        channel = f"detection_results:{self.config.camera_id}"
        pubsub = None
        
        try:
            # Create pubsub connection using async Redis client
            pubsub = self.async_redis_client.pubsub()
            
            # Subscribe to the channel
            await pubsub.subscribe(channel)
            logger.info(f"🔗 Subscribed to detection channel: {channel}")
            
            # Wait for subscription confirmation
            await asyncio.sleep(0.5)
            
            # Listen for messages in a proper loop
            while self.is_active and not self.stop_event.is_set():
                try:
                    # Use async get_message() with longer timeout
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=False),
                        timeout=2.0
                    )
                    
                    if message is not None:
                        if message['type'] == 'message':
                            try:
                                # Process the detection result
                                result_data = pickle.loads(message['data'])
                                await self._handle_detection_result(result_data)
                                self.detection_messages_received += 1
                                
                                # Log periodically
                                if self.detection_messages_received % 10 == 1:
                                    logger.info(f"📨 Received {self.detection_messages_received} detection messages for camera {self.config.camera_id}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error processing detection result: {e}")
                        elif message['type'] == 'subscribe':
                            logger.info(f"✅ Confirmed subscription to: {message['channel'].decode()}")
                            self.pubsub_connected = True
                        
                except asyncio.TimeoutError:
                    # Timeout is normal, continue listening
                    # Check connection health periodically
                    if self.detection_messages_received == 0:
                        logger.debug(f"🔄 Waiting for detection messages on camera {self.config.camera_id}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in detection subscriber: {e}")
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in detection subscriber: {e}")
        finally:
            try:
                if pubsub:
                    await pubsub.unsubscribe(channel)
                    logger.info(f"📤 Unsubscribed from channel: {channel}")
                    await pubsub.aclose()
                    logger.info(f"🔌 Closed pubsub connection for camera {self.config.camera_id}")
            except Exception as e:
                logger.error(f"❌ Error closing pubsub: {e}")
    
    async def _handle_detection_result(self, result_data: Dict[str, Any]):
        """FIXED: Handle incoming detection result and create/update overlay"""
        try:
            # Convert dict to DetectionResult
            result = DetectionResult(
                camera_id=result_data.get('camera_id'),
                session_id=result_data.get('session_id'),
                detected_target=result_data.get('detected_target', False),
                non_target_count=result_data.get('non_target_count', 0),
                processing_time_ms=result_data.get('processing_time_ms', 0),
                timestamp=result_data.get('timestamp', time.time()),
                confidence=result_data.get('confidence'),
                bounding_boxes=result_data.get('bounding_boxes')
            )
            
            # Update performance metrics
            self.frame_processor.update_detection_performance(result.processing_time_ms)
            
            # Store latest result
            self.last_detection_result = result
            self.detection_count += 1
            
            # CRITICAL FIX: Get the processed frame from Redis cache
            processed_frame_data = None
            try:
                cache_key = f"processed_frame:{result.camera_id}:{result.session_id}"
                logger.debug(f"🔍 Looking for processed frame in Redis: {cache_key}")
                
                # Use sync Redis client to get processed frame
                processed_frame_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, cache_key
                )
                
                if processed_frame_data:
                    # Decode the processed frame with detection overlays
                    nparr = np.frombuffer(processed_frame_data, np.uint8)
                    overlay_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if overlay_frame is not None:
                        async with self.frame_lock:
                            # Store the processed frame with overlays
                            self.latest_frame_with_overlay = overlay_frame
                            self.has_recent_detection = True
                            self.overlay_last_updated = time.time()
                            self.processed_frames_retrieved += 1
                            
                        logger.info(f"✅ Updated detection overlay frame for camera {self.config.camera_id} (size: {len(processed_frame_data)} bytes, retrieved: {self.processed_frames_retrieved})")
                    else:
                        logger.warning(f"❌ Failed to decode processed frame for camera {self.config.camera_id}")
                        async with self.frame_lock:
                            self.latest_frame_with_overlay = None
                            self.has_recent_detection = False
                else:
                    logger.warning(f"❌ No processed frame found in Redis cache: {cache_key}")
                    # Clear the overlay if no processed frame is available
                    async with self.frame_lock:
                        self.latest_frame_with_overlay = None
                        self.has_recent_detection = False
                        
            except Exception as e:
                logger.error(f"❌ Error retrieving processed frame from Redis: {e}")
                async with self.frame_lock:
                    self.latest_frame_with_overlay = None
                    self.has_recent_detection = False
            
            # Log detection events
            if result.detected_target:
                logger.info(f"🎯 Target '{self.config.target_label}' detected in camera {self.config.camera_id} "
                           f"(confidence: {result.confidence}, processing: {result.processing_time_ms:.1f}ms, overlays: {self.processed_frames_retrieved})")
                
        except Exception as e:
            logger.error(f"❌ Error handling detection result: {e}")
    
    async def _frame_producer(self):
        """Produce frames from camera and optionally send for detection"""
        logger.info(f"Starting frame producer for camera {self.config.camera_id}")
        
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
                                    # Decode frame
                                    nparr = np.frombuffer(jpeg_data, np.uint8)
                                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if frame is not None and frame.size > 0:
                                        # Resize frame for efficiency
                                        frame = cv2.resize(frame, (640, 480))
                                        
                                        # Process frame
                                        await self._process_frame(frame)
                                        
                                except Exception as e:
                                    logger.debug(f"Frame decode error: {e}")
                                    
                            # Prevent buffer from growing too large
                            if len(buffer) > 100000:
                                buffer = buffer[-50000:]
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.error("Too many consecutive failures, stopping producer")
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"Frame producer stopped for camera {self.config.camera_id}")
    
    async def _process_frame(self, frame: np.ndarray):
        """Process frame and optionally send for detection"""
        try:
            # Always update the latest original frame for streaming
            current_quality = self.frame_processor.get_current_quality()
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, current_quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if success:
                async with self.frame_lock:
                    self.latest_frame_original = buffer.tobytes()
                    self.frame_count += 1
                    self.last_frame_time = time.time()
            
            # Send for detection if enabled and conditions are met
            if (self.config.detection_enabled and 
                self.frame_processor.should_process_for_detection()):
                
                await self._send_frame_for_detection(frame)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    async def _send_frame_for_detection(self, frame: np.ndarray):
        """Send frame to Redis queue for detection processing"""
        try:
            # Encode frame as bytes (no base64 overhead)
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                return
            
            # Create detection request
            request_data = {
                'camera_id': self.config.camera_id,
                'frame_data': buffer.tobytes(),  # Raw bytes, no base64
                'target_label': self.config.target_label,
                'timestamp': time.time(),
                'session_id': self.config.session_id,
                'priority': 1  # Normal priority
            }
            
            # Send to Redis queue (non-blocking) using sync client
            serialized_request = pickle.dumps(request_data)
            queue_length = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis_client.lpush,
                'detection_queue', 
                serialized_request
            )
            
            # Track queue depth for backpressure
            self.frame_processor.detection_backlog = min(queue_length, 10)
            
            # Log detection requests periodically
            if self.frame_count % 25 == 0:
                logger.info(f"📤 Sent frame #{self.frame_count} for detection (camera {self.config.camera_id}, queue: {queue_length}, msgs: {self.detection_messages_received})")
            
        except Exception as e:
            logger.error(f"Error sending frame for detection: {e}")
    
    async def add_consumer(self, consumer_id: str):
        """Add a consumer to this stream"""
        self.consumers.add(consumer_id)
        logger.info(f"👤 Added consumer {consumer_id} to camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"👤 Removed consumer {consumer_id} from camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # Stop stream if no consumers
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for camera {self.config.camera_id}, stopping stream")
            await self.cleanup()
    
    async def get_latest_frame(self) -> Optional[bytes]:
        """FIXED: Get the latest frame (prioritize detection overlay if recent and available)"""
        async with self.frame_lock:
            current_time = time.time()
            
            # Check if we have a recent detection overlay
            if (self.has_recent_detection and 
                self.latest_frame_with_overlay is not None and
                self.last_detection_result and
                current_time - self.last_detection_result.timestamp < 5.0):  # 5 second freshness
                
                try:
                    # Encode the overlay frame
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.frame_processor.get_current_quality()]
                    success, buffer = cv2.imencode('.jpg', self.latest_frame_with_overlay, encode_params)
                    if success:
                        self.overlay_served_count += 1
                        
                        # DEBUG: Log overlay frame serving
                        if self.overlay_served_count % 50 == 1:  # Log every 50 frames
                            overlay_age = current_time - self.overlay_last_updated
                            logger.info(f"📺 Serving overlay frame #{self.overlay_served_count} for camera {self.config.camera_id} (age: {overlay_age:.1f}s, retrieved: {self.processed_frames_retrieved})")
                        
                        return buffer.tobytes()
                except Exception as e:
                    logger.error(f"Error encoding overlay frame: {e}")
            
            # Fall back to original frame if no recent overlay
            if self.latest_frame_original is not None:
                self.original_served_count += 1
                
                # DEBUG: Log original frame serving ratio
                if self.original_served_count % 100 == 1:
                    total_served = self.overlay_served_count + self.original_served_count
                    overlay_ratio = (self.overlay_served_count / total_served * 100) if total_served > 0 else 0
                    logger.info(f"📺 Frame ratio for camera {self.config.camera_id}: overlays={self.overlay_served_count}/{total_served} ({overlay_ratio:.1f}%)")
                
                return self.latest_frame_original
            
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this stream"""
        current_time = time.time()
        overlay_age = current_time - self.overlay_last_updated if self.overlay_last_updated > 0 else 0
        
        return {
            'camera_id': self.config.camera_id,
            'session_id': self.config.session_id,
            'target_label': self.config.target_label,
            'is_active': self.is_active,
            'consumers_count': len(self.consumers),
            'frames_processed': self.frame_count,
            'detections_processed': self.detection_count,
            'avg_detection_time_ms': self.frame_processor.avg_processing_time,
            'detection_interval': self.frame_processor.detection_interval,
            'current_quality': self.frame_processor.get_current_quality(),
            'detection_backlog': self.frame_processor.detection_backlog,
            'last_frame_time': self.last_frame_time,
            'last_detection': self.last_detection_result.timestamp if self.last_detection_result else None,
            'target_detected': self.last_detection_result.detected_target if self.last_detection_result else False,
            'non_target_count': self.last_detection_result.non_target_count if self.last_detection_result else 0,
            'has_recent_overlay': self.has_recent_detection,
            'overlay_age_seconds': overlay_age,
            'overlay_frames_served': self.overlay_served_count,
            'original_frames_served': self.original_served_count,
            'pubsub_connected': self.pubsub_connected,
            'detection_messages_received': self.detection_messages_received,
            'processed_frames_retrieved': self.processed_frames_retrieved
        }
    
    async def cleanup(self):
        """Clean up stream resources"""
        logger.info(f"Cleaning up stream for camera {self.config.camera_id}")
        
        self.is_active = False
        self.stop_event.set()
        
        # Cancel tasks
        if self.frame_producer_task and not self.frame_producer_task.done():
            self.frame_producer_task.cancel()
            try:
                await self.frame_producer_task
            except asyncio.CancelledError:
                pass
        
        if self.detection_subscriber_task and not self.detection_subscriber_task.done():
            self.detection_subscriber_task.cancel()
            try:
                await self.detection_subscriber_task  
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
            
        # Close async Redis client
        if self.async_redis_client:
            try:
                await self.async_redis_client.aclose()
            except Exception as e:
                logger.error(f"Error closing async Redis client: {e}")
            
        # Clear consumers and frames
        self.consumers.clear()
        async with self.frame_lock:
            self.latest_frame_original = None
            self.latest_frame_with_overlay = None
            self.has_recent_detection = False
        
        logger.info(f"Stream cleanup complete for camera {self.config.camera_id}")

# Global optimized stream manager
optimized_stream_manager = OptimizedStreamManager()

async def generate_optimized_video_frames_with_detection(
    camera_id: int,
    target_label: str,
    consumer_id: str = None
) -> AsyncGenerator[bytes, None]:
    """Generate video frames using the optimized Redis-based architecture"""
    
    if consumer_id is None:
        consumer_id = f"consumer_{uuid.uuid4().hex[:8]}"
    
    # Create stream configuration
    config = StreamConfig(
        camera_id=camera_id,
        target_label=target_label,
        detection_enabled=bool(target_label),  # Enable detection only if target_label is provided
        detection_frequency=5.0,  # 5 FPS detection
        session_id=str(uuid.uuid4())
    )
    
    stream_key = None
    
    try:
        # Create or get existing stream
        stream_key = await optimized_stream_manager.create_stream(config)
        stream_state = optimized_stream_manager.get_stream(stream_key)
        
        if not stream_state:
            logger.error(f"Failed to create stream for camera {camera_id}")
            return
        
        # Add consumer
        await stream_state.add_consumer(consumer_id)
        
        # Stream frames
        target_fps = config.target_fps
        frame_time = 1.0 / target_fps
        
        logger.info(f"🚀 Starting optimized stream for camera {camera_id}, consumer {consumer_id}, target: '{target_label}'")
        
        frame_count = 0
        overlay_count = 0
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                frame_bytes = await stream_state.get_latest_frame()
                
                if frame_bytes is not None:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    frame_count += 1
                    
                    # Count overlay frames (check if this was an overlay)
                    if (stream_state.has_recent_detection and 
                        stream_state.last_detection_result and
                        time.time() - stream_state.last_detection_result.timestamp < 3.0):
                        overlay_count += 1
                    
                    # Update manager stats
                    optimized_stream_manager.stats['total_frames_streamed'] += 1
                    if config.detection_enabled:
                        optimized_stream_manager.stats['total_frames_detected'] += 1
                        
                    # DEBUG: Log streaming progress
                    if frame_count % 100 == 0:
                        overlay_ratio = (overlay_count / frame_count * 100) if frame_count > 0 else 0
                        logger.info(f"📊 Stream stats - Camera {camera_id}: frames={frame_count}, overlays={overlay_count}, overlay_ratio={overlay_ratio:.1f}%")
                else:
                    await asyncio.sleep(0.05)
                    continue
                
                # Frame rate control
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info(f"Stream cancelled for camera {camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"Error in frame generation: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in generate_optimized_video_frames_with_detection: {e}")
    finally:
        # Cleanup
        if stream_key and stream_state:
            await stream_state.remove_consumer(consumer_id)
        
        logger.info(f"Optimized video stream stopped for camera {camera_id}, consumer {consumer_id} (streamed {frame_count} frames)")