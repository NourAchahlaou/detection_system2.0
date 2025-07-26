# redis_video_streaming_service.py - FIXED PubSub Connection Issues
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
    """FIXED stream state with reliable pubsub connection management"""
    
    def __init__(self, config: StreamConfig, redis_client: redis.Redis):
        self.config = config
        self.redis_client = redis_client  # Sync Redis client for sending detection requests
        
        # Create ASYNC Redis clients with proper connection management
        self.async_redis_client = None
        self.pubsub_redis_client = None  # Dedicated client for pubsub
        
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
        self.connection_monitor_task = None  # NEW: Monitor pubsub connection
        
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
        self.connection_attempts = 0
        self.last_connection_attempt = 0
        
        # NEW: Connection health monitoring
        self.pubsub = None
        self.subscription_confirmed = False
        self.last_heartbeat = 0
        
    async def initialize(self):
        """Initialize stream state with robust connection management"""
        try:
            logger.info(f"🚀 Initializing stream for camera {self.config.camera_id}")
            
            # Create dedicated async Redis clients
            await self._create_redis_clients()
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Start detection result subscriber FIRST and wait for confirmation
            if self.config.detection_enabled:
                await self._start_pubsub_with_confirmation()
                
            # Start connection monitor
            self.connection_monitor_task = asyncio.create_task(
                self._connection_monitor()
            )
            
            # Start frame producer
            self.frame_producer_task = asyncio.create_task(
                self._frame_producer()
            )
            
            self.is_active = True
            logger.info(f"✅ Initialized optimized stream for camera {self.config.camera_id} (detection: {self.config.detection_enabled})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize stream: {e}")
            await self.cleanup()
            raise
    
    async def _create_redis_clients(self):
        """Create and test Redis clients with proper configuration"""
        try:
            # Main async Redis client
            self.async_redis_client = async_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                retry_on_timeout=True,
                socket_connect_timeout=10,
                socket_timeout=30
            )
            
            # Dedicated pubsub client with optimized settings
            self.pubsub_redis_client = async_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=15,  # More frequent health checks for pubsub
                retry_on_timeout=True,
                socket_connect_timeout=10,
                socket_timeout=60,  # Longer timeout for pubsub
                max_connections=5  # Limit connections for pubsub client
            )
            
            # Test both connections
            await self.async_redis_client.ping()
            await self.pubsub_redis_client.ping()
            
            logger.info(f"✅ Redis clients created successfully for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create Redis clients: {e}")
            raise
    
    async def _start_pubsub_with_confirmation(self):
        """Start pubsub connection and wait for confirmation"""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                self.connection_attempts += 1
                self.last_connection_attempt = time.time()
                
                logger.info(f"🔄 Attempting pubsub connection #{attempt + 1} for camera {self.config.camera_id}")
                
                # Create pubsub connection
                self.pubsub = self.pubsub_redis_client.pubsub()
                
                # Subscribe to channel
                channel = f"detection_results:{self.config.camera_id}"
                await self.pubsub.subscribe(channel)
                
                # Wait for subscription confirmation with timeout
                confirmation_timeout = 5.0
                start_time = time.time()
                
                while time.time() - start_time < confirmation_timeout:
                    try:
                        message = await asyncio.wait_for(
                            self.pubsub.get_message(ignore_subscribe_messages=False),
                            timeout=1.0
                        )
                        
                        if message and message['type'] == 'subscribe':
                            self.subscription_confirmed = True
                            self.pubsub_connected = True
                            logger.info(f"✅ Pubsub subscription CONFIRMED for camera {self.config.camera_id} on channel: {channel}")
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                
                if self.subscription_confirmed:
                    # Start the message listener
                    self.detection_subscriber_task = asyncio.create_task(
                        self._detection_result_subscriber()
                    )
                    
                    # Give a moment for the subscriber to be ready
                    await asyncio.sleep(0.5)
                    
                    # Test the connection by publishing a test message
                    await self._test_pubsub_connection()
                    
                    logger.info(f"🎉 Pubsub connection ESTABLISHED for camera {self.config.camera_id}")
                    return
                else:
                    logger.warning(f"⚠️ Subscription confirmation timeout for camera {self.config.camera_id}")
                    await self._cleanup_pubsub()
                    
            except Exception as e:
                logger.error(f"❌ Pubsub connection attempt {attempt + 1} failed: {e}")
                await self._cleanup_pubsub()
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"❌ Failed to establish pubsub connection after {max_attempts} attempts for camera {self.config.camera_id}")
        raise Exception("Could not establish pubsub connection")
    
    async def _test_pubsub_connection(self):
        """Test pubsub connection by checking subscriber count"""
        try:
            channel = f"detection_results:{self.config.camera_id}"
            
            # Use sync client to check subscriber count
            subscriber_count = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis_client.pubsub_numsub, 
                channel
            )
            
            # pubsub_numsub returns [(channel, count), ...]
            if subscriber_count and len(subscriber_count) > 0:
                count = subscriber_count[0][1] if len(subscriber_count[0]) > 1 else 0
                if count > 0:
                    logger.info(f"✅ Pubsub test PASSED: {count} subscribers on {channel}")
                    return True
            
            logger.warning(f"⚠️ Pubsub test: No subscribers detected on {channel}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Pubsub test failed: {e}")
            return False
    
    async def _connection_monitor(self):
        """Monitor pubsub connection health and reconnect if needed"""
        logger.info(f"🔍 Starting connection monitor for camera {self.config.camera_id}")
        
        reconnect_interval = 30  # Check every 30 seconds
        
        while self.is_active and not self.stop_event.is_set():
            try:
                await asyncio.sleep(reconnect_interval)
                
                if not self.is_active:
                    break
                
                # Check if pubsub is still healthy
                if self.pubsub_connected and self.pubsub:
                    try:
                        # Test connection health
                        await asyncio.wait_for(
                            self.pubsub_redis_client.ping(),
                            timeout=5.0
                        )
                        
                        # Update heartbeat
                        self.last_heartbeat = time.time()
                        
                        # Check if we're receiving messages
                        if (time.time() - self.last_connection_attempt > 60 and 
                            self.detection_messages_received == 0):
                            logger.warning(f"⚠️ No detection messages received for camera {self.config.camera_id}, connection may be stale")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Pubsub health check failed for camera {self.config.camera_id}: {e}")
                        await self._reconnect_pubsub()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in connection monitor: {e}")
        
        logger.info(f"🛑 Connection monitor stopped for camera {self.config.camera_id}")
    
    async def _reconnect_pubsub(self):
        """Attempt to reconnect pubsub"""
        try:
            logger.info(f"🔄 Attempting pubsub reconnection for camera {self.config.camera_id}")
            
            # Cancel existing subscriber task
            if self.detection_subscriber_task and not self.detection_subscriber_task.done():
                self.detection_subscriber_task.cancel()
                try:
                    await self.detection_subscriber_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup old pubsub connection
            await self._cleanup_pubsub()
            
            # Reset connection state
            self.pubsub_connected = False
            self.subscription_confirmed = False
            
            # Recreate pubsub client
            await self._create_redis_clients()
            
            # Restart pubsub connection
            await self._start_pubsub_with_confirmation()
            
            logger.info(f"✅ Pubsub reconnection successful for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Pubsub reconnection failed for camera {self.config.camera_id}: {e}")
    
    async def _cleanup_pubsub(self):
        """Clean up pubsub resources"""
        try:
            if self.pubsub:
                try:
                    await self.pubsub.unsubscribe()
                    await asyncio.sleep(0.1)  # Give time for unsubscribe
                except:
                    pass
                try:
                    await self.pubsub.aclose()
                except:
                    pass
                self.pubsub = None
            
            self.pubsub_connected = False
            self.subscription_confirmed = False
            
        except Exception as e:
            logger.debug(f"Error cleaning up pubsub: {e}")
    
    async def _detection_result_subscriber(self):
        """FIXED: Robust detection result subscriber with proper error handling"""
        logger.info(f"🔊 Starting detection result subscriber for camera {self.config.camera_id}")
        
        try:
            while self.is_active and not self.stop_event.is_set():
                try:
                    if not self.pubsub:
                        logger.error("❌ Pubsub connection is None, exiting subscriber")
                        break
                    
                    # Listen for messages with timeout
                    message = await asyncio.wait_for(
                        self.pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=5.0
                    )
                    
                    if message is not None and message['type'] == 'message':
                        try:
                            # Process the detection result
                            result_data = pickle.loads(message['data'])
                            await self._handle_detection_result(result_data)
                            self.detection_messages_received += 1
                            
                            # Update heartbeat
                            self.last_heartbeat = time.time()
                            
                            # Log periodically
                            if self.detection_messages_received % 5 == 0:
                                logger.info(f"📨 Processed {self.detection_messages_received} detection messages for camera {self.config.camera_id}")
                                
                        except Exception as e:
                            logger.error(f"❌ Error processing detection result: {e}")
                
                except asyncio.TimeoutError:
                    # Timeout is normal, check connection health
                    if self.pubsub_connected and time.time() - self.last_heartbeat > 30:
                        logger.warning(f"⚠️ No messages received for 30s on camera {self.config.camera_id}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in detection subscriber: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            logger.info(f"🛑 Detection subscriber cancelled for camera {self.config.camera_id}")
        except Exception as e:
            logger.error(f"❌ Fatal error in detection subscriber: {e}")
        finally:
            logger.info(f"🛑 Detection subscriber stopped for camera {self.config.camera_id}")
    
    async def _handle_detection_result(self, result_data: Dict[str, Any]):
        """ENHANCED: Handle incoming detection result and create/update overlay with debugging"""
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
            
            # ENHANCED: More aggressive overlay retrieval with multiple attempts
            processed_frame_data = None
            cache_key = f"processed_frame:{result.camera_id}:{result.session_id}"
            
            # Try multiple times to get the processed frame
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    processed_frame_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.get, cache_key
                    )
                    
                    if processed_frame_data:
                        break
                        
                    # If no data, wait a bit and try again
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"❌ Attempt {attempt + 1} to get processed frame failed: {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(0.1)
            
            if processed_frame_data:
                try:
                    # Decode the processed frame with detection overlays
                    nparr = np.frombuffer(processed_frame_data, np.uint8)
                    overlay_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if overlay_frame is not None:
                        async with self.frame_lock:
                            # CRITICAL FIX: Always update overlay frame regardless of detection status
                            self.latest_frame_with_overlay = overlay_frame.copy()  # Make a copy
                            self.has_recent_detection = True
                            self.overlay_last_updated = time.time()
                            self.processed_frames_retrieved += 1
                            
                        logger.info(f"🎨 OVERLAY UPDATED for camera {self.config.camera_id} - "
                                f"Target detected: {result.detected_target}, "
                                f"Frame size: {overlay_frame.shape}, "
                                f"Total overlays: {self.processed_frames_retrieved}")
                        
                        # DEBUGGING: Log frame characteristics
                        if result.detected_target:
                            unique_colors = len(np.unique(overlay_frame.reshape(-1, overlay_frame.shape[2]), axis=0))
                            logger.info(f"🎯 TARGET OVERLAY - Unique colors: {unique_colors}, "
                                    f"Mean brightness: {np.mean(overlay_frame):.1f}")
                    else:
                        logger.error(f"❌ Failed to decode processed frame for camera {self.config.camera_id}")
                except Exception as e:
                    logger.error(f"❌ Error decoding processed frame: {e}")
            else:
                logger.warning(f"⚠️ No processed frame found in cache: {cache_key} after {max_attempts} attempts")
                
                # DEBUGGING: Check if key exists with different session_id
                try:
                    pattern = f"processed_frame:{result.camera_id}:*"
                    keys = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.keys, pattern
                    )
                    if keys:
                        logger.info(f"🔍 Found {len(keys)} processed frame keys for camera {result.camera_id}: "
                                f"{[k.decode() if isinstance(k, bytes) else k for k in keys[:3]]}")
                except Exception as e:
                    logger.error(f"❌ Error checking for alternative keys: {e}")
            
            # Log detection events with more detail
            if result.detected_target:
                logger.info(f"🎯 TARGET '{self.config.target_label}' DETECTED in camera {self.config.camera_id}! "
                        f"Overlay available: {self.has_recent_detection}")
            
        except Exception as e:
            logger.error(f"❌ Error handling detection result: {e}")

    
    async def _frame_producer(self):
        """Produce frames from camera and optionally send for detection"""
        logger.info(f"🎥 Starting frame producer for camera {self.config.camera_id}")
        
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
            logger.info(f"🛑 Frame producer stopped for camera {self.config.camera_id}")
    
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
        """Send frame to Redis queue for detection processing with subscriber verification"""
        try:
            # CRITICAL FIX: Check if we have active subscribers before sending
            if not self.pubsub_connected or not self.subscription_confirmed:
                logger.debug(f"⏸️ Skipping detection for camera {self.config.camera_id} - no active pubsub connection")
                return
            
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
                subscriber_count = await self._get_subscriber_count()
                logger.info(f"📤 Sent frame #{self.frame_count} for detection (camera {self.config.camera_id}, queue: {queue_length}, subscribers: {subscriber_count})")
            
        except Exception as e:
            logger.error(f"Error sending frame for detection: {e}")
    
    async def _get_subscriber_count(self) -> int:
        """Get current subscriber count for this camera's channel"""
        try:
            channel = f"detection_results:{self.config.camera_id}"
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis_client.pubsub_numsub, 
                channel
            )
            
            if result and len(result) > 0:
                return result[0][1] if len(result[0]) > 1 else 0
            return 0
            
        except Exception as e:
            logger.error(f"Error getting subscriber count: {e}")
            return 0
    
    async def add_consumer(self, consumer_id: str):
        """Add a consumer to this stream with pubsub connection verification"""
        self.consumers.add(consumer_id)
        logger.info(f"👤 Added consumer {consumer_id} to camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # CRITICAL FIX: Ensure pubsub is connected when first consumer joins
        if len(self.consumers) == 1 and self.config.detection_enabled:
            if not self.pubsub_connected:
                logger.info(f"🔄 First consumer joined, ensuring pubsub connection for camera {self.config.camera_id}")
                try:
                    await self._reconnect_pubsub()
                except Exception as e:
                    logger.error(f"❌ Failed to establish pubsub on consumer join: {e}")
    
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"👤 Removed consumer {consumer_id} from camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # Stop stream if no consumers
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for camera {self.config.camera_id}, stopping stream")
            await self.cleanup()
    
    async def get_latest_frame(self) -> Optional[bytes]:
        """ENHANCED: Get the latest frame with improved overlay prioritization"""
        async with self.frame_lock:
            current_time = time.time()
            
            # ENHANCED: More lenient overlay freshness check
            overlay_freshness_limit = 10.0  # Increased from 5.0 to 10.0 seconds
            
            # Check if we have a recent detection overlay
            if (self.has_recent_detection and 
                self.latest_frame_with_overlay is not None and
                self.last_detection_result and
                current_time - self.overlay_last_updated < overlay_freshness_limit):
                
                try:
                    # ENHANCED: Use higher quality for overlay frames
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # Increased quality
                    success, buffer = cv2.imencode('.jpg', self.latest_frame_with_overlay, encode_params)
                    if success:
                        self.overlay_served_count += 1
                        
                        # More frequent logging for debugging
                        if self.overlay_served_count % 10 == 1:  # Log every 10 frames instead of 50
                            overlay_age = current_time - self.overlay_last_updated
                            logger.info(f"📺 SERVING OVERLAY frame #{self.overlay_served_count} "
                                    f"for camera {self.config.camera_id} "
                                    f"(age: {overlay_age:.1f}s, target_detected: {self.last_detection_result.detected_target})")
                        
                        return buffer.tobytes()
                except Exception as e:
                    logger.error(f"❌ Error encoding overlay frame: {e}")
            
            # Fall back to original frame
            if self.latest_frame_original is not None:
                self.original_served_count += 1
                
                # More frequent logging for debugging
                if self.original_served_count % 25 == 1:  # Log every 25 frames instead of 100
                    total_served = self.overlay_served_count + self.original_served_count
                    overlay_ratio = (self.overlay_served_count / total_served * 100) if total_served > 0 else 0
                    logger.info(f"📺 SERVING ORIGINAL frame for camera {self.config.camera_id} "
                            f"- Overlay ratio: {overlay_ratio:.1f}% "
                            f"({self.overlay_served_count}/{total_served})")
                
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
            'subscription_confirmed': self.subscription_confirmed,  # NEW
            'detection_messages_received': self.detection_messages_received,
            'processed_frames_retrieved': self.processed_frames_retrieved,
            'connection_attempts': self.connection_attempts,
            'last_heartbeat': self.last_heartbeat
        }
    
    async def cleanup(self):
        """Clean up stream resources"""
        logger.info(f"🧹 Cleaning up stream for camera {self.config.camera_id}")
        
        self.is_active = False
        self.stop_event.set()
        
        # Cancel tasks
        tasks_to_cancel = [
            self.frame_producer_task,
            self.detection_subscriber_task,
            self.connection_monitor_task
        ]
        
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        
        # Clean up pubsub
        await self._cleanup_pubsub()
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
            
        # Close async Redis clients
        for client in [self.async_redis_client, self.pubsub_redis_client]:
            if client:
                try:
                    await client.aclose()
                except Exception as e:
                    logger.debug(f"Error closing Redis client: {e}")
        
        # Clear consumers and frames
        self.consumers.clear()
        async with self.frame_lock:
            self.latest_frame_original = None
            self.latest_frame_with_overlay = None
            self.has_recent_detection = False
        
        logger.info(f"✅ Stream cleanup complete for camera {self.config.camera_id}")

# Global optimized stream manager
optimized_stream_manager = OptimizedStreamManager()

async def generate_optimized_video_frames_with_detection(
    camera_id: int,
    target_label: str,
    consumer_id: str = None
) -> AsyncGenerator[bytes, None]:
    """Generate video frames using the optimized Redis-based architecture with FIXED pubsub"""
    
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
        # CRITICAL FIX: Add startup delay to ensure detection service is ready
        logger.info(f"🚀 Starting optimized stream for camera {camera_id}, target: '{target_label}', consumer: {consumer_id}")
        
        # Create or get existing stream
        stream_key = await optimized_stream_manager.create_stream(config)
        stream_state = optimized_stream_manager.get_stream(stream_key)
        
        if not stream_state:
            logger.error(f"❌ Failed to create stream for camera {camera_id}")
            return
        
        # Add consumer (this will trigger pubsub connection if needed)
        await stream_state.add_consumer(consumer_id)
        
        # CRITICAL FIX: Wait for pubsub connection to be established
        if config.detection_enabled:
            connection_timeout = 10  # 10 second timeout
            start_time = time.time()
            
            while (time.time() - start_time < connection_timeout and 
                   not stream_state.subscription_confirmed):
                await asyncio.sleep(0.5)
                logger.info(f"⏳ Waiting for pubsub connection for camera {camera_id}...")
            
            if stream_state.subscription_confirmed:
                logger.info(f"✅ Pubsub connection confirmed for camera {camera_id}")
            else:
                logger.warning(f"⚠️ Pubsub connection timeout for camera {camera_id}, proceeding anyway")
        
        # Stream frames
        target_fps = config.target_fps
        frame_time = 1.0 / target_fps
        
        frame_count = 0
        overlay_count = 0
        no_frame_count = 0
        max_no_frame = 100  # Max consecutive frames with no data
        
        logger.info(f"🎬 Starting frame streaming for camera {camera_id}")
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                frame_bytes = await stream_state.get_latest_frame()
                
                if frame_bytes is not None:
                    no_frame_count = 0  # Reset counter
                    
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
                        
                    # DEBUG: Log streaming progress and pubsub status
                    if frame_count % 100 == 0:
                        overlay_ratio = (overlay_count / frame_count * 100) if frame_count > 0 else 0
                        subscriber_count = await stream_state._get_subscriber_count()
                        logger.info(f"📊 Stream stats - Camera {camera_id}: frames={frame_count}, overlays={overlay_count}, "
                                   f"overlay_ratio={overlay_ratio:.1f}%, subscribers={subscriber_count}, "
                                   f"pubsub_connected={stream_state.pubsub_connected}")
                else:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.error(f"❌ No frames received for {max_no_frame} attempts, stopping stream for camera {camera_id}")
                        break
                    await asyncio.sleep(0.05)
                    continue
                
                # Frame rate control
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Stream cancelled for camera {camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error in frame generation: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"❌ Error in generate_optimized_video_frames_with_detection: {e}")
    finally:
        # Cleanup
        if stream_key and stream_state:
            await stream_state.remove_consumer(consumer_id)
        
        logger.info(f"🏁 Optimized video stream stopped for camera {camera_id}, consumer {consumer_id} "
                   f"(streamed {frame_count} frames, {overlay_count} with overlays)")

# NEW: Add diagnostic functions to help debug pubsub issues
async def diagnose_pubsub_connection(camera_id: int) -> Dict[str, Any]:
    """Diagnostic function to check pubsub connection health"""
    try:
        # Create test Redis client
        redis_client = async_redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=False
        )
        
        # Test basic connection
        await redis_client.ping()
        
        # Check subscriber count
        channel = f"detection_results:{camera_id}"
        result = await redis_client.pubsub_numsub(channel)
        subscriber_count = result[0][1] if result and len(result) > 0 and len(result[0]) > 1 else 0
        
        # Test publishing
        test_message = pickle.dumps({
            'camera_id': camera_id,
            'test': True,
            'timestamp': time.time(),
            'message': 'Diagnostic test message'
        })
        
        published_count = await redis_client.publish(channel, test_message)
        
        await redis_client.aclose()
        
        return {
            'status': 'success',
            'camera_id': camera_id,
            'channel': channel,
            'subscriber_count': subscriber_count,
            'published_to_subscribers': published_count,
            'redis_connected': True,
            'timestamp': time.time()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'camera_id': camera_id,
            'error': str(e),
            'redis_connected': False,
            'timestamp': time.time()
        }

async def force_pubsub_reconnect(camera_id: int) -> Dict[str, Any]:
    """Force reconnection of pubsub for a specific camera"""
    try:
        # Find the stream
        for stream_key, stream_state in optimized_stream_manager.active_streams.items():
            if stream_state.config.camera_id == camera_id:
                logger.info(f"🔄 Forcing pubsub reconnect for camera {camera_id}")
                await stream_state._reconnect_pubsub()
                
                return {
                    'status': 'success',
                    'message': f'Forced reconnect for camera {camera_id}',
                    'camera_id': camera_id,
                    'pubsub_connected': stream_state.pubsub_connected,
                    'subscription_confirmed': stream_state.subscription_confirmed
                }
        
        return {
            'status': 'error',
            'message': f'No active stream found for camera {camera_id}',
            'camera_id': camera_id
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'camera_id': camera_id
        }