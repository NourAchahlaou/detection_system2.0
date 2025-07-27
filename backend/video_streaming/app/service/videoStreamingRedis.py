# enhanced_redis_video_streaming_service.py - FIXED with Service Readiness Pattern
import cv2
import asyncio
import logging
import redis.asyncio as async_redis
import redis as sync_redis
import redis.exceptions as redis_exceptions
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

# Import the service readiness manager
from video_streaming.app.service.service_readiness_manager import service_readiness_manager

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

class GlobalStreamManager:
    """ENHANCED global stream manager with service readiness pattern"""
    
    def __init__(self):
        # Redis connection with connection pooling (sync for stats)
        self.redis_pool = sync_redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            max_connections=20,
            retry_on_timeout=True
        )
        self.redis_client = sync_redis.Redis(connection_pool=self.redis_pool)
        
        # Active streams
        self.active_streams: Dict[str, 'StreamState'] = {}
        self.stream_lock = asyncio.Lock()
        
        # Performance monitoring
        self.stats = {
            'total_frames_streamed': 0,
            'total_frames_detected': 0,
            'active_streams_count': 0,
            'redis_operations': 0,
            'detection_service_ready_checks': 0,
            'detection_connections_established': 0,
            'detection_connections_failed': 0
        }
        
        # Service readiness tracking
        self.detection_service_ready = False
        self.last_readiness_check = 0
    
    async def create_stream_with_readiness_check(self, config: StreamConfig) -> str:
        """Create a new stream with service readiness verification"""
        stream_key = f"{config.camera_id}_{config.session_id}"
        
        async with self.stream_lock:
            if stream_key not in self.active_streams:
                # CRITICAL: Check detection service readiness before creating detection-enabled streams
                if config.detection_enabled:
                    logger.info(f"🔍 Checking detection service readiness for camera {config.camera_id}...")
                    
                    self.stats['detection_service_ready_checks'] += 1
                    
                    # Use service readiness manager to check if detection service is ready
                    detection_ready = await service_readiness_manager.is_detection_service_ready()
                    
                    if detection_ready:
                        logger.info(f"✅ Detection service is ready for camera {config.camera_id}")
                        self.stats['detection_connections_established'] += 1
                        self.detection_service_ready = True
                    else:
                        logger.warning(f"⚠️ Detection service not ready for camera {config.camera_id}, disabling detection")
                        self.stats['detection_connections_failed'] += 1
                        # Graceful degradation: disable detection but continue streaming
                        config.detection_enabled = False
                        self.detection_service_ready = False
                
                # Create stream state
                stream_state = StreamState(config, self.redis_client)
                await stream_state.initialize()
                self.active_streams[stream_key] = stream_state
                self.stats['active_streams_count'] = len(self.active_streams)
                
                logger.info(f"✅ Created enhanced stream: {stream_key} (detection: {config.detection_enabled})")
            
            return stream_key
    
    async def remove_stream(self, stream_key: str):
        """Remove and cleanup stream"""
        async with self.stream_lock:
            if stream_key in self.active_streams:
                await self.active_streams[stream_key].cleanup()
                del self.active_streams[stream_key]
                self.stats['active_streams_count'] = len(self.active_streams)
                
                logger.info(f"Removed stream: {stream_key}")
    
    def get_stream(self, stream_key: str) -> Optional['StreamState']:
        """Get stream by key"""
        return self.active_streams.get(stream_key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive service health status"""
        try:
            # Get service readiness status
            readiness_status = service_readiness_manager.get_service_status()
            
            # Get Redis connection pool info
            redis_pool_info = self._get_redis_pool_info()
            
            # Count streams by type
            detection_enabled_streams = 0
            detection_disabled_streams = 0
            
            for stream_state in self.active_streams.values():
                if stream_state.config.detection_enabled:
                    detection_enabled_streams += 1
                else:
                    detection_disabled_streams += 1
            
            return {
                **self.stats,
                **redis_pool_info,
                'detection_enabled_streams': detection_enabled_streams,
                'detection_disabled_streams': detection_disabled_streams,
                'service_readiness': readiness_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting service health status: {e}")
            return {
                **self.stats,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_redis_pool_info(self) -> Dict[str, Any]:
        """Safely get Redis connection pool information"""
        try:
            pool = self.redis_pool
            max_connections = pool.connection_kwargs.get('max_connections', 20)
            
            created_connections = getattr(pool, '_created_connections', 0)
            available_connections = getattr(pool, '_available_connections', [])
            
            if hasattr(available_connections, '__len__'):
                available_count = len(available_connections)
            else:
                available_count = 0
            
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

class StreamState:
    """ENHANCED stream state with service readiness integration"""
    
    def __init__(self, config: StreamConfig, redis_client: sync_redis.Redis):
        self.config = config
        self.redis_client = redis_client
        
        # Create ASYNC Redis clients with proper connection management
        self.async_redis_client = None
        self.pubsub_redis_client = None
        
        # Stream management
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.consumers = set()
        
        # Frame processing
        self.frame_processor = AdaptiveFrameProcessor(config)
        self.latest_frame_original = None
        self.latest_frame_with_overlay = None
        self.frame_lock = asyncio.Lock()
        
        # Detection state
        self.last_detection_result = None
        self.detection_subscriber_task = None
        self.has_recent_detection = False
        self.overlay_last_updated = 0
        self.connection_monitor_task = None
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.last_frame_time = 0
        self.overlay_served_count = 0
        self.original_served_count = 0
        
        # HTTP session for camera communication
        self.session = None
        self.frame_producer_task = None
        
        # Enhanced connection tracking
        self.pubsub_connected = False
        self.detection_messages_received = 0
        self.processed_frames_retrieved = 0
        self.connection_attempts = 0
        self.last_connection_attempt = 0
        
        # Service readiness integration
        self.detection_service_ready = False
        self.readiness_check_count = 0
        self.graceful_degradation_active = False
        
        # Connection health monitoring
        self.pubsub = None
        self.subscription_confirmed = False
        self.last_heartbeat = 0
        
    async def initialize(self):
        """Initialize stream state with service readiness checks"""
        try:
            logger.info(f"🚀 Initializing enhanced stream for camera {self.config.camera_id}")
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # CRITICAL: Only initialize detection if service is ready
            if self.config.detection_enabled:
                logger.info(f"🔍 Verifying detection service readiness for camera {self.config.camera_id}")
                
                # Double-check detection service readiness
                detection_ready = await service_readiness_manager.is_detection_service_ready()
                
                if detection_ready:
                    logger.info(f"✅ Detection service confirmed ready, initializing pubsub for camera {self.config.camera_id}")
                    
                    # Create dedicated async Redis clients
                    await self._create_redis_clients()
                    
                    # Start detection result subscriber with confirmation
                    await self._start_pubsub_with_confirmation()
                    
                    # Start connection monitor
                    self.connection_monitor_task = asyncio.create_task(
                        self._connection_monitor()
                    )
                    
                    self.detection_service_ready = True
                    
                else:
                    logger.warning(f"⚠️ Detection service not ready, enabling graceful degradation for camera {self.config.camera_id}")
                    self.config.detection_enabled = False
                    self.graceful_degradation_active = True
                    self.detection_service_ready = False
            
            # Start frame producer (always needed for streaming)
            self.frame_producer_task = asyncio.create_task(
                self._frame_producer()
            )
            
            self.is_active = True
            logger.info(f"✅ Enhanced stream initialized for camera {self.config.camera_id} "
                       f"(detection: {self.config.detection_enabled}, degradation: {self.graceful_degradation_active})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced stream: {e}")
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
                socket_timeout=30,
                max_connections=20
            )
            
            # Dedicated pubsub client with optimized settings
            self.pubsub_redis_client = async_redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=15,
                retry_on_timeout=True,
                socket_connect_timeout=10,
                socket_timeout=60,
                max_connections=5
            )
            
            # Test both connections
            await self.async_redis_client.ping()
            await self.pubsub_redis_client.ping()
            
            logger.info(f"✅ Redis clients created successfully for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create Redis clients: {e}")
            raise
    
    async def _start_pubsub_with_confirmation(self):
        """Start pubsub connection with enhanced readiness verification"""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                self.connection_attempts += 1
                self.last_connection_attempt = time.time()
                
                logger.info(f"🔄 Attempting pubsub connection #{attempt + 1} for camera {self.config.camera_id}")
                
                # CRITICAL: Re-verify detection service is still ready
                if not await service_readiness_manager.is_detection_service_ready():
                    raise Exception("Detection service no longer ready during pubsub setup")
                
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
                    
                    # Test the connection by checking subscriber count
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
        
        # Enable graceful degradation instead of failing completely
        logger.warning(f"🔄 Enabling graceful degradation for camera {self.config.camera_id}")
        self.config.detection_enabled = False
        self.graceful_degradation_active = True
    
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
        """Monitor pubsub connection health with service readiness integration"""
        logger.info(f"🔍 Starting enhanced connection monitor for camera {self.config.camera_id}")
        
        reconnect_interval = 30
        
        while self.is_active and not self.stop_event.is_set():
            try:
                await asyncio.sleep(reconnect_interval)
                
                if not self.is_active:
                    break
                
                # Check if detection service is still ready
                if self.config.detection_enabled and self.pubsub_connected:
                    try:
                        # Quick service readiness check
                        service_ready = await service_readiness_manager.is_detection_service_ready()
                        
                        if not service_ready:
                            logger.warning(f"⚠️ Detection service no longer ready for camera {self.config.camera_id}, enabling graceful degradation")
                            self.config.detection_enabled = False
                            self.graceful_degradation_active = True
                            await self._cleanup_pubsub()
                            continue
                        
                        # Test pubsub connection health
                        await asyncio.wait_for(
                            self.pubsub_redis_client.ping(),
                            timeout=5.0
                        )
                        
                        self.last_heartbeat = time.time()
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Connection health check failed for camera {self.config.camera_id}: {e}")
                        await self._reconnect_pubsub()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in enhanced connection monitor: {e}")
        
        logger.info(f"🛑 Enhanced connection monitor stopped for camera {self.config.camera_id}")
    
    async def _reconnect_pubsub(self):
        """Attempt to reconnect pubsub with service readiness verification"""
        try:
            logger.info(f"🔄 Attempting enhanced pubsub reconnection for camera {self.config.camera_id}")
            
            # CRITICAL: Check if detection service is still ready before reconnecting
            if not await service_readiness_manager.is_detection_service_ready():
                logger.warning(f"⚠️ Detection service not ready during reconnect attempt, enabling graceful degradation")
                self.config.detection_enabled = False
                self.graceful_degradation_active = True
                await self._cleanup_pubsub()
                return
            
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
            
            logger.info(f"✅ Enhanced pubsub reconnection successful for camera {self.config.camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Enhanced pubsub reconnection failed for camera {self.config.camera_id}: {e}")
            # Enable graceful degradation on reconnection failure
            self.config.detection_enabled = False
            self.graceful_degradation_active = True
    
    async def _cleanup_pubsub(self):
        """Clean up pubsub resources"""
        try:
            if self.pubsub:
                try:
                    await self.pubsub.unsubscribe()
                    await asyncio.sleep(0.1)
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
        """Enhanced detection result subscriber with better error handling"""
        logger.info(f"🔊 Starting enhanced detection result subscriber for camera {self.config.camera_id}")
        
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
                    logger.error(f"❌ Error in enhanced detection subscriber: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            logger.info(f"🛑 Enhanced detection subscriber cancelled for camera {self.config.camera_id}")
        except Exception as e:
            logger.error(f"❌ Fatal error in enhanced detection subscriber: {e}")
        finally:
            logger.info(f"🛑 Enhanced detection subscriber stopped for camera {self.config.camera_id}")
    
    async def _handle_detection_result(self, result_data: Dict[str, Any]):
        """Handle incoming detection result with enhanced overlay management"""
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
            
            # Retrieve processed frame with multiple attempts
            processed_frame_data = None
            cache_key = f"processed_frame:{result.camera_id}:{result.session_id}"
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    processed_frame_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.get, cache_key
                    )
                    
                    if processed_frame_data:
                        break
                        
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
                            self.latest_frame_with_overlay = overlay_frame.copy()
                            self.has_recent_detection = True
                            self.overlay_last_updated = time.time()
                            self.processed_frames_retrieved += 1
                            
                        logger.info(f"🎨 OVERLAY UPDATED for camera {self.config.camera_id} - "
                                f"Target detected: {result.detected_target}, "
                                f"Total overlays: {self.processed_frames_retrieved}")
                    else:
                        logger.error(f"❌ Failed to decode processed frame for camera {self.config.camera_id}")
                except Exception as e:
                    logger.error(f"❌ Error decoding processed frame: {e}")
            else:
                logger.warning(f"⚠️ No processed frame found in cache: {cache_key}")
            
            # Log detection events
            if result.detected_target:
                logger.info(f"🎯 TARGET '{self.config.target_label}' DETECTED in camera {self.config.camera_id}!")
            
        except Exception as e:
            logger.error(f"❌ Error handling detection result: {e}")
    
    async def _frame_producer(self):
        """Enhanced frame producer with graceful degradation support"""
        logger.info(f"🎥 Starting enhanced frame producer for camera {self.config.camera_id}")
        
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
                    logger.error(f"Error in enhanced frame producer: {e}")
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.error("Too many consecutive failures, stopping producer")
                        break
                    await asyncio.sleep(1.0)
        
        finally:
            logger.info(f"🛑 Enhanced frame producer stopped for camera {self.config.camera_id}")
    
    async def _process_frame(self, frame: np.ndarray):
        """Enhanced frame processing with service readiness checks"""
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
            
            # Send for detection only if enabled and service is ready
            if (self.config.detection_enabled and 
                not self.graceful_degradation_active and
                self.frame_processor.should_process_for_detection()):
                
                # Double-check detection service readiness before sending frames
                if self.detection_service_ready and self.pubsub_connected:
                    await self._send_frame_for_detection(frame)
                else:
                    # Service might have become unavailable, check occasionally
                    if self.frame_count % 50 == 0:  # Check every 50 frames
                        logger.debug(f"🔍 Periodic service readiness check for camera {self.config.camera_id}")
                        await self._attempt_detection_service_recovery()
                
        except Exception as e:
            logger.error(f"Error in enhanced frame processing: {e}")
    
    async def _attempt_detection_service_recovery(self):
        """Attempt to recover detection service connection"""
        try:
            self.readiness_check_count += 1
            
            # Check if detection service is now ready
            service_ready = await service_readiness_manager.is_detection_service_ready()
            
            if service_ready and self.graceful_degradation_active:
                logger.info(f"🔄 Detection service recovered, attempting to re-enable detection for camera {self.config.camera_id}")
                
                # Try to re-enable detection
                self.config.detection_enabled = True
                self.graceful_degradation_active = False
                
                # Recreate Redis clients and pubsub connection
                try:
                    await self._create_redis_clients()
                    await self._start_pubsub_with_confirmation()
                    
                    # Restart connection monitor
                    if self.connection_monitor_task is None or self.connection_monitor_task.done():
                        self.connection_monitor_task = asyncio.create_task(
                            self._connection_monitor()
                        )
                    
                    self.detection_service_ready = True
                    logger.info(f"✅ Detection service recovery successful for camera {self.config.camera_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Detection service recovery failed: {e}")
                    self.config.detection_enabled = False
                    self.graceful_degradation_active = True
                    self.detection_service_ready = False
            
        except Exception as e:
            logger.error(f"❌ Error during detection service recovery attempt: {e}")
    
    async def _send_frame_for_detection(self, frame: np.ndarray):
        """Enhanced frame sending with service readiness verification"""
        try:
            # CRITICAL: Verify pubsub connection before sending
            if not self.pubsub_connected or not self.subscription_confirmed:
                logger.debug(f"⏸️ Skipping detection for camera {self.config.camera_id} - no active pubsub connection")
                return
            
            # Encode frame as bytes
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                return
            
            # Create detection request
            request_data = {
                'camera_id': self.config.camera_id,
                'frame_data': buffer.tobytes(),
                'target_label': self.config.target_label,
                'timestamp': time.time(),
                'session_id': self.config.session_id,
                'priority': 1
            }
            
            # Send to Redis queue using sync client
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
                logger.info(f"📤 Sent frame #{self.frame_count} for detection "
                           f"(camera {self.config.camera_id}, queue: {queue_length}, subscribers: {subscriber_count})")
            
        except Exception as e:
            logger.error(f"Error sending frame for detection: {e}")
            # Check if this might be a service readiness issue
            if "connection" in str(e).lower() or "redis" in str(e).lower():
                logger.warning(f"⚠️ Connection issue detected, may need to check service readiness")
    
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
        """Add a consumer with enhanced readiness verification"""
        self.consumers.add(consumer_id)
        logger.info(f"👤 Added consumer {consumer_id} to camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # ENHANCED: Ensure detection is available when first consumer joins
        if len(self.consumers) == 1 and self.graceful_degradation_active:
            logger.info(f"🔄 First consumer joined, attempting detection service recovery for camera {self.config.camera_id}")
            await self._attempt_detection_service_recovery()
    
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"👤 Removed consumer {consumer_id} from camera {self.config.camera_id}. Total: {len(self.consumers)}")
        
        # Stop stream if no consumers
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for camera {self.config.camera_id}, stopping stream")
            await self.cleanup()
    
    async def get_latest_frame(self) -> Optional[bytes]:
        """Enhanced frame retrieval with graceful degradation status"""
        async with self.frame_lock:
            current_time = time.time()
            
            # Enhanced overlay freshness check
            overlay_freshness_limit = 10.0
            
            # Check if we have a recent detection overlay (only if detection is enabled)
            if (not self.graceful_degradation_active and
                self.has_recent_detection and 
                self.latest_frame_with_overlay is not None and
                self.last_detection_result and
                current_time - self.overlay_last_updated < overlay_freshness_limit):
                
                try:
                    # Use higher quality for overlay frames
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                    success, buffer = cv2.imencode('.jpg', self.latest_frame_with_overlay, encode_params)
                    if success:
                        self.overlay_served_count += 1
                        
                        # Log overlay serving
                        if self.overlay_served_count % 10 == 1:
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
                
                # Log original frame serving with degradation status
                if self.original_served_count % 25 == 1:
                    total_served = self.overlay_served_count + self.original_served_count
                    overlay_ratio = (self.overlay_served_count / total_served * 100) if total_served > 0 else 0
                    degradation_msg = " (GRACEFUL DEGRADATION)" if self.graceful_degradation_active else ""
                    
                    logger.info(f"📺 SERVING ORIGINAL frame for camera {self.config.camera_id}{degradation_msg} "
                            f"- Overlay ratio: {overlay_ratio:.1f}% "
                            f"({self.overlay_served_count}/{total_served})")
                
                return self.latest_frame_original
            
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Enhanced performance statistics with service readiness info"""
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
            'subscription_confirmed': self.subscription_confirmed,
            'detection_messages_received': self.detection_messages_received,
            'processed_frames_retrieved': self.processed_frames_retrieved,
            'connection_attempts': self.connection_attempts,
            'last_heartbeat': self.last_heartbeat,
            # Enhanced service readiness fields
            'detection_service_ready': self.detection_service_ready,
            'graceful_degradation_active': self.graceful_degradation_active,
            'readiness_check_count': self.readiness_check_count,
            'detection_enabled': self.config.detection_enabled
        }
    
    async def cleanup(self):
        """Enhanced cleanup with service readiness manager integration"""
        logger.info(f"🧹 Cleaning up enhanced stream for camera {self.config.camera_id}")
        
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
        
        logger.info(f"✅ Enhanced stream cleanup complete for camera {self.config.camera_id}")

# Global enhanced stream manager - FIXED instantiation
optimized_stream_manager = GlobalStreamManager()

async def generate_enhanced_video_frames_with_detection(
    camera_id: int,
    target_label: str,
    consumer_id: str = None
) -> AsyncGenerator[bytes, None]:
    """
    Generate video frames using the enhanced Redis-based architecture with service readiness pattern
    """
    
    if consumer_id is None:
        consumer_id = f"enhanced_consumer_{uuid.uuid4().hex[:8]}"
    
    # Create stream configuration
    config = StreamConfig(
        camera_id=camera_id,
        target_label=target_label,
        detection_enabled=bool(target_label),
        detection_frequency=5.0,
        session_id=str(uuid.uuid4())
    )
    
    stream_key = None
    
    try:
        logger.info(f"🚀 Starting ENHANCED stream for camera {camera_id}, target: '{target_label}', consumer: {consumer_id}")
        
        # CRITICAL: Wait for detection service to be ready if detection is enabled
        if config.detection_enabled:
            logger.info(f"⏳ Waiting for detection service readiness...")
            service_ready = await service_readiness_manager.wait_for_service_ready(timeout_seconds=30)
            
            if not service_ready:
                logger.warning(f"⚠️ Detection service not ready after 30s, continuing with graceful degradation")
                # Don't fail completely, just disable detection
                config.detection_enabled = False
        
        # Create stream with readiness check
        stream_key = await optimized_stream_manager.create_stream_with_readiness_check(config)
        stream_state = optimized_stream_manager.get_stream(stream_key)
        
        if not stream_state:
            logger.error(f"❌ Failed to create enhanced stream for camera {camera_id}")
            return
        
        # Add consumer
        await stream_state.add_consumer(consumer_id)
        
        # Stream frames
        target_fps = config.target_fps
        frame_time = 1.0 / target_fps
        
        frame_count = 0
        overlay_count = 0
        no_frame_count = 0
        max_no_frame = 100
        
        logger.info(f"🎬 Starting ENHANCED frame streaming for camera {camera_id}")
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                frame_bytes = await stream_state.get_latest_frame()
                
                if frame_bytes is not None:
                    no_frame_count = 0
                    
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    frame_count += 1
                    
                    # Count overlay frames
                    if (not stream_state.graceful_degradation_active and
                        stream_state.has_recent_detection and 
                        stream_state.last_detection_result and
                        time.time() - stream_state.last_detection_result.timestamp < 3.0):
                        overlay_count += 1
                    
                    # Update manager stats
                    optimized_stream_manager.stats['total_frames_streamed'] += 1
                    if config.detection_enabled and not stream_state.graceful_degradation_active:
                        optimized_stream_manager.stats['total_frames_detected'] += 1
                        
                    # Enhanced logging with service status
                    if frame_count % 100 == 0:
                        overlay_ratio = (overlay_count / frame_count * 100) if frame_count > 0 else 0
                        subscriber_count = await stream_state._get_subscriber_count()
                        degradation_status = "DEGRADATION" if stream_state.graceful_degradation_active else "NORMAL"
                        
                        logger.info(f"📊 ENHANCED Stream stats - Camera {camera_id}: "
                                   f"frames={frame_count}, overlays={overlay_count}, "
                                   f"overlay_ratio={overlay_ratio:.1f}%, subscribers={subscriber_count}, "
                                   f"status={degradation_status}, detection_ready={stream_state.detection_service_ready}")
                else:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.error(f"❌ No frames received for {max_no_frame} attempts, stopping enhanced stream for camera {camera_id}")
                        break
                    await asyncio.sleep(0.05)
                    continue
                
                # Frame rate control
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Enhanced stream cancelled for camera {camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error in enhanced frame generation: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"❌ Error in generate_enhanced_video_frames_with_detection: {e}")
    finally:
        # Cleanup
        if stream_key and stream_state:
            await stream_state.remove_consumer(consumer_id)
        
        # Get final service readiness status for logging
        service_status = service_readiness_manager.get_service_status()
        
        logger.info(f"🏁 ENHANCED video stream stopped for camera {camera_id}, consumer {consumer_id} "
                   f"(streamed {frame_count} frames, {overlay_count} with overlays, "
                   f"service_ready: {service_status['detection_service']['is_ready']})")

# Enhanced diagnostic functions
async def diagnose_enhanced_pubsub_connection(camera_id: int) -> Dict[str, Any]:
    """Enhanced diagnostic function with service readiness integration"""
    try:
        # Get service readiness status
        readiness_status = service_readiness_manager.get_service_status()
        
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
            'message': 'Enhanced diagnostic test message'
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
            'service_readiness': readiness_status,
            'detection_service_ready': readiness_status['detection_service']['is_ready'],
            'circuit_breaker_open': readiness_status['circuit_breaker']['is_open'],
            'timestamp': time.time()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'camera_id': camera_id,
            'error': str(e),
            'redis_connected': False,
            'service_readiness': service_readiness_manager.get_service_status(),
            'timestamp': time.time()
        }

async def force_enhanced_service_recovery(camera_id: int = None) -> Dict[str, Any]:
    """Force service recovery with circuit breaker reset"""
    try:
        logger.info(f"🔄 Forcing enhanced service recovery (camera: {camera_id or 'all'})")
        
        # Reset circuit breaker
        await service_readiness_manager.force_circuit_breaker_reset()
        
        # Force readiness check
        service_ready = await service_readiness_manager.is_detection_service_ready(force_check=True)
        
        recovery_actions = ["Circuit breaker reset", "Forced readiness check"]
        
        if camera_id:
            # Find and recover specific camera stream
            for stream_key, stream_state in optimized_stream_manager.active_streams.items():
                if stream_state.config.camera_id == camera_id:
                    await stream_state._attempt_detection_service_recovery()
                    recovery_actions.append(f"Attempted recovery for camera {camera_id}")
                    break
        else:
            # Recover all streams
            recovery_count = 0
            for stream_state in optimized_stream_manager.active_streams.values():
                try:
                    await stream_state._attempt_detection_service_recovery()
                    recovery_count += 1
                except Exception as e:
                    logger.error(f"Error recovering stream for camera {stream_state.config.camera_id}: {e}")
            
            recovery_actions.append(f"Attempted recovery for {recovery_count} streams")
        
        return {
            'status': 'success',
            'service_ready': service_ready,
            'recovery_actions': recovery_actions,
            'service_readiness': service_readiness_manager.get_service_status(),
            'timestamp': time.time()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }